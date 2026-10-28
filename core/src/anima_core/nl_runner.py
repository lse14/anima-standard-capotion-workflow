from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Callable, Protocol

from .classify_overlay import ClassifyJsonError, parse_annotation_json
from .contracts import (
    CURRENT_JOB_CONFIG_SCHEMA_VERSION,
    MODULE_BATCH_SIZE_BOUNDS,
    NL_MOSS_CAPTION_PRESETS,
    NL_MOSS_PROMPT_VERSION,
    NL_MOSS_RESPONSE_PROTOCOL,
    NL_STANDARD_CAPTION_PRESETS,
    NL_V4_PROMPT_VERSION,
    SampleIssue,
    WorkLease,
    job_config_supports_caption_input_txt_mode,
    job_config_supports_nl_v4,
    job_config_supports_ocr,
    job_config_supports_ocr_device,
    sha256_json,
)
from .db import StateDatabase
from .nl_length import character_name, stable_length_tier
from .nl_overlay import NlOverlayWriter
from .nl_protocol import NlOutcomeV1, NlProtocolError, parse_outcomes
from .ocr_overlay import OcrWorkingSidecarView
from .ocr_sidecar import OcrSidecarError, compact_ocr_context, parse_ocr_sidecar, with_llm_threshold
from .overlay import OverlayError, WorkingAnnotationView
from .path_safety import PathSafetyError, ensure_within, safe_relative_path
from .scheduler import BoundedScheduler, SchedulerError
from .stdio_transport import StdioJsonlTransportError
from .worker_protocol import ProtocolEnvelopeV1, ProtocolError


RUNTIME_ID = "nl"
OWNER = "nl"
NONBLOCKING_TERMINAL_NL_CODES = frozenset({"nl_api_unavailable", "nl_response_invalid", "nl_response_format_invalid"})
DEFAULT_POLICY = {
    "concurrency": 3, "maxRequestsPerMinute": 60, "maxHttpAttempts": 10_000,
    "mainAttempts": 3, "backupEnabled": False, "backupAttempts": 2,
    "connectTimeoutSeconds": 10, "writeTimeoutSeconds": 30, "readTimeoutSeconds": 120,
    "poolTimeoutSeconds": 30, "temperature": 0.7, "topP": 0.95, "maxTokens": 2048,
    "maxImagePixels": 8_000_000, "maxImageSide": 4096, "jpegQuality": 95,
    "maxEncodedImageBytes": 12_582_912, "maxJsonContextBytes": 262_144,
    "maxResponseBodyBytes": 1_048_576, "maxNlBytes": 16_384,
}
WORKER_POLICY_KEYS = frozenset(DEFAULT_POLICY) - {"maxHttpAttempts"}


def build_short_rewrite_item(
    *,
    sample_id: int,
    lease_id: str,
    relative_image_path: str,
    image_path: str | None,
    caption_preset: str,
    primary_character_name: str | None,
    user_supplement: str,
    json_context: str | None,
    ocr_context: dict[str, object] | None,
    current_nl: str,
) -> dict[str, object]:
    """Build one v4 rewrite request without allowing dynamic text into instructions."""
    maximum_context_bytes = DEFAULT_POLICY["maxJsonContextBytes"]
    if type(maximum_context_bytes) is not int:
        raise ValueError("short rewrite context bound is invalid")
    if type(sample_id) is not int or sample_id < 1:
        raise ValueError("short rewrite sample identity is invalid")
    if not isinstance(lease_id, str) or not lease_id or "\x00" in lease_id or len(lease_id.encode("utf-8")) > 128:
        raise ValueError("short rewrite lease identity is invalid")
    try:
        relative_image_path = safe_relative_path(relative_image_path)
    except (PathSafetyError, TypeError) as exc:
        raise ValueError("short rewrite image path is invalid") from exc
    if image_path is not None and (
        not isinstance(image_path, str)
        or "\x00" in image_path
        or len(image_path.encode("utf-8")) > 16_384
        or not PureWindowsPath(image_path).is_absolute()
    ):
        raise ValueError("short rewrite image path is invalid")
    if caption_preset not in NL_STANDARD_CAPTION_PRESETS | NL_MOSS_CAPTION_PRESETS:
        raise ValueError("short rewrite caption preset is invalid")
    if caption_preset in {"character", "moss-character"}:
        if not isinstance(primary_character_name, str) or not primary_character_name or len(primary_character_name.encode("utf-8")) > 512:
            raise ValueError("short rewrite primary character name is invalid")
    elif primary_character_name is not None:
        raise ValueError("short rewrite primary character name is invalid")
    if not isinstance(user_supplement, str) or "\x00" in user_supplement or len(user_supplement.encode("utf-8")) > 16_384:
        raise ValueError("short rewrite user supplement is invalid")
    if not isinstance(current_nl, str) or "\x00" in current_nl or len(current_nl.encode("utf-8")) > 16_384:
        raise ValueError("short rewrite current NL is invalid")
    if json_context is not None and (not isinstance(json_context, str) or len(json_context.encode("utf-8")) > maximum_context_bytes):
        raise ValueError("short rewrite JSON context is invalid")
    try:
        json_value = json.loads(json_context, parse_constant=lambda _: (_ for _ in ()).throw(ValueError())) if json_context is not None else None
        untrusted_context = json.dumps(
            {"jsonContext": json_value, "currentNl": current_nl},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("short rewrite context is invalid") from exc
    if len(untrusted_context.encode("utf-8")) > maximum_context_bytes:
        raise ValueError("short rewrite JSON context is invalid")
    if ocr_context is not None:
        positions = {
            "top-left", "top-center", "top-right", "middle-left", "middle-center", "middle-right",
            "bottom-left", "bottom-center", "bottom-right",
        }
        raw_items = ocr_context.get("items") if isinstance(ocr_context, dict) and set(ocr_context) == {"items"} else None
        if not isinstance(raw_items, list) or len(raw_items) > 1024:
            raise ValueError("short rewrite OCR context is invalid")
        for entry in raw_items:
            if (
                not isinstance(entry, list) or len(entry) != 2 or entry[0] not in positions
                or not isinstance(entry[1], str) or not entry[1] or "\x00" in entry[1]
                or len(entry[1].encode("utf-8")) > 16_384
            ):
                raise ValueError("short rewrite OCR context is invalid")
    combined_context = json.dumps(
        {"jsonContext": untrusted_context, "ocrContext": ocr_context},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(combined_context) > maximum_context_bytes:
        ocr_context = None
        combined_context = json.dumps(
            {"jsonContext": untrusted_context, "ocrContext": ocr_context},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(combined_context) > maximum_context_bytes:
            raise ValueError("short rewrite JSON context is invalid")
    return {
        "schemaVersion": 1,
        "sampleId": sample_id,
        "leaseId": lease_id,
        "relativeImagePath": relative_image_path,
        "imagePath": image_path,
        "captionPreset": caption_preset,
        "lengthTier": "short",
        "primaryCharacterName": primary_character_name,
        "userSupplement": user_supplement,
        "jsonContext": untrusted_context,
        "ocrContext": ocr_context,
    }


def nl_http_attempt_budget(candidate_count: int) -> int:
    """ROADMAP.md:833 default per-job hard budget: candidateCount + ceil(candidateCount * 5%)."""
    if candidate_count < 0:
        raise ValueError("NL candidate count must not be negative")
    return candidate_count + math.ceil(candidate_count * 0.05)


def nl_request_projection(candidate_count: int, policy: dict[str, object] | None = None, *, upload_bytes: int = 0) -> dict[str, int]:
    """Preflight projection: minimum / worst-case request counts and estimated upload bytes."""
    values = dict(DEFAULT_POLICY) | dict(policy or {})
    main = int(values["mainAttempts"])
    backup = int(values["backupAttempts"]) if values["backupEnabled"] is True else 0
    return {
        "candidateCount": candidate_count,
        "minimumRequests": candidate_count,
        "worstCaseRequests": candidate_count * (main + backup),
        "maxHttpAttempts": max(1, nl_http_attempt_budget(candidate_count)),
        "estimatedUploadBytes": upload_bytes,
    }


def pending_api_decisions(database: StateDatabase, job_id: str) -> int:
    """F28: NL requests frozen at request_started await explicit confirmation (possible double billing)."""
    return int(database.connection.execute(
        "SELECT COUNT(*) FROM sample_state WHERE job_id=? AND current_module_id='nl' AND status='request_started'",
        (job_id,),
    ).fetchone()[0])


class NlTransport(Protocol):
    def exchange(self, request: ProtocolEnvelopeV1) -> ProtocolEnvelopeV1: ...


class NlRunnerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NlApiCredentials:
    endpoint: str
    model: str
    apiKey: str
    backupModel: str | None = None
    protocol: str = "openai_chat_completions"
    providerLabel: str = ""
    jsonMode: str = "prompt_only"
    revision: int = 1
    requestPolicy: dict[str, object] | None = None


class NlRunner:
    """Core-owned bounded NL orchestration; API text exists only in staged_nl until overlay commit."""

    def __init__(
        self, database: StateDatabase, scheduler: BoundedScheduler, transport: NlTransport,
        view: WorkingAnnotationView, writer: NlOverlayWriter, *, job_id: str, worker_instance_id: str,
        credentials: NlApiCredentials | None = None,
        runtime_revision_provider: Callable[[], NlApiCredentials | None] | None = None,
        runtime_revision_applied: Callable[[NlApiCredentials], None] | None = None,
    ) -> None:
        self.database = database
        self.scheduler = scheduler
        self.transport = transport
        self.view = view
        self.writer = writer
        self.job_id = job_id
        self.worker_instance_id = worker_instance_id
        self.credentials = credentials
        self.runtime_revision_provider = runtime_revision_provider
        self.runtime_revision_applied = runtime_revision_applied
        self._hello_done = False
        self._nl_batch_size = 3
        self._pending_credentials: NlApiCredentials | None = None
        self._credentials_before_pending: NlApiCredentials | None = credentials

    def _fatal(self, code: str, message: str) -> NlRunnerError:
        return NlRunnerError(code, message)

    def _config(self) -> tuple[str, dict[str, object], Path, int, dict[str, object] | None, str | None]:
        job = self.database.get_job(self.job_id)
        try:
            config = json.loads(str(job["config_json"]))
        except json.JSONDecodeError as exc:
            raise self._fatal("nl_protocol_violation", "frozen JobConfig is invalid JSON") from exc
        config_schema_version = config.get("schemaVersion") if isinstance(config, dict) else None
        expected_prompt_version = (
            "nl-default-prompt-v1" if config_schema_version == 2
            else NL_V4_PROMPT_VERSION if job_config_supports_nl_v4(config_schema_version)
            else "nl-default-prompt-v3" if config_schema_version == 5
            else "nl-default-prompt-v2"
        )
        actual_prompt_version = config.get("nl", {}).get("promptVersion") if isinstance(config, dict) and isinstance(config.get("nl"), dict) else None
        prompt_version_valid = (
            actual_prompt_version in {expected_prompt_version, NL_MOSS_PROMPT_VERSION}
            if job_config_supports_nl_v4(config_schema_version)
            else actual_prompt_version == expected_prompt_version
        )
        if (
            not isinstance(config, dict)
            or sha256_json(config) != job["config_hash"]
            or config_schema_version != CURRENT_JOB_CONFIG_SCHEMA_VERSION
            or "profile" in config
            or not isinstance(config.get("nl"), dict)
            or config_schema_version != int(job["config_schema_version"])
            or not prompt_version_valid
        ):
            raise self._fatal("nl_protocol_violation", "frozen NL configuration is invalid")
        batches = config.get("moduleBatchSize")
        if (
            not isinstance(batches, dict)
            or set(batches) != set(MODULE_BATCH_SIZE_BOUNDS)
            or type(batches.get("nl")) is not int
            or not MODULE_BATCH_SIZE_BOUNDS["nl"][0] <= batches["nl"] <= MODULE_BATCH_SIZE_BOUNDS["nl"][1]
        ):
            raise self._fatal("nl_protocol_violation", "frozen module batch configuration is invalid")
        self._nl_batch_size = int(batches["nl"])
        raw_api_policy = config["nl"].get("apiPolicy") if isinstance(config.get("nl"), dict) else None
        if isinstance(raw_api_policy, dict) and "concurrency" in raw_api_policy:
            raise self._fatal("nl_protocol_violation", "NL concurrency is controlled by moduleBatchSize.nl")
        input_txt_mode: str | None = None
        if job_config_supports_caption_input_txt_mode(config_schema_version):
            caption = config.get("caption")
            if (
                not isinstance(caption, dict)
                or caption.get("inputTxtMode") not in {"tag", "nl"}
                or type(caption.get("taggerFallbackOnMissingTxt")) is not bool
            ):
                raise self._fatal("nl_protocol_violation", "frozen caption TXT input mode is invalid")
            input_txt_mode = caption["inputTxtMode"]
        ocr: dict[str, object] | None = None
        if job_config_supports_ocr(config_schema_version):
            candidate = config.get("ocr")
            if (
                not isinstance(candidate, dict)
                or not {"enabled", "llmMinConfidence", "forceReprocess", "resourceId"}.issubset(candidate)
                or set(candidate) - {"enabled", "llmMinConfidence", "forceReprocess", "resourceId", "resourceManifestRelativePath", "resourceFingerprint", "device"}
                or type(candidate["enabled"]) is not bool
                or type(candidate["forceReprocess"]) is not bool
                or isinstance(candidate["llmMinConfidence"], bool)
                or not isinstance(candidate["llmMinConfidence"], (int, float))
                or not math.isfinite(float(candidate["llmMinConfidence"]))
                or not 0 <= float(candidate["llmMinConfidence"]) <= 1
                or (job_config_supports_ocr_device(config_schema_version) and candidate.get("device") not in {"auto", "cuda", "cpu"})
                or (not job_config_supports_ocr_device(config_schema_version) and "device" in candidate)
            ):
                raise self._fatal("nl_protocol_violation", "frozen OCR configuration is invalid")
            ocr = dict(candidate)
        return (
            str(job["config_hash"]),
            dict(config["nl"]),
            Path(str(job["dataset_root"])),
            config_schema_version,
            ocr,
            input_txt_mode,
        )

    def _exchange(self, method: str, payload: dict[str, object], config_hash: str) -> dict[str, object]:
        request = ProtocolEnvelopeV1(
            protocolVersion="1.0", kind="request", messageId=f"nl-{uuid.uuid4().hex}", runtimeId=RUNTIME_ID,
            owner=OWNER, method=method, payload=payload, jobId=self.job_id, configHash=config_hash,
        )
        try:
            response = self.transport.exchange(request)
        except StdioJsonlTransportError:
            raise
        except Exception as exc:
            raise self._fatal("nl_protocol_violation", "NL transport failed") from exc
        if not isinstance(response, ProtocolEnvelopeV1) or response.kind != "response" or response.replyTo != request.messageId or response.runtimeId != RUNTIME_ID or response.owner != OWNER or response.jobId != self.job_id or response.configHash != config_hash:
            raise self._fatal("nl_protocol_violation", "NL response envelope identity mismatch")
        if response.method == "error":
            raise self._fatal("nl_protocol_violation", "NL worker returned a fatal protocol error")
        if response.method != ("hello" if method == "hello" else "result"):
            raise self._fatal("nl_protocol_violation", "NL worker response method mismatch")
        return response.payload

    def _hello(self, config_hash: str, nl: dict[str, object], *, response_protocol: str, prompt_version: str) -> None:
        if self.credentials is None:
            raise self._fatal("nl_credentials_unavailable", "API-enabled NL module has no resolved credential")
        policy = self._policy(nl)
        request_payload: dict[str, object] = {
            "schemaVersion": 1, "payloadType": "nl_hello_request", "jobId": self.job_id, "configHash": config_hash,
            "endpoint": self.credentials.endpoint, "model": self.credentials.model, "backupModel": self.credentials.backupModel,
            "apiKey": self.credentials.apiKey, "systemPrompt": nl.get("systemPrompt"),
            "apiPolicy": {key: policy[key] for key in WORKER_POLICY_KEYS},
            "protocol": self.credentials.protocol, "providerLabel": self.credentials.providerLabel,
            "jsonMode": self.credentials.jsonMode,
        }
        if response_protocol in {"nl-count-v2", NL_MOSS_RESPONSE_PROTOCOL}:
            request_payload["responseProtocol"] = response_protocol
            if prompt_version in {"nl-default-prompt-v3", NL_V4_PROMPT_VERSION, NL_MOSS_PROMPT_VERSION}:
                request_payload["promptVersion"] = prompt_version
        payload = self._exchange("hello", request_payload, config_hash)
        if payload != {"schemaVersion": 1, "payloadType": "nl_hello_result", "ready": True, "concurrency": policy.get("concurrency", 3)}:
            raise self._fatal("nl_protocol_violation", "NL hello result is invalid")
        self._hello_done = True

    def _refresh_runtime_revision(self) -> None:
        if self.credentials is None or self.runtime_revision_provider is None:
            return
        candidate = self.runtime_revision_provider()
        if candidate is None or candidate.revision <= self.credentials.revision:
            return
        self._credentials_before_pending = self.credentials
        self._pending_credentials = candidate
        self.credentials = candidate
        self._hello_done = False

    def _confirm_runtime_revision(self) -> None:
        if self._pending_credentials is None:
            return
        candidate = self._pending_credentials
        if self.runtime_revision_applied is not None:
            self.runtime_revision_applied(candidate)
        self._pending_credentials = None

    def _policy(self, nl: dict[str, object]) -> dict[str, object]:
        value = self.credentials.requestPolicy if self.credentials and self.credentials.requestPolicy is not None else nl.get("apiPolicy", {})
        if not isinstance(value, dict) or set(value) - set(DEFAULT_POLICY) or "concurrency" in value:
            raise self._fatal("nl_protocol_violation", "NL API policy is invalid")
        policy = {key: value.get(key, default) for key, default in DEFAULT_POLICY.items()}
        policy["concurrency"] = self._nl_batch_size
        if "maxHttpAttempts" not in value:
            # F30: an unfrozen budget follows the dataset size instead of a flat constant.
            policy["maxHttpAttempts"] = max(1, nl_http_attempt_budget(int(self.database.get_job(self.job_id)["sample_count"])))
        if type(policy["concurrency"]) is not int or not 1 <= policy["concurrency"] <= 16:
            raise self._fatal("nl_protocol_violation", "NL concurrency is invalid")
        rpm = policy["maxRequestsPerMinute"]
        if rpm != "unlimited" and (type(rpm) is not int or not 1 <= rpm <= 100_000):
            raise self._fatal("nl_protocol_violation", "NL RPM is invalid")
        for name, maximum in (("maxHttpAttempts", 10_000_000), ("mainAttempts", 3), ("backupAttempts", 2), ("connectTimeoutSeconds", 120), ("writeTimeoutSeconds", 120), ("readTimeoutSeconds", 600), ("poolTimeoutSeconds", 120), ("maxTokens", 16_384)):
            if type(policy[name]) is not int or not 1 <= policy[name] <= maximum:
                raise self._fatal("nl_protocol_violation", f"NL policy {name} is invalid")
        if type(policy["backupEnabled"]) is not bool:
            raise self._fatal("nl_protocol_violation", "NL backup policy is invalid")
        for name, expected in (("maxImagePixels", 8_000_000), ("maxImageSide", 4096), ("jpegQuality", 95), ("maxEncodedImageBytes", 12_582_912), ("maxJsonContextBytes", 262_144), ("maxResponseBodyBytes", 1_048_576), ("maxNlBytes", 16_384)):
            if policy[name] != expected:
                raise self._fatal("nl_protocol_violation", f"NL policy {name} is frozen")
        for name in ("temperature", "topP"):
            if isinstance(policy[name], bool) or not isinstance(policy[name], (int, float)) or not 0 <= float(policy[name]) <= 2:
                raise self._fatal("nl_protocol_violation", f"NL policy {name} is invalid")
        return policy

    def _pause(self, code: str) -> None:
        self.database.increment_module_diagnostic(self.job_id, "nl", code, severity="warning", amount=1)
        self.database.set_module_summary(self.job_id, "nl", status="paused")
        self.database.set_job_status(self.job_id, "paused", current_module_id="nl", resume_status="running")

    def _issue(self, lease: WorkLease, row: object, code: str, message: str, *, severity: str = "error", blocking: bool = True, retriable: bool = False, repair_start: str | None = None, allowed_statuses: tuple[str, ...] = ("leased",)) -> None:
        self.scheduler.fail_with_issue(lease, SampleIssue(
            issueId=hashlib.sha256(f"{self.job_id}\0{lease.sampleId}\0nl\0{code}".encode("utf-8")).hexdigest(), jobId=self.job_id,
            sampleId=lease.sampleId, relativeImagePath=str(row["relative_image_path"]), moduleId="nl", code=code,
            severity=severity, blocking=blocking, retriable=retriable, message=message[:1024], attempt=lease.attempt,
            repairStartModule=repair_start,  # type: ignore[arg-type]
        ), allowed_statuses=allowed_statuses)

    def _not_requested(self, lease: WorkLease, reason: str) -> None:
        self.database.record_count_observation_not_requested(
            self.job_id,
            lease.sampleId,
            lease_id=str(lease.leaseId),
            reason=reason,
        )

    def _baseline_json(self, row: object) -> tuple[bytes | None, str | None]:
        """NL-09 / ROADMAP.md:768: verify the manifest fingerprint before reading the baseline JSON."""
        raw = self.view.baseline.read(str(row["annotation_key"]), ".json")
        expected = row["original_json_sha256"]
        if row["original_json_state"] == "nonblank" and isinstance(expected, str) and len(expected) == 64:
            if raw is None or hashlib.sha256(raw).hexdigest() != expected:
                return None, "baseline JSON no longer matches its manifest fingerprint"
        return raw, None

    @staticmethod
    def _original_nl(raw: bytes | None) -> tuple[str | None, str | None]:
        try:
            original = parse_annotation_json(raw)
        except ClassifyJsonError:
            return None, "baseline JSON is invalid"
        if original is None or "nl" not in original:
            return None, None
        value = original["nl"]
        if not isinstance(value, str):
            return None, "baseline NL must be a string"
        return value if value.strip() else None, None

    @staticmethod
    def _canonical_context_bytes(json_context: str | None, ocr_context: dict[str, object] | None) -> bytes:
        return json.dumps(
            {"jsonContext": json_context, "ocrContext": ocr_context},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def _ocr_context(self, relative_image_path: str, ocr: dict[str, object]) -> dict[str, object] | None:
        if ocr["enabled"] is not True:
            return None
        try:
            raw = OcrWorkingSidecarView(self.view.baseline.dataset_root, self.view.overlay).read_bytes(relative_image_path)
        except (OSError, OverlayError):
            self.database.increment_module_diagnostic(self.job_id, "nl", "nl_ocr_sidecar_invalid", severity="warning", amount=1)
            return None
        if raw is None:
            self.database.increment_module_diagnostic(self.job_id, "nl", "nl_ocr_sidecar_missing", severity="warning", amount=1)
            return None
        try:
            sidecar = parse_ocr_sidecar(raw, expected_relative_image_path=relative_image_path)
        except OcrSidecarError:
            self.database.increment_module_diagnostic(self.job_id, "nl", "nl_ocr_sidecar_invalid", severity="warning", amount=1)
            return None
        if sidecar.status == "failed":
            self.database.increment_module_diagnostic(self.job_id, "nl", "nl_ocr_sidecar_failed", severity="warning", amount=1)
            return None
        try:
            return compact_ocr_context(with_llm_threshold(sidecar, float(ocr["llmMinConfidence"])))
        except OcrSidecarError:
            self.database.increment_module_diagnostic(self.job_id, "nl", "nl_ocr_sidecar_invalid", severity="warning", amount=1)
            return None

    def _worker_item(
        self,
        lease: WorkLease,
        row: object,
        projection: dict[str, object],
        *,
        use_image: bool,
        use_json: bool,
        dataset_root: Path,
        config_schema_version: int,
        ocr: dict[str, object] | None,
        nl: dict[str, object],
    ) -> dict[str, object]:
        image_path: str | None = None
        if use_image:
            try:
                image_path = str(ensure_within(dataset_root, dataset_root / str(row["relative_image_path"])))
            except PathSafetyError as exc:
                raise self._fatal("nl_protocol_violation", "leased image path escapes dataset root") from exc
        context: str | None = None
        if use_json:
            copy = dict(projection)
            copy["nl"] = ""
            context = json.dumps(copy, ensure_ascii=False, separators=(",", ":"))
            if len(context.encode("utf-8")) > 262_144:
                raise self._fatal("nl_context_too_large", "working JSON context exceeds 256 KiB")
        item = {
            "schemaVersion": 1,
            "sampleId": lease.sampleId,
            "leaseId": lease.leaseId,
            "relativeImagePath": row["relative_image_path"],
            "imagePath": image_path,
            "jsonContext": context,
        }
        if job_config_supports_nl_v4(config_schema_version):
            try:
                preset = str(nl["captionPreset"])
                expected_presets = NL_MOSS_CAPTION_PRESETS if nl.get("promptVersion") == NL_MOSS_PROMPT_VERSION else NL_STANDARD_CAPTION_PRESETS
                if preset not in expected_presets:
                    raise ValueError("NL caption preset family does not match prompt version")
                distribution = dict(nl["lengthDistribution"])
                tier = stable_length_tier(
                    seed=str(nl["lengthSeed"]),
                    relative_image_path=str(row["relative_image_path"]),
                    distribution=distribution,
                )
                primary = character_name(str(row["relative_image_path"])) if preset in {"character", "moss-character"} else None
                if primary is not None and len(primary.encode("utf-8")) > 512:
                    raise ValueError("primary character name exceeds its bound")
                supplement = nl.get("systemPrompt")
                if not isinstance(supplement, str) or len(supplement.encode("utf-8")) > 16_384:
                    raise ValueError("NL user supplement exceeds its bound")
            except (KeyError, TypeError, ValueError) as exc:
                raise self._fatal("nl_protocol_violation", "frozen NL v4 routing is invalid") from exc
            item.update({
                "captionPreset": preset,
                "lengthTier": tier,
                "primaryCharacterName": primary,
                "userSupplement": supplement,
                "ocrContext": None,
            })
        if job_config_supports_ocr(config_schema_version):
            assert ocr is not None
            ocr_context = self._ocr_context(str(row["relative_image_path"]), ocr)
            if len(self._canonical_context_bytes(context, ocr_context)) > 262_144 and ocr_context is not None:
                ocr_context = None
                self.database.increment_module_diagnostic(
                    self.job_id,
                    "nl",
                    "nl_ocr_context_omitted_too_large",
                    severity="warning",
                    amount=1,
                )
            item["ocrContext"] = ocr_context
        return item

    def run(self) -> str:
        config_hash, nl, dataset_root, config_schema_version, ocr, input_txt_mode = self._config()
        prompt_version = str(nl["promptVersion"])
        moss_mode = prompt_version == NL_MOSS_PROMPT_VERSION
        response_protocol = NL_MOSS_RESPONSE_PROTOCOL if moss_mode else "nl-v1" if config_schema_version == 2 else "nl-count-v2"
        if nl.get("enabled") is not True:
            raise self._fatal("nl_protocol_violation", "NL runner cannot execute a disabled module")
        reuse = nl.get("reuseOriginalNl") is True and not moss_mode
        input_txt_nl = input_txt_mode == "nl" and not moss_mode
        api_enabled = nl.get("apiEnabled") is True and not input_txt_nl
        use_image, use_json = nl.get("useImage") is True, nl.get("useFullJson") is True
        if moss_mode and (not api_enabled or not use_image):
            raise self._fatal("nl_protocol_violation", "MOSS output requires the NL API and image input")
        if api_enabled and not use_image:
            raise self._fatal("nl_protocol_violation", "API-enabled NL requires image input")
        policy = self._policy(nl) if api_enabled else None
        active: list[WorkLease] = []
        try:
            while True:
                job = self.database.get_job(self.job_id)
                if job["status"] in {"cancelling", "paused"}:
                    return str(job["status"])
                if job["status"] != "running" or job["current_module_id"] != "nl":
                    raise self._fatal("nl_protocol_violation", "NL module is not active")
                self._refresh_runtime_revision()
                if api_enabled:
                    policy = self._policy(nl)
                active = self.scheduler.claim_batch(self.job_id, "nl", self.worker_instance_id, config_hash)
                if not active:
                    if self.database.count_module_unsettled(self.job_id, "nl"):
                        raise self._fatal("nl_protocol_violation", "NL has unsettled but unclaimable work")
                    summary = self.database.module_summary(self.job_id, "nl")
                    return self.scheduler.finish_module(self.job_id, "nl", with_issues=int(summary["issue_count"]) > 0)
                request_items: list[tuple[WorkLease, object, dict[str, object]]] = []
                for lease in active[:]:
                    row = self.database.get_leased_sample(self.job_id, "nl", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id=self.worker_instance_id)
                    if moss_mode and not use_json:
                        projection: dict[str, object] = {}
                    else:
                        try:
                            projection = parse_annotation_json(self.view.read(str(row["annotation_key"]), ".json"))
                        except ClassifyJsonError as exc:
                            self._not_requested(lease, "working_json_invalid")
                            self._issue(lease, row, "nl_working_json_invalid", str(exc), repair_start="classify")
                            active.remove(lease)
                            continue
                        if projection is None:
                            self._not_requested(lease, "working_json_missing")
                            self._issue(lease, row, "nl_working_json_missing", "working JSON is missing or blank", severity="warning", blocking=False, repair_start="classify")
                            active.remove(lease)
                            continue
                    if moss_mode:
                        request_items.append((lease, row, projection))
                        continue
                    if input_txt_nl:
                        target = projection.get("nl")
                        if not isinstance(target, str):
                            self._not_requested(lease, "working_json_invalid")
                            self._issue(
                                lease,
                                row,
                                "nl_working_json_invalid",
                                "working JSON must contain a string nl",
                                repair_start="classify",
                            )
                        else:
                            if target.strip():
                                self.database.archive_nl_for_sample(
                                    self.job_id, lease.sampleId, source="nl_txt", text=target,
                                )
                            self._not_requested(lease, "input_txt_nl")
                            self.scheduler.complete(lease)
                        active.remove(lease)
                        continue
                    baseline_raw, fingerprint_error = self._baseline_json(row)
                    if fingerprint_error is not None:
                        self._not_requested(lease, "original_fingerprint_mismatch")
                        self._issue(lease, row, "nl_original_fingerprint_mismatch", fingerprint_error, retriable=True)
                        active.remove(lease)
                        continue
                    original, original_error = self._original_nl(baseline_raw)
                    if original is not None:
                        self.database.archive_nl_for_sample(
                            self.job_id, lease.sampleId, source="original_json", text=original,
                        )
                    rebuilt_working_json = self.view.overlay.annotation_path(str(row["annotation_key"]), ".json").is_file()
                    # A rebuilt working JSON is the current repair input.  Its
                    # NL value must be sent to the API when the immutable
                    # original NL is malformed, regardless of which field
                    # validation failed.
                    if original_error is not None and not rebuilt_working_json:
                        self._not_requested(lease, "original_invalid")
                        self._issue(lease, row, "nl_original_invalid", original_error)
                        active.remove(lease)
                        continue
                    # Preserve a non-blank value already present in a rebuilt
                    # working overlay (for example a human repair).  It is
                    # never replaced by an older original/history revision.
                    working_nl = projection.get("nl")
                    protected_working = (
                        rebuilt_working_json and original_error is None
                        and isinstance(working_nl, str) and bool(working_nl.strip())
                    )
                    if protected_working:
                        target = str(working_nl)
                        history_source = "manual"
                    else:
                        target = None
                        history_source = "api"
                        if reuse and original is not None:
                            target = original
                            history_source = "original_json"
                        elif reuse:
                            history = self.database.verified_nl_history_for_sample(self.job_id, lease.sampleId)
                            if history is not None:
                                target = str(history["text"])
                                history_source = str(history["source"])
                        if target is None and not api_enabled:
                            target = ""
                    if target is not None:
                        self._not_requested(
                            lease,
                            "reused_original_nl" if reuse and original is not None else "reused_history_nl" if reuse and target else "api_disabled",
                        )
                        if target != projection["nl"]:
                            self.writer.write_value(sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]), projection=projection, nl=target, history_source=history_source)
                        self.scheduler.complete(lease)
                        active.remove(lease)
                        continue
                    if api_enabled and use_image:
                        image_path = dataset_root / str(row["relative_image_path"])
                        try:
                            safe_image_path = ensure_within(dataset_root, image_path)
                            with safe_image_path.open("rb") as image_file:
                                if not image_file.read(1):
                                    raise OSError("image is empty")
                        except PathSafetyError as exc:
                            raise self._fatal("nl_protocol_violation", "leased image path escapes dataset root") from exc
                        except OSError:
                            self._not_requested(lease, "image_missing")
                            self._issue(
                                lease,
                                row,
                                "nl_image_missing",
                                "local image is missing or unreadable; no API request was sent",
                                retriable=False,
                            )
                            active.remove(lease)
                            continue
                    request_items.append((lease, row, projection))
                if not request_items:
                    continue
                if not self._hello_done:
                    try:
                        self._hello(
                            config_hash,
                            nl,
                            response_protocol=response_protocol,
                            prompt_version=str(nl["promptVersion"]),
                        )
                    except NlRunnerError:
                        if self._pending_credentials is None:
                            raise
                        previous = self._credentials_before_pending
                        self._pending_credentials = None
                        if previous is None:
                            raise
                        self.credentials = previous
                        self._credentials_before_pending = previous
                        self._hello_done = False
                        self.database.increment_module_diagnostic(
                            self.job_id, "nl", "nl_api_revision_switch_failed", severity="warning", amount=1,
                        )
                        self._hello(
                            config_hash,
                            nl,
                            response_protocol=response_protocol,
                            prompt_version=str(nl["promptVersion"]),
                        )
                    else:
                        self._confirm_runtime_revision()
                payload_items: list[dict[str, object]] = []
                expected: dict[tuple[int, str], str] = {}
                for lease, row, projection in request_items:
                    payload_items.append(
                        self._worker_item(
                            lease,
                            row,
                            projection,
                            use_image=use_image,
                            use_json=use_json,
                            dataset_root=dataset_root,
                            config_schema_version=config_schema_version,
                            ocr=ocr,
                            nl=nl,
                        )
                    )
                    expected[(lease.sampleId, str(lease.leaseId))] = str(row["relative_image_path"])
                assert policy is not None
                used = self.database.module_diagnostic_count(self.job_id, "nl", "nl_http_attempts")
                maximum = int(policy["maxHttpAttempts"]) + int(self.database.get_job(self.job_id)["api_budget_extra"])
                if used >= maximum:
                    for lease, _, _ in request_items:
                        self.scheduler.release_unstarted(lease)
                    self._pause("nl_http_budget_paused")
                    return "paused"
                for lease, _, _ in request_items:
                    if moss_mode:
                        self._not_requested(lease, "moss_output")
                    self.database.mark_nl_request_started(self.job_id, lease.sampleId, lease_id=str(lease.leaseId))
                allowance = min(maximum - used, len(request_items) * (int(policy["mainAttempts"]) + (int(policy["backupAttempts"]) if policy["backupEnabled"] else 0)))
                outcomes = parse_outcomes(
                    self._exchange("process_batch", {
                        "schemaVersion": 1,
                        "payloadType": "nl_process_request",
                        "items": payload_items,
                        "httpAttemptAllowance": allowance,
                    }, config_hash),
                    expected,
                    response_protocol=response_protocol,
                )
                by_lease = {(lease.sampleId, str(lease.leaseId)): (lease, row) for lease, row, _ in request_items}
                pause_code: str | None = None
                for outcome in outcomes:
                    lease, row = by_lease[(outcome.sampleId, outcome.leaseId)]
                    if outcome.httpAttempts:
                        self.database.increment_module_diagnostic(self.job_id, "nl", "nl_http_attempts", severity="info", amount=outcome.httpAttempts)
                    if outcome.code in {"nl_budget_exhausted", "nl_cancelled"} and outcome.httpAttempts == 0:
                        self.database.return_unsubmitted_nl_request(self.job_id, lease.sampleId, lease_id=outcome.leaseId)
                        if outcome.code == "nl_budget_exhausted":
                            pause_code = "nl_http_budget_paused"
                        active.remove(lease)
                        continue
                    if outcome.code is not None:
                        code = outcome.code or "nl_processing_failed"
                        nonblocking_terminal = code in NONBLOCKING_TERMINAL_NL_CODES
                        self._issue(
                            lease,
                            row,
                            code,
                            outcome.message or "NL worker could not produce a caption",
                            blocking=not nonblocking_terminal,
                            retriable=outcome.retriable,
                            repair_start="nl",
                            allowed_statuses=("request_started",),
                        )
                        processed, failed = self.database.record_nl_outcome(self.job_id, succeeded=False)
                        format_failed = code == "nl_response_format_invalid"
                        format_processed, format_failures = self.database.record_nl_format_outcome(
                            self.job_id, format_failed=format_failed,
                        )
                        if format_failed:
                            self.database.increment_module_diagnostic(
                                self.job_id, "nl", "nl_format_failures", severity="warning", amount=1,
                            )
                            self.database.increment_module_diagnostic(
                                self.job_id, "nl", "nl_format_consecutive_failures", severity="warning", amount=1,
                            )
                        else:
                            self.database.set_module_diagnostic_count(
                                self.job_id, "nl", "nl_format_consecutive_failures", severity="warning", count=0,
                            )
                        if outcome.code == "nl_auth_failed":
                            pause_code = "nl_auth_paused"
                        elif code == "nl_response_format_invalid" and (
                            self.database.module_diagnostic_count(self.job_id, "nl", "nl_format_consecutive_failures") >= 10
                            or (format_processed >= 20 and format_failures * 2 >= format_processed)
                        ):
                            pause_code = "nl_format_circuit_breaker_paused"
                        elif failed >= 10 and self.database.module_diagnostic_count(self.job_id, "nl", "nl_consecutive_failures") >= 9:
                            pause_code = "nl_circuit_breaker_paused"
                        elif processed >= 20 and failed * 2 >= processed:
                            pause_code = "nl_circuit_breaker_paused"
                        self.database.increment_module_diagnostic(self.job_id, "nl", "nl_consecutive_failures", severity="warning", amount=1)
                    else:
                        if outcome.mossJson is not None:
                            self.writer.write_moss_value(
                                sample_id=lease.sampleId,
                                lease_id=outcome.leaseId,
                                relative_image_path=outcome.relativeImagePath,
                                moss=outcome.mossJson,
                                allowed_statuses=("request_started",),
                            )
                        elif outcome.nl is not None:
                            self.database.stage_nl_response(
                                self.job_id,
                                lease.sampleId,
                                lease_id=outcome.leaseId,
                                nl=outcome.nl,
                                sha256=hashlib.sha256(outcome.nl.encode("utf-8")).hexdigest(),
                                observation=outcome.observation.to_dict() if outcome.observation is not None else None,
                            )
                            self.writer.commit_staged(self.job_id, lease.sampleId, outcome.leaseId)
                        else:
                            raise self._fatal("nl_protocol_violation", "NL worker returned an empty success outcome")
                        self.scheduler.complete(lease)
                        self.database.record_nl_outcome(self.job_id, succeeded=True)
                        self.database.set_module_diagnostic_count(self.job_id, "nl", "nl_consecutive_failures", severity="warning", count=0)
                    active.remove(lease)
                if self.database.module_diagnostic_count(self.job_id, "nl", "nl_http_attempts") >= maximum:
                    pause_code = pause_code or "nl_http_budget_paused"
                if pause_code is not None:
                    self._pause(pause_code)
                    return "paused"
        except StdioJsonlTransportError:
            for lease in active:
                state = self.database.get_sample_state(self.job_id, lease.sampleId)
                if state["status"] == "leased":
                    self.scheduler.release_unstarted(lease)
            if pending_api_decisions(self.database, self.job_id):
                self._pause("nl_api_outcome_unknown")
                return "paused"
            raise
        except (NlRunnerError, NlProtocolError, SchedulerError, ProtocolError) as exc:
            for lease in active:
                state = self.database.get_sample_state(self.job_id, lease.sampleId)
                if state["status"] == "leased":
                    self.scheduler.release_unstarted(lease)
            self.database.set_module_summary(self.job_id, "nl", status="failed", finished=True)
            self.database.set_job_status(self.job_id, "failed", current_module_id="nl")
            raise
        except Exception as exc:
            for lease in active:
                state = self.database.get_sample_state(self.job_id, lease.sampleId)
                if state["status"] == "leased":
                    self.scheduler.release_unstarted(lease)
            self.database.set_module_summary(self.job_id, "nl", status="failed", finished=True)
            self.database.set_job_status(self.job_id, "failed", current_module_id="nl")
            raise self._fatal("nl_result_persistence_failed", "NL result persistence failed") from exc
