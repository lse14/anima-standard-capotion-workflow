"""Core-owned Token Budget review API boundary.

The implementation is intentionally separate from the worker and the trusted
Token Budget record path.  Review proposals are never a replacement for a
working annotation until a later explicit apply operation validates them again.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping
import uuid

from anima_caption_format import normalize_annotation
from anima_caption_format.normalizer import CaptionDisplayPolicy, FIELDS

from .classify_overlay import ClassifyJsonError, parse_annotation_json
from .contracts import NL_MOSS_CAPTION_PRESETS, NL_STANDARD_CAPTION_PRESETS, canonical_json, job_config_supports_token_budget
from .db import StateDatabase
from .launcher import WorkerLauncher, WorkerLaunchError
from .overlay import BaselineView, OverlayError, OverlayLayout, WorkingAnnotationView
from .path_safety import PathSafetyError, ensure_within
from .nl_length import character_name
from .nl_protocol import NlOutcomeV1
from .nl_runner import DEFAULT_POLICY, build_short_rewrite_item, nl_http_attempt_budget
from .ocr_overlay import OcrWorkingSidecarView
from .ocr_sidecar import OcrSidecarError, compact_ocr_context, parse_ocr_sidecar, with_llm_threshold
from .stdio_transport import StdioJsonlTransport, StdioJsonlTransportError
from .token_budget_overlay import TokenBudgetOverlayError, TokenBudgetOverlayWriter, TokenBudgetRecord
from .token_budget_protocol import TokenBudgetOutcomeV1, TokenBudgetProtocolError, validate_token_budget_outcome
from .worker_protocol import ProtocolEnvelopeV1, ProtocolError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REMOVABLE_FIELDS = ("quality", "environment", "tags", "appearance")
_TOKENIZER_IDS = frozenset({"tokenizer-qwen3-0.6b-anima-v1", "tokenizer-qwen3-vl-4b-krea2-v1"})
_NL_REVIEW_RESERVED = "nl_review_http_reserved"


class TokenBudgetReviewError(RuntimeError):
    pass


class TokenBudgetReviewConflictError(ValueError):
    pass


@dataclass(frozen=True)
class OverflowReviewV1:
    sampleId: int
    leaseId: str
    status: str
    originalTokens: int
    finalTokens: int
    removed: dict[str, list[str]]
    maxTokens: int
    resourceId: str
    resourceFingerprint: str


class IsolatedTokenBudgetCounter:
    """One bounded, isolated worker exchange for an explicit review action."""

    def __init__(self, database: StateDatabase, job_id: str, launcher: WorkerLauncher) -> None:
        self.database = database
        self.job_id = job_id
        self.launcher = launcher

    def __call__(self, sample_id: int, lease_id: str, annotation: dict[str, object], review: OverflowReviewV1) -> dict[str, object]:
        job = self.database.get_job(self.job_id)
        try:
            config = json.loads(str(job["config_json"]))
            budget = config["tokenBudget"]
            caption_format = config["captionFormat"]
            if not isinstance(budget, dict) or not isinstance(caption_format, dict):
                raise ValueError("invalid frozen configuration")
            resource_path = budget["resourceManifestRelativePath"]
            context_limit = budget["contextLimit"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TokenBudgetReviewError("frozen Token Budget configuration is invalid") from exc
        if not isinstance(resource_path, str) or type(context_limit) is not int:
            raise TokenBudgetReviewError("frozen Token Budget configuration is invalid")

        def exchange(transport: StdioJsonlTransport, method: str, payload: dict[str, object]) -> dict[str, object]:
            request = ProtocolEnvelopeV1(
                "1.0", "request", f"token-budget-review-{uuid.uuid4().hex}", "token-budget", "token-budget",
                method, payload, jobId=self.job_id, configHash=str(job["config_hash"]),
            )
            response = transport.exchange(request)
            if (
                response.kind != "response" or response.replyTo != request.messageId
                or response.runtimeId != "token-budget" or response.owner != "token-budget"
                or response.jobId != self.job_id or response.configHash != job["config_hash"]
                or response.method == "error"
            ):
                raise TokenBudgetReviewError("Token Budget worker response is invalid")
            return response.payload

        try:
            with StdioJsonlTransport(self.launcher.spawn("token-budget", expected_owner="token-budget")) as transport:
                hello = exchange(transport, "hello", {
                    "schemaVersion": 1, "payloadType": "token_budget_hello_request", "jobId": self.job_id,
                    "configHash": str(job["config_hash"]), "resourceId": review.resourceId,
                    "resourceManifestRelativePath": resource_path, "resourceFingerprint": review.resourceFingerprint,
                    "contextLimit": context_limit, "maxTokens": review.maxTokens,
                })
                if hello != {
                    "schemaVersion": 1, "payloadType": "token_budget_hello_result", "ready": True,
                    "resourceFingerprint": review.resourceFingerprint, "contextLimit": context_limit,
                }:
                    raise TokenBudgetReviewError("Token Budget worker hello is invalid")
                result = exchange(transport, "process_batch", {
                    "schemaVersion": 1, "payloadType": "token_budget_process_request", "captionFormat": caption_format,
                    "items": [{"schemaVersion": 1, "sampleId": sample_id, "leaseId": lease_id, "annotation": annotation}],
                })
                if set(result) != {"schemaVersion", "payloadType", "outcomes"} or result.get("schemaVersion") != 1 or result.get("payloadType") != "token_budget_process_result" or not isinstance(result.get("outcomes"), list) or len(result["outcomes"]) != 1 or not isinstance(result["outcomes"][0], dict):
                    raise TokenBudgetReviewError("Token Budget worker result is invalid")
                return result["outcomes"][0]
        except (WorkerLaunchError, StdioJsonlTransportError, ProtocolError, OSError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget worker is unavailable") from exc


def parse_overflow_review(value: object) -> OverflowReviewV1:
    """Read the Task 2.5 overflow-only review document without caption text."""
    expected = {
        "schemaVersion", "sampleId", "leaseId", "status", "originalTokens", "finalTokens",
        "removed", "maxTokens", "resourceId", "resourceFingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schemaVersion") != 1:
        raise TokenBudgetReviewError("Token Budget overflow review fields are invalid")
    sample_id = value["sampleId"]
    lease_id = value["leaseId"]
    original = value["originalTokens"]
    final = value["finalTokens"]
    maximum = value["maxTokens"]
    resource_id = value["resourceId"]
    fingerprint = value["resourceFingerprint"]
    removed = value["removed"]
    if (
        type(sample_id) is not int or sample_id < 1
        or not isinstance(lease_id, str) or not lease_id or len(lease_id.encode("utf-8")) > 128
        or value["status"] != "overflow"
        or type(original) is not int or original < 0
        or type(final) is not int or final < 0
        or type(maximum) is not int or maximum < 1 or final <= maximum
        or resource_id not in _TOKENIZER_IDS
        or not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None
        or not isinstance(removed, Mapping) or set(removed) != set(_REMOVABLE_FIELDS)
    ):
        raise TokenBudgetReviewError("Token Budget overflow review is invalid")
    result: dict[str, list[str]] = {}
    for field in _REMOVABLE_FIELDS:
        entries = removed[field]
        if not isinstance(entries, list) or any(not isinstance(entry, str) or not entry for entry in entries):
            raise TokenBudgetReviewError("Token Budget overflow removal audit is invalid")
        result[field] = list(entries)
    return OverflowReviewV1(sample_id, lease_id, "overflow", original, final, result, maximum, resource_id, fingerprint)


class TokenBudgetReviewService:
    """Owns overflow-review lifecycle operations for one immutable job."""

    def __init__(
        self,
        database: StateDatabase,
        job_id: str,
        *,
        counter: Callable[[int, str, dict[str, object], OverflowReviewV1], dict[str, object]] | None = None,
        rewriter: Callable[[dict[str, object]], str | NlOutcomeV1] | None = None,
    ) -> None:
        self.database = database
        self.job_id = job_id
        self._counter = counter
        self._rewriter = rewriter

    def _job_and_layout(self) -> tuple[object, OverlayLayout]:
        job = self.database.get_job(self.job_id)
        if not isinstance(job["overlay_root"], str):
            raise TokenBudgetReviewError("Token Budget review overlay is unavailable")
        try:
            layout = OverlayLayout.open_existing(str(job["overlay_root"]), self.job_id)
        except (OSError, OverlayError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget review overlay is unavailable") from exc
        if job_config_supports_token_budget(job["config_schema_version"]) and job["status"] == "reviewing" and job["current_module_id"] == "token_budget":
            self._recover_prepared_applies(job, layout)
            job = self.database.get_job(self.job_id)
        if (
            not job_config_supports_token_budget(job["config_schema_version"])
            or job["status"] in {"succeeded", "failed", "cancelled", "cancelling"}
            or job["status"] != "reviewing"
            or job["current_module_id"] != "token_budget"
        ):
            raise TokenBudgetReviewError("Token Budget review is unavailable for this task")
        return job, layout

    def _reset_prepared_apply_to_failed(self, sample_id: int, lease_id: str) -> None:
        with self.database.transaction(immediate=True):
            self.database.connection.execute(
                """UPDATE sample_state SET status='failed',lease_id=NULL,worker_instance_id=NULL,
                   lease_expires_at=NULL,prepared_artifact_relative_path=NULL,prepared_artifact_sha256=NULL,
                   updated_at=datetime('now') WHERE job_id=? AND sample_id=? AND current_module_id='token_budget'
                   AND status IN ('leased','prepared') AND lease_id=? AND worker_instance_id='token-budget-review'""",
                (self.job_id, sample_id, lease_id),
            )

    def _finalize_prepared_apply(
        self,
        job: object,
        layout: OverlayLayout,
        *,
        sample_id: int,
        lease_id: str,
    ) -> TokenBudgetRecord:
        try:
            config = json.loads(str(job["config_json"]))
            caption_format = config["captionFormat"]
            budget = config["tokenBudget"]
            if not isinstance(caption_format, dict) or not isinstance(budget, dict) or type(budget.get("maxTokens")) is not int:
                raise ValueError("frozen configuration is invalid")
            writer = TokenBudgetOverlayWriter(
                self.database,
                layout,
                WorkingAnnotationView(BaselineView(Path(str(job["dataset_root"]))), layout),
                self.job_id,
            )
            record = writer.record_for_export(
                sample_id=sample_id,
                annotation_key=str(self.database.get_sample_with_state(self.job_id, sample_id)["annotation_key"]),
                caption_format=caption_format,
                max_tokens=int(budget["maxTokens"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, TokenBudgetOverlayError) as exc:
            raise TokenBudgetReviewConflictError("Token Budget prepared apply evidence is invalid") from exc
        if record.lease_id != lease_id:
            raise TokenBudgetReviewConflictError("Token Budget prepared apply lease is invalid")
        with self.database.transaction(immediate=True):
            issue = self.database.connection.execute(
                """SELECT issue_id FROM issues WHERE job_id=? AND sample_id=? AND module_id='token_budget'
                   AND code='token_budget_overflow' AND resolved_at IS NULL""",
                (self.job_id, sample_id),
            ).fetchone()
            current = self.database.get_sample_with_state(self.job_id, sample_id)
            if issue is None or (
                current["current_module_id"] != "token_budget" or current["status"] != "prepared"
                or current["lease_id"] != lease_id or current["worker_instance_id"] != "token-budget-review"
            ):
                raise TokenBudgetReviewConflictError("Token Budget prepared apply state changed")
            self.database.connection.execute(
                """UPDATE sample_state SET current_module_id='export',status='pending',lease_id=NULL,
                   worker_instance_id=NULL,lease_expires_at=NULL,prepared_artifact_relative_path=NULL,
                   prepared_artifact_sha256=NULL,updated_at=datetime('now')
                   WHERE job_id=? AND sample_id=? AND current_module_id='token_budget' AND status='prepared' AND lease_id=?""",
                (self.job_id, sample_id, lease_id),
            )
            self.database.connection.execute(
                "UPDATE issues SET resolved_at=datetime('now'),updated_at=datetime('now') WHERE issue_id=? AND resolved_at IS NULL",
                (issue["issue_id"],),
            )
            summary = self.database.connection.execute(
                "SELECT 1 FROM module_summary WHERE job_id=? AND module_id='token_budget'", (self.job_id,)
            ).fetchone()
            if summary is not None:
                self.database.connection.execute(
                    """UPDATE module_summary SET completed=completed+1,failed=CASE WHEN failed>0 THEN failed-1 ELSE 0 END,
                       issue_count=CASE WHEN issue_count>0 THEN issue_count-1 ELSE 0 END WHERE job_id=? AND module_id='token_budget'""",
                    (self.job_id,),
                )
        return record

    def _recover_leased_applies(self) -> None:
        rows = list(self.database.connection.execute(
            """SELECT sample_id,lease_id
                 FROM sample_state WHERE job_id=? AND current_module_id='token_budget'
                   AND status='leased' AND worker_instance_id='token-budget-review'
                 ORDER BY sample_id LIMIT 500""",
            (self.job_id,),
        ))
        for row in rows:
            sample_id = int(row["sample_id"])
            lease_id = row["lease_id"]
            if not isinstance(lease_id, str):
                continue
            # A process died before SQLite reached prepared.  It cannot prove an
            # annotation commit, so preserve any artifacts as diagnostics and retry from review.
            self._reset_prepared_apply_to_failed(sample_id, lease_id)

    def _recover_prepared_applies(self, job: object, layout: OverlayLayout) -> None:
        self._recover_leased_applies()
        rows = list(self.database.connection.execute(
            """SELECT sample_id,lease_id,prepared_artifact_relative_path,prepared_artifact_sha256
                 FROM sample_state WHERE job_id=? AND current_module_id='token_budget'
                   AND status='prepared' AND worker_instance_id='token-budget-review'
                 ORDER BY sample_id LIMIT 500""",
            (self.job_id,),
        ))
        for row in rows:
            sample_id = int(row["sample_id"])
            lease_id = row["lease_id"]
            if not isinstance(lease_id, str):
                continue
            try:
                self._recover_prepared_apply(
                    job,
                    layout,
                    sample_id=sample_id,
                    lease_id=lease_id,
                    prepared_relative_path=row["prepared_artifact_relative_path"],
                    expected_sha256=row["prepared_artifact_sha256"],
                )
            except TokenBudgetReviewConflictError:
                self._reset_prepared_apply_to_failed(sample_id, lease_id)

    def _recover_prepared_apply(
        self,
        job: object,
        layout: OverlayLayout,
        *,
        sample_id: int,
        lease_id: str,
        prepared_relative_path: object,
        expected_sha256: object,
    ) -> str:
        """Finalize a review-owned apply without generic scheduler completion."""
        if not isinstance(prepared_relative_path, str) or not isinstance(expected_sha256, str):
            self._reset_prepared_apply_to_failed(sample_id, lease_id)
            return "settled"
        writer = TokenBudgetOverlayWriter(
            self.database,
            layout,
            WorkingAnnotationView(BaselineView(Path(str(job["dataset_root"]))), layout),
            self.job_id,
        )
        if not writer.recover_prepared(self.job_id, sample_id, prepared_relative_path, expected_sha256):
            self._reset_prepared_apply_to_failed(sample_id, lease_id)
            return "settled"
        try:
            self._finalize_prepared_apply(job, layout, sample_id=sample_id, lease_id=lease_id)
        except TokenBudgetReviewConflictError:
            self._reset_prepared_apply_to_failed(sample_id, lease_id)
        return "settled"

    @staticmethod
    def _read_review(layout: OverlayLayout, sample_id: int) -> OverflowReviewV1:
        try:
            raw = layout.resource_path(f"token-budget\\reviews\\{sample_id}.json").read_bytes()
            value = json.loads(raw.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget overflow review is unreadable") from exc
        review = parse_overflow_review(value)
        if review.sampleId != sample_id:
            raise TokenBudgetReviewError("Token Budget overflow review identity is invalid")
        return review

    @staticmethod
    def _proposal_relative(sample_id: int) -> str:
        if type(sample_id) is not int or sample_id < 1:
            raise TokenBudgetReviewError("Token Budget proposal identity is invalid")
        return f"token-budget\\proposals\\{sample_id}.json"

    @staticmethod
    def _rewrite_relative(operation_id: str) -> str:
        if not isinstance(operation_id, str) or _SHA256.fullmatch(operation_id) is None:
            raise TokenBudgetReviewError("Token Budget rewrite operation is invalid")
        return f"token-budget\\rewrites\\{operation_id}.json"

    @staticmethod
    def _rewrite_proposal_relative(sample_id: int) -> str:
        if type(sample_id) is not int or sample_id < 1:
            raise TokenBudgetReviewError("Token Budget rewrite proposal identity is invalid")
        return f"token-budget\\rewrite-proposals\\{sample_id}.json"

    @staticmethod
    def _read_rewrite_proposal(layout: OverlayLayout, sample_id: int) -> dict[str, object] | None:
        path = layout.resource_path(TokenBudgetReviewService._rewrite_proposal_relative(sample_id))
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget rewrite proposal is unreadable") from exc
        if (
            not isinstance(value, dict) or set(value) != {"schemaVersion", "operationId", "proposal"}
            or value.get("schemaVersion") != 1 or not isinstance(value.get("operationId"), str)
            or _SHA256.fullmatch(value["operationId"]) is None or not isinstance(value.get("proposal"), dict)
        ):
            raise TokenBudgetReviewError("Token Budget rewrite proposal is invalid")
        return value

    @staticmethod
    def _read_proposal(layout: OverlayLayout, sample_id: int) -> dict[str, object] | None:
        path = layout.resource_path(TokenBudgetReviewService._proposal_relative(sample_id))
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget proposal is unreadable") from exc
        expected = {
            "schemaVersion", "sampleId", "version", "status", "baseAnnotationSha256", "originalTokens",
            "finalTokens", "removed", "annotation", "flatTextSha256", "maxTokens", "resourceId",
            "resourceFingerprint",
        }
        if not isinstance(value, dict) or set(value) != expected or value.get("schemaVersion") != 1:
            raise TokenBudgetReviewError("Token Budget proposal fields are invalid")
        if (
            value.get("sampleId") != sample_id
            or type(value.get("version")) is not int or value["version"] < 2
            or value.get("status") not in {"within_budget", "trimmed", "overflow", "failed"}
            or not isinstance(value.get("baseAnnotationSha256"), str) or _SHA256.fullmatch(value["baseAnnotationSha256"]) is None
            or type(value.get("originalTokens")) is not int or value["originalTokens"] < 0
            or type(value.get("finalTokens")) is not int or value["finalTokens"] < 0
            or type(value.get("maxTokens")) is not int or value["maxTokens"] < 1
            or value.get("resourceId") not in _TOKENIZER_IDS
            or not isinstance(value.get("resourceFingerprint"), str) or _SHA256.fullmatch(value["resourceFingerprint"]) is None
        ):
            raise TokenBudgetReviewError("Token Budget proposal values are invalid")
        if value["status"] in {"within_budget", "trimmed"}:
            if not isinstance(value.get("annotation"), dict) or not isinstance(value.get("flatTextSha256"), str) or _SHA256.fullmatch(value["flatTextSha256"]) is None or value["finalTokens"] > value["maxTokens"]:
                raise TokenBudgetReviewError("Token Budget proposal outcome is invalid")
        elif value.get("annotation") is not None or value.get("flatTextSha256") is not None:
            raise TokenBudgetReviewError("Token Budget proposal outcome is invalid")
        try:
            parse_overflow_review({
                "schemaVersion": 1, "sampleId": value["sampleId"], "leaseId": "proposal-read",
                "status": "overflow", "originalTokens": max(value["originalTokens"], value["maxTokens"] + 1),
                "finalTokens": max(value["finalTokens"], value["maxTokens"] + 1), "removed": value["removed"],
                "maxTokens": value["maxTokens"], "resourceId": value["resourceId"],
                "resourceFingerprint": value["resourceFingerprint"],
            })
        except TokenBudgetReviewError as exc:
            raise TokenBudgetReviewError("Token Budget proposal removal audit is invalid") from exc
        return value

    @staticmethod
    def _strict_annotation(annotation: object, caption_format: Mapping[str, object]) -> tuple[dict[str, object], str]:
        if not isinstance(annotation, dict) or set(annotation) != set(FIELDS):
            raise TokenBudgetReviewError("Token Budget annotation must contain exactly the nine fields")
        try:
            raw = json.dumps(annotation, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            policy = CaptionDisplayPolicy.from_mapping(caption_format)
            normalized = normalize_annotation(raw, policy, export_format="both")
        except (TypeError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget annotation is invalid") from exc
        if not normalized.valid or normalized.payload is None:
            raise TokenBudgetReviewError("Token Budget annotation is invalid")
        return normalized.payload, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _frozen_config(job: object, review: OverflowReviewV1) -> tuple[dict[str, object], dict[str, object]]:
        try:
            config = json.loads(str(job["config_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise TokenBudgetReviewError("frozen Token Budget configuration is invalid") from exc
        if not isinstance(config, dict) or not job_config_supports_token_budget(config.get("schemaVersion")):
            raise TokenBudgetReviewError("frozen Token Budget configuration is invalid")
        caption_format = config.get("captionFormat")
        budget = config.get("tokenBudget")
        if not isinstance(caption_format, dict) or not isinstance(budget, dict):
            raise TokenBudgetReviewError("frozen Token Budget configuration is invalid")
        if (
            budget.get("enabled") is not True or budget.get("maxTokens") != review.maxTokens
            or budget.get("resourceId") != review.resourceId
            or budget.get("resourceFingerprint") != review.resourceFingerprint
        ):
            raise TokenBudgetReviewError("frozen Token Budget review identity is invalid")
        return caption_format, budget

    def _current_annotation(
        self,
        job: object,
        layout: OverlayLayout,
        sample_id: int,
        caption_format: Mapping[str, object],
    ) -> tuple[dict[str, object], str]:
        try:
            sample = self.database.get_sample_with_state(self.job_id, sample_id)
            if not bool(sample["in_processing_scope"]):
                raise TokenBudgetReviewError("Token Budget review sample is out of scope")
            raw = WorkingAnnotationView(
                baseline=BaselineView(Path(str(job["dataset_root"]))),
                overlay=layout,
            ).read(str(sample["annotation_key"]), ".json")
            annotation = parse_annotation_json(raw) if raw is not None else None
        except (ClassifyJsonError, KeyError, OSError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget working annotation is unavailable") from exc
        if annotation is None:
            raise TokenBudgetReviewError("Token Budget working annotation is unavailable")
        normalized, _ = self._strict_annotation(annotation, caption_format)
        return normalized, hashlib.sha256(raw).hexdigest()

    def _require_open_overflow(self, sample_id: int, *, concurrent: bool = False) -> None:
        issue = self.database.connection.execute(
            """SELECT 1 FROM issues WHERE job_id=? AND sample_id=? AND module_id='token_budget'
               AND code='token_budget_overflow' AND resolved_at IS NULL AND blocking=1""",
            (self.job_id, sample_id),
        ).fetchone()
        if issue is None:
            error = "Token Budget overflow has already been resolved"
            if concurrent:
                raise TokenBudgetReviewConflictError(error)
            raise TokenBudgetReviewError("Token Budget overflow review is unavailable")

    def recount(
        self,
        sample_id: int,
        *,
        expected_version: int,
        annotation: object,
        _after_request: bool = False,
        _expected_base_digest: str | None = None,
    ) -> dict[str, object]:
        """Count a client candidate and save only a versioned overlay proposal."""
        if type(sample_id) is not int or sample_id < 1 or type(expected_version) is not int or expected_version < 1:
            raise TokenBudgetReviewError("Token Budget proposal version is invalid")
        job, layout = self._job_and_layout()
        self.database.get_sample_with_state(self.job_id, sample_id)
        review = self._read_review(layout, sample_id)
        caption_format, _ = self._frozen_config(job, review)
        self._require_open_overflow(sample_id, concurrent=_after_request)
        normalized, _ = self._strict_annotation(annotation, caption_format)
        _, current_digest = self._current_annotation(job, layout, sample_id, caption_format)
        if _expected_base_digest is not None:
            if current_digest != _expected_base_digest:
                raise TokenBudgetReviewConflictError("Token Budget working annotation changed before proposal")
            base_digest = _expected_base_digest
        else:
            base_digest = current_digest
        existing = self._read_proposal(layout, sample_id)
        current_version = 1 if existing is None else int(existing["version"])
        if expected_version != current_version:
            raise TokenBudgetReviewConflictError("Token Budget review version conflict")
        if self._counter is None:
            raise TokenBudgetReviewError("Token Budget counter is unavailable")
        lease_id = f"review-{sample_id}-{current_version}"
        try:
            raw_outcome = self._counter(sample_id, lease_id, normalized, review)
            outcome = validate_token_budget_outcome(
                raw_outcome,
                expected_sample_id=sample_id,
                expected_lease_id=lease_id,
                original_annotation=normalized,
                caption_format=caption_format,
                max_tokens=review.maxTokens,
            )
        except (TokenBudgetProtocolError, TypeError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget counter returned an invalid proposal") from exc
        if outcome.status not in {"within_budget", "trimmed", "overflow", "failed"}:
            raise TokenBudgetReviewError("Token Budget counter returned an invalid proposal")
        proposal = {
            "schemaVersion": 1, "sampleId": sample_id, "version": current_version + 1,
            "status": outcome.status, "baseAnnotationSha256": base_digest,
            "originalTokens": outcome.original_tokens, "finalTokens": outcome.final_tokens,
            "removed": outcome.removed, "annotation": outcome.annotation,
            "flatTextSha256": outcome.flat_text_sha256, "maxTokens": review.maxTokens,
            "resourceId": review.resourceId, "resourceFingerprint": review.resourceFingerprint,
        }
        if any(proposal[name] is None for name in ("originalTokens", "finalTokens", "removed")):
            raise TokenBudgetReviewError("Token Budget counter returned an incomplete proposal")
        with self.database.transaction(immediate=True):
            # The SQLite immediate lock makes the version check and atomic overlay replacement one operation.
            current = self._read_proposal(layout, sample_id)
            if expected_version != (1 if current is None else int(current["version"])):
                raise TokenBudgetReviewConflictError("Token Budget review version conflict")
            self._require_open_overflow(sample_id, concurrent=True)
            _, current_digest = self._current_annotation(job, layout, sample_id, caption_format)
            if current_digest != base_digest:
                raise TokenBudgetReviewConflictError("Token Budget working annotation changed before proposal")
            layout.write_resource(self._proposal_relative(sample_id), (canonical_json(proposal) + "\n").encode("utf-8"))
        return proposal

    def apply(self, sample_id: int, *, expected_version: int) -> dict[str, object]:
        """Re-count a saved proposal before it becomes the one trusted Export record."""
        if type(sample_id) is not int or sample_id < 1 or type(expected_version) is not int or expected_version < 2:
            raise TokenBudgetReviewError("Token Budget proposal version is invalid")
        job, layout = self._job_and_layout()
        self.database.get_sample_with_state(self.job_id, sample_id)
        review = self._read_review(layout, sample_id)
        caption_format, _ = self._frozen_config(job, review)
        proposal = self._read_proposal(layout, sample_id)
        if proposal is None or proposal["version"] != expected_version:
            raise TokenBudgetReviewConflictError("Token Budget review version conflict")
        if proposal["status"] not in {"within_budget", "trimmed"} or not isinstance(proposal["annotation"], dict):
            raise TokenBudgetReviewError("Token Budget proposal cannot be applied")
        self._require_open_overflow(sample_id)
        _, current_digest = self._current_annotation(job, layout, sample_id, caption_format)
        if current_digest != proposal["baseAnnotationSha256"]:
            raise TokenBudgetReviewConflictError("Token Budget working annotation changed before apply")
        annotation, _ = self._strict_annotation(proposal["annotation"], caption_format)
        if self._counter is None:
            raise TokenBudgetReviewError("Token Budget counter is unavailable")
        lease_id = f"review-apply-{sample_id}-{expected_version}"
        try:
            raw_outcome = self._counter(sample_id, lease_id, annotation, review)
            outcome = validate_token_budget_outcome(
                raw_outcome,
                expected_sample_id=sample_id,
                expected_lease_id=lease_id,
                original_annotation=annotation,
                caption_format=caption_format,
                max_tokens=review.maxTokens,
            )
        except (TokenBudgetProtocolError, TypeError, ValueError) as exc:
            raise TokenBudgetReviewError("Token Budget counter returned an invalid apply result") from exc
        same_final_annotation = (
            outcome.annotation == annotation
            and outcome.final_tokens == proposal["finalTokens"]
            and outcome.flat_text_sha256 == proposal["flatTextSha256"]
        )
        unchanged = (
            outcome.status == proposal["status"]
            and outcome.original_tokens == proposal["originalTokens"]
            and outcome.removed == proposal["removed"]
        )
        trimmed_confirmation = (
            proposal["status"] == "trimmed"
            and outcome.status == "within_budget"
            and outcome.original_tokens == proposal["finalTokens"]
            and outcome.removed == {field: [] for field in _REMOVABLE_FIELDS}
        )
        if outcome.status not in {"within_budget", "trimmed"} or not same_final_annotation or not (unchanged or trimmed_confirmation):
            raise TokenBudgetReviewConflictError("Token Budget proposal changed before apply")
        assert outcome.annotation is not None and outcome.flat_text_sha256 is not None
        trusted_outcome = TokenBudgetOutcomeV1(
            str(proposal["status"]),
            int(proposal["originalTokens"]),
            int(proposal["finalTokens"]),
            dict(proposal["removed"]),
            annotation,
            str(proposal["flatTextSha256"]),
        )
        with self.database.transaction(immediate=True):
            current = self._read_proposal(layout, sample_id)
            if current is None or current["version"] != expected_version:
                raise TokenBudgetReviewConflictError("Token Budget review version conflict")
            _, current_digest = self._current_annotation(job, layout, sample_id, caption_format)
            if current_digest != current["baseAnnotationSha256"]:
                raise TokenBudgetReviewConflictError("Token Budget working annotation changed before apply")
            issue = self.database.connection.execute(
                """SELECT issue_id FROM issues WHERE job_id=? AND sample_id=? AND module_id='token_budget'
                   AND code='token_budget_overflow' AND resolved_at IS NULL""",
                (self.job_id, sample_id),
            ).fetchone()
            current_sample = self.database.get_sample_with_state(self.job_id, sample_id)
            if issue is None:
                raise TokenBudgetReviewConflictError("Token Budget overflow has already been resolved")
            if current_sample["current_module_id"] != "token_budget" or current_sample["status"] != "failed":
                raise TokenBudgetReviewConflictError("Token Budget sample state changed before apply")
            self.database.connection.execute(
                """UPDATE sample_state SET status='leased',lease_id=?,worker_instance_id='token-budget-review',
                   lease_expires_at=NULL,prepared_artifact_relative_path=NULL,prepared_artifact_sha256=NULL,
                   updated_at=datetime('now') WHERE job_id=? AND sample_id=? AND current_module_id='token_budget'
                   AND status='failed'""",
                (lease_id, self.job_id, sample_id),
            )
        try:
            writer = TokenBudgetOverlayWriter(
                self.database,
                layout,
                WorkingAnnotationView(BaselineView(Path(str(job["dataset_root"]))), layout),
                self.job_id,
            )
            writer.prepare_and_commit(
                sample_id=sample_id,
                lease_id=lease_id,
                annotation_key=str(current_sample["annotation_key"]),
                outcome=trusted_outcome,
                caption_format=caption_format,
                max_tokens=review.maxTokens,
            )
            return self._finalize_prepared_apply(job, layout, sample_id=sample_id, lease_id=lease_id).to_dict()
        except TokenBudgetOverlayError as exc:
            raise TokenBudgetReviewError("Token Budget apply preparation failed") from exc

    def _rewrite_ocr_context(
        self,
        *,
        layout: OverlayLayout,
        dataset_root: Path,
        relative_image_path: str,
        ocr: object,
    ) -> dict[str, object] | None:
        if ocr is None:
            return None
        if not isinstance(ocr, dict) or ocr.get("enabled") is not True:
            return None
        threshold = ocr.get("llmMinConfidence")
        if type(threshold) not in {int, float} or not 0 <= float(threshold) <= 1:
            raise TokenBudgetReviewError("frozen v6 OCR configuration is invalid")
        try:
            raw = OcrWorkingSidecarView(dataset_root, layout).read_bytes(relative_image_path)
            if raw is None:
                return None
            sidecar = parse_ocr_sidecar(raw, expected_relative_image_path=relative_image_path)
            if sidecar.status == "failed":
                return None
            context = compact_ocr_context(with_llm_threshold(sidecar, float(threshold)))
        except (OSError, OverlayError, OcrSidecarError, PathSafetyError):
            return None
        return context

    def _short_rewrite_budget(self, job: object, nl: Mapping[str, object]) -> tuple[int, int]:
        """Return the frozen task allowance and the per-sample retry ceiling."""
        configured = nl.get("apiPolicy", {})
        if not isinstance(configured, Mapping) or set(configured) - set(DEFAULT_POLICY):
            raise TokenBudgetReviewError("frozen v6 NL API policy is invalid")
        policy = {name: configured.get(name, default) for name, default in DEFAULT_POLICY.items()}
        maximum = policy["maxHttpAttempts"]
        if "maxHttpAttempts" not in configured:
            maximum = max(1, nl_http_attempt_budget(int(job["sample_count"])))
        main_attempts = policy["mainAttempts"]
        backup_attempts = policy["backupAttempts"]
        backup_enabled = policy["backupEnabled"]
        extra = job["api_budget_extra"]
        if (
            type(maximum) is not int or not 1 <= maximum <= 10_000_000
            or type(main_attempts) is not int or not 1 <= main_attempts <= 3
            or type(backup_attempts) is not int or not 1 <= backup_attempts <= 2
            or type(backup_enabled) is not bool
            or type(extra) is not int or extra < 0
        ):
            raise TokenBudgetReviewError("frozen v6 NL API policy is invalid")
        return maximum + extra, main_attempts + (backup_attempts if backup_enabled else 0)

    def _reserve_short_rewrite_attempts(self, *, maximum: int, per_sample: int) -> int:
        with self.database.transaction(immediate=True):
            used = self.database.module_diagnostic_count(self.job_id, "nl", "nl_http_attempts")
            reserved = self.database.module_diagnostic_count(self.job_id, "nl", _NL_REVIEW_RESERVED)
            remaining = maximum - used - reserved
            if remaining < 1:
                raise TokenBudgetReviewError("NL HTTP attempt budget is exhausted")
            allowance = min(remaining, per_sample)
            self.database.increment_module_diagnostic(
                self.job_id, "nl", _NL_REVIEW_RESERVED, severity="info", amount=allowance,
            )
        return allowance

    def _settle_short_rewrite_attempts(self, *, reserved: int, charged: int) -> None:
        if type(charged) is not int or not 1 <= charged <= reserved:
            raise TokenBudgetReviewError("NL short rewrite attempt accounting is invalid")
        with self.database.transaction(immediate=True):
            pending = self.database.module_diagnostic_count(self.job_id, "nl", _NL_REVIEW_RESERVED)
            if pending < reserved:
                raise TokenBudgetReviewError("NL short rewrite attempt reservation is missing")
            self.database.set_module_diagnostic_count(
                self.job_id, "nl", _NL_REVIEW_RESERVED, severity="info", count=pending - reserved,
            )
            self.database.increment_module_diagnostic(
                self.job_id, "nl", "nl_http_attempts", severity="info", amount=charged,
            )

    def _call_short_rewriter(self, item: dict[str, object], *, allowance: int) -> str | NlOutcomeV1:
        assert self._rewriter is not None
        bounded = getattr(self._rewriter, "with_http_attempt_allowance", None)
        if callable(bounded):
            return bounded(item, http_attempt_allowance=allowance)
        return self._rewriter(item)

    def rewrite_short(self, *, sample_ids: object, expected_versions: object) -> dict[str, object]:
        """Issue one user-selected v4 short rewrite per stable operation, never a retry loop."""
        if (
            not isinstance(sample_ids, list) or not 1 <= len(sample_ids) <= 500
            or any(type(sample_id) is not int or sample_id < 1 for sample_id in sample_ids)
            or len(set(sample_ids)) != len(sample_ids) or not isinstance(expected_versions, dict)
            or set(expected_versions) != {str(sample_id) for sample_id in sample_ids}
            or any(type(version) is not int or version < 1 for version in expected_versions.values())
        ):
            raise TokenBudgetReviewError("Token Budget rewrite selection is invalid")
        job, layout = self._job_and_layout()
        try:
            config = json.loads(str(job["config_json"]))
            nl = config["nl"]
            if not isinstance(nl, dict) or nl.get("captionPreset") not in NL_STANDARD_CAPTION_PRESETS | NL_MOSS_CAPTION_PRESETS:
                raise ValueError("invalid NL configuration")
            supplement = nl.get("systemPrompt")
            if not isinstance(supplement, str):
                raise ValueError("invalid NL supplement")
            ocr = config.get("ocr")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TokenBudgetReviewError("frozen v6 NL configuration is invalid") from exc
        maximum_attempts, per_sample_attempts = self._short_rewrite_budget(job, nl)
        operation_input = {
            "jobId": self.job_id, "configHash": str(job["config_hash"]),
            "sampleIds": sorted(sample_ids),
            "expectedVersions": {str(sample_id): expected_versions[str(sample_id)] for sample_id in sorted(sample_ids)},
        }
        operation_id = hashlib.sha256(canonical_json(operation_input).encode("utf-8")).hexdigest()
        operation_path = layout.resource_path(self._rewrite_relative(operation_id))
        if operation_path.is_file():
            try:
                previous = json.loads(operation_path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise TokenBudgetReviewError("Token Budget rewrite operation is unreadable") from exc
            if not isinstance(previous, dict) or previous.get("operationId") != operation_id:
                raise TokenBudgetReviewError("Token Budget rewrite operation is invalid")
            if previous.get("status") == "completed":
                proposals = previous.get("proposals")
                if not isinstance(proposals, list) or len(proposals) != len(sample_ids) or any(not isinstance(proposal, dict) for proposal in proposals):
                    raise TokenBudgetReviewError("Token Budget rewrite replay is incomplete")
                return {"operationId": operation_id, "sampleIds": sorted(sample_ids), "proposals": proposals}
            if previous.get("status") == "conflict":
                raise TokenBudgetReviewConflictError("Token Budget rewrite working annotation changed")
            raise TokenBudgetReviewError("Token Budget rewrite outcome is unknown and will not be retried automatically")
        if self._rewriter is None:
            raise TokenBudgetReviewError("Token Budget short rewrite runner is unavailable")
        prepared_items: list[tuple[int, dict[str, object], dict[str, object], str]] = []
        for sample_id in sorted(sample_ids):
            row = self.database.get_sample_with_state(self.job_id, sample_id)
            if not bool(row["in_processing_scope"]):
                raise TokenBudgetReviewError("Token Budget rewrite sample is out of scope")
            review = self._read_review(layout, sample_id)
            caption_format, _ = self._frozen_config(job, review)
            self._require_open_overflow(sample_id)
            current, base_digest = self._current_annotation(job, layout, sample_id, caption_format)
            proposal = self._read_proposal(layout, sample_id)
            if expected_versions[str(sample_id)] != (1 if proposal is None else proposal["version"]):
                raise TokenBudgetReviewConflictError("Token Budget review version conflict")
            if row["current_module_id"] not in {None, "token_budget"} or row["status"] not in {"pending", "failed"}:
                raise TokenBudgetReviewConflictError("Token Budget rewrite sample state changed")
            preset = str(nl["captionPreset"])
            primary = character_name(str(row["relative_image_path"])) if preset in {"character", "moss-character"} else None
            context = dict(current)
            context["nl"] = ""
            try:
                image = ensure_within(
                    Path(str(job["dataset_root"])),
                    Path(str(job["dataset_root"])) / str(row["relative_image_path"]),
                )
                image_path = str(image) if image.is_file() else None
            except PathSafetyError as exc:
                raise TokenBudgetReviewError("Token Budget rewrite image path is invalid") from exc
            try:
                item = build_short_rewrite_item(
                    sample_id=sample_id, lease_id=f"review-rewrite-{sample_id}-{expected_versions[str(sample_id)]}",
                    relative_image_path=str(row["relative_image_path"]), caption_preset=preset,
                    image_path=image_path,
                    primary_character_name=primary, user_supplement=supplement,
                    json_context=json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                    ocr_context=self._rewrite_ocr_context(
                        layout=layout,
                        dataset_root=Path(str(job["dataset_root"])),
                        relative_image_path=str(row["relative_image_path"]),
                        ocr=ocr,
                    ),
                    current_nl=str(current["nl"]),
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TokenBudgetReviewError("Token Budget rewrite request is invalid") from exc
            prepared_items.append((sample_id, item, current, base_digest))
        base_digests = {str(sample_id): digest for sample_id, _, _, digest in prepared_items}

        def write_operation(status: str, http_attempts: int, *, proposals: list[dict[str, object]] | None = None) -> None:
            value: dict[str, object] = {
                "schemaVersion": 1,
                "operationId": operation_id,
                "status": status,
                "httpAttempts": http_attempts,
                "sampleIds": sorted(sample_ids),
                "expectedVersions": operation_input["expectedVersions"],
                "baseAnnotationSha256": base_digests,
            }
            if proposals is not None:
                value["proposals"] = proposals
            layout.write_resource(self._rewrite_relative(operation_id), (canonical_json(value) + "\n").encode("utf-8"))

        write_operation("request_started", 0)
        proposals: list[dict[str, object]] = []
        attempts = 0
        for sample_id, item, current, base_digest in prepared_items:
            allowance = self._reserve_short_rewrite_attempts(
                maximum=maximum_attempts,
                per_sample=per_sample_attempts,
            )
            try:
                raw_rewrite = self._call_short_rewriter(item, allowance=allowance)
            except Exception as exc:
                self._settle_short_rewrite_attempts(reserved=allowance, charged=allowance)
                attempts += allowance
                write_operation("outcome_unknown", attempts)
                raise TokenBudgetReviewError("Token Budget short rewrite has an unknown outcome") from exc
            if isinstance(raw_rewrite, NlOutcomeV1):
                outcome_attempts = raw_rewrite.httpAttempts if type(raw_rewrite.httpAttempts) is int and 1 <= raw_rewrite.httpAttempts <= 5 else 1
                if (
                    raw_rewrite.sampleId != sample_id
                    or raw_rewrite.leaseId != item["leaseId"]
                    or raw_rewrite.relativeImagePath != item["relativeImagePath"]
                    or raw_rewrite.code is not None
                    or raw_rewrite.retriable
                    or not isinstance(raw_rewrite.nl, str)
                    or not raw_rewrite.nl
                    or outcome_attempts != raw_rewrite.httpAttempts
                    or outcome_attempts > allowance
                ):
                    self._settle_short_rewrite_attempts(reserved=allowance, charged=allowance)
                    attempts += allowance
                    write_operation("outcome_unknown", attempts)
                    raise TokenBudgetReviewError("Token Budget short rewrite has an unknown outcome")
                self._settle_short_rewrite_attempts(reserved=allowance, charged=outcome_attempts)
                attempts += outcome_attempts
                rewritten = raw_rewrite.nl
            elif isinstance(raw_rewrite, str) and raw_rewrite:
                self._settle_short_rewrite_attempts(reserved=allowance, charged=1)
                attempts += 1
                rewritten = raw_rewrite
            else:
                self._settle_short_rewrite_attempts(reserved=allowance, charged=allowance)
                attempts += allowance
                write_operation("outcome_unknown", attempts)
                raise TokenBudgetReviewError("Token Budget short rewrite has an unknown outcome")
            _, current_digest = self._current_annotation(job, layout, sample_id, caption_format)
            if current_digest != base_digest:
                write_operation("conflict", attempts)
                raise TokenBudgetReviewConflictError("Token Budget rewrite working annotation changed")
            candidate = dict(current)
            candidate["nl"] = rewritten
            try:
                proposal = self.recount(
                    sample_id,
                    expected_version=expected_versions[str(sample_id)],
                    annotation=candidate,
                    _after_request=True,
                    _expected_base_digest=base_digest,
                )
            except TokenBudgetReviewConflictError:
                write_operation("conflict", attempts)
                raise
            except Exception as exc:
                write_operation("outcome_unknown", attempts)
                raise TokenBudgetReviewError("Token Budget short rewrite has an unknown outcome") from exc
            layout.write_resource(self._rewrite_proposal_relative(sample_id), (canonical_json({
                "schemaVersion": 1, "operationId": operation_id, "proposal": proposal,
            }) + "\n").encode("utf-8"))
            proposals.append(proposal)
        write_operation("completed", attempts, proposals=proposals)
        return {"operationId": operation_id, "sampleIds": sorted(sample_ids), "proposals": proposals}

    def page(self, *, after_sample_id: int | None = None, limit: int = 200) -> dict[str, object]:
        if type(limit) is not int or not 1 <= limit <= 500 or (after_sample_id is not None and (type(after_sample_id) is not int or after_sample_id < 1)):
            raise TokenBudgetReviewError("Token Budget review page is invalid")
        job, layout = self._job_and_layout()
        predicate = "" if after_sample_id is None else " AND issue.sample_id>?"
        parameters: list[object] = [self.job_id]
        if after_sample_id is not None:
            parameters.append(after_sample_id)
        parameters.append(limit)
        rows = list(self.database.connection.execute(
            """SELECT issue.sample_id,sample.relative_image_path
                 FROM issues AS issue JOIN samples AS sample
                   ON sample.job_id=issue.job_id AND sample.sample_id=issue.sample_id
                WHERE issue.job_id=? AND issue.module_id='token_budget'
                  AND issue.code='token_budget_overflow' AND issue.resolved_at IS NULL
                  AND issue.blocking=1 AND sample.in_processing_scope=1""" + predicate + " ORDER BY issue.sample_id LIMIT ?",
            parameters,
        ))
        items: list[dict[str, object]] = []
        for row in rows:
            review = self._read_review(layout, int(row["sample_id"]))
            caption_format, _ = self._frozen_config(job, review)
            annotation, _ = self._current_annotation(job, layout, review.sampleId, caption_format)
            proposal = self._read_proposal(layout, review.sampleId)
            rewrite_proposal = self._read_rewrite_proposal(layout, review.sampleId)
            items.append({
                "sampleId": review.sampleId,
                "relativeImagePath": str(row["relative_image_path"]),
                "review": {
                    "status": review.status, "originalTokens": review.originalTokens,
                    "finalTokens": review.finalTokens, "removed": review.removed,
                    "maxTokens": review.maxTokens, "resourceId": review.resourceId,
                    "resourceFingerprint": review.resourceFingerprint,
                    "version": 1 if proposal is None else proposal["version"],
                },
                "annotation": annotation,
                "proposal": proposal,
                "rewriteProposal": rewrite_proposal,
            })
        target_count = int(self.database.connection.execute(
            """SELECT COUNT(*) FROM issues AS issue JOIN samples AS sample
                   ON sample.job_id=issue.job_id AND sample.sample_id=issue.sample_id
                 WHERE issue.job_id=? AND issue.module_id='token_budget'
                   AND issue.code='token_budget_overflow' AND issue.resolved_at IS NULL
                   AND issue.blocking=1 AND sample.in_processing_scope=1""",
            (self.job_id,),
        ).fetchone()[0])
        return {
            "items": items,
            "targetCount": target_count,
            "nextAfterSampleId": int(rows[-1]["sample_id"]) if len(rows) == limit else None,
        }
