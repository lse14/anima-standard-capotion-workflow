from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from . import PROTOCOL_VERSION, SCHEMA_VERSION


Profile = Literal["e621", "danbooru"]
WorkMode = Literal["in_place", "full_copy"]
OverwriteMode = Literal["incremental", "rebuild"]
ModuleId = Literal["workspace", "caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"]
SampleModuleId = Literal["caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"]
LEGACY_PIPELINE_MODULE_IDS = ("caption", "classify", "replace", "nl", "dropout", "export")
CURRENT_PIPELINE_MODULE_IDS = ("caption", "classify", "replace", "nl", "count_review", "dropout", "export")
OCR_PIPELINE_MODULE_IDS = ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "export")
TOKEN_BUDGET_PIPELINE_MODULE_IDS = ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export")
CURRENT_JOB_CONFIG_SCHEMA_VERSION = 10
COUNT_REVIEW_SCHEMA_VERSIONS = frozenset({3, 4, 5, 6, 7, 8, 9, 10})
OCR_JOB_CONFIG_SCHEMA_VERSIONS = frozenset({5, 6, 7, 8, 9, 10})
OCR_DEVICE_JOB_CONFIG_SCHEMA_VERSIONS = frozenset({7, 8, 9, 10})
CAPTION_INPUT_TXT_MODE_SCHEMA_VERSIONS = frozenset({8, 9, 10})
NL_V4_JOB_CONFIG_SCHEMA_VERSIONS = frozenset({6, 7, 8, 9, 10})
NL_V4_PROMPT_VERSION = "nl-default-prompt-v4"
NL_MOSS_PROMPT_VERSION = "nl-default-prompt-moss-v1"
NL_MOSS_RESPONSE_PROTOCOL = "nl-moss-v1"
NL_STANDARD_CAPTION_PRESETS = frozenset({"general", "style", "character"})
NL_MOSS_CAPTION_PRESETS = frozenset({"moss-general", "moss-style", "moss-character"})
TOKEN_BUDGET_JOB_CONFIG_SCHEMA_VERSIONS = frozenset({6, 7, 8, 9, 10})
MODULE_BATCH_SIZE_BOUNDS: dict[str, tuple[int, int]] = {
    "caption": (1, 64),
    "classify": (1, 500),
    "replace": (1, 500),
    "ocr": (1, 1024),
    "nl": (1, 16),
    "countReview": (1, 500),
    "dropout": (1, 16),
    "tokenBudget": (1, 500),
    "export": (1, 500),
}
DEFAULT_MODULE_BATCH_SIZE: dict[str, int] = {
    "caption": 4,
    "classify": 128,
    "replace": 128,
    "ocr": 4,
    "nl": 3,
    "countReview": 100,
    "dropout": 4,
    "tokenBudget": 128,
    "export": 500,
}
RESOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
TOKENIZER_RESOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
RESOURCE_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
CATEGORY_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RESOURCE_REFERENCE_FIELDS = {"resourceId", "resourceManifestRelativePath", "resourceFingerprint"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _schema_capability(schema_version: object, supported: frozenset[int]) -> bool:
    return type(schema_version) is int and schema_version in supported


def job_config_supports_ocr(schema_version: object) -> bool:
    return _schema_capability(schema_version, OCR_JOB_CONFIG_SCHEMA_VERSIONS)


def job_config_supports_ocr_device(schema_version: object) -> bool:
    return _schema_capability(schema_version, OCR_DEVICE_JOB_CONFIG_SCHEMA_VERSIONS)


def job_config_supports_caption_input_txt_mode(schema_version: object) -> bool:
    return _schema_capability(schema_version, CAPTION_INPUT_TXT_MODE_SCHEMA_VERSIONS)


def job_config_supports_nl_v4(schema_version: object) -> bool:
    return _schema_capability(schema_version, NL_V4_JOB_CONFIG_SCHEMA_VERSIONS)


def job_config_supports_token_budget(schema_version: object) -> bool:
    return _schema_capability(schema_version, TOKEN_BUDGET_JOB_CONFIG_SCHEMA_VERSIONS)


def pipeline_module_ids(config_schema_version: int) -> tuple[str, ...]:
    if config_schema_version == 2:
        return LEGACY_PIPELINE_MODULE_IDS
    if config_schema_version == 5:
        return OCR_PIPELINE_MODULE_IDS
    if job_config_supports_token_budget(config_schema_version):
        return TOKEN_BUDGET_PIPELINE_MODULE_IDS
    if config_schema_version in COUNT_REVIEW_SCHEMA_VERSIONS:
        return CURRENT_PIPELINE_MODULE_IDS
    raise ValueError(f"unsupported JobConfig schemaVersion: {config_schema_version}")


def profile_supports_job_config_schema(profile: object, schema_version: object) -> bool:
    """Keep legacy tasks runnable while allowing shared OCR in v5."""
    if type(schema_version) is not int:
        return False
    return (
        profile == "e621" and schema_version in {2, 3, 4, 5, 6, 7, 8}
    ) or (
        profile == "danbooru" and schema_version in {4, 5, 6, 7, 8}
    )


def caption_display_term(term: str, *, replace_underscores: bool, preserve_escapes: bool) -> str:
    """Mirror the Caption TXT display transform so preflight sees the real term."""
    value = term.replace("_", " ") if replace_underscores else term
    if preserve_escapes:
        value = "".join(f"\\{character}" if character in "\\()" else character for character in value)
    return value


@dataclass(frozen=True)
class CaptionFormatPolicy:
    replaceUnderscoresWithSpaces: bool = True
    preserveEscapes: bool = True
    triggersEnabled: bool = False
    triggerTerms: tuple[str, ...] = ()
    flatTxtLayout: Literal["single_line", "nl_newline"] = "nl_newline"


@dataclass(frozen=True)
class ImageDecodePolicy:
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    rejectMultiFrame: bool = True
    applyExifTranspose: bool = True
    alphaBackground: str = "#FFFFFF"
    invalidImageAction: Literal["block", "skip"] = "block"


class _DefaultNlConfig(dict[str, Any]):
    """Marks a generated NL section so the schema can select its frozen prompt."""


def _default_nl_config() -> dict[str, Any]:
    return _DefaultNlConfig({
        "enabled": True,
        "reuseOriginalNl": True,
        "apiEnabled": True,
        "useImage": True,
        "useFullJson": False,
        "systemPrompt": "",
        "promptVersion": "nl-default-prompt-v2",
    })


@dataclass(frozen=True)
class JobConfig:
    workMode: WorkMode
    overwriteMode: OverwriteMode
    sourceRoot: str
    outputRoot: str | None = None
    annotationBackup: Literal["required"] = "required"
    recursive: bool = False
    captionFormat: CaptionFormatPolicy = field(default_factory=CaptionFormatPolicy)
    imageDecode: ImageDecodePolicy = field(default_factory=ImageDecodePolicy)
    caption: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "thresholdMode": "model_default",
        "overwriteTxt": False,
        "resourceId": "caption-e621-eva02-large-full-v1",
    })
    classify: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "indexMode": "bundled",
        "overwriteJson": False,
        "overwriteCount": False,
        "resourceId": "classify-e621-20260724-v1",
    })
    replace: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "indexMode": "bundled",
        "resourceId": "replace-e621-20260726-v2",
    })
    ocr: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "llmMinConfidence": 0.5,
        "forceReprocess": False,
        "resourceId": "ocr-ppocrv5-server-paddle-v1",
    })
    nl: dict[str, Any] = field(default_factory=_default_nl_config)
    countReview: dict[str, Any] | None = field(default_factory=lambda: {
        "enabled": True,
        "protocolVersion": "count-review-v1",
    })
    dropout: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "policyVersion": "dataset-batch-policy-v1",
        "seed": "anima-policy-default-v1",
        "artist": {"enabled": True, "dropoutProbability": 0.0, "addAtPrefix": True},
        "quality": {
            "enabled": True,
            "dropoutProbability": 0.0,
            "device": "auto",
            "batchSize": 4,
            "resourceId": "lse14-scorer-5k-v1",
        },
        "appearanceNl": {
            "enabled": True,
            "solo": {"dropNl": 0.70, "dropAppearance": 0.05},
            "nonSolo": {"dropNl": 0.05, "dropAppearance": 0.70},
            "unknown": {"dropNl": 0.15, "dropAppearance": 0.15},
        },
    })
    export: dict[str, Any] = field(default_factory=lambda: {"format": "both"})
    tokenBudget: dict[str, Any] | None = None
    moduleBatchSize: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MODULE_BATCH_SIZE))
    schemaVersion: int = CURRENT_JOB_CONFIG_SCHEMA_VERSION
    profile: Profile | None = None

    def __post_init__(self) -> None:
        if type(self.nl) is _DefaultNlConfig:
            nl = dict(self.nl)
            if self.schemaVersion == 5:
                nl["promptVersion"] = "nl-default-prompt-v3"
            elif job_config_supports_nl_v4(self.schemaVersion):
                nl.update({
                    "promptVersion": "nl-default-prompt-v4",
                    "captionPreset": "general",
                    "lengthDistribution": {"short": 33, "medium": 34, "long": 33},
                    "lengthSeed": "anima-nl-length-v1",
                })
            object.__setattr__(self, "nl", nl)
        if job_config_supports_token_budget(self.schemaVersion) and self.tokenBudget is None:
            object.__setattr__(self, "tokenBudget", {
                "enabled": True,
                "maxTokens": 490,
                "resourceId": "tokenizer-qwen3-0.6b-anima-v1",
            })
        if job_config_supports_caption_input_txt_mode(self.schemaVersion) and isinstance(self.caption, dict):
            object.__setattr__(self, "caption", {
                **self.caption,
                "inputTxtMode": self.caption.get("inputTxtMode", "tag"),
                "taggerFallbackOnMissingTxt": self.caption.get("taggerFallbackOnMissingTxt", True),
            })
        if job_config_supports_ocr_device(self.schemaVersion) and isinstance(self.ocr, dict) and "device" not in self.ocr:
            object.__setattr__(self, "ocr", {**self.ocr, "device": "auto"})

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.schemaVersion == CURRENT_JOB_CONFIG_SCHEMA_VERSION:
            value.pop("profile")
        value["captionFormat"]["triggerTerms"] = list(self.captionFormat.triggerTerms)
        if self.schemaVersion != CURRENT_JOB_CONFIG_SCHEMA_VERSION:
            value["captionFormat"].pop("flatTxtLayout", None)
            value.pop("moduleBatchSize", None)
        if not job_config_supports_ocr(self.schemaVersion):
            value.pop("ocr")
        if not job_config_supports_token_budget(self.schemaVersion):
            value.pop("tokenBudget")
        if self.schemaVersion == 2:
            value.pop("countReview")
        return value

    @property
    def config_hash(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class SampleRecord:
    sampleId: int
    relativeImagePath: str
    annotationKey: str
    source: Profile
    inProcessingScope: bool
    imageFormat: Literal["jpeg", "png", "webp", "bmp"]
    imageFrameCount: int
    originalTxtState: Literal["missing_or_blank", "nonblank"]
    originalJsonState: Literal["missing_or_blank", "nonblank"]


@dataclass(frozen=True)
class SampleManifest:
    jobId: str
    recursive: bool
    sampleCount: int
    generatedAt: str
    schemaVersion: int = 1


@dataclass(frozen=True)
class SampleRunState:
    sampleId: int
    txtProvenance: Literal["missing", "original_preserved", "module1_written"] = "missing"
    currentModuleId: SampleModuleId | None = None
    status: Literal[
        "pending", "leased", "prepared", "request_started", "response_staged", "completed", "failed", "skipped"
    ] = "pending"
    attempt: int = 0
    leaseId: str | None = None
    workerInstanceId: str | None = None
    leaseExpiresAt: str | None = None
    preparedArtifactRelativePath: str | None = None
    preparedArtifactSha256: str | None = None


@dataclass(frozen=True)
class WorkLease:
    """Bounded unit of work handed to one worker instance."""

    jobId: str
    moduleId: SampleModuleId
    sampleId: int
    status: Literal["pending", "leased", "prepared", "completed", "failed", "skipped"]
    attempt: int
    configHash: str
    leaseId: str | None = None
    workerInstanceId: str | None = None
    leaseExpiresAt: str | None = None


MODULE_STATUS_VALUES = frozenset({
    "pending", "running", "paused", "completed", "completed_with_issues", "failed",
    "skipped", "skipped_not_available",
})


@dataclass(frozen=True)
class JobState:
    jobId: str
    configHash: str
    status: Literal[
        "draft", "preflighting", "ready", "preparing_workspace", "running", "paused", "interrupted",
        "reviewing", "exporting", "committing", "cancelling", "cancelled_recoverable", "succeeded", "succeeded_with_issues", "failed", "discarded"
    ]
    currentModuleId: ModuleId | None = None
    manifestSchemaVersion: int = 1
    lastEventId: int = 0
    pinned: bool = False
    createdAt: str = field(default_factory=utc_now)
    startedAt: str | None = None
    cancelRequestedAt: str | None = None
    finishedAt: str | None = None
    resumeStatus: str | None = None


@dataclass(frozen=True)
class SampleIssue:
    issueId: str
    jobId: str
    sampleId: int | None
    relativeImagePath: str | None
    moduleId: SampleModuleId
    code: str
    severity: Literal["info", "warning", "error"]
    blocking: bool
    retriable: bool
    message: str
    attempt: int = 0
    repairStartModule: SampleModuleId | None = None
    fieldErrors: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class ProgressEvent:
    jobId: str
    eventId: int
    moduleId: ModuleId
    status: Literal[
        "pending", "running", "paused", "completed", "completed_with_issues", "failed", "skipped", "skipped_not_available"
    ]
    completed: int
    total: int
    configHash: str
    attempt: int
    occurredAt: str = field(default_factory=utc_now)
    sampleId: int | None = None
    issueCode: str | None = None
    message: str | None = None


def validate_job_config(
    config: JobConfig,
    *,
    allow_unavailable_profile: bool = False,
    adjustable_categories: tuple[str, ...] | None = None,
) -> None:
    if config.schemaVersion not in {2, 3, 4, 5, 6, 7, 8, 9, CURRENT_JOB_CONFIG_SCHEMA_VERSION}:
        raise ValueError("unsupported JobConfig schemaVersion")
    if config.schemaVersion == 2:
        if config.countReview is not None:
            raise ValueError("JobConfig v2 must not contain countReview")
        if config.nl.get("promptVersion") != "nl-default-prompt-v1":
            raise ValueError("JobConfig v2 must use NL prompt v1")
    else:
        if not isinstance(config.countReview, dict) or set(config.countReview) != {"enabled", "protocolVersion"}:
            raise ValueError("countReview configuration is invalid")
        if type(config.countReview.get("enabled")) is not bool:
            raise ValueError("countReview enabled must be boolean")
        if config.countReview.get("protocolVersion") != "count-review-v1":
            raise ValueError("countReview protocolVersion is invalid")
        expected_prompt_version = (
            NL_V4_PROMPT_VERSION if job_config_supports_nl_v4(config.schemaVersion)
            else "nl-default-prompt-v3" if config.schemaVersion == 5
            else "nl-default-prompt-v2"
        )
        actual_prompt_version = config.nl.get("promptVersion")
        if job_config_supports_nl_v4(config.schemaVersion):
            if actual_prompt_version not in {expected_prompt_version, NL_MOSS_PROMPT_VERSION}:
                raise ValueError("NL prompt version is invalid")
        elif actual_prompt_version != expected_prompt_version:
            raise ValueError(f"JobConfig v{config.schemaVersion} must use {expected_prompt_version}")
    nl = config.nl
    base_nl_fields = {
        "enabled", "reuseOriginalNl", "apiEnabled", "useImage", "useFullJson", "systemPrompt", "promptVersion",
        "apiProfileId", "apiPolicy", "apiRuntime",
    }
    required_nl_fields = base_nl_fields - {"apiProfileId", "apiPolicy", "apiRuntime"}
    v6_nl_fields = {"captionPreset", "lengthDistribution", "lengthSeed"}
    allowed_nl_fields = base_nl_fields | (v6_nl_fields if job_config_supports_nl_v4(config.schemaVersion) else set())
    if not isinstance(nl, dict) or not required_nl_fields.issubset(nl) or set(nl) - allowed_nl_fields:
        raise ValueError("nl configuration fields are invalid")
    if "apiRuntime" in nl:
        runtime = nl["apiRuntime"]
        if runtime is not None:
            if not isinstance(runtime, dict) or set(runtime) != {
                "revision", "protocol", "providerLabel", "endpoint", "model", "backupModel",
                "credentialRef", "jsonMode", "policy", "appliedAt",
            }:
                raise ValueError("NL apiRuntime is invalid")
            if type(runtime["revision"]) is not int or runtime["revision"] < 1:
                raise ValueError("NL apiRuntime revision is invalid")
            if runtime["protocol"] != "openai_chat_completions" or runtime["jsonMode"] not in {"prompt_only", "json_object"}:
                raise ValueError("NL apiRuntime protocol or jsonMode is invalid")
            for field in ("providerLabel", "endpoint", "model", "credentialRef", "appliedAt"):
                if not isinstance(runtime[field], str) or not runtime[field].strip() or "\x00" in runtime[field]:
                    raise ValueError(f"NL apiRuntime {field} is invalid")
            if runtime["backupModel"] is not None and not isinstance(runtime["backupModel"], str):
                raise ValueError("NL apiRuntime backupModel is invalid")
            if not isinstance(runtime["policy"], dict):
                raise ValueError("NL apiRuntime policy is invalid")
            if any("key" in str(key).casefold() or "token" in str(key).casefold() or "secret" in str(key).casefold() for key in runtime["policy"]):
                raise ValueError("NL apiRuntime policy must not contain credentials")
    if job_config_supports_nl_v4(config.schemaVersion):
        if not v6_nl_fields.issubset(nl):
            raise ValueError("JobConfig v6 requires NL routing fields")
        preset_values = NL_MOSS_CAPTION_PRESETS if nl.get("promptVersion") == NL_MOSS_PROMPT_VERSION else NL_STANDARD_CAPTION_PRESETS
        if nl.get("captionPreset") not in preset_values:
            raise ValueError("nl captionPreset is invalid")
        if (
            nl.get("enabled") is True
            and nl.get("promptVersion") == NL_MOSS_PROMPT_VERSION
            and (nl.get("apiEnabled") is not True or nl.get("useImage") is not True)
        ):
            raise ValueError("MOSS NL requires API generation with image input")
        distribution = nl.get("lengthDistribution")
        if not isinstance(distribution, dict) or set(distribution) != {"short", "medium", "long"}:
            raise ValueError("nl lengthDistribution is invalid")
        if any(type(value) is not int or not 0 <= value <= 100 for value in distribution.values()) or sum(distribution.values()) != 100:
            raise ValueError("nl lengthDistribution must contain integer percentages totaling 100")
        length_seed = nl.get("lengthSeed")
        if not isinstance(length_seed, str) or not length_seed.strip() or len(length_seed.encode("utf-8")) > 256:
            raise ValueError("nl lengthSeed must be non-blank and at most 256 UTF-8 bytes")
    if config.schemaVersion == CURRENT_JOB_CONFIG_SCHEMA_VERSION:
        api_policy = nl.get("apiPolicy")
        if api_policy is not None and (not isinstance(api_policy, dict) or "concurrency" in api_policy):
            raise ValueError("NL concurrency is controlled by moduleBatchSize.nl")
        if config.profile is not None:
            raise ValueError("task profile is not supported by JobConfig v10")
        if not isinstance(config.moduleBatchSize, dict) or set(config.moduleBatchSize) != set(MODULE_BATCH_SIZE_BOUNDS):
            raise ValueError("moduleBatchSize must contain each module exactly once")
        for module_name, (minimum, maximum) in MODULE_BATCH_SIZE_BOUNDS.items():
            value = config.moduleBatchSize[module_name]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"moduleBatchSize.{module_name} must be between {minimum} and {maximum}")
    elif config.schemaVersion != 9:
        if config.profile not in ("e621", "danbooru"):
            raise ValueError("unsupported profile")
        if config.profile == "danbooru" and config.schemaVersion in {2, 3} and not allow_unavailable_profile:
            raise ValueError("profile_not_available:danbooru")
    if config.workMode == "in_place" and config.outputRoot is not None:
        raise ValueError("in_place must omit outputRoot")
    if config.workMode == "full_copy" and not config.outputRoot:
        raise ValueError("full_copy requires outputRoot")
    caption_keys = set(config.caption)
    if not {"enabled", "thresholdMode", "overwriteTxt"}.issubset(caption_keys):
        raise ValueError("caption configuration is incomplete")
    caption_input_txt_mode_fields = {
        "inputTxtMode", "taggerFallbackOnMissingTxt",
    } if job_config_supports_caption_input_txt_mode(config.schemaVersion) else set()
    if caption_keys - {
        "enabled", "thresholdMode", "overwriteTxt", "uniformThreshold", "categoryThresholds",
        *caption_input_txt_mode_fields,
        *RESOURCE_REFERENCE_FIELDS,
    }:
        raise ValueError("caption configuration contains unknown fields")
    if type(config.caption.get("enabled")) is not bool or type(config.caption.get("overwriteTxt")) is not bool:
        raise ValueError("caption enabled and overwriteTxt must be boolean")
    if job_config_supports_caption_input_txt_mode(config.schemaVersion):
        if not caption_input_txt_mode_fields.issubset(caption_keys):
            raise ValueError("JobConfig v8 caption TXT input mode is incomplete")
        if config.caption.get("inputTxtMode") not in {"tag", "nl"}:
            raise ValueError("caption inputTxtMode is invalid")
        if type(config.caption.get("taggerFallbackOnMissingTxt")) is not bool:
            raise ValueError("caption taggerFallbackOnMissingTxt must be boolean")
    threshold_mode = config.caption.get("thresholdMode")
    if threshold_mode not in {"model_default", "uniform", "per_category"}:
        raise ValueError("invalid thresholdMode")
    raw_categories = config.caption.get("categoryThresholds")
    if raw_categories is not None and not isinstance(raw_categories, dict):
        raise ValueError("categoryThresholds must be an object")
    categories = raw_categories or {}
    if threshold_mode == "model_default":
        if "uniformThreshold" in config.caption or "categoryThresholds" in config.caption:
            raise ValueError("model_default must not include threshold overrides")
    if threshold_mode == "uniform":
        if "uniformThreshold" not in config.caption:
            raise ValueError("uniform threshold is required")
        if "categoryThresholds" in config.caption:
            raise ValueError("uniform threshold must not include category thresholds")
    if threshold_mode == "per_category":
        if "uniformThreshold" in config.caption:
            raise ValueError("per_category must not include a uniform threshold")
        actual = set(categories)
        if config.schemaVersion in {2, 3}:
            required = {"general", "character", "species", "rating"}
            if actual != required:
                raise ValueError("all category thresholds are required")
        else:
            if not actual or len(actual) > 32 or any(not isinstance(key, str) or not CATEGORY_ID.fullmatch(key) for key in actual):
                raise ValueError("v4 category thresholds must use bounded category identifiers")
            if adjustable_categories is not None and actual != set(adjustable_categories):
                raise ValueError("category thresholds must exactly match the selected tagging resource")
    for value in [config.caption.get("uniformThreshold"), *categories.values()]:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("threshold must be numeric")
        if value is not None and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
            raise ValueError("threshold must be between 0 and 1")
    def resource_reference(section: dict[str, Any], field_name: str, *, resource_id_required: bool = False) -> None:
        resource_id = section.get("resourceId")
        if resource_id is None and resource_id_required:
            raise ValueError(f"{field_name} resourceId is required")
        if resource_id is not None and (not isinstance(resource_id, str) or not RESOURCE_ID.fullmatch(resource_id)):
            raise ValueError(f"{field_name} resourceId is invalid")
        relative = section.get("resourceManifestRelativePath")
        fingerprint = section.get("resourceFingerprint")
        if (relative is None) != (fingerprint is None):
            raise ValueError(f"{field_name} frozen resource reference is incomplete")
        if relative is not None:
            if (
                not isinstance(relative, str) or not relative or "\x00" in relative
                or relative.startswith(("\\", "/")) or ":" in relative
                or any(part in {"", ".", ".."} for part in relative.replace("/", "\\").split("\\"))
            ):
                raise ValueError(f"{field_name} resource manifest path is invalid")
            if not isinstance(fingerprint, str) or not RESOURCE_FINGERPRINT.fullmatch(fingerprint):
                raise ValueError(f"{field_name} resource fingerprint is invalid")

    resource_reference(config.caption, "caption")
    classify = config.classify
    classify_identity_fields = {
        "resourceId", "resourceFingerprint", "wikiDataSourceId",
        "dictionaryEntryCount", "resourceProfile",
    }
    classify_frozen_fields = classify_identity_fields | {"resourceManifestRelativePath"}
    classify_base_fields = {
        "enabled", "indexMode", "overwriteJson", "overwriteCount",
        "resourceId", "customResourcePath", "customResourceContentSha256",
    }
    if config.schemaVersion in {9, CURRENT_JOB_CONFIG_SCHEMA_VERSION}:
        if (
            not isinstance(classify, dict)
            or set(classify) - (classify_base_fields | classify_frozen_fields)
            or not {"enabled", "indexMode", "overwriteJson", "overwriteCount"}.issubset(classify)
        ):
            raise ValueError("classify configuration fields are invalid")
        index_mode = classify.get("indexMode")
        if index_mode not in {"bundled", "custom"}:
            raise ValueError("classify indexMode is invalid")
        if index_mode == "bundled":
            if (
                "customResourcePath" in classify
                or "customResourceContentSha256" in classify
                or "resourceId" not in classify
            ):
                raise ValueError("bundled classify configuration is invalid")
        else:
            has_external_path = "customResourcePath" in classify
            has_any_identity = bool(classify_identity_fields & set(classify))
            has_frozen_reference = classify_frozen_fields <= set(classify)
            has_any_frozen_metadata = has_any_identity or "resourceManifestRelativePath" in classify
            has_content_digest = "customResourceContentSha256" in classify
            valid_input = has_external_path and not has_any_frozen_metadata and not has_content_digest
            valid_inspected = (
                has_external_path
                and classify_identity_fields <= set(classify)
                and has_content_digest
                and not has_frozen_reference
            )
            valid_frozen = not has_external_path and has_frozen_reference and not has_content_digest
            if sum((valid_input, valid_inspected, valid_frozen)) != 1:
                raise ValueError(
                    "custom classify configuration must contain an external input, inspected identity, or frozen reference"
                )
            content_digest = classify.get("customResourceContentSha256")
            if has_content_digest and (
                not isinstance(content_digest, str) or not RESOURCE_FINGERPRINT.fullmatch(content_digest)
            ):
                raise ValueError("custom classify content digest is invalid")
    elif not isinstance(classify, dict) or set(classify) - {
        "enabled", "overwriteJson", "overwriteCount", "wikiDataSourceId", *RESOURCE_REFERENCE_FIELDS
    } or not {"enabled", "overwriteJson", "overwriteCount", "wikiDataSourceId"}.issubset(classify):
        raise ValueError("classify configuration fields are invalid")
    for field_name in ("enabled", "overwriteJson", "overwriteCount"):
        if type(classify.get(field_name)) is not bool:
            raise ValueError(f"classify {field_name} must be boolean")
    wiki_data_source_id = classify.get("wikiDataSourceId")
    if wiki_data_source_id is not None and (
        not isinstance(wiki_data_source_id, str) or not RESOURCE_ID.fullmatch(wiki_data_source_id)
        or len(wiki_data_source_id) > 128
    ):
        raise ValueError("classify wikiDataSourceId is invalid")
    if config.schemaVersion not in {9, CURRENT_JOB_CONFIG_SCHEMA_VERSION}:
        resource_reference(classify, "classify")
    elif "resourceManifestRelativePath" in classify:
        resource_reference(classify, "classify", resource_id_required=True)
        if type(classify.get("dictionaryEntryCount")) is not int or classify["dictionaryEntryCount"] < 1:
            raise ValueError("classify dictionaryEntryCount is invalid")
        if classify.get("resourceProfile") not in {"e621", "danbooru"}:
            raise ValueError("classify resourceProfile is invalid")
    if config.countReview is not None and config.countReview["enabled"] and not classify["enabled"]:
        raise ValueError("countReview requires classify to be enabled")

    if not job_config_supports_ocr_device(config.schemaVersion) and isinstance(config.ocr, dict) and "device" in config.ocr:
        raise ValueError("ocr device is only supported by JobConfig v7 and v8")
    if job_config_supports_ocr(config.schemaVersion):
        ocr = config.ocr
        if not isinstance(ocr, dict) or set(ocr) - {
            "enabled", "llmMinConfidence", "forceReprocess", "device", *RESOURCE_REFERENCE_FIELDS,
        } or not {"enabled", "llmMinConfidence", "forceReprocess", "resourceId"}.issubset(ocr):
            raise ValueError("ocr configuration fields are invalid")
        if job_config_supports_ocr_device(config.schemaVersion):
            if ocr.get("device") not in {"auto", "cuda", "cpu"}:
                raise ValueError("ocr device is invalid")
        elif "device" in ocr:
            raise ValueError("ocr device is only supported by JobConfig v7 and v8")
        if type(ocr.get("enabled")) is not bool:
            raise ValueError("ocr enabled must be boolean")
        if type(ocr.get("forceReprocess")) is not bool:
            raise ValueError("ocr forceReprocess must be boolean")
        confidence = ocr.get("llmMinConfidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("ocr llmMinConfidence must be numeric")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("ocr llmMinConfidence must be between 0 and 1")
        resource_reference(ocr, "ocr", resource_id_required=True)
    if job_config_supports_token_budget(config.schemaVersion):
        token_budget = config.tokenBudget
        frozen_token_budget_fields = {
            "resourceManifestRelativePath", "resourceFingerprint", "contextLimit",
        }
        if not isinstance(token_budget, dict) or set(token_budget) - {
            "enabled", "maxTokens", "resourceId", *frozen_token_budget_fields,
        } or not {"enabled", "maxTokens", "resourceId"}.issubset(token_budget):
            raise ValueError("tokenBudget configuration fields are invalid")
        if type(token_budget.get("enabled")) is not bool:
            raise ValueError("tokenBudget enabled must be boolean")
        if type(token_budget.get("maxTokens")) is not int or token_budget["maxTokens"] < 1:
            raise ValueError("tokenBudget maxTokens must be a positive integer")
        token_budget_resource_id = token_budget.get("resourceId")
        if not isinstance(token_budget_resource_id, str) or not TOKENIZER_RESOURCE_ID.fullmatch(token_budget_resource_id):
            raise ValueError("tokenBudget resourceId is invalid")
        present_frozen_fields = frozen_token_budget_fields & set(token_budget)
        if present_frozen_fields and present_frozen_fields != frozen_token_budget_fields:
            raise ValueError("tokenBudget frozen resource reference is incomplete")
        if present_frozen_fields:
            if token_budget["enabled"] is not True:
                raise ValueError("disabled tokenBudget must not contain a frozen resource reference")
            relative = token_budget["resourceManifestRelativePath"]
            fingerprint = token_budget["resourceFingerprint"]
            context_limit = token_budget["contextLimit"]
            if (
                not isinstance(relative, str) or not relative or "\x00" in relative
                or relative.startswith(("\\", "/")) or ":" in relative
                or any(part in {"", ".", ".."} for part in relative.replace("/", "\\").split("\\"))
            ):
                raise ValueError("tokenBudget resource manifest path is invalid")
            if not isinstance(fingerprint, str) or not RESOURCE_FINGERPRINT.fullmatch(fingerprint):
                raise ValueError("tokenBudget resource fingerprint is invalid")
            if type(context_limit) is not int or context_limit < 1:
                raise ValueError("tokenBudget contextLimit must be a positive integer")
            if token_budget["maxTokens"] > context_limit:
                raise ValueError("tokenBudget maxTokens must not exceed contextLimit")
    elif config.tokenBudget is not None:
        raise ValueError("tokenBudget is only supported by JobConfig v6")
    dropout = config.dropout
    if set(dropout) != {"enabled", "policyVersion", "seed", "artist", "quality", "appearanceNl"}:
        raise ValueError("dropout configuration fields are invalid")
    if type(dropout.get("enabled")) is not bool:
        raise ValueError("dropout enabled must be boolean")
    if dropout.get("policyVersion") != "dataset-batch-policy-v1":
        raise ValueError("dropout policyVersion is invalid")
    seed = dropout.get("seed")
    if not isinstance(seed, str) or not seed.strip() or len(seed.encode("utf-8")) > 256:
        raise ValueError("dropout seed must be non-blank and at most 256 UTF-8 bytes")
    artist = dropout.get("artist")
    quality = dropout.get("quality")
    appearance_nl = dropout.get("appearanceNl")
    if not isinstance(artist, dict) or set(artist) - {"enabled", "dropoutProbability", "addAtPrefix"} or not {"enabled", "dropoutProbability"}.issubset(artist):
        raise ValueError("dropout artist configuration is invalid")
    if not isinstance(quality, dict) or set(quality) - {
        "enabled", "dropoutProbability", "device", "batchSize", *RESOURCE_REFERENCE_FIELDS
    } or not {"enabled", "dropoutProbability", "device", "batchSize"}.issubset(quality):
        raise ValueError("dropout quality configuration is invalid")
    if not isinstance(appearance_nl, dict) or set(appearance_nl) != {
        "enabled", "solo", "nonSolo", "unknown"
    }:
        raise ValueError("dropout appearanceNl configuration is invalid")
    for name, section in (("artist", artist), ("quality", quality), ("appearanceNl", appearance_nl)):
        if type(section.get("enabled")) is not bool:
            raise ValueError(f"dropout {name} enabled must be boolean")
    if "addAtPrefix" in artist and type(artist["addAtPrefix"]) is not bool:
        raise ValueError("dropout artist addAtPrefix must be boolean")

    def probability(value: object, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{field_name} must be between 0 and 1")
        return result

    probability(artist.get("dropoutProbability"), "dropout artist probability")
    probability(quality.get("dropoutProbability"), "dropout quality probability")
    if quality.get("device") not in {"auto", "cuda", "cpu"}:
        raise ValueError("dropout quality device is invalid")
    if type(quality.get("batchSize")) is not int or not 1 <= quality["batchSize"] <= 16:
        raise ValueError("dropout quality batchSize must be between 1 and 16")
    resource_reference(quality, "dropout quality")
    for count_group in ("solo", "nonSolo", "unknown"):
        probabilities = appearance_nl.get(count_group)
        if not isinstance(probabilities, dict) or set(probabilities) != {"dropNl", "dropAppearance"}:
            raise ValueError(f"dropout appearanceNl {count_group} is invalid")
        drop_nl = probability(probabilities.get("dropNl"), f"dropout {count_group} dropNl")
        drop_appearance = probability(
            probabilities.get("dropAppearance"), f"dropout {count_group} dropAppearance"
        )
        if drop_nl + drop_appearance > 1.0 + 1e-12:
            raise ValueError(f"dropout {count_group} probabilities must not exceed 1")
    replace = config.replace
    if not isinstance(replace, dict) or type(replace.get("enabled")) is not bool or replace.get("indexMode") not in {"bundled", "custom"}:
        raise ValueError("replace configuration is invalid")
    replace_mode = replace["indexMode"]
    custom_fields = {"customIndexPath", "customIndexSha256", "customIndexRuleCount"}
    if set(replace) - {"enabled", "indexMode", *custom_fields, *RESOURCE_REFERENCE_FIELDS}:
        raise ValueError("replace configuration contains unknown fields")
    if replace_mode == "bundled":
        if custom_fields & set(replace):
            raise ValueError("bundled replace configuration must not contain custom index fields")
        resource_reference(replace, "replace")
    if replace_mode == "custom":
        if RESOURCE_REFERENCE_FIELDS & set(replace):
            raise ValueError("custom replace configuration must not contain bundled resource fields")
        path = replace.get("customIndexPath")
        if not isinstance(path, str) or not path.strip() or "\x00" in path or len(path.encode("utf-8")) > 32_768:
            raise ValueError("custom replace index path is invalid")
        resolved = custom_fields - {"customIndexPath"}
        if resolved & set(replace) and not resolved <= set(replace):
            raise ValueError("custom replace index resolution is incomplete")
        if "customIndexSha256" in replace and (not isinstance(replace["customIndexSha256"], str) or len(replace["customIndexSha256"]) != 64 or any(c not in "0123456789abcdef" for c in replace["customIndexSha256"])):
            raise ValueError("custom replace index digest is invalid")
        if "customIndexRuleCount" in replace and (type(replace["customIndexRuleCount"]) is not int or not 1 <= replace["customIndexRuleCount"] <= 250_000):
            raise ValueError("custom replace index rule count is invalid")
    if config.annotationBackup != "required":
        raise ValueError("annotation backup is required")
    if tuple(config.imageDecode.extensions) != (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        raise ValueError("image extension policy is frozen for first release")
    if config.imageDecode.rejectMultiFrame is not True or config.imageDecode.applyExifTranspose is not True:
        raise ValueError("multi-frame rejection and EXIF transpose must be enabled")
    if config.imageDecode.alphaBackground != "#FFFFFF":
        raise ValueError("transparent images must use a white background")
    if config.imageDecode.invalidImageAction not in {"block", "skip"}:
        raise ValueError("invalid image action must be block or skip")
    for name, value in (
        ("replaceUnderscoresWithSpaces", config.captionFormat.replaceUnderscoresWithSpaces),
        ("preserveEscapes", config.captionFormat.preserveEscapes),
        ("triggersEnabled", config.captionFormat.triggersEnabled),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
    if config.captionFormat.flatTxtLayout not in {"single_line", "nl_newline"}:
        raise ValueError("caption flatTxtLayout is invalid")
    if len(config.captionFormat.triggerTerms) > 64:
        raise ValueError("at most 64 trigger terms are allowed")
    if config.captionFormat.triggersEnabled and not config.captionFormat.triggerTerms:
        raise ValueError("at least one trigger term is required")
    for term in config.captionFormat.triggerTerms:
        if not isinstance(term, str) or not term or term != term.strip():
            raise ValueError("trigger terms must be non-blank and trimmed")
        if len(term.encode("utf-8")) > 512 or any(character in term for character in ",\r\n\x00"):
            raise ValueError("trigger terms exceed limits or contain forbidden separators")
        displayed = caption_display_term(
            term,
            replace_underscores=config.captionFormat.replaceUnderscoresWithSpaces,
            preserve_escapes=config.captionFormat.preserveEscapes,
        )
        if not displayed or displayed != displayed.strip() or any(character in displayed for character in ",\r\n\x00"):
            raise ValueError("trigger terms must stay non-blank and trimmed after display formatting")
    if "ocr" in config.nl:
        raise ValueError("nl.ocr is not supported")
    if config.nl.get("apiEnabled") and config.nl.get("useImage") is not True:
        raise ValueError("API-enabled NL requires image input")
    if config.export.get("format") not in {"json", "flat_txt", "both"}:
        raise ValueError("invalid export format")


def validate_manifest_record(record: SampleRecord) -> None:
    if record.sampleId < 1:
        raise ValueError("sampleId must be positive")
    if not record.relativeImagePath or not record.annotationKey:
        raise ValueError("manifest paths must be non-empty")
    if record.imageFrameCount < 1:
        raise ValueError("imageFrameCount must be positive")
    if record.imageFormat not in {"jpeg", "png", "webp", "bmp"}:
        raise ValueError("unsupported image format")


def envelope_dict(**kwargs: Any) -> dict[str, Any]:
    value = {"protocolVersion": PROTOCOL_VERSION, **kwargs}
    if "payload" not in value:
        value["payload"] = {}
    return value
