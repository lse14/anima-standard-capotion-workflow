from __future__ import annotations

from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any

from .validation import NlValidationError, normalize_endpoint


MAX_BATCH = 32
MAX_JSON_CONTEXT_BYTES = 262_144
MAX_OCR_CONTEXT_ITEMS = 1024
MAX_OCR_TEXT_BYTES = 16_384
MAX_USER_SUPPLEMENT_BYTES = 16_384
MAX_CHARACTER_NAME_BYTES = 512
CAPTION_PRESETS = frozenset({"general", "style", "character"})
LENGTH_TIERS = frozenset({"short", "medium", "long"})
OCR_POSITIONS = frozenset({
    "top-left", "top-center", "top-right",
    "middle-left", "middle-center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
})


class NlProtocolError(ValueError):
    pass


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NlProtocolError(f"{field} must be an object")
    return value


def _exact(item: dict[str, object], fields: set[str], field: str) -> None:
    if set(item) != fields:
        raise NlProtocolError(f"{field} fields are invalid")


def _string(value: object, field: str, limit: int, *, blank: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value or len(value.encode("utf-8")) > limit or (not blank and not value):
        raise NlProtocolError(f"{field} is invalid")
    return value


def _positive(value: object, field: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise NlProtocolError(f"{field} is invalid")
    return value


@dataclass(frozen=True)
class NlApiPolicyV1:
    concurrency: int
    maxRequestsPerMinute: int | str
    mainAttempts: int
    backupEnabled: bool
    backupAttempts: int
    connectTimeoutSeconds: int
    writeTimeoutSeconds: int
    readTimeoutSeconds: int
    poolTimeoutSeconds: int
    temperature: float
    topP: float
    maxTokens: int

    @classmethod
    def from_dict(cls, value: object) -> "NlApiPolicyV1":
        item = _object(value, "apiPolicy")
        fields = {"concurrency", "maxRequestsPerMinute", "mainAttempts", "backupEnabled", "backupAttempts", "connectTimeoutSeconds", "writeTimeoutSeconds", "readTimeoutSeconds", "poolTimeoutSeconds", "temperature", "topP", "maxTokens", "maxImagePixels", "maxImageSide", "jpegQuality", "maxEncodedImageBytes", "maxJsonContextBytes", "maxResponseBodyBytes", "maxNlBytes"}
        _exact(item, fields, "apiPolicy")
        if type(item["backupEnabled"]) is not bool:
            raise NlProtocolError("backupEnabled is invalid")
        for name in ("temperature", "topP"):
            if isinstance(item[name], bool) or not isinstance(item[name], (int, float)) or not 0 <= float(item[name]) <= 2:
                raise NlProtocolError(f"apiPolicy.{name} is invalid")
        rpm = item["maxRequestsPerMinute"]
        if rpm != "unlimited" and (type(rpm) is not int or not 1 <= rpm <= 100_000):
            raise NlProtocolError("maxRequestsPerMinute is invalid")
        fixed = {
            "maxImagePixels": 8_000_000, "maxImageSide": 4096, "jpegQuality": 95,
            "maxEncodedImageBytes": 12_582_912, "maxJsonContextBytes": 262_144,
            "maxResponseBodyBytes": 1_048_576, "maxNlBytes": 16_384,
        }
        if any(item[name] != expected for name, expected in fixed.items()):
            raise NlProtocolError("frozen API limit is invalid")
        return cls(
            _positive(item["concurrency"], "concurrency", 16), rpm, _positive(item["mainAttempts"], "mainAttempts", 3), item["backupEnabled"],
            _positive(item["backupAttempts"], "backupAttempts", 2), _positive(item["connectTimeoutSeconds"], "connectTimeoutSeconds", 120),
            _positive(item["writeTimeoutSeconds"], "writeTimeoutSeconds", 120), _positive(item["readTimeoutSeconds"], "readTimeoutSeconds", 600),
            _positive(item["poolTimeoutSeconds"], "poolTimeoutSeconds", 120), float(item["temperature"]), float(item["topP"]),
            _positive(item["maxTokens"], "maxTokens", 16_384),
        )


@dataclass(frozen=True)
class NlHelloV1:
    jobId: str
    configHash: str
    endpoint: str
    model: str
    backupModel: str | None
    apiKey: str
    systemPrompt: str
    policy: NlApiPolicyV1
    responseProtocol: str = "nl-v1"
    promptVersion: str | None = None
    protocol: str = "openai_chat_completions"
    providerLabel: str = ""
    jsonMode: str = "prompt_only"


@dataclass(frozen=True)
class NlWorkItemV1:
    sampleId: int
    leaseId: str
    relativeImagePath: str
    imagePath: str | None
    jsonContext: str | None
    ocrContext: dict[str, object] | None = None
    hasOcrContext: bool = False
    captionPreset: str | None = None
    lengthTier: str | None = None
    primaryCharacterName: str | None = None
    userSupplement: str | None = None


def parse_hello(value: object) -> NlHelloV1:
    item = _object(value, "nl hello")
    fields = {"schemaVersion", "payloadType", "jobId", "configHash", "endpoint", "model", "backupModel", "apiKey", "systemPrompt", "apiPolicy"}
    optional = {"responseProtocol", "promptVersion", "protocol", "providerLabel", "jsonMode"}
    if set(item) not in {
        frozenset(fields),
        frozenset(fields | {"responseProtocol"}),
        frozenset(fields | {"responseProtocol", "promptVersion"}),
    } and not (set(item) - set(fields) <= optional):
        raise NlProtocolError("nl hello fields are invalid")
    if item["schemaVersion"] != 1 or item["payloadType"] != "nl_hello_request":
        raise NlProtocolError("nl hello identity is invalid")
    response_protocol = item.get("responseProtocol", "nl-v1")
    if response_protocol not in {"nl-v1", "nl-count-v2"} or (
        response_protocol == "nl-v1" and "responseProtocol" in item
    ):
        raise NlProtocolError("NL response protocol is invalid")
    prompt_version = item.get("promptVersion")
    if prompt_version is not None and (
        response_protocol != "nl-count-v2"
        or prompt_version not in {"nl-default-prompt-v3", "nl-default-prompt-v4"}
    ):
        raise NlProtocolError("NL prompt version is invalid")
    protocol = item.get("protocol", "openai_chat_completions")
    if protocol != "openai_chat_completions":
        raise NlProtocolError("NL provider protocol is invalid")
    provider_label = item.get("providerLabel", "")
    if not isinstance(provider_label, str) or len(provider_label.encode("utf-8")) > 128:
        raise NlProtocolError("NL provider label is invalid")
    json_mode = item.get("jsonMode", "prompt_only")
    if json_mode not in {"prompt_only", "json_object"}:
        raise NlProtocolError("NL JSON mode is invalid")
    backup = item["backupModel"]
    if backup is not None:
        backup = _string(backup, "backupModel", 512)
    try:
        endpoint = normalize_endpoint(item["endpoint"])
    except NlValidationError as exc:
        raise NlProtocolError(str(exc)) from exc
    return NlHelloV1(
        _string(item["jobId"], "jobId", 128), _string(item["configHash"], "configHash", 64), endpoint,
        _string(item["model"], "model", 512), backup, _string(item["apiKey"], "apiKey", 16_384),
        _string(item["systemPrompt"], "systemPrompt", 65_536), NlApiPolicyV1.from_dict(item["apiPolicy"]),
        response_protocol, prompt_version, protocol, provider_label, json_mode,
    )


def _ocr_context(value: object) -> dict[str, object]:
    context = _object(value, "ocrContext")
    _exact(context, {"items"}, "ocrContext")
    raw_items = context["items"]
    if not isinstance(raw_items, list) or len(raw_items) > MAX_OCR_CONTEXT_ITEMS:
        raise NlProtocolError("ocrContext.items is invalid")
    items: list[list[str]] = []
    for index, entry in enumerate(raw_items):
        if not isinstance(entry, list) or len(entry) != 2:
            raise NlProtocolError(f"ocrContext.items[{index}] is invalid")
        position = entry[0]
        if position not in OCR_POSITIONS:
            raise NlProtocolError(f"ocrContext.items[{index}] position is invalid")
        items.append([position, _string(entry[1], f"ocrContext.items[{index}] text", MAX_OCR_TEXT_BYTES)])
    return {"items": items}


def parse_process(value: object) -> tuple[tuple[NlWorkItemV1, ...], int]:
    item = _object(value, "nl process")
    _exact(item, {"schemaVersion", "payloadType", "items", "httpAttemptAllowance"}, "nl process")
    if item["schemaVersion"] != 1 or item["payloadType"] != "nl_process_request" or not isinstance(item["items"], list) or not 1 <= len(item["items"]) <= MAX_BATCH or type(item["httpAttemptAllowance"]) is not int or not 1 <= item["httpAttemptAllowance"] <= 160:
        raise NlProtocolError("nl process request is invalid")
    parsed: list[NlWorkItemV1] = []
    for raw in item["items"]:
        work = _object(raw, "nl work item")
        fields = {"schemaVersion", "sampleId", "leaseId", "relativeImagePath", "imagePath", "jsonContext"}
        v6_fields = fields | {"captionPreset", "lengthTier", "primaryCharacterName", "userSupplement", "ocrContext"}
        if set(work) not in (fields, fields | {"ocrContext"}, v6_fields):
            raise NlProtocolError("nl work item fields are invalid")
        if work["schemaVersion"] != 1 or type(work["sampleId"]) is not int or work["sampleId"] < 1:
            raise NlProtocolError("nl work identity is invalid")
        image_path = work["imagePath"]
        if image_path is not None:
            image_path = _string(image_path, "imagePath", 16_384)
            if not PureWindowsPath(image_path).is_absolute():
                raise NlProtocolError("imagePath must be an absolute Windows path")
        context = work["jsonContext"]
        if context is not None:
            context = _string(context, "jsonContext", MAX_JSON_CONTEXT_BYTES, blank=True)
        if image_path is None and context is None:
            raise NlProtocolError("nl work item requires image or JSON context")
        has_ocr_context = "ocrContext" in work
        ocr_context = None if work.get("ocrContext") is None else _ocr_context(work["ocrContext"])
        caption_preset = length_tier = primary_character_name = user_supplement = None
        if set(work) == v6_fields:
            caption_preset = work["captionPreset"]
            if caption_preset not in CAPTION_PRESETS:
                raise NlProtocolError("captionPreset is invalid")
            length_tier = work["lengthTier"]
            if length_tier not in LENGTH_TIERS:
                raise NlProtocolError("lengthTier is invalid")
            primary_character_name = work["primaryCharacterName"]
            if primary_character_name is not None:
                primary_character_name = _string(primary_character_name, "primaryCharacterName", MAX_CHARACTER_NAME_BYTES)
            if caption_preset == "character" and primary_character_name is None:
                raise NlProtocolError("character preset requires primaryCharacterName")
            if caption_preset != "character" and primary_character_name is not None:
                raise NlProtocolError("non-character preset must not include primaryCharacterName")
            user_supplement = _string(work["userSupplement"], "userSupplement", MAX_USER_SUPPLEMENT_BYTES, blank=True)
        parsed.append(NlWorkItemV1(
            work["sampleId"],
            _string(work["leaseId"], "leaseId", 128),
            _string(work["relativeImagePath"], "relativeImagePath", 16_384),
            image_path,
            context,
            ocr_context,
            has_ocr_context,
            caption_preset,
            length_tier,
            primary_character_name,
            user_supplement,
        ))
    if len({(work.sampleId, work.leaseId) for work in parsed}) != len(parsed):
        raise NlProtocolError("nl process items are duplicated")
    return tuple(parsed), item["httpAttemptAllowance"]


def process_result(
    item: NlWorkItemV1,
    *,
    nl: str,
    request_id: str | None,
    usage: dict[str, int],
    http_attempts: int,
    observation: dict[str, object] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schemaVersion": 1,
        "payloadType": "nl_result" if observation is None else "nl_result_v2",
        "sampleId": item.sampleId,
        "leaseId": item.leaseId,
        "relativeImagePath": item.relativeImagePath,
        "nl": nl,
        "requestId": request_id,
        "usage": usage,
        "httpAttempts": http_attempts,
    }
    if observation is not None:
        result["observation"] = observation
    return result


def process_issue(item: NlWorkItemV1, code: str, message: str, *, retriable: bool, http_attempts: int = 0) -> dict[str, Any]:
    return {"schemaVersion": 1, "payloadType": "nl_issue", "sampleId": item.sampleId, "leaseId": item.leaseId, "relativeImagePath": item.relativeImagePath, "code": code, "severity": "error", "blocking": True, "retriable": retriable, "message": message[:1024], "httpAttempts": http_attempts}
