from __future__ import annotations

from pathlib import Path

from .replacement import ReplacementError, replace_projection
from .resource import ReplaceResource, ReplaceResourceError, load_custom_replace_resource, load_replace_resource


def bounded_message(message: str, limit: int = 1_024) -> str:
    """Truncate an error message to the parent's UTF-8 byte limit.

    The core validates `message` by encoded byte length, while slicing a Python
    string counts characters. Text with three-byte characters would therefore
    pass the slice and still be rejected, which fails a whole worker batch.
    """
    raw = message.encode("utf-8")
    if len(raw) <= limit:
        return message
    return raw[:limit].decode("utf-8", errors="ignore")


class ReplaceWorkerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ReplaceWorker:
    def __init__(self) -> None:
        self.resource: ReplaceResource | None = None
        self.hello: dict[str, object] | None = None

    def initialize(self, payload: object, *, install_root: Path) -> dict[str, object]:
        if self.resource is not None or not isinstance(payload, dict):
            raise ReplaceWorkerError("replace_protocol_violation", "replace hello payload is invalid")
        common = {"schemaVersion", "payloadType", "jobId", "configHash"}
        bundled = common | {"resourceManifestRelativePath", "resourceFingerprint"}
        custom = common | {"customIndexOverlayRoot", "customIndexPath", "customIndexSha256", "customIndexRuleCount"}
        if (set(payload) != bundled and set(payload) != custom) or payload["schemaVersion"] != 1 or payload["payloadType"] != "replace_hello_request":
            raise ReplaceWorkerError("replace_protocol_violation", "replace hello identity is invalid")
        try:
            if set(payload) == bundled:
                if not isinstance(payload["resourceManifestRelativePath"], str) or not isinstance(payload["resourceFingerprint"], str):
                    raise ReplaceResourceError("replace resource identity is invalid")
                resource = load_replace_resource(install_root, payload["resourceManifestRelativePath"], payload["resourceFingerprint"])
            else:
                resource = load_custom_replace_resource(Path(str(payload["customIndexOverlayRoot"])), str(payload["customIndexPath"]), str(payload["customIndexSha256"]), payload["customIndexRuleCount"])
        except ReplaceResourceError as exc:
            raise ReplaceWorkerError("replace_resource_invalid", str(exc)) from exc
        self.resource = resource
        self.hello = dict(payload)
        return {
            "schemaVersion": 1, "payloadType": "replace_hello_result", "ready": True,
            "indexLoads": 1, "ruleCount": len(resource.rules), "resourceFingerprint": resource.fingerprint,
            "keepNonCanonical": resource.keep_non_canonical,
            "canonicalDirectionConflict": resource.canonical_direction_conflicts,
        }

    def process(self, item: object) -> dict[str, object]:
        if self.resource is None or not isinstance(item, dict) or set(item) != {
            "schemaVersion", "sampleId", "leaseId", "source", "relativeImagePath", "projection",
        }:
            raise ReplaceWorkerError("replace_protocol_violation", "replace work item is invalid")
        if item["schemaVersion"] != 1 or item["source"] != "e621" or type(item["sampleId"]) is not int or not isinstance(item["leaseId"], str) or not isinstance(item["relativeImagePath"], str) or not isinstance(item["projection"], dict):
            raise ReplaceWorkerError("replace_protocol_violation", "replace work item identity is invalid")
        try:
            projection, summary = replace_projection(item["projection"], self.resource.rules)
        except ReplacementError as exc:
            return {
                "schemaVersion": 1, "payloadType": "replace_issue", "sampleId": item["sampleId"], "leaseId": item["leaseId"],
                "source": "e621", "relativeImagePath": item["relativeImagePath"], "code": "replace_json_invalid",
                "severity": "error", "blocking": True, "retriable": False, "message": bounded_message(str(exc)),
            }
        return {
            "schemaVersion": 1, "payloadType": "replace_result", "sampleId": item["sampleId"], "leaseId": item["leaseId"],
            "source": "e621", "relativeImagePath": item["relativeImagePath"], "projection": projection,
            "replaced": summary.replaced, "dropped": summary.dropped, "passthrough": summary.passthrough,
            "keepRewritten": summary.keep_rewritten,
        }

    def process_batch(self, items: object) -> list[dict[str, object]]:
        if not isinstance(items, list) or not 1 <= len(items) <= 500:
            raise ReplaceWorkerError("replace_protocol_violation", "replace batch is invalid")
        identities: set[tuple[int, str]] = set()
        for item in items:
            if not isinstance(item, dict) or type(item.get("sampleId")) is not int or not isinstance(item.get("leaseId"), str):
                raise ReplaceWorkerError("replace_protocol_violation", "replace work item identity is invalid")
            identity = (item["sampleId"], item["leaseId"])
            if identity in identities:
                raise ReplaceWorkerError("replace_protocol_violation", "replace batch has duplicate sample/lease identities")
            identities.add(identity)
        return [self.process(item) for item in items]
