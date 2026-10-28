from __future__ import annotations

import sys
from pathlib import Path

from .formatting import CaptionFormattingError, format_caption
from .image import (
    CaptionImageDecodeError,
    CaptionSourceFingerprintError,
    load_image_rgb,
    validate_dataset_root,
)
from .model import (
    CaptionInferenceError,
    CaptionMetadataError,
    CaptionModelError,
    SessionFactory,
    TaggerAdapter,
    create_tagger_adapter,
    resolve_thresholds,
)
from .protocol import CaptionPayloadError, validate_hello_payload, validate_work_item
from .resource import CaptionResourceError, WorkerCaptionResource, load_caption_resource


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


class CaptionWorkerInitializationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CaptionWorker:
    def __init__(self) -> None:
        self.hello: dict[str, object] | None = None
        self.resource: WorkerCaptionResource | None = None
        self.model: TaggerAdapter | None = None
        self.dataset_root: Path | None = None
        self.thresholds: dict[str, float] | None = None

    def initialize(
        self,
        payload: object,
        *,
        install_root: Path,
        session_factory: SessionFactory | None = None,
    ) -> dict[str, object]:
        if self.model is not None or self.hello is not None:
            raise CaptionWorkerInitializationError("caption_protocol_violation", "caption worker is already initialized")
        try:
            hello = validate_hello_payload(payload)
        except CaptionPayloadError as exc:
            raise CaptionWorkerInitializationError("caption_protocol_violation", str(exc)) from exc
        try:
            dataset_root = validate_dataset_root(str(hello["datasetRoot"]))
        except CaptionSourceFingerprintError as exc:
            raise CaptionWorkerInitializationError("caption_source_fingerprint_mismatch", str(exc)) from exc
        try:
            resource = load_caption_resource(
                install_root,
                str(hello["resourceManifestRelativePath"]),
                str(hello["resourceFingerprint"]),
            )
        except (CaptionResourceError, OSError, ValueError) as exc:
            raise CaptionWorkerInitializationError("caption_resource_invalid", str(exc)) from exc
        if resource.profile != hello["profile"]:
            raise CaptionWorkerInitializationError(
                "caption_profile_mismatch",
                "caption resource profile does not match hello",
            )
        try:
            model = create_tagger_adapter(resource, session_factory=session_factory)
            thresholds = resolve_thresholds(hello["thresholdPolicy"], model.metadata.default_thresholds)
        except CaptionMetadataError as exc:
            raise CaptionWorkerInitializationError("caption_metadata_mismatch", str(exc)) from exc
        except CaptionModelError as exc:
            raise CaptionWorkerInitializationError("caption_model_load_failed", str(exc)) from exc
        self.hello = hello
        self.resource = resource
        self.model = model
        self.dataset_root = dataset_root
        self.thresholds = thresholds
        return {
            "schemaVersion": 1,
            "payloadType": "caption_hello_result",
            "executable": sys.executable,
            "pythonVersion": ".".join(map(str, sys.version_info[:3])),
            "ready": True,
            "provider": model.provider,
            "gpuFallback": model.gpu_fallback,
            "modelSessionLoads": model.session_loads,
            "tagCount": len(model.metadata.tag_names),
            "resourceFingerprint": resource.fingerprint,
        }

    @staticmethod
    def _issue(item: dict[str, object], code: str, message: str, *, retriable: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "schemaVersion": 1,
            "payloadType": "caption_issue",
            "sampleId": item["sampleId"],
            "leaseId": item["leaseId"],
            "source": item["source"],
            "relativeImagePath": item["relativeImagePath"],
            "code": code,
            "severity": "error",
            "blocking": True,
            "retriable": retriable,
            "message": bounded_message(message),
        }
        if retriable:
            result["repairStartModule"] = "caption"
        return result

    def process(self, value: object) -> dict[str, object]:
        if self.hello is None or self.model is None or self.dataset_root is None or self.thresholds is None:
            raise CaptionWorkerInitializationError("caption_protocol_violation", "caption worker is not initialized")
        item = validate_work_item(value)
        try:
            image = load_image_rgb(self.dataset_root, item)
            tensor = self.model.preprocess(image)
        except CaptionSourceFingerprintError:
            raise
        except CaptionImageDecodeError as exc:
            return self._issue(
                item,
                "caption_image_decode_failed",
                f"Image decode failed: {exc}",
                retriable=True,
            )
        try:
            predictions = self.model.predict(tensor, self.thresholds)
        except CaptionInferenceError as exc:
            return self._issue(
                item,
                "caption_inference_failed",
                f"Caption inference failed: {exc}",
                retriable=True,
            )
        if not predictions:
            return self._issue(
                item,
                "caption_no_tags",
                "No model tags matched the frozen thresholds.",
                retriable=False,
            )
        try:
            formatted = format_caption(predictions, self.hello["captionFormat"])
        except CaptionFormattingError as exc:
            return self._issue(
                item,
                "caption_inference_failed",
                f"Caption formatting failed: {exc}",
                retriable=True,
            )
        return {
            "schemaVersion": 1,
            "payloadType": "caption_result",
            "sampleId": item["sampleId"],
            "leaseId": item["leaseId"],
            "source": item["source"],
            "relativeImagePath": item["relativeImagePath"],
            "tags": [
                {"rawTag": prediction.raw_tag, "score": prediction.score, "category": prediction.category}
                for prediction in predictions
            ],
            "formattedTxt": formatted,
            "provider": self.model.provider,
            "modelSessionLoads": self.model.session_loads,
        }

    def process_batch(self, items: list[dict[str, object]]) -> list[dict[str, object]]:
        """Keep item-local failures in their own outcomes while reusing this session."""
        return [self.process(item) for item in items]
