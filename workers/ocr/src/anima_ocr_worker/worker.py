from __future__ import annotations

import json
import math
import sys
import unicodedata
from collections.abc import Mapping
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Literal

from .image import (
    DecodedImage,
    OcrImageDecodeError,
    OcrImageTooLargeError,
    OcrSourceFingerprintError,
    decode_and_verify,
    image_exceeds_limits,
    verify_source_fingerprint,
)
from .model import ModelFactory, OcrModelError, PaddleOcrModel, create_paddle_engine, set_offline_environment
from .protocol import (
    MAX_PROCESS_ITEMS,
    MAX_TEXT_BYTES,
    OcrHelloRequest,
    OcrPayloadError,
    OcrWorkItem,
    parse_hello,
    parse_process,
    process_result,
)
from .resource import OcrResource, OcrResourceError, load_ocr_resource


MAX_OUTPUT_PAYLOAD_BYTES = 1_000_000
INFERENCE_FAILURE_MESSAGE = "OCR inference failed for this image."
IMAGE_DECODE_FAILURE_MESSAGE = "OCR could not decode this image."
MODEL_INFERENCE_FAILURE_MESSAGE = "OCR model inference failed for this image."
MODEL_OUTPUT_FAILURE_MESSAGE = "OCR model output was invalid for this image."
UNEXPECTED_FAILURE_MESSAGE = "OCR encountered an unexpected error for this image."
OUTPUT_ENCODING_FAILURE_MESSAGE = "OCR result could not be encoded safely."
OUTPUT_SIZE_FAILURE_MESSAGE = "OCR result exceeds the output safety limit."
OVERSIZE_MESSAGE = "OCR image dimensions exceed the first-release safety limit."


class OcrWorkerInitializationError(RuntimeError):
    pass


class OcrInferenceError(RuntimeError):
    pass


def _runtime_evidence() -> dict[str, object]:
    """Read the worker's own manifest and Paddle state; never trust a hello request."""
    executable = Path(sys.executable).resolve()
    runtime_root = executable.parent
    runtime_id = runtime_root.name
    if runtime_id not in {"ocr-paddle", "ocr-paddle-gpu"}:
        raise OcrModelError("OCR runtime identity is invalid")
    install_root = runtime_root.parent.parent
    manifest_path = install_root / "manifests" / "runtimes" / f"{runtime_id}.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        runtime = manifest["runtime"]
        expected_executable = install_root / Path(str(runtime["interpreterRelativePath"]).replace("\\", "/"))
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OcrModelError("OCR runtime identity is invalid") from exc
    if (
        runtime.get("runtimeId") != runtime_id
        or runtime.get("owner") != "ocr"
        or expected_executable.resolve() != executable
    ):
        raise OcrModelError("OCR runtime identity is invalid")
    try:
        import paddle

        paddle_version = metadata.version("paddlepaddle-gpu" if runtime_id == "ocr-paddle-gpu" else "paddlepaddle")
        expected_paddle_version = "3.3.0" if runtime_id == "ocr-paddle-gpu" else "3.2.2"
        if paddle_version != expected_paddle_version or metadata.version("paddleocr") != "3.7.0" or metadata.version("paddlex") != "3.7.2":
            raise OcrModelError("OCR runtime dependency versions are invalid")
        compiled = paddle.device.is_compiled_with_cuda()
    except OcrModelError:
        raise
    except Exception as exc:
        raise OcrModelError("OCR runtime dependency evidence is invalid") from exc
    if type(compiled) is not bool:
        raise OcrModelError("OCR runtime device evidence is invalid")
    evidence: dict[str, object] = {
        "runtimeId": runtime_id,
        "runtimeFingerprint": sha256(manifest_bytes).hexdigest(),
        "paddleVersion": paddle_version,
        "compiledWithCuda": compiled,
    }
    if runtime_id == "ocr-paddle":
        if compiled:
            raise OcrModelError("OCR runtime device evidence is invalid")
        return {**evidence, "cudaVersion": None, "gpuName": None, "totalVramBytes": None}
    try:
        if not compiled or paddle.device.cuda.device_count() < 1:
            raise OcrModelError("CUDA is unavailable in the OCR GPU runtime")
        cuda_version = paddle.version.cuda()
        gpu_name = paddle.device.cuda.get_device_name()
    except OcrModelError:
        raise
    except Exception as exc:
        raise OcrModelError("CUDA is unavailable in the OCR GPU runtime") from exc
    for value in (cuda_version, gpu_name):
        if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 128:
            raise OcrModelError("OCR runtime device evidence is invalid")
    try:
        total_vram_bytes = int(paddle.device.cuda.get_device_properties(0).total_memory)
    except Exception:
        total_vram_bytes = None
    if total_vram_bytes is not None and total_vram_bytes < 0:
        total_vram_bytes = None
    return {
        **evidence,
        "cudaVersion": cuda_version,
        "gpuName": gpu_name,
        "totalVramBytes": total_vram_bytes,
    }


def _finite_number(value: object, field: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, (bool, str, bytes, bytearray, Mapping)):
        raise OcrInferenceError(f"{field} is invalid")
    try:
        result = float(value)  # Supports finite NumPy scalar values without importing NumPy.
    except (TypeError, ValueError, OverflowError) as exc:
        raise OcrInferenceError(f"{field} is invalid") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise OcrInferenceError(f"{field} is invalid")
    return result


def _sequence(value: object, field: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise OcrInferenceError(f"{field} is invalid")
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise OcrInferenceError(f"{field} is invalid") from exc


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        raise OcrInferenceError("rec_texts is invalid")
    try:
        if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
            raise OcrInferenceError("rec_texts is invalid")
    except UnicodeEncodeError as exc:
        raise OcrInferenceError("rec_texts is invalid") from exc
    result = "".join(character for character in value if unicodedata.category(character) != "Cc")
    try:
        if not result.strip():
            return None
        if len(result.encode("utf-8")) > MAX_TEXT_BYTES:
            raise OcrInferenceError("rec_texts is invalid")
    except UnicodeEncodeError as exc:
        raise OcrInferenceError("rec_texts is invalid") from exc
    return result


def _polygon(value: object, image: DecodedImage) -> list[list[float]]:
    points = _sequence(value, "rec_polys")
    if len(points) != 4:
        raise OcrInferenceError("rec_polys is invalid")
    result: list[list[float]] = []
    for point in points:
        coordinates = _sequence(point, "rec_polys")
        if len(coordinates) != 2:
            raise OcrInferenceError("rec_polys is invalid")
        x = _finite_number(coordinates[0], "rec_polys")
        y = _finite_number(coordinates[1], "rec_polys")
        if not 0 <= x <= image.width or not 0 <= y <= image.height:
            raise OcrInferenceError("rec_polys is outside the image")
        result.append([x, y])
    return result


def _bbox(value: object, polygon: list[list[float]], image: DecodedImage) -> list[float]:
    values = _sequence(value, "rec_boxes")
    if len(values) != 4:
        raise OcrInferenceError("rec_boxes is invalid")
    left, top, right, bottom = [_finite_number(entry, "rec_boxes") for entry in values]
    if not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height:
        raise OcrInferenceError("rec_boxes is outside the image")
    tolerance = 1e-6
    for x, y in polygon:
        if x < left - tolerance or x > right + tolerance or y < top - tolerance or y > bottom + tolerance:
            raise OcrInferenceError("rec_polys escapes rec_boxes")
    return [left, top, right, bottom]


def _orientation(value: object) -> float:
    raw = _finite_number(value, "textline_orientation_angles")
    if raw == 0:
        return 0.0
    if raw == 1:
        return 180.0
    raise OcrInferenceError("textline_orientation_angles is invalid")


def _raw_items(value: object, image: DecodedImage) -> list[dict[str, object]]:
    if not isinstance(value, Mapping):
        raise OcrInferenceError("PaddleOCR result is invalid")
    required = ("rec_texts", "rec_scores", "rec_polys", "rec_boxes")
    if any(name not in value for name in required):
        raise OcrInferenceError("PaddleOCR result fields are incomplete")
    sequences = {name: _sequence(value[name], name) for name in required}
    count = len(sequences["rec_texts"])
    if count > MAX_PROCESS_ITEMS or any(len(values) != count for values in sequences.values()):
        raise OcrInferenceError("PaddleOCR result field lengths are invalid")
    if count == 0 and "textline_orientation_angles" not in value:
        return []
    if "textline_orientation_angles" not in value:
        raise OcrInferenceError("PaddleOCR result fields are incomplete")
    sequences["textline_orientation_angles"] = _sequence(
        value["textline_orientation_angles"],
        "textline_orientation_angles",
    )
    if len(sequences["textline_orientation_angles"]) != count:
        raise OcrInferenceError("PaddleOCR result field lengths are invalid")
    items: list[dict[str, object]] = []
    for index in range(count):
        text = _clean_text(sequences["rec_texts"][index])
        if text is None:
            continue
        polygon = _polygon(sequences["rec_polys"][index], image)
        items.append({
            "text": text,
            "confidence": _finite_number(sequences["rec_scores"][index], "rec_scores", minimum=0, maximum=1),
            "polygonPixels": polygon,
            "bboxPixels": _bbox(sequences["rec_boxes"][index], polygon, image),
            "textlineOrientationDegrees": _orientation(sequences["textline_orientation_angles"][index]),
        })
    return items


def _identity(item: OcrWorkItem, width: int, height: int) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "sampleId": item.sample_id,
        "leaseId": item.lease_id,
        "relativeImagePath": item.relative_image_path,
        "image": {
            "width": max(1, width),
            "height": max(1, height),
            "sizeBytes": item.image_size,
            "sha256": item.image_sha256,
        },
    }


def _inference_failure(
    item: OcrWorkItem,
    width: int,
    height: int,
    *,
    message: str = INFERENCE_FAILURE_MESSAGE,
) -> dict[str, object]:
    return {
        **_identity(item, width, height),
        "status": "failed",
        "items": [],
        "error": {
            "code": "ocr_inference_failed",
            "message": message,
            "retriable": True,
        },
    }


def _oversize_failure(item: OcrWorkItem, width: int, height: int) -> dict[str, object]:
    return {
        **_identity(item, width, height),
        "status": "failed",
        "items": [],
        "error": {
            "code": "ocr_image_too_large",
            "message": OVERSIZE_MESSAGE,
            "retriable": False,
            "details": {
                "actualPixels": width * height,
                "maxPixels": 40_000_000,
                "maxSide": 16_384,
            },
        },
    }


def _decode_failure(item: OcrWorkItem, error: OcrImageDecodeError) -> dict[str, object]:
    """Preserve an oversize classification when header parsing reports dimensions."""
    if (
        type(error.width) is int
        and type(error.height) is int
        and error.width > 0
        and error.height > 0
        and image_exceeds_limits(error.width, error.height)
    ):
        return _oversize_failure(item, error.width, error.height)
    return _inference_failure(
        item,
        error.width or 1,
        error.height or 1,
        message=IMAGE_DECODE_FAILURE_MESSAGE,
    )


class OcrWorker:
    def __init__(self, *, model_factory: ModelFactory = create_paddle_engine) -> None:
        self._model_factory = model_factory
        self.hello: OcrHelloRequest | None = None
        self.resource: OcrResource | None = None
        self.model: PaddleOcrModel | None = None

    def initialize(self, payload: object, *, resource_root: Path) -> dict[str, object]:
        if self.hello is not None or self.model is not None:
            raise OcrWorkerInitializationError("OCR worker is already initialized")
        try:
            hello = parse_hello(payload)
            resource = load_ocr_resource(
                resource_root,
                hello.resource_manifest_relative_path,
                hello.resource_fingerprint,
            )
            set_offline_environment()
            evidence: dict[str, object] | None = None
            device: Literal["cpu", "cuda"] = "cpu"
            if hello.expected_runtime_id is not None:
                evidence = _runtime_evidence()
                if (
                    evidence.get("runtimeId") != hello.expected_runtime_id
                    or evidence.get("runtimeFingerprint") != hello.expected_runtime_fingerprint
                ):
                    raise OcrModelError("OCR runtime identity does not match the frozen request")
                device = "cuda" if evidence["runtimeId"] == "ocr-paddle-gpu" else "cpu"
            model = self._model_factory(
                resource,
                device=device,
                text_det_limit_side_len=(
                    1920 if hello.text_det_limit_side_len is None else hello.text_det_limit_side_len
                ),
                text_batch_size=1 if hello.text_batch_size is None else hello.text_batch_size,
            )
        except (OcrPayloadError, OcrResourceError, OcrModelError) as exc:
            raise OcrWorkerInitializationError("OCR worker initialization failed") from exc
        except Exception as exc:
            raise OcrWorkerInitializationError("OCR worker initialization failed") from exc
        self.hello = hello
        self.resource = resource
        self.model = model
        result: dict[str, object] = {
            "schemaVersion": 1,
            "payloadType": "ocr_hello_result",
            "ready": True,
            "executable": sys.executable,
            "pythonVersion": ".".join(map(str, sys.version_info[:3])),
            "modelSessionLoads": 1,
            "resourceFingerprint": resource.fingerprint,
        }
        if evidence is not None:
            observed = "cuda" if evidence["runtimeId"] == "ocr-paddle-gpu" else "cpu"
            result.update({"requestedDevice": hello.requested_device, "observedDevice": observed, **evidence})
        return result

    def _process_item(self, item: OcrWorkItem) -> dict[str, object]:
        if self.model is None:
            raise OcrWorkerInitializationError("OCR worker is not initialized")
        try:
            decoded = decode_and_verify(item)
        except OcrImageTooLargeError as exc:
            return _oversize_failure(item, exc.width, exc.height)
        except OcrImageDecodeError as exc:
            return _decode_failure(item, exc)
        try:
            results = self.model.predict(decoded.image)
            if len(results) != 1:
                raise OcrInferenceError("PaddleOCR must return one result per image")
        except OcrModelError:
            return _inference_failure(
                item,
                decoded.width,
                decoded.height,
                message=MODEL_INFERENCE_FAILURE_MESSAGE,
            )
        except Exception:
            return _inference_failure(
                item,
                decoded.width,
                decoded.height,
                message=UNEXPECTED_FAILURE_MESSAGE,
            )
        try:
            raw_items = _raw_items(results[0], decoded)
            verify_source_fingerprint(item)
        except OcrSourceFingerprintError:
            raise
        except OcrInferenceError:
            return _inference_failure(
                item,
                decoded.width,
                decoded.height,
                message=MODEL_OUTPUT_FAILURE_MESSAGE,
            )
        except Exception:
            return _inference_failure(
                item,
                decoded.width,
                decoded.height,
                message=UNEXPECTED_FAILURE_MESSAGE,
            )
        if not raw_items:
            return {
                **_identity(item, decoded.width, decoded.height),
                "status": "no_text",
                "items": [],
            }
        return {
            **_identity(item, decoded.width, decoded.height),
            "status": "success",
            "items": raw_items,
        }

    def _process_prediction(
        self,
        item: OcrWorkItem,
        decoded: DecodedImage,
        result: object,
    ) -> dict[str, object]:
        try:
            raw_items = _raw_items(result, decoded)
            verify_source_fingerprint(item)
        except OcrSourceFingerprintError:
            raise
        except OcrInferenceError:
            return _inference_failure(
                item,
                decoded.width,
                decoded.height,
                message=MODEL_OUTPUT_FAILURE_MESSAGE,
            )
        except Exception:
            return _inference_failure(
                item,
                decoded.width,
                decoded.height,
                message=UNEXPECTED_FAILURE_MESSAGE,
            )
        if not raw_items:
            return {
                **_identity(item, decoded.width, decoded.height),
                "status": "no_text",
                "items": [],
            }
        return {
            **_identity(item, decoded.width, decoded.height),
            "status": "success",
            "items": raw_items,
        }

    def process(self, payload: object) -> dict[str, object]:
        if self.model is None or self.hello is None:
            raise OcrWorkerInitializationError("OCR worker is not initialized")
        request = parse_process(payload)
        outcomes_by_identity: dict[tuple[int, str], dict[str, object]] = {}
        decoded_items: list[tuple[OcrWorkItem, DecodedImage]] = []
        for item in request.items:
            try:
                decoded_items.append((item, decode_and_verify(item)))
            except OcrImageTooLargeError as exc:
                outcomes_by_identity[(item.sample_id, item.lease_id)] = _oversize_failure(item, exc.width, exc.height)
            except OcrImageDecodeError as exc:
                outcomes_by_identity[(item.sample_id, item.lease_id)] = _decode_failure(item, exc)

        predictions: list[object | None] = []
        if decoded_items:
            try:
                predictions = self.model.predict_batch([decoded.image for _, decoded in decoded_items])
                if len(predictions) != len(decoded_items):
                    raise OcrInferenceError("PaddleOCR returned the wrong number of image results")
            except OcrModelError:
                predictions = [None] * len(decoded_items)
            except Exception:
                predictions = [None] * len(decoded_items)

        for (item, decoded), prediction in zip(decoded_items, predictions, strict=True):
            if prediction is None:
                outcomes_by_identity[(item.sample_id, item.lease_id)] = _inference_failure(
                    item,
                    decoded.width,
                    decoded.height,
                    message=MODEL_INFERENCE_FAILURE_MESSAGE,
                )
            else:
                outcomes_by_identity[(item.sample_id, item.lease_id)] = self._process_prediction(item, decoded, prediction)

        outcomes: list[dict[str, object]] = []
        for item in request.items:
            outcome = outcomes_by_identity[(item.sample_id, item.lease_id)]
            candidate = [*outcomes, outcome]
            try:
                encoded = json.dumps(
                    {"schemaVersion": 1, "payloadType": "ocr_process_result", "items": candidate},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, UnicodeEncodeError, ValueError):
                outcome = _inference_failure(item, 1, 1, message=OUTPUT_ENCODING_FAILURE_MESSAGE)
                encoded = b""
            if len(encoded) > MAX_OUTPUT_PAYLOAD_BYTES:
                image = outcome["image"]
                assert isinstance(image, dict)
                outcome = _inference_failure(
                    item,
                    int(image["width"]),
                    int(image["height"]),
                    message=OUTPUT_SIZE_FAILURE_MESSAGE,
                )
            outcomes.append(outcome)
        return process_result(outcomes)
