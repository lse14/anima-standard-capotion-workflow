from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))
sys.path.insert(0, str(ROOT / "workers" / "ocr" / "src"))

from anima_core.ocr_protocol import OcrHelloResultV1, parse_ocr_process_result
import anima_ocr_worker.worker as worker_module
from anima_ocr_worker.protocol import OcrPayloadError, hello_to_dict, parse_hello
try:
    from anima_ocr_worker.entry import RUNTIME_ID, run
    from anima_ocr_worker import image as image_module
    from anima_ocr_worker.image import image_exceeds_limits, scaled_dimensions
    from anima_ocr_worker.model import OcrModelError, PaddleOcrModel, create_paddle_engine
    from anima_ocr_worker.resource import OcrResource, OcrResourceError, load_ocr_resource
    from anima_ocr_worker.worker import OcrSourceFingerprintError, OcrWorker, OcrWorkerInitializationError
except ModuleNotFoundError as exc:
    _WORKER_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _WORKER_IMPORT_ERROR = None


RESOURCE_ID = "ocr-ppocrv5-server-paddle-v1"
INFERENCE = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": True,
    "textRecScoreThresh": 0,
    "textDetLimitSideLen": 1920,
    "textDetLimitType": "max",
}
ENTRYPOINTS = {
    "detection": r"detection\inference.json",
    "recognition": r"recognition\inference.json",
    "textlineOrientation": r"textline-orientation\inference.json",
}
MODEL_FILES = {
    r"detection\inference.json": b'{"model":"det"}',
    r"detection\model.pdiparams": b"det-params",
    r"recognition\inference.json": b'{"model":"rec"}',
    r"recognition\model.pdiparams": b"rec-params",
    r"textline-orientation\inference.json": b'{"model":"orientation"}',
    r"textline-orientation\model.pdiparams": b"orientation-params",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_resource(
    root: Path,
    *,
    distribution: dict[str, object] | None = None,
) -> tuple[Path, str, str]:
    package = root / "ocr-models" / RESOURCE_ID
    package.mkdir(parents=True)
    files: dict[str, dict[str, object]] = {}
    for relative, content in MODEL_FILES.items():
        target = package / Path(relative.replace("\\", os.sep))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[relative] = {"sizeBytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    metadata = {
        "models": {
            "detection": "PP-OCRv5_server_det",
            "recognition": "PP-OCRv5_server_rec",
            "textlineOrientation": "PP-LCNet_x1_0_textline_ori",
        },
        "inference": dict(INFERENCE),
    }
    manifest = {
        "schemaVersion": 2,
        "kind": "ocr-model",
        "resourceId": RESOURCE_ID,
        "resourceVersion": "ppocrv5-server-paddle-v1",
        "profile": "shared",
        "displayName": {"en": "OCR test", "zh-CN": "OCR test"},
        "description": {"en": "OCR test", "zh-CN": "OCR test"},
        "runtimeFormat": "ppocrv5-server-paddle-v1",
        "distribution": distribution or {
            "mode": "local-only",
            "sourceUrl": "https://example.invalid/ppocrv5",
            "licenseStatus": "unverified",
        },
        "entrypoints": dict(ENTRYPOINTS),
        "files": files,
        "metadata": metadata,
        "documentation": [],
    }
    unsigned = {
        name: manifest[name]
        for name in (
            "schemaVersion", "kind", "resourceId", "resourceVersion", "profile", "runtimeFormat",
            "entrypoints", "files", "metadata", "distribution",
        )
    }
    fingerprint = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    manifest_path = package / "resource.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return root, r"ocr-models\ocr-ppocrv5-server-paddle-v1\resource.json", fingerprint


def _hello(manifest_relative: str, fingerprint: str) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "payloadType": "ocr_hello_request",
        "jobId": "ocr-job",
        "configHash": "a" * 64,
        "resourceId": RESOURCE_ID,
        "resourceManifestRelativePath": manifest_relative,
        "resourceFingerprint": fingerprint,
        "inference": dict(INFERENCE),
    }


def _work_item(path: Path, *, sample_id: int = 1, lease_id: str = "lease-1") -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "sampleId": sample_id,
        "leaseId": lease_id,
        "relativeImagePath": path.name,
        "imagePath": str(path),
        "imageSize": path.stat().st_size,
        "imageSha256": _sha256(path),
    }


def _request(items: list[dict[str, object]]) -> dict[str, object]:
    return {"schemaVersion": 1, "payloadType": "ocr_process_request", "items": items}


def _expected_entry_runtime_id() -> str:
    runtime_id = Path(sys.executable).resolve().parent.name
    return runtime_id if runtime_id in {"ocr-paddle", "ocr-paddle-gpu"} else "ocr-paddle"


def _entry_frame(
    message_id: str,
    method: str,
    payload: dict[str, object],
    *,
    job_id: str | None = None,
    config_hash: str | None = None,
) -> bytes:
    frame: dict[str, object] = {
        "protocolVersion": "1.0",
        "kind": "request",
        "messageId": message_id,
        "runtimeId": _expected_entry_runtime_id(),
        "owner": "ocr",
        "method": method,
        "payload": payload,
    }
    if job_id is not None:
        frame["jobId"] = job_id
    if config_hash is not None:
        frame["configHash"] = config_hash
    return json.dumps(frame, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _raw_result(*, texts: list[object] | None = None, scores: list[object] | None = None, polys: list[object] | None = None, boxes: list[object] | None = None, angles: list[object] | None = None) -> dict[str, object]:
    return {
        "rec_texts": ["same", "same"] if texts is None else texts,
        "rec_scores": [0.8, 0.9] if scores is None else scores,
        "rec_polys": [
            [[0, 0], [2, 0], [2, 1], [0, 1]],
            [[0, 1], [2, 1], [2, 2], [0, 2]],
        ] if polys is None else polys,
        "rec_boxes": [[0, 0, 2, 1], [0, 1, 2, 2]] if boxes is None else boxes,
        "textline_orientation_angles": [0, 1] if angles is None else angles,
    }


class _FakeEngine:
    def __init__(self, result: object) -> None:
        self.result = result
        self.images: list[object] = []

    def predict(self, image: object) -> list[object]:
        self.images.append(image)
        return [self.result]


class _FakeFactory:
    def __init__(self, engine: _FakeEngine, *, convert_input_to_array: bool = False) -> None:
        self.engine = engine
        self.convert_input_to_array = convert_input_to_array
        self.calls = 0
        self.resources: list[object] = []
        self.devices: list[str | None] = []
        self.text_det_limit_side_lengths: list[int] = []
        self.text_batch_sizes: list[int] = []
        self.environments: list[dict[str, str | None]] = []

    def __call__(
        self,
        resource: object,
        *,
        device: str | None = None,
        text_det_limit_side_len: int = 1920,
        text_batch_size: int = 1,
    ) -> PaddleOcrModel:
        self.calls += 1
        self.resources.append(resource)
        self.devices.append(device)
        self.text_det_limit_side_lengths.append(text_det_limit_side_len)
        self.text_batch_sizes.append(text_batch_size)
        self.environments.append({
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        })
        if self.convert_input_to_array:
            return PaddleOcrModel(self.engine, convert_input_to_array=True)
        return PaddleOcrModel(self.engine)


class OcrWorkerTests(unittest.TestCase):
    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_empty_candidates_are_ignored_without_failing_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            resource_root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
            image_path = temporary / "image.png"
            Image.new("RGB", (4, 4), "white").save(image_path)
            worker = OcrWorker(model_factory=_FakeFactory(_FakeEngine(_raw_result(
                texts=["valid", "   "],
                scores=[0.9, 0.8],
            ))))
            worker.initialize(_hello(manifest_relative, fingerprint), resource_root=resource_root)

            outcome = parse_ocr_process_result(worker.process(_request([_work_item(image_path)])))[0]

        self.assertEqual("success", outcome.status)
        self.assertEqual(["valid"], [item.text for item in outcome.items])

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_all_empty_candidates_produce_no_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            resource_root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
            image_path = temporary / "image.png"
            Image.new("RGB", (4, 4), "white").save(image_path)
            worker = OcrWorker(model_factory=_FakeFactory(_FakeEngine(_raw_result(
                texts=["", "\t"],
                scores=[0.9, 0.8],
            ))))
            worker.initialize(_hello(manifest_relative, fingerprint), resource_root=resource_root)

            outcome = parse_ocr_process_result(worker.process(_request([_work_item(image_path)])))[0]

        self.assertEqual("no_text", outcome.status)

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_empty_paddle_result_without_orientation_is_no_text(self) -> None:
        """PaddleOCR omits orientation output when it detects no text lines."""
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            resource_root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
            image_path = temporary / "image.png"
            Image.new("RGB", (4, 4), "white").save(image_path)
            empty_result = _raw_result(texts=[], scores=[], polys=[], boxes=[], angles=[])
            del empty_result["textline_orientation_angles"]
            worker = OcrWorker(model_factory=_FakeFactory(_FakeEngine(empty_result)))
            worker.initialize(_hello(manifest_relative, fingerprint), resource_root=resource_root)

            outcome = parse_ocr_process_result(worker.process(_request([_work_item(image_path)])))[0]

        self.assertEqual("no_text", outcome.status)
        self.assertEqual((), outcome.items)

    def test_runtime_evidence_uses_runtime_specific_paddle_versions(self) -> None:
        cases = (
            ("ocr-paddle", "paddlepaddle", "3.2.2", False),
            ("ocr-paddle-gpu", "paddlepaddle-gpu", "3.3.0", True),
        )
        for runtime_id, distribution, paddle_version, compiled_with_cuda in cases:
            with self.subTest(runtime_id=runtime_id), tempfile.TemporaryDirectory() as temporary_name:
                install_root = Path(temporary_name)
                executable = install_root / "runtimes" / runtime_id / "python.exe"
                executable.parent.mkdir(parents=True)
                executable.write_bytes(b"")
                manifest = install_root / "manifests" / "runtimes" / f"{runtime_id}.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps({
                    "runtime": {
                        "runtimeId": runtime_id,
                        "owner": "ocr",
                        "interpreterRelativePath": f"runtimes/{runtime_id}/python.exe",
                    },
                }), encoding="utf-8")
                fake_cuda = types.SimpleNamespace(
                    device_count=lambda: 1,
                    get_device_name=lambda: "fixture GPU",
                    get_device_properties=lambda _index: types.SimpleNamespace(total_memory=8589934592),
                )
                fake_paddle = types.ModuleType("paddle")
                fake_paddle.device = types.SimpleNamespace(  # type: ignore[attr-defined]
                    is_compiled_with_cuda=lambda: compiled_with_cuda,
                    cuda=fake_cuda,
                )
                fake_paddle.version = types.SimpleNamespace(cuda=lambda: "12.9")  # type: ignore[attr-defined]
                versions = {
                    distribution: paddle_version,
                    "paddleocr": "3.7.0",
                    "paddlex": "3.7.2",
                }
                with mock.patch.object(worker_module.sys, "executable", str(executable)), mock.patch.dict(
                    sys.modules, {"paddle": fake_paddle}
                ), mock.patch.object(worker_module.metadata, "version", side_effect=versions.__getitem__) as version:
                    evidence = worker_module._runtime_evidence()

                self.assertEqual(paddle_version, evidence["paddleVersion"])
                self.assertEqual(
                    [distribution, "paddleocr", "paddlex"],
                    [call.args[0] for call in version.call_args_list],
                )

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_paddle_engine_passes_tuning_only_to_detection_and_textline_batches(self) -> None:
        required = {"text_det_limit_side_len", "text_batch_size"}
        supported = required <= set(inspect.signature(create_paddle_engine).parameters)
        self.assertTrue(supported, "Task 3.4 requires bounded PaddleOCR execution tuning")
        if not supported:
            return
        observed: dict[str, object] = {}

        class FakePaddleOcr:
            def __init__(self, **kwargs: object) -> None:
                observed["kwargs"] = kwargs

            def predict(self, image: object) -> list[object]:
                return []

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name) / "models"
            for directory in ("detection", "recognition", "textline-orientation"):
                (root / directory).mkdir(parents=True)
            resource = OcrResource(
                RESOURCE_ID, "0" * 64, root, root / "detection", root / "recognition", root / "textline-orientation", {},
            )
            fake_paddle = types.ModuleType("paddle")
            fake_paddleocr = types.ModuleType("paddleocr")
            fake_paddleocr.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
            with mock.patch.dict(sys.modules, {"paddle": fake_paddle, "paddleocr": fake_paddleocr}):
                create_paddle_engine(resource, device="cpu", text_det_limit_side_len=2560, text_batch_size=4)
        kwargs = observed["kwargs"]
        assert isinstance(kwargs, dict)
        self.assertEqual(
            {
                "text_det_limit_side_len": 2560,
                "textline_orientation_batch_size": 4,
                "text_recognition_batch_size": 4,
            },
            {name: kwargs[name] for name in (
                "text_det_limit_side_len", "textline_orientation_batch_size", "text_recognition_batch_size",
            )},
        )

    def test_worker_hello_preserves_legacy_and_parses_complete_device_request_only(self) -> None:
        legacy = _hello(r"ocr-models\ocr-ppocrv5-server-paddle-v1\resource.json", "b" * 64)
        self.assertEqual(legacy, hello_to_dict(parse_hello(legacy)))
        extended = {
            **legacy,
            "requestedDevice": "auto",
            "expectedRuntimeId": "ocr-paddle",
            "expectedRuntimeFingerprint": "c" * 64,
        }
        try:
            parsed = parse_hello(extended)
        except OcrPayloadError as exc:
            self.fail(f"valid extended OCR hello request was rejected: {exc}")
        self.assertEqual(extended, hello_to_dict(parsed))
        tuned = {
            **extended,
            "inference": {**INFERENCE, "textDetLimitSideLen": 2560},
            "executionTuning": {"textDetLimitSideLen": 2560, "textBatchSize": 4},
        }
        try:
            parsed_tuned = parse_hello(tuned)
        except OcrPayloadError as exc:
            self.fail(f"Task 3.4 worker execution tuning was rejected: {exc}")
        self.assertEqual(tuned, hello_to_dict(parsed_tuned))
        self.assertEqual((2560, 4), (parsed_tuned.text_det_limit_side_len, parsed_tuned.text_batch_size))
        for payload in (
            {**legacy, "requestedDevice": "cpu"},
            {**extended, "requestedDevice": "GPU"},
            {**extended, "expectedRuntimeFingerprint": "C" * 64},
            {**extended, "requestedDevice": "cuda", "expectedRuntimeId": "ocr-paddle"},
            {**extended, "requestedDevice": "cpu", "expectedRuntimeId": "ocr-paddle-gpu"},
        ):
            with self.subTest(payload=payload), self.assertRaises(OcrPayloadError):
                parse_hello(payload)

    def test_worker_source_package_is_available(self) -> None:
        self.assertIsNone(_WORKER_IMPORT_ERROR, "anima_ocr_worker source package is missing")

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_paddle_engine_requires_verified_cuda_before_constructing_gpu_engine(self) -> None:
        calls: list[str] = []

        class FakePaddleOcr:
            def __init__(self, **kwargs: object) -> None:
                calls.append("construct")
                self.kwargs = kwargs

            def predict(self, image: object) -> list[object]:
                return []

        class FakeCuda:
            def device_count(self) -> int:
                calls.append("count")
                return 1

        class FakeDevice:
            cuda = FakeCuda()

            @staticmethod
            def is_compiled_with_cuda() -> bool:
                calls.append("compiled")
                return True

            @staticmethod
            def set_device(value: str) -> None:
                calls.append(f"set:{value}")

        fake_paddle = types.ModuleType("paddle")
        fake_paddle.device = FakeDevice()  # type: ignore[attr-defined]
        fake_paddleocr = types.ModuleType("paddleocr")
        fake_paddleocr.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name) / "models"
            for directory in ("detection", "recognition", "textline-orientation"):
                (root / directory).mkdir(parents=True)
            resource = OcrResource(RESOURCE_ID, "0" * 64, root, root / "detection", root / "recognition", root / "textline-orientation", {})
            with mock.patch.dict(sys.modules, {"paddle": fake_paddle, "paddleocr": fake_paddleocr}):
                try:
                    model = create_paddle_engine(resource, device="cuda")
                except TypeError as exc:
                    self.fail(f"CUDA device selection is not part of the Paddle engine boundary: {exc}")
            self.assertIsInstance(model, PaddleOcrModel)
        self.assertEqual(["compiled", "count", "set:gpu:0", "construct"], calls)

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_extended_hello_uses_frozen_runtime_identity_for_engine_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
            factory = _FakeFactory(_FakeEngine(_raw_result()))
            extended = {
                **_hello(manifest_relative, fingerprint),
                "requestedDevice": "auto",
                "expectedRuntimeId": "ocr-paddle-gpu",
                "expectedRuntimeFingerprint": "d" * 64,
                "inference": {**INFERENCE, "textDetLimitSideLen": 2560},
                "executionTuning": {"textDetLimitSideLen": 2560, "textBatchSize": 4},
            }
            expected_evidence = {
                "runtimeId": "ocr-paddle-gpu",
                "runtimeFingerprint": "d" * 64,
                "paddleVersion": "3.2.2",
                "compiledWithCuda": True,
                "cudaVersion": "12.6",
                "gpuName": "NVIDIA GPU",
                "totalVramBytes": 24 * 1024 ** 3,
            }
            with mock.patch.object(worker_module, "_runtime_evidence", return_value=expected_evidence, create=True):
                hello = OcrWorker(model_factory=factory).initialize(extended, resource_root=root)
        self.assertEqual(["cuda"], factory.devices)
        self.assertEqual([2560], factory.text_det_limit_side_lengths)
        self.assertEqual([4], factory.text_batch_sizes)
        self.assertEqual({"requestedDevice": "auto", "observedDevice": "cuda", **expected_evidence}, {
            name: hello[name]
            for name in (
                "requestedDevice", "observedDevice", "runtimeId", "runtimeFingerprint",
                "paddleVersion", "compiledWithCuda", "cudaVersion", "gpuName", "totalVramBytes",
            )
        })

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_paddle_engine_uses_relative_ascii_model_directories_and_explicit_names(self) -> None:
        observed: dict[str, object] = {}

        class FakePaddleOcr:
            def __init__(self, **kwargs: object) -> None:
                observed["cwd"] = os.getcwd()
                observed["kwargs"] = kwargs

            def predict(self, image: object) -> list[object]:
                return []

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name) / "OCR资源"
            for directory in ("detection", "recognition", "textline-orientation"):
                (root / directory).mkdir(parents=True)
            resource = OcrResource(
                RESOURCE_ID,
                "0" * 64,
                root,
                root / "detection",
                root / "recognition",
                root / "textline-orientation",
                {},
            )
            before = os.getcwd()
            fake_paddle = types.ModuleType("paddle")
            fake_paddleocr = types.ModuleType("paddleocr")
            fake_paddleocr.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
            with mock.patch.dict(sys.modules, {"paddle": fake_paddle, "paddleocr": fake_paddleocr}):
                model = create_paddle_engine(resource, device="cpu")

        self.assertIsInstance(model, PaddleOcrModel)
        self.assertEqual(before, os.getcwd())
        self.assertEqual(os.path.normcase(str(root)), os.path.normcase(str(observed["cwd"])))
        kwargs = observed["kwargs"]
        assert isinstance(kwargs, dict)
        self.assertEqual(
            {
                "text_detection_model_name": "PP-OCRv5_server_det",
                "text_recognition_model_name": "PP-OCRv5_server_rec",
                "textline_orientation_model_name": "PP-LCNet_x1_0_textline_ori",
                "text_detection_model_dir": "detection",
                "text_recognition_model_dir": "recognition",
                "textline_orientation_model_dir": "textline-orientation",
            },
            {name: kwargs[name] for name in (
                "text_detection_model_name",
                "text_recognition_model_name",
                "textline_orientation_model_name",
                "text_detection_model_dir",
                "text_recognition_model_dir",
                "textline_orientation_model_dir",
            )},
        )
        self.assertTrue(all(str(kwargs[name]).isascii() for name in (
            "text_detection_model_dir",
            "text_recognition_model_dir",
            "textline_orientation_model_dir",
        )))
        self.assertEqual(
            {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "text_rec_score_thresh": 0,
                "text_det_limit_side_len": 1920,
                "text_det_limit_type": "max",
                "device": "cpu",
            },
            {name: kwargs[name] for name in (
                "use_doc_orientation_classify",
                "use_doc_unwarping",
                "use_textline_orientation",
                "text_rec_score_thresh",
                "text_det_limit_side_len",
                "text_det_limit_type",
                "device",
            )},
        )

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_paddle_engine_restores_cwd_when_initialization_fails(self) -> None:
        observed: dict[str, object] = {}

        class FailingPaddleOcr:
            def __init__(self, **kwargs: object) -> None:
                observed["cwd"] = os.getcwd()
                raise RuntimeError("fixture initialization failure")

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name) / "OCR资源"
            for directory in ("detection", "recognition", "textline-orientation"):
                (root / directory).mkdir(parents=True)
            resource = OcrResource(
                RESOURCE_ID,
                "0" * 64,
                root,
                root / "detection",
                root / "recognition",
                root / "textline-orientation",
                {},
            )
            before = os.getcwd()
            fake_paddle = types.ModuleType("paddle")
            fake_paddleocr = types.ModuleType("paddleocr")
            fake_paddleocr.PaddleOCR = FailingPaddleOcr  # type: ignore[attr-defined]
            with mock.patch.dict(sys.modules, {"paddle": fake_paddle, "paddleocr": fake_paddleocr}):
                with self.assertRaises(OcrModelError):
                    create_paddle_engine(resource, device="cpu")

        self.assertEqual(before, os.getcwd())
        self.assertEqual(os.path.normcase(str(root)), os.path.normcase(str(observed["cwd"])))

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_ocr_resource_allows_only_unverified_local_only_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            root, manifest_relative, fingerprint = _write_resource(
                temporary / "valid",
                distribution={
                    "mode": "local-only",
                    "sourceUrl": "https://example.invalid/ppocrv5",
                    "licenseStatus": "unverified",
                },
            )
            resource = load_ocr_resource(root, manifest_relative, fingerprint)
            self.assertEqual(RESOURCE_ID, resource.resource_id)

            invalid_distributions = (
                {"mode": "bundled"},
                {
                    "mode": "local-only",
                    "sourceUrl": "https://example.invalid/ppocrv5",
                    "licenseUrl": "https://example.invalid/license",
                },
                {
                    "mode": "local-only",
                    "sourceUrl": "https://example.invalid/ppocrv5",
                    "licenseStatus": "verified",
                },
                {
                    "mode": "local-only",
                    "sourceUrl": "https://example.invalid/ppocrv5",
                    "licenseStatus": "unverified",
                    "extra": True,
                },
                {
                    "mode": "local-only",
                    "sourceUrl": "http://example.invalid/ppocrv5",
                    "licenseStatus": "unverified",
                },
                {
                    "mode": "local-only",
                    "sourceUrl": "https://user:secret@example.invalid/ppocrv5",
                    "licenseStatus": "unverified",
                },
            )
            for index, distribution in enumerate(invalid_distributions, start=1):
                with self.subTest(distribution=distribution):
                    invalid_root, invalid_relative, invalid_fingerprint = _write_resource(
                        temporary / f"invalid-{index}",
                        distribution=distribution,
                    )
                    with self.assertRaises(OcrResourceError):
                        load_ocr_resource(invalid_root, invalid_relative, invalid_fingerprint)

    def _ready_worker(self, temporary: Path, result: object | None = None) -> tuple[OcrWorker, _FakeEngine, _FakeFactory]:
        root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
        engine = _FakeEngine(_raw_result() if result is None else result)
        factory = _FakeFactory(engine)
        worker = OcrWorker(model_factory=factory)
        hello = worker.initialize(_hello(manifest_relative, fingerprint), resource_root=root)
        OcrHelloResultV1.from_dict(hello)
        self.assertEqual(1, factory.calls)
        self.assertEqual(1, hello["modelSessionLoads"])
        return worker, engine, factory

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_worker_loads_one_fake_model_and_preserves_duplicate_result_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            path = temporary / "transparent.png"
            Image.new("RGBA", (3, 2), (255, 0, 0, 128)).save(path)
            source_before = path.read_bytes()
            worker, engine, factory = self._ready_worker(temporary)
            outcome = worker.process(_request([_work_item(path)]))
            parsed = parse_ocr_process_result(outcome)
            self.assertEqual("success", parsed[0].status)
            self.assertEqual(["same", "same"], [item.text for item in parsed[0].items])
            self.assertEqual([0.0, 180.0], [item.textlineOrientationDegrees for item in parsed[0].items])
            self.assertEqual(source_before, path.read_bytes())
            self.assertEqual(1, factory.calls)
            self.assertEqual("RGB", engine.images[0].mode)
            self.assertEqual((255, 127, 127), engine.images[0].getpixel((0, 0)))
            with self.assertRaises(OcrWorkerInitializationError):
                worker.initialize({}, resource_root=temporary)

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_worker_accepts_finite_numeric_scalars_from_the_engine(self) -> None:
        decimal = Decimal
        result = _raw_result(
            scores=[decimal("0.8"), decimal("0.9")],
            polys=[
                [[decimal("0"), decimal("0")], [decimal("2"), decimal("0")], [decimal("2"), decimal("1")], [decimal("0"), decimal("1")]],
                [[decimal("0"), decimal("1")], [decimal("2"), decimal("1")], [decimal("2"), decimal("2")], [decimal("0"), decimal("2")]],
            ],
            boxes=[
                [decimal("0"), decimal("0"), decimal("2"), decimal("1")],
                [decimal("0"), decimal("1"), decimal("2"), decimal("2")],
            ],
            angles=[decimal("0"), decimal("1")],
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            path = temporary / "numeric.png"
            Image.new("RGB", (3, 2), "red").save(path)
            worker, _, _ = self._ready_worker(temporary, result)
            outcome = parse_ocr_process_result(worker.process(_request([_work_item(path)])))[0]
            self.assertEqual("success", outcome.status)
            self.assertEqual([0.8, 0.9], [item.confidence for item in outcome.items])

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_production_wrapper_lazily_converts_rgb_input_to_an_array(self) -> None:
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.asarray = lambda image: ("array", image)  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            path = temporary / "array-input.png"
            Image.new("RGB", (3, 2), "red").save(path)
            engine = _FakeEngine(_raw_result())
            factory = _FakeFactory(engine, convert_input_to_array=True)
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy}):
                try:
                    worker = OcrWorker(model_factory=factory)
                    root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
                    worker.initialize(_hello(manifest_relative, fingerprint), resource_root=root)
                    outcome = parse_ocr_process_result(worker.process(_request([_work_item(path)])))[0]
                    converted = outcome.status == "success" and engine.images[0][0] == "array"
                except OcrWorkerInitializationError:
                    converted = False
            self.assertTrue(converted, "production Paddle wrapper must convert the RGB copy lazily")

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_empty_result_is_no_text_and_invalid_model_fields_keep_a_safe_reason(self) -> None:
        cases = {
            "length": _raw_result(scores=[0.8]),
            "nan": _raw_result(scores=[float("nan"), 0.9]),
            "geometry": _raw_result(polys=[[[0, 0], [4, 0], [4, 1], [0, 1]], [[0, 1], [2, 1], [2, 2], [0, 2]]]),
            "limit": _raw_result(
                texts=["x"] * 1025,
                scores=[0.5] * 1025,
                polys=[[[0, 0], [2, 0], [2, 1], [0, 1]]] * 1025,
                boxes=[[0, 0, 2, 1]] * 1025,
                angles=[0] * 1025,
            ),
        }
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            path = temporary / "sample.png"
            Image.new("RGB", (3, 2), "red").save(path)
            worker, engine, _ = self._ready_worker(temporary, _raw_result(texts=[], scores=[], polys=[], boxes=[], angles=[]))
            no_text = parse_ocr_process_result(worker.process(_request([_work_item(path)])))[0]
            self.assertEqual("no_text", no_text.status)
            for index, (name, result) in enumerate(cases.items(), start=2):
                with self.subTest(name=name):
                    engine.result = result
                    outcome = parse_ocr_process_result(
                        worker.process(_request([_work_item(path, sample_id=index, lease_id=f"lease-{index}")]))
                    )[0]
                    self.assertEqual("failed", outcome.status)
                    self.assertEqual("ocr_inference_failed", outcome.error.code)
                    self.assertEqual("OCR model output was invalid for this image.", outcome.error.message)
                    self.assertNotIn("details", outcome.error.to_dict())
                    self.assertNotRegex(outcome.error.message, r"[A-Za-z]:[\\/]")

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_image_policy_enforces_exif_format_hash_and_inclusive_limits(self) -> None:
        self.assertFalse(image_exceeds_limits(16_000, 2_500))
        self.assertFalse(image_exceeds_limits(16_384, 2_441))
        self.assertTrue(image_exceeds_limits(16_385, 1))
        self.assertTrue(image_exceeds_limits(20_000, 2_001))
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            oriented = temporary / "oriented.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (2, 3), "red").save(oriented, exif=exif)
            worker, _, _ = self._ready_worker(temporary)
            oriented_outcome = parse_ocr_process_result(worker.process(_request([_work_item(oriented)])))[0]
            self.assertEqual((3, 2), (oriented_outcome.image.width, oriented_outcome.image.height))

            disguised = temporary / "disguised.jpg"
            Image.new("RGB", (3, 2), "red").save(disguised, format="PNG")
            mismatch = parse_ocr_process_result(
                worker.process(_request([_work_item(disguised, sample_id=2, lease_id="lease-2")]))
            )[0]
            self.assertEqual("ocr_inference_failed", mismatch.error.code)
            self.assertNotIn("details", mismatch.error.to_dict())

            changed = _work_item(oriented, sample_id=3, lease_id="lease-3")
            oriented.write_bytes(oriented.read_bytes() + b"changed")
            with self.assertRaises(OcrSourceFingerprintError):
                worker.process(_request([changed]))

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_oversize_dimensions_are_scaled_to_safe_ocr_size(self) -> None:
        self.assertEqual((5412, 7390), scaled_dimensions(5859, 8000))
        width, height = scaled_dimensions(12553, 6099)
        self.assertLessEqual(width * height, 40_000_000)
        self.assertLessEqual(max(width, height), 16_384)

        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            path = temporary / "large.jpg"
            Image.new("RGB", (200, 100), "white").save(path)
            source_bytes = path.read_bytes()
            worker, engine, _ = self._ready_worker(temporary)
            with mock.patch.object(image_module, "MAX_PIXELS", 10_000):
                outcome = parse_ocr_process_result(worker.process(_request([_work_item(path)])))[0]

            self.assertEqual("success", outcome.status)
            self.assertLessEqual(outcome.image.width * outcome.image.height, 10_000)
            self.assertLessEqual(max(outcome.image.width, outcome.image.height), 16_384)
            self.assertEqual(source_bytes, path.read_bytes())
            self.assertEqual(1, len(engine.images))

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_malformed_oversize_header_returns_decode_failure_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            path = temporary / "too-large.bmp"
            width, height = 20_000, 2_001
            file_header = struct.pack("<2sIHHI", b"BM", 54, 0, 0, 54)
            dib_header = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 0, 0, 0, 0, 0)
            path.write_bytes(file_header + dib_header)
            worker, engine, _ = self._ready_worker(temporary)
            outcome = parse_ocr_process_result(worker.process(_request([_work_item(path)])))[0]
            self.assertEqual("failed", outcome.status)
            self.assertEqual("ocr_image_too_large", outcome.error.code)
            self.assertFalse(outcome.error.retriable)
            self.assertEqual(width * height, outcome.error.to_dict()["details"]["actualPixels"])
            self.assertEqual([], engine.images)

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_malformed_huge_header_returns_decode_failure_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            path = temporary / "very-large.bmp"
            width, height = 50_000, 2_001
            file_header = struct.pack("<2sIHHI", b"BM", 54, 0, 0, 54)
            dib_header = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 0, 0, 0, 0, 0)
            path.write_bytes(file_header + dib_header)
            worker, engine, _ = self._ready_worker(temporary)
            outcome = parse_ocr_process_result(worker.process(_request([_work_item(path)])))[0]
            self.assertEqual("failed", outcome.status)
            self.assertEqual("ocr_image_too_large", outcome.error.code)
            self.assertFalse(outcome.error.retriable)
            self.assertEqual(width * height, outcome.error.to_dict()["details"]["actualPixels"])
            self.assertEqual([], engine.images)

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_resource_hashes_and_offline_guards_are_checked_before_fake_model_factory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
            tampered = root / "ocr-models" / RESOURCE_ID / "recognition" / "model.pdiparams"
            tampered.write_bytes(b"tampered")
            with self.assertRaises(OcrWorkerInitializationError):
                OcrWorker(model_factory=_FakeFactory(_FakeEngine(_raw_result()))).initialize(
                    _hello(manifest_relative, fingerprint), resource_root=root
                )

            root, manifest_relative, fingerprint = _write_resource(temporary / "second-library")
            factory = _FakeFactory(_FakeEngine(_raw_result()))
            with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network disabled")), mock.patch.object(
                socket.socket, "connect_ex", side_effect=AssertionError("network disabled")
            ):
                OcrWorker(model_factory=factory).initialize(_hello(manifest_relative, fingerprint), resource_root=root)
            self.assertEqual(
                {
                    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                factory.environments[0],
            )

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_entry_uses_bounded_jsonl_frames_and_source_imports_do_not_require_paddle(self) -> None:
        self.assertEqual(_expected_entry_runtime_id(), RUNTIME_ID)
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT / 'workers' / 'ocr' / 'src')!r}); "
            "import anima_ocr_worker.worker; "
            "assert not {'PIL','numpy','paddle','paddleocr','paddlex'} & set(sys.modules); print('ok')"
        )
        imported = subprocess.run(
            [sys.executable, "-B", "-I", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(0, imported.returncode, imported.stderr)
        self.assertEqual("ok", imported.stdout.strip())

        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            root, manifest_relative, fingerprint = _write_resource(temporary / "resource-library")
            path = temporary / "entry.png"
            Image.new("RGB", (3, 2), "red").save(path)
            hello_payload = _hello(manifest_relative, fingerprint)
            process_payload = _request([_work_item(path)])
            frames = b"".join((
                _entry_frame("hello-1", "hello", hello_payload, job_id="ocr-job", config_hash="a" * 64),
                _entry_frame("process-1", "process_batch", process_payload, job_id="ocr-job", config_hash="a" * 64),
                _entry_frame("shutdown-1", "shutdown", {}),
            ))
            output = io.BytesIO()
            errors = io.StringIO()
            factory = _FakeFactory(_FakeEngine(_raw_result()))
            with mock.patch.dict(os.environ, {"ANIMA_RESOURCE_ROOT": str(root)}, clear=False):
                self.assertEqual(0, run(io.BytesIO(frames), output, errors, model_factory=factory))
            replies = output.getvalue().splitlines(keepends=True)
            self.assertEqual(3, len(replies))
            self.assertTrue(all(len(frame) <= 1_048_576 for frame in replies))
            result = json.loads(replies[1].decode("utf-8"))
            self.assertEqual("1.0", result["protocolVersion"])
            self.assertEqual("response", result["kind"])
            self.assertEqual("reply-process-1", result["messageId"])
            self.assertEqual("process-1", result["replyTo"])
            self.assertEqual(_expected_entry_runtime_id(), result["runtimeId"])
            self.assertEqual("ocr", result["owner"])
            self.assertEqual("result", result["method"])
            self.assertEqual("success", parse_ocr_process_result(result["payload"])[0].status)
            self.assertEqual("", errors.getvalue())

    @unittest.skipIf(_WORKER_IMPORT_ERROR is not None, "OCR worker source is not implemented")
    def test_requirements_are_exact_and_formal_lock_is_published(self) -> None:
        requirements = ROOT / "packaging" / "requirements" / "ocr-paddle.in"
        self.assertEqual(
            ["paddlepaddle==3.2.2", "paddleocr==3.7.0", "paddlex[ocr-core]==3.7.2"],
            requirements.read_text(encoding="utf-8").splitlines(),
        )
        self.assertTrue((ROOT / "packaging" / "requirements" / "ocr-paddle.lock").is_file())


if __name__ == "__main__":
    unittest.main()
