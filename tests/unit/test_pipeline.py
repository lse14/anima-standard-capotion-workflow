from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from PIL import Image

from anima_caption_format import flat_txt_sha256
from anima_caption_format.normalizer import CaptionDisplayPolicy
from anima_core.commit_journal import CommitJournal, load as load_commit_journal, write as write_commit_journal
from anima_core.contracts import JobConfig
from anima_core.db import StateDatabase
from anima_core.dataset_workspace import DatasetWorkspace
from anima_core.job_preflight import JobPreparationService
from anima_core.repair import RepairPreparationService
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.path_safety import file_fingerprint, windows_key
from anima_core.pipeline import PipelineService
from anima_core import pipeline_recovery as pipeline_recovery_module
from anima_core.pipeline_dispatch import PipelineError
from anima_core.ocr_runtime_binding import (
    OcrRuntimeBindingV1,
    OcrExecutionRequestV1,
    read_execution_request,
    read_runtime_binding,
    write_execution_request,
    write_runtime_binding,
)
from anima_core.scheduler import BoundedScheduler
from anima_core.contracts import SampleIssue, SampleRunState
from anima_core.classify_overlay import serialize_annotation_json
from anima_core.ocr_sidecar import FIXED_OCR_INFERENCE_SETTINGS, parse_ocr_sidecar, serialize_ocr_sidecar
from anima_core.stdio_transport import StdioJsonlTransportError
from anima_core.token_budget_overlay import TokenBudgetOverlayWriter
from anima_core.token_budget_protocol import validate_token_budget_outcome


class PipelineTests(unittest.TestCase):
    @staticmethod
    def _control_job_config(root: Path) -> dict[str, object]:
        config = JobConfig(
            workMode="in_place",
            overwriteMode="incremental",
            sourceRoot=str(root),
        )
        config.nl["systemPrompt"] = "Describe the visible image."
        return {
            "config_schema_version": config.schemaVersion,
            "config_json": json.dumps(config.to_dict()),
            "config_hash": config.config_hash,
        }

    @staticmethod
    def _v7_cpu_ocr_binding() -> OcrRuntimeBindingV1:
        return OcrRuntimeBindingV1.from_dict({
            "schemaVersion": 1,
            "requested": {
                "device": "auto",
                "textDetLimitSideLen": {"mode": "auto", "value": None},
                "textBatchSize": {"mode": "auto", "value": None},
            },
            "recommended": {"source": "cpu", "totalVramBytes": None, "textDetLimitSideLen": 1920, "textBatchSize": 1},
            "effective": {"textDetLimitSideLen": 1920, "textBatchSize": 1},
            "runtime": {
                "runtimeId": "ocr-paddle", "runtimeFingerprint": "a" * 64, "observedDevice": "cpu",
                "paddleVersion": "3.2.2", "compiledWithCuda": False, "cudaVersion": None, "gpuName": None,
            },
            "resourceFingerprint": "b" * 64,
            "startupReason": "gpu_runtime_unavailable",
        })

    @staticmethod
    def _v7_gpu_ocr_binding() -> OcrRuntimeBindingV1:
        return OcrRuntimeBindingV1.from_dict({
            "schemaVersion": 1,
            "requested": {
                "device": "auto",
                "textDetLimitSideLen": {"mode": "auto", "value": None},
                "textBatchSize": {"mode": "auto", "value": None},
            },
            "recommended": {
                "source": "gpu_vram_table", "totalVramBytes": 8589934592,
                "textDetLimitSideLen": 2048, "textBatchSize": 4,
            },
            "effective": {"textDetLimitSideLen": 2048, "textBatchSize": 4},
            "runtime": {
                "runtimeId": "ocr-paddle-gpu", "runtimeFingerprint": "c" * 64, "observedDevice": "cuda",
                "paddleVersion": "3.2.2", "compiledWithCuda": True, "cudaVersion": "12.6", "gpuName": "test-gpu",
            },
            "resourceFingerprint": "d" * 64,
            "startupReason": None,
        })

    def test_v7_resume_and_repair_use_the_existing_binding_without_runtime_reselection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = self._v7_cpu_ocr_binding()
            parent_path = root / "parent" / "ocr-runtime-binding-v1.json"
            write_runtime_binding(parent_path, binding)
            dispatch = object.__new__(PipelineService)
            dispatch.install_root = root
            database = SimpleNamespace(
                repair_parent_job_id=lambda _job_id: "parent-job",
                get_job=lambda _job_id: {"overlay_root": str(root / "parent-overlay")},
            )
            child_path = root / "child" / "ocr-runtime-binding-v1.json"
            child_layout = SimpleNamespace(resource_path=lambda _filename: child_path)
            parent_layout = SimpleNamespace(resource_path=lambda _filename: parent_path)
            with patch("anima_core.pipeline_dispatch.OverlayLayout.open_existing", return_value=parent_layout):
                dispatch._resolve_ocr_runtime = lambda runtime_id: (SimpleNamespace(), binding.runtimeFingerprint)  # type: ignore[method-assign]
                self.assertEqual(child_path, dispatch._ocr_binding_path(database, "repair-job", child_layout))
            self.assertEqual(binding, read_runtime_binding(child_path))

            original = child_path.read_bytes()
            dispatch._resolve_ocr_runtime = lambda _runtime_id: self.fail("an existing binding must bypass runtime resolution")  # type: ignore[method-assign]
            dispatch._probe_ocr_gpu_runtime = lambda *_args: self.fail("an existing binding must bypass the GPU probe")  # type: ignore[method-assign]
            selection = dispatch._select_ocr_runtime(database, "repair-job", {"ocr": {"device": "auto"}}, child_path)
            self.assertEqual(("ocr-paddle", "a" * 64, None, "gpu_runtime_unavailable"), (
                selection.runtime_id, selection.runtime_fingerprint, selection.total_vram_bytes, selection.startup_reason,
            ))
            self.assertEqual(original, child_path.read_bytes())

    def test_v7_repair_does_not_copy_a_stale_parent_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_binding = self._v7_cpu_ocr_binding()
            parent_path = root / "parent" / "ocr-runtime-binding-v1.json"
            write_runtime_binding(parent_path, parent_binding)
            dispatch = object.__new__(PipelineService)
            dispatch.install_root = root
            database = SimpleNamespace(
                repair_parent_job_id=lambda _job_id: "parent-job",
                get_job=lambda _job_id: {"overlay_root": str(root / "parent-overlay")},
            )
            child_path = root / "child" / "ocr-runtime-binding-v1.json"
            child_layout = SimpleNamespace(resource_path=lambda _filename: child_path)
            parent_layout = SimpleNamespace(resource_path=lambda _filename: parent_path)
            with patch("anima_core.pipeline_dispatch.OverlayLayout.open_existing", return_value=parent_layout):
                dispatch._resolve_ocr_runtime = lambda runtime_id: (SimpleNamespace(), f"{runtime_id}-current")  # type: ignore[method-assign]
                self.assertEqual(child_path, dispatch._ocr_binding_path(database, "repair-job", child_layout))
            self.assertFalse(child_path.exists())

    def test_v7_stale_repair_binding_inherits_execution_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_binding_path = root / "parent" / "ocr-runtime-binding-v1.json"
            parent_request_path = root / "parent" / "ocr-execution-request-v1.json"
            child_binding_path = root / "child" / "ocr-runtime-binding-v1.json"
            child_request_path = root / "child" / "ocr-execution-request-v1.json"
            write_runtime_binding(parent_binding_path, self._v7_cpu_ocr_binding())
            request = OcrExecutionRequestV1.auto()
            write_execution_request(parent_request_path, request)
            dispatch = object.__new__(PipelineService)
            dispatch.install_root = root
            database = SimpleNamespace(
                repair_parent_job_id=lambda _job_id: "parent-job",
                get_job=lambda _job_id: {"overlay_root": str(root / "parent-overlay")},
            )
            child_layout = SimpleNamespace(
                resource_path=lambda name: child_binding_path if name == "ocr-runtime-binding-v1.json" else child_request_path,
            )
            parent_layout = SimpleNamespace(
                resource_path=lambda name: parent_binding_path if name == "ocr-runtime-binding-v1.json" else parent_request_path,
            )
            with patch("anima_core.pipeline_dispatch.OverlayLayout.open_existing", return_value=parent_layout):
                dispatch._resolve_ocr_runtime = lambda runtime_id: (SimpleNamespace(), f"{runtime_id}-current")  # type: ignore[method-assign]
                self.assertEqual(child_binding_path, dispatch._ocr_binding_path(database, "repair-job", child_layout))
            self.assertFalse(child_binding_path.exists())
            self.assertEqual(request, read_execution_request(child_request_path))

    def test_v7_auto_ocr_falls_back_to_cpu_when_gpu_operation_probe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dispatch = object.__new__(PipelineService)
            dispatch.install_root = root
            resolved: list[str] = []

            def resolve(runtime_id: str) -> tuple[object, str]:
                resolved.append(runtime_id)
                return SimpleNamespace(), f"{runtime_id}-fingerprint"

            def probe(_launch: object, _install_root: Path) -> int | None:
                raise PipelineError("CUDA OCR runtime is unavailable")

            dispatch._resolve_ocr_runtime = resolve  # type: ignore[method-assign]
            dispatch._probe_ocr_gpu_runtime = probe  # type: ignore[method-assign]
            selection = dispatch._select_ocr_runtime(
                SimpleNamespace(), "job", {"ocr": {"device": "auto"}}, root / "ocr-runtime-binding-v1.json",
            )

            self.assertEqual(("ocr-paddle", "ocr-paddle-fingerprint", None, "gpu_runtime_unavailable"), (
                selection.runtime_id, selection.runtime_fingerprint, selection.total_vram_bytes, selection.startup_reason,
            ))
            self.assertEqual(["ocr-paddle-gpu", "ocr-paddle"], resolved)

    def test_v7_auto_ocr_selects_gpu_after_a_successful_operation_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dispatch = object.__new__(PipelineService)
            dispatch.install_root = root
            launch = SimpleNamespace()
            resolved: list[str] = []
            probed: list[tuple[object, Path]] = []

            def resolve(runtime_id: str) -> tuple[object, str]:
                resolved.append(runtime_id)
                return launch, "ocr-paddle-gpu-fingerprint"

            def probe(actual_launch: object, actual_root: Path) -> int:
                probed.append((actual_launch, actual_root))
                return 8589934592

            dispatch._resolve_ocr_runtime = resolve  # type: ignore[method-assign]
            dispatch._probe_ocr_gpu_runtime = probe  # type: ignore[method-assign]
            selection = dispatch._select_ocr_runtime(
                SimpleNamespace(), "job", {"ocr": {"device": "auto"}}, root / "ocr-runtime-binding-v1.json",
            )

            self.assertEqual(("ocr-paddle-gpu", "ocr-paddle-gpu-fingerprint", 8589934592, None), (
                selection.runtime_id, selection.runtime_fingerprint, selection.total_vram_bytes, selection.startup_reason,
            ))
            self.assertEqual(["ocr-paddle-gpu"], resolved)
            self.assertEqual([(launch, root)], probed)

    def test_v7_forced_cuda_reports_stable_message_without_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dispatch = object.__new__(PipelineService)
            dispatch.install_root = root
            resolved: list[str] = []

            def resolve(runtime_id: str) -> tuple[object, str]:
                resolved.append(runtime_id)
                return SimpleNamespace(), f"{runtime_id}-fingerprint"

            def probe(_launch: object, _install_root: Path) -> int | None:
                raise PipelineError("CUDA OCR runtime is unavailable")

            dispatch._resolve_ocr_runtime = resolve  # type: ignore[method-assign]
            dispatch._probe_ocr_gpu_runtime = probe  # type: ignore[method-assign]
            with self.assertRaises(PipelineError) as raised:
                dispatch._select_ocr_runtime(
                    SimpleNamespace(), "job", {"ocr": {"device": "cuda"}}, root / "ocr-runtime-binding-v1.json",
                )

            self.assertEqual(
                "The OCR CUDA runtime is unavailable or incompatible with this GPU. Choose Auto or CPU.",
                str(raised.exception),
            )
            self.assertEqual(["ocr-paddle-gpu"], resolved)

    def test_v7_cpu_ocr_never_probes_the_gpu_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dispatch = object.__new__(PipelineService)
            dispatch.install_root = root
            resolved: list[str] = []

            def resolve(runtime_id: str) -> tuple[object, str]:
                resolved.append(runtime_id)
                return SimpleNamespace(), f"{runtime_id}-fingerprint"

            dispatch._resolve_ocr_runtime = resolve  # type: ignore[method-assign]
            dispatch._probe_ocr_gpu_runtime = lambda *_args: self.fail("CPU selection must not probe CUDA")  # type: ignore[method-assign]
            selection = dispatch._select_ocr_runtime(
                SimpleNamespace(), "job", {"ocr": {"device": "cpu"}}, root / "ocr-runtime-binding-v1.json",
            )

            self.assertEqual(("ocr-paddle", "ocr-paddle-fingerprint", None, None), (
                selection.runtime_id, selection.runtime_fingerprint, selection.total_vram_bytes, selection.startup_reason,
            ))
            self.assertEqual(["ocr-paddle"], resolved)

    def test_v7_existing_gpu_binding_bypasses_runtime_resolution_and_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding_path = root / "ocr-runtime-binding-v1.json"
            write_runtime_binding(binding_path, self._v7_gpu_ocr_binding())
            original = binding_path.read_bytes()
            dispatch = object.__new__(PipelineService)
            dispatch._resolve_ocr_runtime = lambda _runtime_id: self.fail("an existing GPU binding must bypass runtime resolution")  # type: ignore[method-assign]
            dispatch._probe_ocr_gpu_runtime = lambda *_args: self.fail("an existing GPU binding must bypass the GPU probe")  # type: ignore[method-assign]

            selection = dispatch._select_ocr_runtime(
                SimpleNamespace(), "job", {"ocr": {"device": "auto"}}, binding_path,
            )

            self.assertEqual(("ocr-paddle-gpu", "c" * 64, 8589934592, None), (
                selection.runtime_id, selection.runtime_fingerprint, selection.total_vram_bytes, selection.startup_reason,
            ))
            self.assertEqual(original, binding_path.read_bytes())

    def test_v7_gpu_probe_runs_a_synchronized_cuda_matrix_operation(self) -> None:
        launch = SimpleNamespace(interpreter=Path("gpu-runtime-python.exe"), environment={"PATH": "runtime-bin"})
        completed = SimpleNamespace(returncode=0, stdout='{"totalVramBytes":8589934592}', stderr="")

        with patch("anima_core.pipeline_dispatch.subprocess.run", return_value=completed) as run:
            total = PipelineService._probe_ocr_gpu_runtime(launch, ROOT)

        self.assertEqual(8589934592, total)
        command = run.call_args.args[0]
        self.assertEqual(["gpu-runtime-python.exe", "-B", "-I", "-c"], command[:4])
        probe = command[4]
        self.assertIn("paddle.set_device('gpu:0')", probe)
        self.assertIn("paddle.matmul", probe)
        self.assertIn("paddle.device.synchronize", probe)
        self.assertIn(".numpy()", probe)
        self.assertIn("expected = 23.0", probe)
        self.assertIn("value != expected", probe)
        self.assertLess(probe.index("paddle.matmul"), probe.index("paddle.device.synchronize"))
        self.assertLess(probe.index("paddle.device.synchronize"), probe.index(".numpy()"))
        self.assertEqual(1, probe.count("print("))
        self.assertIn("print(json.dumps({'totalVramBytes'", probe)
        self.assertEqual(15, run.call_args.kwargs["timeout"])
        self.assertLessEqual(len(completed.stdout), 1024)

        overlong = SimpleNamespace(returncode=0, stdout="x" * 1025, stderr="")
        with patch("anima_core.pipeline_dispatch.subprocess.run", return_value=overlong):
            with self.assertRaisesRegex(PipelineError, "CUDA OCR runtime is unavailable"):
                PipelineService._probe_ocr_gpu_runtime(launch, ROOT)

    def test_v7_repair_missing_parent_binding_is_reported_as_a_pipeline_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dispatch = object.__new__(PipelineService)
            database = SimpleNamespace(
                repair_parent_job_id=lambda _job_id: "parent-job",
                get_job=lambda _job_id: {"overlay_root": None},
            )
            layout = SimpleNamespace(resource_path=lambda filename: root / "child" / filename)

            try:
                dispatch._ocr_binding_path(database, "repair-job", layout)
            except Exception as exc:
                self.assertIsInstance(exc, PipelineError)
            else:
                self.fail("a repair without its parent OCR binding must be blocked")

    def _interrupted_caption_job(self, root: Path) -> tuple[JobPreparationService, str, Path]:
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
        config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
        config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        preparation = JobPreparationService(root / "state.db")
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        database = StateDatabase.open(root / "state.db")
        try:
            scheduler = BoundedScheduler(database)
            scheduler.start_module(job_id, "caption", enabled=True, profile="e621")
            lease = scheduler.claim_batch(job_id, "caption", "caption-test", str(database.get_job(job_id)["config_hash"]))[0]
            layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
            prepared, digest = layout.write_prepared("caption", str(lease.leaseId), ".txt", b"tag_a, tag_b")
            relative = str(prepared.relative_to(layout.root)).replace("/", "\\")
            database.stage_prepared_artifact(job_id, lease.sampleId, lease_id=str(lease.leaseId), relative_path=relative, sha256=digest)
            database.mark_interrupted(job_id)
        finally:
            database.close()
        return preparation, job_id, dataset

    def _cancelled_caption_job(self, root: Path) -> tuple[str, Path]:
        dataset = root / "dataset"
        dataset.mkdir()
        image = dataset / "image.png"
        Image.new("RGB", (3, 3), "white").save(image)
        config = JobConfig(
            schemaVersion=10, workMode="in_place",
            overwriteMode="incremental", sourceRoot=str(dataset),
        )
        config.nl["promptVersion"] = "nl-default-prompt-v4"
        config.nl["apiEnabled"] = False
        job_id = "cancelled-caption-job"
        layout = OverlayLayout.create(dataset, job_id)
        fingerprint = file_fingerprint(image)
        database = StateDatabase.open(root / "state.db")
        try:
            database.insert_job({
                "job_id": job_id, "config_schema_version": 10, "config_json": json.dumps(config.to_dict()),
                "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place",
                "overwrite_mode": "incremental", "source_root": str(dataset), "output_root": None,
                "dataset_root": str(dataset), "dataset_root_key": windows_key(dataset),
                "manifest_schema_version": 1, "recursive": 0, "sample_count": 1,
                "manifest_generated_at": "2026-08-16T00:00:00Z", "status": "cancelled_recoverable",
                "current_module_id": "caption", "last_event_id": 0, "pinned": 0,
                "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": str(layout.root),
                "commit_journal_path": None, "resume_status": "running", "created_at": "2026-08-16T00:00:00Z",
                "started_at": "2026-08-16T00:00:00Z", "cancel_requested_at": "2026-08-16T00:00:00Z",
                "finished_at": "2026-08-16T00:01:00Z",
            })
            database.insert_samples(job_id, [{
                "sample_id": 1, "relative_image_path": "image.png", "annotation_key": "image", "source": "e621",
                "in_processing_scope": True, "image_format": "png", "image_frame_count": 1,
                "original_txt_state": "missing_or_blank", "original_json_state": "missing_or_blank",
                "image_file_id": fingerprint["file_id"], "image_size": fingerprint["size"],
                "image_mtime_ns": fingerprint["mtime_ns"],
            }])
            prepared, digest = layout.write_prepared("caption", "cancelled-caption-lease", ".txt", b"tag_a")
            database.set_sample_state(job_id, 1, SampleRunState(
                sampleId=1, currentModuleId="caption", status="prepared", attempt=1,
                leaseId="cancelled-caption-lease", workerInstanceId="caption-worker",
                leaseExpiresAt="2030-01-01T00:00:00Z",
                preparedArtifactRelativePath=str(prepared.relative_to(layout.root)).replace("/", "\\\\"),
                preparedArtifactSha256=digest,
            ))
            database.initialize_module_summary(job_id, "caption", total=1, status="running")
        finally:
            database.close()
        return job_id, dataset

    @staticmethod
    def _dropout_only_job(root: Path) -> tuple[JobPreparationService, str]:
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
        (dataset / "image.json").write_bytes(serialize_annotation_json({
            "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
            "appearance": ["white hair"], "tags": ["smile"], "environment": [], "nl": "A person smiles.",
        }))
        config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        config.dropout["enabled"] = True
        config.dropout["artist"]["enabled"] = False
        config.dropout["quality"]["enabled"] = False
        # These tests exercise Dropout/Export processing of the existing JSON
        # annotation; opt into that mutation explicitly under incremental mode.
        config.classify["overwriteJson"] = True
        config.tokenBudget["enabled"] = False  # type: ignore[index]
        preparation = JobPreparationService(root / "state.db")
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        return preparation, job_id

    @staticmethod
    def _interrupted_export_commit(
        root: Path,
        *,
        state: str = "staged",
        phase: str = "writing_backup",
        freeze: bool = False,
    ) -> tuple[JobPreparationService, str, Path, OverlayLayout, Path]:
        preparation, job_id = PipelineTests._dropout_only_job(root)
        database = StateDatabase.open(root / "state.db")
        try:
            scheduler = BoundedScheduler(database)
            for module_id in (
                "caption", "classify", "replace", "ocr", "nl",
                "count_review", "dropout", "token_budget",
            ):
                scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
            scheduler.start_module(job_id, "export", enabled=True, profile="e621")
            database.set_module_summary(
                job_id, "export", status="completed", completed=1, finished=True,
            )
            database.set_job_status(job_id, "committing", current_module_id="export")
            job = database.get_job(job_id)
            dataset = Path(str(job["dataset_root"]))
            layout = OverlayLayout.open_existing(str(job["overlay_root"]), job_id)
            token = job_id.replace("-", "")[:24]
            staging = dataset.parent / f".{dataset.name}.anima-stage-{token}"
            rollback = dataset.parent / f".{dataset.name}.anima-rollback-{token}"
            backup = dataset.parent / f".{dataset.name}.anima-backups" / f"{job_id}.zip"
            staging.mkdir()
            (staging / "owned-stage.txt").write_text("owned", encoding="ascii")
            write_commit_journal(layout, CommitJournal.v2(
                job_id, state, dataset, staging, rollback, backup, phase=phase,
            ))  # type: ignore[arg-type]
            if freeze:
                database.mark_interrupted(job_id)
        finally:
            database.close()
        return preparation, job_id, dataset, layout, staging

    def test_export_recovery_classification_separates_pre_and_post_swap(self) -> None:
        classify = getattr(pipeline_recovery_module, "classify_export_recovery", None)
        self.assertTrue(callable(classify), "export recovery classifier is missing")
        root = Path(tempfile.gettempdir()) / "anima-export-recovery-classification"
        paths = (
            root / "dataset", root / ".dataset.stage", root / ".dataset.rollback",
            root / ".dataset.anima-backups" / "job.zip",
        )
        for state, phase, expected in (
            ("declared", "creating_stage", "manual_recommit"),
            ("staging", "writing_annotations", "manual_recommit"),
            ("staged", "writing_backup", "manual_recommit"),
            ("backup_verified", "switching_directories", "manual_recommit"),
            ("rollback_created", "switching_directories", "automatic_switch_recovery"),
            ("rollback_required", "switching_directories", "automatic_switch_recovery"),
            ("committed", "cleanup", "resolved"),
            ("rolled_back", "cleanup", "resolved"),
        ):
            with self.subTest(state=state):
                journal = CommitJournal.v2("job", state, *paths, phase=phase)  # type: ignore[arg-type]
                self.assertEqual(expected, classify(journal))
        legacy = CommitJournal("job", "prepared", *paths)
        self.assertEqual("manual_recommit", classify(legacy))

    def test_startup_recovery_preserves_every_pre_swap_stage(self) -> None:
        for state, phase in (
            ("declared", "creating_stage"),
            ("staging", "writing_annotations"),
            ("staged", "writing_backup"),
            ("backup_verified", "switching_directories"),
        ):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                preparation, job_id, dataset, layout, staging = self._interrupted_export_commit(
                    root, state=state, phase=phase,
                )
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                try:
                    pipeline.startup_recovery()
                    self.assertTrue((staging / "owned-stage.txt").is_file())
                    self.assertTrue(dataset.is_dir())
                    journal = load_commit_journal(layout)
                    self.assertIsNotNone(journal)
                    assert journal is not None
                    self.assertEqual((state, phase), (journal.state, journal.phase))
                    database = StateDatabase.open(root / "state.db")
                    try:
                        job = database.get_job(job_id)
                        self.assertEqual(("interrupted", "committing", "export"), (
                            job["status"], job["resume_status"], job["current_module_id"],
                        ))
                    finally:
                        database.close()
                finally:
                    pipeline.close()
                    preparation.close()

    def test_completed_export_recovery_returns_recommit_without_starting_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id, _dataset, _layout, staging = self._interrupted_export_commit(
                root, freeze=True,
            )
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with patch("anima_core.pipeline_recovery.BoundedScheduler.recover") as recover, patch(
                    "anima_core.pipeline.BoundedScheduler.start_module",
                ) as start_module, patch.object(
                    StateDatabase, "resolve_repaired_parent_issues",
                ) as resolve_parent_issues:
                    response = pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual((False, "interrupted", "export_recommit"), (
                    response["started"], response["status"], response["recoveryKind"],
                ))
                recover.assert_not_called()
                start_module.assert_not_called()
                resolve_parent_issues.assert_not_called()
                self.assertTrue((staging / "owned-stage.txt").is_file())
                self.assertFalse(pipeline.is_running(job_id))
            finally:
                pipeline.close()
                preparation.close()

    @staticmethod
    def _prepared_ocr_job(
        root: Path,
        *,
        schema_version: int = 10,
        enabled: bool = True,
        device: str = "cuda",
    ) -> tuple[JobPreparationService, str]:
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
        config = JobConfig(
            schemaVersion=schema_version,
            workMode="in_place",
            overwriteMode="incremental",
            sourceRoot=str(dataset),
        )
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
        config.nl["enabled"] = config.dropout["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        config.ocr["enabled"] = enabled
        if schema_version in {7, 8, 9, 10}:
            config.ocr["device"] = device
        preparation = JobPreparationService(root / "state.db")
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        return preparation, job_id

    def test_forced_cuda_start_gate_rejects_a_failed_probe_before_starting_a_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._prepared_ocr_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    frozen = json.loads(str(job["config_json"]))
                    layout = OverlayLayout.open_existing(str(job["overlay_root"]), job_id)
                    binding_path = layout.resource_path("ocr-runtime-binding-v1.json")
                finally:
                    database.close()
                self.assertEqual("cuda", frozen["ocr"]["device"])
                self.assertFalse(binding_path.exists())
                launch = SimpleNamespace()
                with patch.object(pipeline, "_resolve_ocr_runtime", return_value=(launch, "gpu-fingerprint")) as resolve:
                    with patch.object(pipeline, "_probe_ocr_gpu_runtime", side_effect=PipelineError("CUDA probe failed")) as probe:
                        with patch("anima_core.pipeline.threading.Thread.start") as thread_start:
                            with self.assertRaisesRegex(PipelineError, "Choose Auto or CPU"):
                                pipeline.start(job_id)
                self.assertEqual([(("ocr-paddle-gpu",), {})], resolve.call_args_list)
                self.assertEqual([((launch, ROOT / ".runtime-build"), {})], probe.call_args_list)
                thread_start.assert_not_called()
                self.assertFalse(binding_path.exists())
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("preparing_workspace", database.get_job(job_id)["status"])
                finally:
                    database.close()
            finally:
                pipeline._threads.pop(job_id, None)
                preparation.close()

    def test_forced_cuda_start_gate_rejects_a_tampered_frozen_config_before_probing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._prepared_ocr_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                database = StateDatabase.open(root / "state.db")
                try:
                    frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                    frozen["ocr"]["textBatchSize"] = {"mode": "manual", "value": 2}
                    database.connection.execute(
                        "UPDATE jobs SET config_json=? WHERE job_id=?",
                        (json.dumps(frozen, separators=(",", ":")), job_id),
                    )
                finally:
                    database.close()
                with patch.object(pipeline, "_resolve_ocr_runtime", side_effect=self.fail) as resolve:
                    with patch.object(pipeline, "_probe_ocr_gpu_runtime", side_effect=self.fail) as probe:
                        with patch("anima_core.pipeline.threading.Thread.start") as thread_start:
                            with self.assertRaisesRegex(PipelineError, "hash does not match"):
                                pipeline.start(job_id)
                resolve.assert_not_called()
                probe.assert_not_called()
                thread_start.assert_not_called()
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("preparing_workspace", database.get_job(job_id)["status"])
                finally:
                    database.close()
            finally:
                pipeline._threads.pop(job_id, None)
                preparation.close()

    def test_forced_cuda_start_gate_skips_an_existing_ocr_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._prepared_ocr_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    layout = OverlayLayout.open_existing(str(job["overlay_root"]), job_id)
                    binding_path = layout.resource_path("ocr-runtime-binding-v1.json")
                    write_runtime_binding(binding_path, self._v7_gpu_ocr_binding())
                finally:
                    database.close()
                original = binding_path.read_bytes()
                with patch.object(pipeline, "_resolve_ocr_runtime", side_effect=self.fail) as resolve:
                    with patch.object(pipeline, "_probe_ocr_gpu_runtime", side_effect=self.fail) as probe:
                        with patch("anima_core.pipeline.threading.Thread.start") as thread_start:
                            pipeline.start(job_id)
                resolve.assert_not_called()
                probe.assert_not_called()
                thread_start.assert_called_once_with()
                self.assertEqual(original, binding_path.read_bytes())
            finally:
                pipeline._threads.pop(job_id, None)
                preparation.close()

    def test_start_gate_skips_auto_and_disabled_ocr_configurations(self) -> None:
        cases = (
            ("auto", 10, True, "auto"),
            ("disabled", 10, False, "cuda"),
        )
        for label, schema_version, enabled, device in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                preparation, job_id = self._prepared_ocr_job(
                    root,
                    schema_version=schema_version,
                    enabled=enabled,
                    device=device,
                )
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                try:
                    with patch.object(pipeline, "_resolve_ocr_runtime", side_effect=self.fail) as resolve:
                        with patch.object(pipeline, "_probe_ocr_gpu_runtime", side_effect=self.fail) as probe:
                            with patch("anima_core.pipeline.threading.Thread.start") as thread_start:
                                pipeline.start(job_id)
                    resolve.assert_not_called()
                    probe.assert_not_called()
                    thread_start.assert_called_once_with()
                finally:
                    pipeline._threads.pop(job_id, None)
                    preparation.close()

    def test_initial_thread_start_failure_unregisters_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._dropout_only_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with patch("anima_core.pipeline.threading.Thread.start", side_effect=RuntimeError("thread start failed")):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        pipeline.start(job_id)
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("preparing_workspace", database.get_job(job_id)["status"])
                finally:
                    database.close()
            finally:
                pipeline._threads.pop(job_id, None)
                preparation.close()

    def test_abnormal_worker_exit_restarts_twice_then_skips_the_remainder(self) -> None:
        for crashes, expected in ((2, "paused"), (9, "succeeded")):
            with self.subTest(crashes=crashes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                preparation, job_id = self._dropout_only_job(root)
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                attempts: list[str] = []

                def crashing(_database, scheduler, current_job_id, module_id, _config, attempts=attempts) -> str:
                    attempts.append(module_id)
                    if module_id == "dropout" and sum(item == "dropout" for item in attempts) <= crashes:
                        raise StdioJsonlTransportError("worker exited before request")
                    if module_id == "export":
                        return scheduler.finish_module(current_job_id, "export")
                    return "paused"

                pipeline._run_active_module = crashing  # type: ignore[method-assign]
                try:
                    original = (root / "dataset" / "image.json").read_bytes()
                    pipeline._run(job_id)
                    database = StateDatabase.open(root / "state.db")
                    try:
                        summary = database.module_summary(job_id, "dropout")
                        self.assertEqual(3, sum(item == "dropout" for item in attempts))
                        self.assertEqual(min(crashes, 3), int(summary["worker_restart_count"]))
                        if expected == "succeeded":
                            self.assertEqual("completed_with_issues", summary["status"])
                            self.assertEqual((1, 1), (summary["failed"], summary["issue_count"]))
                            self.assertEqual("succeeded", database.get_job(job_id)["status"])
                            state = database.get_sample_state(job_id, 1)
                            self.assertEqual(("export", "skipped"), (state["current_module_id"], state["status"]))
                            self.assertEqual(original, (root / "dataset" / "image.json").read_bytes())
                        else:
                            self.assertEqual("running", summary["status"])
                    finally:
                        database.close()
                finally:
                    preparation.close()

    def test_export_worker_exhaustion_skips_unfinished_samples_and_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._dropout_only_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            export_attempts = 0

            def crash_export(database, scheduler, current_job_id, module_id, _config) -> str:
                nonlocal export_attempts
                if module_id == "dropout":
                    leases = scheduler.claim_batch(
                        current_job_id,
                        "dropout",
                        "dropout-worker",
                        database.get_job(current_job_id)["config_hash"],
                    )
                    for lease in leases:
                        scheduler.complete(lease)
                    return scheduler.finish_module(current_job_id, "dropout")
                if module_id == "export":
                    export_attempts += 1
                    raise StdioJsonlTransportError("export worker exited before request")
                return "paused"

            pipeline._run_active_module = crash_export  # type: ignore[method-assign]
            try:
                original = (root / "dataset" / "image.json").read_bytes()
                pipeline._run(job_id)

                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(3, export_attempts)
                    self.assertEqual("succeeded", database.get_job(job_id)["status"])
                    summary = database.module_summary(job_id, "export")
                    self.assertEqual(("completed", 1, 1), (
                        summary["status"], summary["failed"], summary["issue_count"],
                    ))
                    state = database.get_sample_state(job_id, 1)
                    self.assertEqual(("export", "skipped"), (state["current_module_id"], state["status"]))
                    self.assertEqual(original, (root / "dataset" / "image.json").read_bytes())
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_caption_production_wiring_passes_install_root_to_resource_coverage_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")

            class TransportContext:
                def __enter__(self):
                    return object()

                def __exit__(self, *_args):
                    return False

            pipeline._spawn_transport = lambda _module_id: TransportContext()  # type: ignore[method-assign]
            try:
                scheduler = BoundedScheduler(database)
                scheduler.start_module(job_id, "caption", enabled=True, profile="e621")
                frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                with patch("anima_core.pipeline_dispatch.CaptionRunner") as runner_type:
                    runner_type.return_value.run.return_value = SimpleNamespace(status="completed")
                    result = pipeline._run_active_module(database, scheduler, job_id, "caption", frozen)
                self.assertEqual("completed", result)
                self.assertEqual(ROOT / "resource-library", runner_type.call_args.kwargs["install_root"])
            finally:
                database.close()
                pipeline.close()
                preparation.close()

    def test_blocking_issues_for_skipped_samples_do_not_hold_the_job_in_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                sample = database.page_samples(job_id, limit=1)[0]
                database.upsert_issue(SampleIssue(
                    issueId="blocking-classify", jobId=job_id, sampleId=int(sample["sample_id"]),
                    relativeImagePath=str(sample["relative_image_path"]), moduleId="classify",
                    code="classify_input_missing", severity="error", blocking=True, retriable=True,
                    repairStartModule="classify", message="blocking", attempt=1,
                ))
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                pipeline._run(job_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("succeeded_with_issues", database.get_job(job_id)["status"])
                    self.assertIn(
                        "export", {str(row["module_id"]) for row in database.module_summaries(job_id)}
                    )
                    issue = database.page_issues(job_id, limit=10)
                    self.assertTrue(any(row["issue_id"] == "blocking-classify" for row in issue))
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_startup_recovery_freezes_active_jobs_and_sweeps_finished_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._dropout_only_job(root)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(job_id, "running", current_module_id="dropout")
                self.assertEqual(1, len(list(database.connection.execute("SELECT 1 FROM dataset_claims"))))
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                result = pipeline.startup_recovery()
                self.assertEqual(1, result["interruptedJobs"])
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    self.assertEqual(("interrupted", "running"), (job["status"], job["resume_status"]))
                    self.assertEqual(1, len(list(database.connection.execute("SELECT 1 FROM dataset_claims"))))
                    database.connection.execute("UPDATE jobs SET status='succeeded' WHERE job_id=?", (job_id,))
                finally:
                    database.close()
                self.assertEqual(1, pipeline.startup_recovery()["clearedDatasetClaims"])
            finally:
                preparation.close()

    def test_commit_recovery_tolerates_an_overlay_directory_that_no_longer_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._dropout_only_job(root)
            database = StateDatabase.open(root / "state.db")
            try:
                overlay = Path(str(database.get_job(job_id)["overlay_root"]))
                database.connection.execute(
                    "UPDATE jobs SET commit_journal_path=? WHERE job_id=?",
                    (str(overlay / "commit-journal.json"), job_id),
                )
            finally:
                database.close()
            import shutil

            shutil.rmtree(overlay)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                # A pruned overlay used to abort the whole startup scan.
                pipeline.recover_pending_commits()
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    self.assertIsNone(job["overlay_root"])
                    self.assertIsNone(job["commit_journal_path"])
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_recovery_commits_real_prepared_overlay_before_resuming_current_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id, _dataset = self._interrupted_caption_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._run = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
            try:
                result = pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual((1, 0, 0, True, "running"), (
                    result["committedPrepared"], result["returnedLeases"], result["pendingApiDecisions"], result["started"], result["status"],
                ))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("completed", database.get_sample_state(job_id, 1)["status"])
                    overlay = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                    self.assertEqual(b"tag_a, tag_b", overlay.annotation_path("image", ".txt").read_bytes())
                finally:
                    database.close()
            finally:
                pipeline.close()
                preparation.close()

    def test_recovery_target_status_is_derived_from_the_persisted_current_module(self) -> None:
        for module_id, expected_status in (
            ("caption", "running"),
            ("export", "exporting"),
            ("count_review", "reviewing"),
            ("token_budget", "reviewing"),
            (None, "preparing_workspace"),
        ):
            with self.subTest(module_id=module_id):
                self.assertEqual(expected_status, PipelineService._recovery_target_status(module_id))

    def test_recovery_does_not_replace_an_existing_pipeline_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            existing_thread = SimpleNamespace()
            job_id = "job-already-running"
            pipeline._threads[job_id] = existing_thread  # type: ignore[assignment]
            try:
                with self.assertRaisesRegex(PipelineError, "already running"):
                    pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual({job_id: existing_thread}, pipeline._threads)
            finally:
                pipeline._threads.pop(job_id, None)
                pipeline.close()

    def test_recovery_start_failure_restores_interrupted_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id, _dataset = self._cancelled_caption_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with patch("anima_core.pipeline_recovery.threading.Thread.start", side_effect=RuntimeError("thread start failed")):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        pipeline.recover_job(job_id, confirmed=True)
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    self.assertEqual(("interrupted", "running", "caption"), (
                        job["status"], job["resume_status"], job["current_module_id"],
                    ))
                finally:
                    database.close()
            finally:
                pipeline._threads.pop(job_id, None)

    def test_cancelled_recovery_retains_metadata_until_validation_then_clears_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id, dataset = self._cancelled_caption_job(root)
            Image.new("RGB", (4, 4), "black").save(dataset / "image.png")
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with self.assertRaisesRegex(Exception, "fingerprints"):
                    pipeline.recover_job(job_id, confirmed=True)
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    self.assertEqual(
                        ("cancelled_recoverable", "2026-08-16T00:00:00Z", "2026-08-16T00:01:00Z"),
                        (job["status"], job["cancel_requested_at"], job["finished_at"]),
                    )
                    self.assertEqual("prepared", database.get_sample_state(job_id, 1)["status"])
                finally:
                    database.close()
            finally:
                pipeline.close()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id, _dataset = self._cancelled_caption_job(root)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._run = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
            try:
                result = pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual((True, "running"), (result["started"], result["status"]))
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    self.assertEqual((None, None), (job["cancel_requested_at"], job["finished_at"]))
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_cancelled_recovery_with_pending_api_decision_returns_to_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id, _dataset = self._cancelled_caption_job(root)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_sample_state(job_id, 1, SampleRunState(
                    sampleId=1, currentModuleId="nl", status="request_started", attempt=1,
                    leaseId="unknown-api", workerInstanceId="nl-worker", leaseExpiresAt="2030-01-01T00:00:00Z",
                ))
                database.connection.execute(
                    "UPDATE jobs SET current_module_id='nl',resume_status='running' WHERE job_id=?", (job_id,)
                )
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                result = pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual((1, False, "interrupted"), (
                    result["pendingApiDecisions"], result["started"], result["status"],
                ))
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    self.assertEqual(
                        ("interrupted", "running", "2026-08-16T00:00:00Z", "2026-08-16T00:01:00Z"),
                        (job["status"], job["resume_status"], job["cancel_requested_at"], job["finished_at"]),
                    )
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_cancelled_recovery_rejects_incompatible_target_without_touching_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id, _dataset = self._cancelled_caption_job(root)
            database = StateDatabase.open(root / "state.db")
            try:
                database.connection.execute("UPDATE jobs SET resume_status='exporting' WHERE job_id=?", (job_id,))
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with self.assertRaisesRegex(PipelineError, "incompatible"):
                    pipeline.recover_job(job_id, confirmed=True)
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    self.assertEqual(
                        ("cancelled_recoverable", "exporting", "2026-08-16T00:00:00Z", "2026-08-16T00:01:00Z"),
                        (job["status"], job["resume_status"], job["cancel_requested_at"], job["finished_at"]),
                    )
                    self.assertEqual("prepared", database.get_sample_state(job_id, 1)["status"])
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_cancelled_recovery_to_review_does_not_start_a_thread(self) -> None:
        for module_id, resume_status in (
            ("count_review", "reviewing"),
            ("count_review", "running"),
            ("token_budget", "reviewing"),
            ("token_budget", "running"),
        ):
            with self.subTest(module_id=module_id, resume_status=resume_status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                job_id, _dataset = self._cancelled_caption_job(root)
                database = StateDatabase.open(root / "state.db")
                try:
                    database.connection.execute(
                        """UPDATE jobs SET current_module_id=?,resume_status=?
                           WHERE job_id=?""",
                        (module_id, resume_status, job_id),
                    )
                    database.connection.execute(
                        """UPDATE sample_state SET current_module_id=?,status='completed',lease_id=NULL,
                           worker_instance_id=NULL,lease_expires_at=NULL WHERE job_id=?""",
                        (module_id, job_id),
                    )
                finally:
                    database.close()
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                try:
                    result = pipeline.recover_job(job_id, confirmed=True)
                    self.assertEqual((False, "reviewing"), (result["started"], result["status"]))
                    self.assertFalse(pipeline.is_running(job_id))
                    database = StateDatabase.open(root / "state.db")
                    try:
                        job = database.get_job(job_id)
                        self.assertEqual(("reviewing", None, None), (
                            job["status"], job["cancel_requested_at"], job["finished_at"],
                        ))
                    finally:
                        database.close()
                finally:
                    pipeline.close()

    def test_recovery_commits_or_reuses_prepared_ocr_sidecar_before_resuming(self) -> None:
        for already_committed in (False, True):
            with self.subTest(already_committed=already_committed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / "dataset"
                dataset.mkdir()
                image = dataset / "image.png"
                Image.new("RGB", (10, 8), "white").save(image)
                image_bytes = image.read_bytes()
                config = JobConfig(
                    workMode="in_place", overwriteMode="incremental",
                    sourceRoot=str(dataset), schemaVersion=10,
                )
                config.nl["promptVersion"] = "nl-default-prompt-v4"
                config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
                config.nl["enabled"] = config.dropout["enabled"] = False
                config.countReview["enabled"] = False  # type: ignore[index]
                config.ocr["enabled"] = True
                preparation = JobPreparationService(root / "state.db")
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    scheduler = BoundedScheduler(database, lease_id_factory=lambda: "ocr-recovery-lease")
                    for module_id in ("caption", "classify", "replace"):
                        scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                    scheduler.start_module(job_id, "ocr", enabled=True, profile="e621")
                    lease = scheduler.claim_batch(job_id, "ocr", "ocr-recovery-worker", str(database.get_job(job_id)["config_hash"]))[0]
                    layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                    frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                    sidecar = {
                        "schemaVersion": 1,
                        "relativeImagePath": "image.png",
                        "image": {"width": 10, "height": 8, "sizeBytes": len(image_bytes), "sha256": hashlib.sha256(image_bytes).hexdigest()},
                        "status": "no_text",
                        "engine": {"backend": "paddle", "resourceId": "ocr-ppocrv5-server-paddle-v1", "resourceFingerprint": frozen["ocr"]["resourceFingerprint"]},
                        "settings": {"llmMinConfidence": 0.5, "inference": dict(FIXED_OCR_INFERENCE_SETTINGS)},
                        "items": [],
                        "error": None,
                    }
                    payload = serialize_ocr_sidecar(parse_ocr_sidecar(json.dumps(sidecar).encode("utf-8")))
                    prepared, digest = layout.write_ocr_prepared(str(lease.leaseId), payload)
                    relative = str(prepared.relative_to(layout.root)).replace("/", "\\\\")
                    database.stage_prepared_artifact(job_id, lease.sampleId, lease_id=str(lease.leaseId), relative_path=relative, sha256=digest)
                    if already_committed:
                        layout.commit_ocr_prepared(relative, digest, "image.png")
                    database.mark_interrupted(job_id)
                finally:
                    database.close()

                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                pipeline._run = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
                try:
                    result = pipeline.recover_job(job_id, confirmed=True)
                    self.assertEqual((1, 0), (result["committedPrepared"], result["repeatedPrepared"]))
                    database = StateDatabase.open(root / "state.db")
                    try:
                        self.assertEqual("completed", database.get_sample_state(job_id, 1)["status"])
                        layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                        self.assertEqual(payload, layout.ocr_sidecar_path("image.png").read_bytes())
                    finally:
                        database.close()
                finally:
                    pipeline.close()
                    preparation.close()

    def test_recovery_settles_a_prepared_ocr_failure_as_one_nonblocking_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            image = dataset / "image.png"
            Image.new("RGB", (10, 8), "white").save(image)
            image_bytes = image.read_bytes()
            config = JobConfig(
                workMode="in_place", overwriteMode="incremental",
                sourceRoot=str(dataset), schemaVersion=10,
            )
            config.nl["promptVersion"] = "nl-default-prompt-v4"
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.ocr["enabled"] = True
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=lambda: "ocr-failure-recovery-lease")
                for module_id in ("caption", "classify", "replace"):
                    scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                scheduler.start_module(job_id, "ocr", enabled=True, profile="e621")
                lease = scheduler.claim_batch(
                    job_id, "ocr", "ocr-recovery-worker", str(database.get_job(job_id)["config_hash"])
                )[0]
                layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                sidecar = {
                    "schemaVersion": 1,
                    "relativeImagePath": "image.png",
                    "image": {
                        "width": 10, "height": 8, "sizeBytes": len(image_bytes),
                        "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    },
                    "status": "failed",
                    "engine": {
                        "backend": "paddle", "resourceId": "ocr-ppocrv5-server-paddle-v1",
                        "resourceFingerprint": frozen["ocr"]["resourceFingerprint"],
                    },
                    "settings": {
                        "llmMinConfidence": 0.5,
                        "inference": dict(FIXED_OCR_INFERENCE_SETTINGS),
                    },
                    "items": [],
                    "error": {
                        "code": "ocr_inference_failed",
                        "message": "OCR inference failed for this image.",
                        "retriable": True,
                    },
                }
                payload = serialize_ocr_sidecar(parse_ocr_sidecar(json.dumps(sidecar).encode("utf-8")))
                prepared, digest = layout.write_ocr_prepared(str(lease.leaseId), payload)
                relative = str(prepared.relative_to(layout.root)).replace("/", "\\\\")
                database.stage_prepared_artifact(
                    job_id, lease.sampleId, lease_id=str(lease.leaseId), relative_path=relative, sha256=digest
                )
                database.mark_interrupted(job_id)
            finally:
                database.close()

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._run = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
            try:
                result = pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual((1, 0), (result["committedPrepared"], result["repeatedPrepared"]))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("completed", database.get_sample_state(job_id, 1)["status"])
                    issue = database.page_issues(job_id, limit=1)[0]
                    self.assertEqual(("ocr_inference_failed", 0, 1), (
                        issue["code"], int(issue["blocking"]), int(issue["retriable"]),
                    ))
                    summary = database.module_summary(job_id, "ocr")
                    self.assertEqual((1, 1), (int(summary["completed"]), int(summary["issue_count"])))
                finally:
                    database.close()
            finally:
                pipeline.close()
                preparation.close()

    def test_shutdown_cancels_only_jobs_owned_by_live_pipeline_threads(self) -> None:
        class JoinedThread:
            joined = False

            def join(self) -> None:
                self.joined = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_dataset = root / "active"
            inactive_dataset = root / "inactive"
            active_dataset.mkdir()
            inactive_dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(active_dataset / "image.png")
            Image.new("RGB", (3, 3), "white").save(inactive_dataset / "image.png")
            preparation = JobPreparationService(root / "state.db")
            def config_for(dataset: Path) -> JobConfig:
                config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
                config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
                config.countReview["enabled"] = False  # type: ignore[index]
                return config

            active = preparation.preflight(config_for(active_dataset).to_dict()).jobId
            inactive = preparation.preflight(config_for(inactive_dataset).to_dict()).jobId
            preparation.confirm_workspace(active, confirmed=True, confirmed_rebuild=False)
            preparation.confirm_workspace(inactive, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(inactive, "running", current_module_id="caption")
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            joined = JoinedThread()
            pipeline._threads[active] = joined  # type: ignore[assignment]
            try:
                pipeline.shutdown()
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertTrue(joined.joined)
                    self.assertEqual("cancelled_recoverable", database.get_job(active)["status"])
                    self.assertEqual("running", database.get_job(inactive)["status"])
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_recovery_rejects_changed_manifest_image_before_touching_prepared_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id, dataset = self._interrupted_caption_job(root)
            Image.new("RGB", (4, 4), "black").save(dataset / "image.png")
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with self.assertRaisesRegex(Exception, "fingerprints"):
                    pipeline.recover_job(job_id, confirmed=True)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("interrupted", database.get_job(job_id)["status"])
                    self.assertEqual("prepared", database.get_sample_state(job_id, 1)["status"])
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_recovery_keeps_nl_request_started_waiting_for_explicit_api_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.nl["enabled"] = True
            config.nl["apiEnabled"] = True
            config.nl["systemPrompt"] = "describe"
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                scheduler = BoundedScheduler(database)
                for module_id in ("caption", "classify", "replace", "ocr"):
                    scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                scheduler.start_module(job_id, "nl", enabled=True, profile="e621")
                lease = scheduler.claim_batch(job_id, "nl", "nl-test", str(database.get_job(job_id)["config_hash"]))[0]
                database.set_sample_state(job_id, lease.sampleId, SampleRunState(
                    sampleId=lease.sampleId, currentModuleId="nl", status="request_started", attempt=lease.attempt,
                    leaseId=lease.leaseId, workerInstanceId="nl-test", leaseExpiresAt=lease.leaseExpiresAt,
                ))
                database.mark_interrupted(job_id)
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                result = pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual((1, False, "interrupted"), (result["pendingApiDecisions"], result["started"], result["status"]))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("request_started", database.get_sample_state(job_id, 1)["status"])
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_recovery_commits_prepared_policy_output_before_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            artist = dataset / "1_Artist"
            artist.mkdir(parents=True)
            Image.new("RGB", (3, 3), "white").save(artist / "image.png")
            baseline = {
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": ["white hair"], "tags": ["smile"], "environment": [], "nl": "A person smiles.",
            }
            (artist / "image.json").write_bytes(serialize_annotation_json(baseline))
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset), recursive=True)
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.dropout["enabled"] = True
            config.dropout["quality"]["enabled"] = False
            config.classify["overwriteJson"] = True
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=lambda: "policy-recovery-lease")
                for module_id in ("caption", "classify", "replace", "ocr", "nl"):
                    scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                scheduler.start_module(job_id, "count_review", enabled=False, profile="e621")
                scheduler.start_module(job_id, "dropout", enabled=True, profile="e621")
                lease = scheduler.claim_batch(
                    job_id, "dropout", "policy-test", str(database.get_job(job_id)["config_hash"])
                )[0]
                layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                repaired = {**baseline, "artist": "@Artist"}
                prepared, digest = layout.write_prepared("dropout", str(lease.leaseId), ".json", serialize_annotation_json(repaired))
                database.stage_prepared_artifact(
                    job_id, lease.sampleId, lease_id=str(lease.leaseId),
                    relative_path=str(prepared.relative_to(layout.root)).replace("/", "\\"), sha256=digest,
                )
                database.mark_interrupted(job_id)
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._run = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
            try:
                result = pipeline.recover_job(job_id, confirmed=True)
                self.assertEqual((1, 0, "running"), (result["committedPrepared"], result["repeatedPrepared"], result["status"]))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("completed", database.get_sample_state(job_id, 1)["status"])
                    layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                    self.assertEqual("@Artist", __import__("json").loads(layout.annotation_path("1_Artist\\image", ".json").read_text(encoding="utf-8"))["artist"])
                finally:
                    database.close()
            finally:
                pipeline.close()
                preparation.close()

    def test_cancellation_return_from_a_module_settles_to_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            def cancel_after_start(database, _scheduler, candidate_job_id, _module_id, _config):
                database.begin_cancellation(candidate_job_id)
                return "cancelling"
            pipeline._run_active_module = cancel_after_start  # type: ignore[method-assign]
            try:
                pipeline._run(job_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("cancelled_recoverable", database.get_job(job_id)["status"])
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_resume_starts_a_paused_policy_module_from_its_persisted_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            artist = dataset / "1_Artist"
            artist.mkdir(parents=True)
            Image.new("RGB", (3, 3), "white").save(artist / "image.png")
            (artist / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": ["white hair"], "tags": ["smile"], "environment": [], "nl": "A person smiles.",
            }))
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.dropout["enabled"] = True
            config.dropout["quality"]["enabled"] = False
            config.classify["overwriteJson"] = True
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                scheduler = BoundedScheduler(database)
                for module_id in ("caption", "classify", "replace", "ocr", "nl"):
                    scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                scheduler.start_module(job_id, "count_review", enabled=False, profile="e621")
                scheduler.start_module(job_id, "dropout", enabled=True, profile="e621")
                database.set_module_summary(job_id, "dropout", status="paused")
                database.set_job_status(job_id, "paused", current_module_id="dropout", resume_status="running")
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                self.assertTrue(pipeline.resume(job_id))
                for _ in range(500):
                    if not pipeline.is_running(job_id):
                        break
                    time.sleep(0.01)
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("succeeded", database.get_job(job_id)["status"])
                    self.assertEqual("completed", database.module_summary(job_id, "dropout")["status"])
                    self.assertEqual("completed", database.module_summary(job_id, "export")["status"])
                finally:
                    database.close()
            finally:
                pipeline.close()
                preparation.close()

    def test_pause_persists_running_worker_module_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-pause"
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, **self._control_job_config(root),
                    "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                    "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                    "recursive": 0, "sample_count": 1, "manifest_generated_at": "now", "status": "running",
                    "current_module_id": "dropout", "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                    "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": None,
                    "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
                database.initialize_module_summary(job_id, "dropout", total=1, status="running")
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._threads[job_id] = SimpleNamespace()  # type: ignore[assignment]
            try:
                self.assertTrue(pipeline.pause(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    summary = database.module_summary(job_id, "dropout")
                    self.assertEqual(("paused", "paused", "dropout", "running"), (
                        job["status"], summary["status"], job["current_module_id"], job["resume_status"],
                    ))
                finally:
                    database.close()
                pipeline._threads.pop(job_id)
                with self.assertRaisesRegex(PipelineError, "running worker"):
                    pipeline.pause(job_id)
            finally:
                pipeline._threads.pop(job_id, None)

    def test_pause_preserves_the_missing_job_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = PipelineService(Path(temporary) / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with self.assertRaises(KeyError):
                    pipeline.pause("missing-job")
            finally:
                pipeline.close()

    def test_pause_module_maps_an_atomic_active_state_conflict_to_pipeline_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-module-pause-conflict"
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, **self._control_job_config(root),
                    "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                    "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                    "recursive": 0, "sample_count": 0, "manifest_generated_at": "now", "status": "running",
                    "current_module_id": "dropout", "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                    "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": None,
                    "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
                database.initialize_module_summary(job_id, "dropout", total=0, status="running")
            finally:
                database.close()

            original_pause = StateDatabase.pause_active_module

            def finish_before_pause(database: StateDatabase, job_id: str, module_id: str, *, active_status: str) -> None:
                concurrent = StateDatabase.open(root / "state.db")
                try:
                    BoundedScheduler(concurrent).finish_module(job_id, module_id)
                finally:
                    concurrent.close()
                original_pause(database, job_id, module_id, active_status=active_status)

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._threads[job_id] = SimpleNamespace()  # type: ignore[assignment]
            try:
                with patch.object(StateDatabase, "pause_active_module", new=finish_before_pause):
                    with self.assertRaisesRegex(PipelineError, "active module state changed before pause"):
                        pipeline.pause_module(job_id, "dropout")
            finally:
                pipeline._threads.pop(job_id, None)
                pipeline.close()

    def test_resume_future_module_rejects_a_terminating_task_without_removing_its_pre_pause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-module-resume-cancelling"
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(root))
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, "config_schema_version": config.schemaVersion, "config_json": json.dumps(config.to_dict()),
                    "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental",
                    "source_root": str(root), "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root),
                    "manifest_schema_version": 1, "recursive": 0, "sample_count": 0, "manifest_generated_at": "now",
                    "status": "cancelling", "current_module_id": "caption", "last_event_id": 0, "pinned": 0,
                    "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None,
                    "resume_status": "running", "created_at": "now", "started_at": "now", "cancel_requested_at": "now", "finished_at": None,
                })
                database.initialize_module_summary(job_id, "nl", total=0, status="paused")
            finally:
                database.close()

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with self.assertRaisesRegex(PipelineError, "only a running task"):
                    pipeline.resume_module(job_id, "nl")
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("paused", database.module_summary(job_id, "nl")["status"])
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_future_module_pause_does_not_win_a_race_with_global_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-module-pause-cancellation-race"
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(root))
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, "config_schema_version": config.schemaVersion, "config_json": json.dumps(config.to_dict()),
                    "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental",
                    "source_root": str(root), "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root),
                    "manifest_schema_version": 1, "recursive": 0, "sample_count": 0, "manifest_generated_at": "now",
                    "status": "running", "current_module_id": "caption", "last_event_id": 0, "pinned": 0,
                    "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None,
                    "resume_status": None, "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
            finally:
                database.close()

            original_pause = StateDatabase.pause_future_module

            def cancel_before_future_pause(
                database: StateDatabase, job_id: str, module_id: str, *, total: int, current_module_id: str, active_status: str,
            ) -> None:
                concurrent = StateDatabase.open(root / "state.db")
                try:
                    concurrent.begin_cancellation(job_id)
                finally:
                    concurrent.close()
                original_pause(
                    database, job_id, module_id, total=total, current_module_id=current_module_id, active_status=active_status,
                )

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with patch.object(StateDatabase, "pause_future_module", new=cancel_before_future_pause):
                    with self.assertRaisesRegex(PipelineError, "task state changed before future pause"):
                        pipeline.pause_module(job_id, "nl")
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("cancelling", database.get_job(job_id)["status"])
                    with self.assertRaises(KeyError):
                        database.module_summary(job_id, "nl")
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_active_module_pause_and_resume_starts_exactly_one_thread(self) -> None:
        class DeferredThread:
            created: list[object] = []

            def __init__(self, *, target, args, daemon, name) -> None:
                self.target = target
                self.args = args
                self.daemon = daemon
                self.name = name
                self.started = False
                self.created.append(self)

            def start(self) -> None:
                self.started = True

            def join(self, timeout=None) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-module-active-resume"
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, **self._control_job_config(root),
                    "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                    "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                    "recursive": 0, "sample_count": 0, "manifest_generated_at": "now", "status": "running",
                    "current_module_id": "dropout", "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                    "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": None,
                    "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
                database.initialize_module_summary(job_id, "dropout", total=0, status="running")
            finally:
                database.close()

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._threads[job_id] = SimpleNamespace()  # type: ignore[assignment]
            try:
                self.assertEqual("paused", pipeline.pause_module(job_id, "dropout"))
                pipeline._threads.pop(job_id)
                with patch("anima_core.pipeline.threading.Thread", DeferredThread):
                    self.assertEqual("running", pipeline.resume_module(job_id, "dropout"))
                self.assertEqual(1, len(DeferredThread.created))
                self.assertTrue(DeferredThread.created[0].started)
            finally:
                pipeline._threads.pop(job_id, None)
                pipeline.close()

    def test_future_module_pause_and_resume_do_not_start_a_thread_or_stop_the_current_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-module-future-prepause"
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(root))
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, "config_schema_version": config.schemaVersion, "config_json": json.dumps(config.to_dict()),
                    "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental",
                    "source_root": str(root), "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root),
                    "manifest_schema_version": 1, "recursive": 0, "sample_count": 0, "manifest_generated_at": "now",
                    "status": "running", "current_module_id": "caption", "last_event_id": 0, "pinned": 0,
                    "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None,
                    "resume_status": None, "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
                database.initialize_module_summary(job_id, "caption", total=0, status="running")
            finally:
                database.close()

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                self.assertEqual("paused", pipeline.pause_module(job_id, "nl"))
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(("running", "caption", "paused"), (
                        database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"],
                        database.module_summary(job_id, "nl")["status"],
                    ))
                finally:
                    database.close()
                self.assertEqual("pending", pipeline.resume_module(job_id, "nl"))
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    with self.assertRaises(KeyError):
                        database.module_summary(job_id, "nl")
                    self.assertEqual(("running", "caption"), (
                        database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"],
                    ))
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_resume_arrived_prepaused_module_initializes_its_samples_after_thread_start(self) -> None:
        class DeferredThread:
            def __init__(self, *, target, args, daemon, name) -> None:
                self.target = target
                self.args = args
                self.daemon = daemon
                self.name = name

            def start(self) -> None:
                return None

            def join(self, timeout=None) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-prepaused-nl"
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(root))
            config.caption["enabled"] = True
            config.nl["enabled"] = True
            config.nl["systemPrompt"] = "Describe the visible image."
            frozen = config.to_dict()
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, "config_schema_version": config.schemaVersion, "config_json": json.dumps(frozen),
                    "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental",
                    "source_root": str(root), "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root),
                    "manifest_schema_version": 1, "recursive": 0, "sample_count": 1, "manifest_generated_at": "now",
                    "status": "running", "current_module_id": "caption", "last_event_id": 0, "pinned": 0,
                    "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None,
                    "resume_status": None, "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
                database.insert_samples(job_id, [{
                    "sample_id": 1, "relative_image_path": "image.png", "annotation_key": "image", "source": "e621",
                    "in_processing_scope": True, "image_format": "png", "image_frame_count": 1, "original_txt_state": "missing_or_blank",
                    "original_json_state": "missing_or_blank", "image_file_id": "volume:1", "image_size": 1, "image_mtime_ns": 1,
                }])
                scheduler = BoundedScheduler(database, lease_id_factory=lambda: "caption-lease")
                scheduler.start_module(job_id, "caption", enabled=True, profile="e621")
                caption_lease = scheduler.claim_batch(job_id, "caption", "caption-worker", str(config.config_hash))[0]
                scheduler.complete(caption_lease)
                scheduler.finish_module(job_id, "caption")
                scheduler.start_module(job_id, "classify", enabled=False, profile="e621")
                scheduler.start_module(job_id, "replace", enabled=False, profile="e621")
                scheduler.start_module(job_id, "ocr", enabled=False, profile="e621")
                scheduler.pause_future_module(job_id, "nl", total=1)
                self.assertEqual("paused", scheduler.start_module(job_id, "nl", enabled=True, profile="e621"))
            finally:
                database.close()

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._run_module_with_restarts = lambda *_args: "paused"  # type: ignore[method-assign]
            try:
                with patch("anima_core.pipeline.threading.Thread", DeferredThread):
                    self.assertEqual("running", pipeline.resume_module(job_id, "nl"))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(("running", "nl"), (database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"]))
                    self.assertEqual("pending", database.module_summary(job_id, "nl")["status"])
                    self.assertEqual(("caption", "completed", None), (
                        database.get_sample_state(job_id, 1)["current_module_id"],
                        database.get_sample_state(job_id, 1)["status"],
                        database.get_sample_state(job_id, 1)["lease_id"],
                    ))
                finally:
                    database.close()
                pipeline._run(job_id, resume_current=True)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(("nl", "pending", None), (
                        database.get_sample_state(job_id, 1)["current_module_id"],
                        database.get_sample_state(job_id, 1)["status"],
                        database.get_sample_state(job_id, 1)["lease_id"],
                    ))
                finally:
                    database.close()
            finally:
                pipeline._threads.pop(job_id, None)
                pipeline.close()

    def test_pause_and_resume_cover_count_review_and_export(self) -> None:
        class DeferredThread:
            def __init__(self, *, target, args, daemon, name) -> None:
                self.target = target
                self.args = args
                self.daemon = daemon
                self.name = name
                self.started = False

            def start(self) -> None:
                self.started = True

            def join(self, timeout=None) -> None:
                return None

        for module_id, active_status in (("count_review", "running"), ("export", "exporting")):
            with self.subTest(module_id=module_id), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                job_id = f"job-pause-{module_id}"
                database = StateDatabase.open(root / "state.db")
                try:
                    database.insert_job({
                        "job_id": job_id, **self._control_job_config(root),
                        "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                        "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                        "recursive": 0, "sample_count": 1, "manifest_generated_at": "now", "status": active_status,
                        "current_module_id": module_id, "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                        "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": None,
                        "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                    })
                    database.initialize_module_summary(job_id, module_id, total=1, status="running")
                finally:
                    database.close()
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                pipeline._threads[job_id] = SimpleNamespace()  # type: ignore[assignment]
                try:
                    self.assertTrue(pipeline.pause(job_id))
                    database = StateDatabase.open(root / "state.db")
                    try:
                        job = database.get_job(job_id)
                        self.assertEqual(("paused", active_status), (job["status"], job["resume_status"]))
                        self.assertEqual("paused", database.module_summary(job_id, module_id)["status"])
                    finally:
                        database.close()
                    pipeline._threads.pop(job_id)
                    with patch("anima_core.pipeline.threading.Thread", DeferredThread):
                        self.assertTrue(pipeline.resume(job_id))
                    database = StateDatabase.open(root / "state.db")
                    try:
                        self.assertEqual(active_status, database.get_job(job_id)["status"])
                        self.assertEqual("running", database.module_summary(job_id, module_id)["status"])
                    finally:
                        database.close()
                finally:
                    pipeline._threads.pop(job_id, None)
                    pipeline.close()

    def test_paused_resume_start_failure_preserves_concurrent_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-paused-start-failure"
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, **self._control_job_config(root),
                    "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                    "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                    "recursive": 0, "sample_count": 0, "manifest_generated_at": "now", "status": "paused",
                    "current_module_id": "dropout", "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                    "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": "running",
                    "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
                database.initialize_module_summary(job_id, "dropout", total=0, status="paused")
                database.connection.execute(
                    "UPDATE module_summary SET started_at=? WHERE job_id=? AND module_id='dropout'",
                    ("2026-08-18T00:00:00Z", job_id),
                )
            finally:
                database.close()

            def fail_start(*_args: object, **_kwargs: object) -> None:
                concurrent = StateDatabase.open(root / "state.db")
                try:
                    concurrent.begin_cancellation(job_id)
                    concurrent.settle_cancellation(job_id)
                finally:
                    concurrent.close()
                raise RuntimeError("thread start failed")

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with patch("anima_core.pipeline.threading.Thread.start", side_effect=fail_start):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        pipeline.resume(job_id)
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("cancelled_recoverable", database.get_job(job_id)["status"])
                    self.assertEqual("running", database.module_summary(job_id, "dropout")["status"])
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_zero_work_paused_resume_start_failure_preserves_summary_after_concurrent_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-zero-work-paused-start-failure"
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, **self._control_job_config(root),
                    "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                    "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                    "recursive": 0, "sample_count": 0, "manifest_generated_at": "now", "status": "paused",
                    "current_module_id": "dropout", "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                    "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": "running",
                    "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
                database.initialize_module_summary(job_id, "dropout", total=0, status="paused")
            finally:
                database.close()

            def fail_start(*_args: object, **_kwargs: object) -> None:
                concurrent = StateDatabase.open(root / "state.db")
                try:
                    concurrent.begin_cancellation(job_id)
                    concurrent.settle_cancellation(job_id)
                finally:
                    concurrent.close()
                raise RuntimeError("thread start failed")

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                with patch("anima_core.pipeline.threading.Thread.start", side_effect=fail_start):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        pipeline.resume(job_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    summary = database.module_summary(job_id, "dropout")
                    self.assertEqual(0, int(summary["total"]))
                    self.assertEqual("pending", summary["status"])
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_token_budget_resume_start_failure_preserves_concurrent_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-token-budget-start-failure"
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, **self._control_job_config(root),
                    "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                    "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                    "recursive": 0, "sample_count": 0, "manifest_generated_at": "now", "status": "reviewing",
                    "current_module_id": "token_budget", "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                    "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": None,
                    "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
            finally:
                database.close()

            def fail_start(*_args: object, **_kwargs: object) -> None:
                concurrent = StateDatabase.open(root / "state.db")
                try:
                    concurrent.begin_cancellation(job_id)
                finally:
                    concurrent.close()
                raise RuntimeError("thread start failed")

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._token_budget_export_gate = lambda *_: True  # type: ignore[method-assign]
            try:
                with patch("anima_core.pipeline.threading.Thread.start", side_effect=fail_start):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        pipeline.resume(job_id)
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(("cancelling", "export"), (
                        database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"],
                    ))
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_token_budget_gate_does_not_block_concurrent_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job-token-budget-gate-cancellation"
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job({
                    "job_id": job_id, **self._control_job_config(root),
                    "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(root),
                    "output_root": None, "dataset_root": str(root), "dataset_root_key": str(root), "manifest_schema_version": 1,
                    "recursive": 0, "sample_count": 0, "manifest_generated_at": "now", "status": "reviewing",
                    "current_module_id": "token_budget", "last_event_id": 0, "pinned": 0, "api_budget_extra": 0,
                    "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": None,
                    "created_at": "now", "started_at": "now", "cancel_requested_at": None, "finished_at": None,
                })
            finally:
                database.close()

            def gate_and_cancel(*_args: object) -> bool:
                concurrent = StateDatabase.open(root / "state.db")
                try:
                    concurrent.begin_cancellation(job_id)
                    concurrent.settle_cancellation(job_id)
                finally:
                    concurrent.close()
                return True

            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._token_budget_export_gate = gate_and_cancel  # type: ignore[method-assign]
            try:
                self.assertFalse(pipeline.resume(job_id))
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(("cancelled_recoverable", "token_budget"), (
                        database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"],
                    ))
                finally:
                    database.close()
            finally:
                pipeline.close()

    def test_resume_rejects_settling_thread_before_persisted_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, job_id = self._dropout_only_job(root)
            database = StateDatabase.open(root / "state.db")
            try:
                scheduler = BoundedScheduler(database)
                for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review"):
                    scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                scheduler.start_module(job_id, "dropout", enabled=True, profile="e621")
                database.set_module_summary(job_id, "dropout", status="paused")
                database.set_job_status(job_id, "paused", current_module_id="dropout", resume_status="running")
            finally:
                database.close()
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline._threads[job_id] = SimpleNamespace()  # type: ignore[assignment]
            try:
                with self.assertRaisesRegex(PipelineError, "settling"):
                    pipeline.resume(job_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(job_id)
                    summary = database.module_summary(job_id, "dropout")
                    self.assertEqual(("paused", "paused", "running"), (
                        job["status"], summary["status"], job["resume_status"],
                    ))
                finally:
                    database.close()
            finally:
                pipeline._threads.pop(job_id, None)
                preparation.close()

    def test_policy_overlay_is_committed_only_by_export_after_the_pipeline_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            artist = dataset / "1_Artist"
            artist.mkdir(parents=True)
            Image.new("RGB", (3, 3), "white").save(artist / "image.png")
            baseline = {
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": ["white hair"], "tags": ["smile"], "environment": [], "nl": "A person smiles.",
            }
            (artist / "image.json").write_bytes(serialize_annotation_json(baseline))
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset), recursive=True)
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.dropout["enabled"] = True
            config.dropout["quality"]["enabled"] = False
            config.dropout["appearanceNl"]["enabled"] = False
            config.classify["overwriteJson"] = True
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                pipeline.start(job_id)
                for _ in range(500):
                    if not pipeline.is_running(job_id):
                        break
                    time.sleep(0.01)
                self.assertFalse(pipeline.is_running(job_id))
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("succeeded", database.get_job(job_id)["status"])
                    self.assertEqual("completed", database.module_summary(job_id, "dropout")["status"])
                    self.assertEqual("completed", database.module_summary(job_id, "export")["status"])
                    overlay = Path(str(database.get_job(job_id)["overlay_root"]))
                    self.assertFalse((overlay / "annotations" / "1_Artist" / "image.json").exists())
                finally:
                    database.close()
                committed = json.loads((artist / "image.json").read_text(encoding="utf-8"))
                self.assertEqual("@Artist", committed["artist"])
                self.assertEqual(baseline["count"], committed["count"])
            finally:
                pipeline.close()
                preparation.close()

    def test_v10_pipeline_traverses_the_fixed_module_order(self) -> None:
        cases = (
            (10, ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export")),
        )
        for schema_version, expected_order in cases:
            with self.subTest(schema_version=schema_version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / "dataset"
                dataset.mkdir()
                Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
                (dataset / "image.json").write_bytes(serialize_annotation_json({
                    "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                    "appearance": ["white hair"], "tags": ["smile"], "environment": [],
                    "nl": "A person smiles.",
                }))
                config = JobConfig(
                    workMode="in_place",
                    overwriteMode="incremental",
                    sourceRoot=str(dataset),
                    schemaVersion=schema_version,
                    countReview={"enabled": False, "protocolVersion": "count-review-v1"},
                )
                config.nl["promptVersion"] = "nl-default-prompt-v4"
                config.tokenBudget["enabled"] = False  # type: ignore[index]
                config.caption["enabled"] = False
                config.classify["enabled"] = False
                config.replace["enabled"] = False
                config.nl["enabled"] = False
                config.dropout["enabled"] = False
                preparation = JobPreparationService(root / "state.db")
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                try:
                    job_id = preparation.preflight(config.to_dict()).jobId
                    preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                    visited: list[str] = []
                    original_start_module = BoundedScheduler.start_module

                    def recording_start_module(
                        scheduler, candidate_job_id, module_id, *, enabled, profile=None,
                    ):
                        if candidate_job_id == job_id:
                            visited.append(module_id)
                        return original_start_module(
                            scheduler, candidate_job_id, module_id, enabled=enabled, profile=profile
                        )

                    with patch("anima_core.pipeline.BoundedScheduler.start_module", new=recording_start_module):
                        pipeline.start(job_id)
                        for _ in range(200):
                            if not pipeline.is_running(job_id):
                                break
                            time.sleep(0.01)
                    self.assertFalse(pipeline.is_running(job_id))
                    database = StateDatabase.open(root / "state.db")
                    try:
                        self.assertEqual("succeeded", database.get_job(job_id)["status"])
                        self.assertEqual(list(expected_order), visited)
                        summaries = {str(row["module_id"]): str(row["status"]) for row in database.module_summaries(job_id)}
                        self.assertEqual(set(expected_order), set(summaries))
                        self.assertEqual("completed", summaries["export"])
                        self.assertTrue(all(summaries[module_id] == "skipped" for module_id in expected_order[:-1]))
                    finally:
                        database.close()
                finally:
                    pipeline.close()
                    preparation.close()

    def test_moss_nl_skips_dropout_even_when_dropout_features_are_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            artist = dataset / "1_Artist"
            artist.mkdir(parents=True)
            Image.new("RGB", (3, 3), "white").save(artist / "image.png")
            (artist / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": [], "environment": [], "nl": "A person.",
            }))
            config = JobConfig(
                workMode="in_place",
                overwriteMode="incremental",
                sourceRoot=str(dataset),
            )
            config.caption["enabled"] = False
            config.classify["enabled"] = False
            config.replace["enabled"] = False
            config.ocr["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.nl.update({
                "enabled": True,
                "apiEnabled": True,
                "reuseOriginalNl": False,
                "useImage": True,
                "useFullJson": False,
                "promptVersion": "nl-default-prompt-moss-v1",
                "captionPreset": "moss-general",
                "systemPrompt": "MOSS prompt is composed from the selected preset.",
            })
            config.dropout["enabled"] = True
            config.dropout["artist"]["enabled"] = True
            config.dropout["quality"]["enabled"] = False
            config.dropout["appearanceNl"]["enabled"] = False
            config.classify["overwriteJson"] = True
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    scheduler = BoundedScheduler(database)
                    for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review"):
                        scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                finally:
                    database.close()

                with patch("anima_core.pipeline.pipeline_module_ids", return_value=("dropout",)):
                    pipeline._run(job_id)

                database = StateDatabase.open(root / "state.db")
                try:
                    summary = database.module_summary(job_id, "dropout")
                    self.assertEqual("skipped", summary["status"])
                finally:
                    database.close()
            finally:
                pipeline.close()
                preparation.close()

    def test_v10_input_nl_dispatches_without_resolving_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(
                schemaVersion=10,
                workMode="in_place",
                overwriteMode="incremental",
                sourceRoot=str(dataset),
            )
            config.caption["inputTxtMode"] = "nl"
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.ocr["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            config.nl.update({"enabled": True, "apiEnabled": True, "systemPrompt": ""})
            preparation = JobPreparationService(root / "state.db")
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            database: StateDatabase | None = None
            try:
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                scheduler = BoundedScheduler(database)
                for module_id in ("caption", "classify", "replace", "ocr"):
                    scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                scheduler.start_module(job_id, "nl", enabled=True, profile="e621")
                frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                pipeline._nl_credentials = lambda _: self.fail("v8 input NL must not resolve credentials")  # type: ignore[method-assign]
                with patch("anima_core.pipeline_dispatch.NlRunner") as runner_type:
                    runner_type.return_value.run.return_value = "completed"
                    self.assertEqual("completed", pipeline._run_active_module(database, scheduler, job_id, "nl", frozen))
                    self.assertEqual("_NoExchangeTransport", type(runner_type.call_args.args[2]).__name__)
            finally:
                if database is not None:
                    database.close()
                pipeline.close()
                preparation.close()

    def test_disabled_modules_commit_missing_annotations_to_unlabeled_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = False
            config.classify["enabled"] = False
            config.replace["enabled"] = False
            config.nl["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.dropout["enabled"] = True
            config.dropout["artist"]["enabled"] = False
            config.dropout["quality"]["enabled"] = False
            config.dropout["appearanceNl"]["enabled"] = False
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            pipeline.start(job_id)
            for _ in range(100):
                if not pipeline.is_running(job_id):
                    break
                time.sleep(0.01)
            self.assertFalse(pipeline.is_running(job_id))
            database = StateDatabase.open(root / "state.db")
            try:
                self.assertEqual("succeeded_with_issues", database.get_job(job_id)["status"])
                summaries = {str(row["module_id"]): str(row["status"]) for row in database.module_summaries(job_id)}
                self.assertEqual(
                    {"caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"},
                    set(summaries),
                )
                self.assertEqual({
                    "caption": "skipped", "classify": "skipped", "replace": "skipped",
                    "ocr": "skipped", "nl": "skipped", "count_review": "skipped",
                    "dropout": "skipped", "token_budget": "skipped",
                    "export": "completed",
                }, summaries)
                self.assertEqual(1, database.count("issues", job_id))
                self.assertFalse((dataset / "image.json").exists())
                self.assertFalse((dataset / "image.txt").exists())
                unlabeled = dataset / "_unlabeled" / job_id.replace("-", "")[:24] / "image.png"
                self.assertTrue(unlabeled.is_file())
                self.assertTrue(DatasetWorkspace.for_dataset(dataset).backup_zip(job_id).is_file())
                self.assertIsNone(database.get_job(job_id)["overlay_root"])
            finally:
                database.close()
                preparation.close()

    def test_v10_enabled_missing_token_budget_record_blocks_before_export_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(job_id, "running", current_module_id="token_budget")
                self.assertFalse(PipelineService._token_budget_export_gate(database, job_id, json.loads(str(database.get_job(job_id)["config_json"]))))
                issue = database.page_issues(job_id, limit=10)[0]
                self.assertEqual(("reviewing", "token_budget"), (database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"]))
                self.assertEqual(("token_budget", "token_budget_export_gate_failed", "token_budget"), (issue["module_id"], issue["code"], issue["repair_start_module"]))
            finally:
                database.close()
                preparation.close()

    def test_v10_missing_token_budget_record_moves_the_job_to_reviewing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(job_id, "running", current_module_id="token_budget")
                frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                self.assertFalse(PipelineService._token_budget_export_gate(database, job_id, frozen))
                self.assertEqual(("reviewing", "token_budget"), (database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"]))
            finally:
                database.close()
                preparation.close()

    def test_review_resume_restores_token_budget_reviewing_state_when_thread_start_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            pipeline: PipelineService | None = None
            try:
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    database.initialize_module_summary(job_id, "token_budget", total=1, status="completed_with_issues")
                    database.set_job_status(job_id, "running", current_module_id="token_budget")
                    database.set_job_status(job_id, "reviewing", current_module_id="token_budget")
                finally:
                    database.close()
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                pipeline._token_budget_export_gate = lambda *_: True  # type: ignore[method-assign]
                with patch("anima_core.pipeline.threading.Thread.start", side_effect=RuntimeError("thread start failed")):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        pipeline.resume(job_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(
                        ("reviewing", "token_budget"),
                        (database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"]),
                    )
                finally:
                    database.close()
            finally:
                if pipeline is not None:
                    pipeline.close()
                preparation.close()

    def test_v10_overflow_blocker_does_not_create_a_second_export_gate_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(job_id, "running", current_module_id="token_budget")
                database.upsert_issue(SampleIssue(
                    issueId="overflow", jobId=job_id, sampleId=1, relativeImagePath="image.png",
                    moduleId="token_budget", code="token_budget_overflow", severity="error", blocking=True,
                    retriable=True, repairStartModule="token_budget", message="Token Budget exceeded the frozen maximum", attempt=1,
                ))
                self.assertFalse(PipelineService._token_budget_export_gate(database, job_id, json.loads(str(database.get_job(job_id)["config_json"]))))
                self.assertEqual(1, len(database.page_issues(job_id, limit=10)))
                self.assertEqual("token_budget_overflow", database.page_issues(job_id, limit=10)[0]["code"])
            finally:
                database.close()
                preparation.close()

    def test_v10_token_budget_failure_is_retained_but_skipped_by_the_export_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(job_id, "running", current_module_id="token_budget")
                database.set_sample_state(job_id, 1, SampleRunState(
                    sampleId=1, currentModuleId="token_budget", status="failed", attempt=1,
                ))
                database.upsert_issue(SampleIssue(
                    issueId="overflow-failed", jobId=job_id, sampleId=1, relativeImagePath="image.png",
                    moduleId="token_budget", code="token_budget_overflow", severity="error", blocking=True,
                    retriable=True, repairStartModule="token_budget", message="Token Budget exceeded the frozen maximum", attempt=1,
                ))

                frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                self.assertTrue(PipelineService._token_budget_export_gate(database, job_id, frozen))
                self.assertEqual("running", database.get_job(job_id)["status"])
                self.assertEqual(1, len(database.page_issues(job_id, limit=10)))
            finally:
                database.close()
                preparation.close()

    def test_v10_repair_export_gate_only_checks_repair_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            Image.new("RGB", (3, 3), "black").save(dataset / "untargeted.png")
            (dataset / "target.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(
                schemaVersion=10,
                workMode="in_place",
                overwriteMode="incremental",
                sourceRoot=str(dataset),
            )
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            repair = RepairPreparationService(root / "state.db")
            try:
                parent_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    target = next(
                        row for row in database.page_samples(parent_id, limit=10)
                        if row["relative_image_path"] == "target.png"
                    )
                    database.set_job_status(parent_id, "running", current_module_id="token_budget")
                    database.set_job_status(parent_id, "reviewing", current_module_id="token_budget")
                    database.upsert_issue(SampleIssue(
                        issueId="target-token-budget", jobId=parent_id,
                        sampleId=int(target["sample_id"]), relativeImagePath="target.png",
                        moduleId="token_budget", code="token_budget_overflow", severity="error",
                        blocking=True, retriable=True, repairStartModule="token_budget",
                        message="retry", attempt=1,
                    ))
                finally:
                    database.close()
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                repair_job_id = repair.prepare(parent_id).repairJobId
                database = StateDatabase.open(root / "state.db")
                try:
                    seen: list[int] = []
                    frozen = json.loads(str(database.get_job(repair_job_id)["config_json"]))
                    with patch.object(
                        TokenBudgetOverlayWriter,
                        "record_for_export",
                        side_effect=lambda **kwargs: seen.append(int(kwargs["sample_id"])),
                    ):
                        self.assertTrue(PipelineService._token_budget_export_gate(database, repair_job_id, frozen))
                    target_id = int(database.connection.execute(
                        "SELECT sample_id FROM repair_targets WHERE repair_job_id=?",
                        (repair_job_id,),
                    ).fetchone()["sample_id"])
                    self.assertEqual([target_id], seen)
                    self.assertEqual([], database.page_issues(repair_job_id, limit=10))
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_v10_disabled_token_budget_skips_runtime_and_overlay_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.tokenBudget["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            called: list[str] = []
            pipeline._run_module_with_restarts = lambda _database, _scheduler, _job_id, module_id, _config: called.append(module_id) or "paused"  # type: ignore[method-assign]
            try:
                pipeline._run(job_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(["export"], called)
                    self.assertEqual("skipped", database.module_summary(job_id, "token_budget")["status"])
                    overlay = Path(str(database.get_job(job_id)["overlay_root"]))
                    self.assertFalse((overlay / "resources" / "token-budget").exists())
                finally:
                    database.close()
            finally:
                pipeline.close()
                preparation.close()

    def test_v10_token_budget_export_gate_accepts_json_and_both_with_the_same_flat_text_record(self) -> None:
        for export_format in ("json", "both"):
            with self.subTest(export_format=export_format), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / "dataset"
                dataset.mkdir()
                Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
                annotation = {"quality": [], "count": "solo", "character": "", "series": "", "artist": "", "appearance": [], "tags": ["smile"], "environment": [], "nl": ""}
                (dataset / "image.json").write_bytes(serialize_annotation_json(annotation))
                config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
                config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
                config.classify["overwriteJson"] = True
                config.countReview["enabled"] = False  # type: ignore[index]
                config.export["format"] = export_format
                preparation = JobPreparationService(root / "state.db")
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    scheduler = BoundedScheduler(database, lease_id_factory=lambda: "lease-1")
                    for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout"):
                        scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                    scheduler.start_module(job_id, "token_budget", enabled=True, profile="e621")
                    lease = scheduler.claim_batch(job_id, "token_budget", "worker", str(database.get_job(job_id)["config_hash"]))[0]
                    row = database.get_leased_sample(job_id, "token_budget", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id="worker")
                    frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                    layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                    view = WorkingAnnotationView(BaselineView(dataset), layout)
                    policy = CaptionDisplayPolicy.from_mapping(frozen["captionFormat"])
                    outcome = validate_token_budget_outcome({"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": lease.sampleId, "leaseId": lease.leaseId, "status": "within_budget", "originalTokens": 1, "finalTokens": 1, "removed": {"quality": [], "environment": [], "tags": [], "appearance": []}, "annotation": annotation, "flatTextSha256": flat_txt_sha256(annotation, policy)}, expected_sample_id=lease.sampleId, expected_lease_id=str(lease.leaseId), original_annotation=annotation, caption_format=frozen["captionFormat"], max_tokens=int(frozen["tokenBudget"]["maxTokens"]))
                    TokenBudgetOverlayWriter(database, layout, view, job_id).prepare_and_commit(sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]), outcome=outcome, caption_format=frozen["captionFormat"], max_tokens=int(frozen["tokenBudget"]["maxTokens"]))
                    scheduler.complete(lease)
                    self.assertTrue(PipelineService._token_budget_export_gate(database, job_id, frozen))
                    self.assertEqual([], database.page_issues(job_id, limit=10))
                finally:
                    database.close()
                    preparation.close()

    def test_v10_token_budget_export_gate_rejects_over_limit_record_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            annotation = {"quality": [], "count": "solo", "character": "", "series": "", "artist": "", "appearance": [], "tags": ["smile"], "environment": [], "nl": ""}
            (dataset / "image.json").write_bytes(serialize_annotation_json(annotation))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.classify["overwriteJson"] = True
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            job_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=lambda: "lease-1")
                for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout"):
                    scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                scheduler.start_module(job_id, "token_budget", enabled=True, profile="e621")
                lease = scheduler.claim_batch(job_id, "token_budget", "worker", str(database.get_job(job_id)["config_hash"]))[0]
                row = database.get_leased_sample(job_id, "token_budget", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id="worker")
                frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                view = WorkingAnnotationView(BaselineView(dataset), layout)
                policy = CaptionDisplayPolicy.from_mapping(frozen["captionFormat"])
                outcome = validate_token_budget_outcome({"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": lease.sampleId, "leaseId": lease.leaseId, "status": "within_budget", "originalTokens": 1, "finalTokens": 1, "removed": {"quality": [], "environment": [], "tags": [], "appearance": []}, "annotation": annotation, "flatTextSha256": flat_txt_sha256(annotation, policy)}, expected_sample_id=lease.sampleId, expected_lease_id=str(lease.leaseId), original_annotation=annotation, caption_format=frozen["captionFormat"], max_tokens=int(frozen["tokenBudget"]["maxTokens"]))
                writer = TokenBudgetOverlayWriter(database, layout, view, job_id)
                writer.prepare_and_commit(sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]), outcome=outcome, caption_format=frozen["captionFormat"], max_tokens=int(frozen["tokenBudget"]["maxTokens"]))
                record_path = layout.resource_path("token-budget\\records\\1.json")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record["finalTokens"] = int(frozen["tokenBudget"]["maxTokens"]) + 1
                record_path.write_text(json.dumps(record), encoding="utf-8")
                self.assertFalse(PipelineService._token_budget_export_gate(database, job_id, frozen))
                self.assertEqual(("reviewing", "token_budget"), (database.get_job(job_id)["status"], database.get_job(job_id)["current_module_id"]))
            finally:
                database.close()
                preparation.close()

    def test_recover_prepared_token_budget_review_apply_finalizes_once_for_export(self) -> None:
        """Recovery must not settle a review apply as a generic Token Budget completion."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            annotation = {
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }
            (dataset / "image.json").write_bytes(serialize_annotation_json(annotation))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.classify["overwriteJson"] = True
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            pipeline: PipelineService | None = None
            try:
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    scheduler = BoundedScheduler(database, lease_id_factory=lambda: "review-apply-1-2")
                    for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout"):
                        scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                    scheduler.start_module(job_id, "token_budget", enabled=True, profile="e621")
                    lease = scheduler.claim_batch(job_id, "token_budget", "token-budget-review", str(database.get_job(job_id)["config_hash"]))[0]
                    row = database.get_leased_sample(job_id, "token_budget", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id="token-budget-review")
                    frozen = json.loads(str(database.get_job(job_id)["config_json"]))
                    layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
                    view = WorkingAnnotationView(BaselineView(dataset), layout)
                    policy = CaptionDisplayPolicy.from_mapping(frozen["captionFormat"])
                    outcome = validate_token_budget_outcome(
                        {
                            "schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": lease.sampleId,
                            "leaseId": lease.leaseId, "status": "within_budget", "originalTokens": 1, "finalTokens": 1,
                            "removed": {"quality": [], "environment": [], "tags": [], "appearance": []},
                            "annotation": annotation, "flatTextSha256": flat_txt_sha256(annotation, policy),
                        },
                        expected_sample_id=lease.sampleId, expected_lease_id=str(lease.leaseId), original_annotation=annotation,
                        caption_format=frozen["captionFormat"], max_tokens=int(frozen["tokenBudget"]["maxTokens"]),
                    )
                    TokenBudgetOverlayWriter(database, layout, view, job_id).prepare_and_commit(
                        sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]),
                        outcome=outcome, caption_format=frozen["captionFormat"], max_tokens=int(frozen["tokenBudget"]["maxTokens"]),
                    )
                    database.upsert_issue(SampleIssue(
                        issueId="overflow", jobId=job_id, sampleId=lease.sampleId, relativeImagePath="image.png",
                        moduleId="token_budget", code="token_budget_overflow", severity="error", blocking=True,
                        retriable=True, repairStartModule="token_budget", message="Token Budget exceeded the frozen maximum", attempt=1,
                    ))
                    database.set_module_summary(job_id, "token_budget", status="completed_with_issues", completed=0, failed=1, issue_count=1)
                    database.set_job_status(job_id, "reviewing", current_module_id="token_budget")
                    database.mark_interrupted(job_id)
                finally:
                    database.close()

                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                first = pipeline.recover_job(job_id, confirmed=True)
                database = StateDatabase.open(root / "state.db")
                try:
                    state = database.get_sample_state(job_id, 1)
                    issue = database.connection.execute(
                        "SELECT resolved_at FROM issues WHERE job_id=? AND sample_id=1 AND code='token_budget_overflow'",
                        (job_id,),
                    ).fetchone()
                    summary = database.module_summary(job_id, "token_budget")
                    record = TokenBudgetOverlayWriter(database, layout, view, job_id).record_for_export(
                        sample_id=1, annotation_key="image", caption_format=frozen["captionFormat"],
                        max_tokens=int(frozen["tokenBudget"]["maxTokens"]),
                    )
                    self.assertEqual((1, 0, "reviewing"), (first["committedPrepared"], first["repeatedPrepared"], first["status"]))
                    self.assertEqual(("pending", "export"), (state["status"], state["current_module_id"]))
                    self.assertIsNotNone(issue["resolved_at"])
                    self.assertEqual((1, 0, 0), (summary["completed"], summary["failed"], summary["issue_count"]))
                    self.assertEqual(flat_txt_sha256(annotation, policy), record.flat_text_sha256)
                    database.mark_interrupted(job_id)
                finally:
                    database.close()
                second = pipeline.recover_job(job_id, confirmed=True)
                database = StateDatabase.open(root / "state.db")
                try:
                    state = database.get_sample_state(job_id, 1)
                    summary = database.module_summary(job_id, "token_budget")
                    self.assertEqual((0, 0, "reviewing"), (second["committedPrepared"], second["repeatedPrepared"], second["status"]))
                    self.assertEqual(("pending", "export"), (state["status"], state["current_module_id"]))
                    self.assertEqual((1, 0, 0), (summary["completed"], summary["failed"], summary["issue_count"]))
                finally:
                    database.close()
            finally:
                if pipeline is not None:
                    pipeline.close()
                preparation.close()

    def test_recover_leased_token_budget_review_apply_returns_to_failed_for_retry(self) -> None:
        """A crash before SQLite reaches prepared must preserve the review apply path."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["smile"], "environment": [], "nl": "",
            }))
            config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.classify["overwriteJson"] = True
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            pipeline: PipelineService | None = None
            try:
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    scheduler = BoundedScheduler(database, lease_id_factory=lambda: "review-crash-before-prepared")
                    for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout"):
                        scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
                    scheduler.start_module(job_id, "token_budget", enabled=True, profile="e621")
                    lease = scheduler.claim_batch(job_id, "token_budget", "token-budget-review", str(database.get_job(job_id)["config_hash"]))[0]
                    database.upsert_issue(SampleIssue(
                        issueId="overflow", jobId=job_id, sampleId=lease.sampleId, relativeImagePath="image.png",
                        moduleId="token_budget", code="token_budget_overflow", severity="error", blocking=True,
                        retriable=True, repairStartModule="token_budget", message="Token Budget exceeded the frozen maximum", attempt=1,
                    ))
                    database.set_job_status(job_id, "reviewing", current_module_id="token_budget")
                    database.mark_interrupted(job_id)
                finally:
                    database.close()

                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                response = pipeline.recover_job(job_id, confirmed=True)
                database = StateDatabase.open(root / "state.db")
                try:
                    state = database.get_sample_state(job_id, 1)
                    issue = database.connection.execute(
                        "SELECT resolved_at FROM issues WHERE job_id=? AND sample_id=1 AND code='token_budget_overflow'",
                        (job_id,),
                    ).fetchone()
                    self.assertEqual((0, "reviewing"), (response["returnedLeases"], response["status"]))
                    self.assertEqual(("failed", "token_budget", None, None), (
                        state["status"], state["current_module_id"], state["lease_id"], state["worker_instance_id"],
                    ))
                    self.assertIsNone(issue["resolved_at"])
                finally:
                    database.close()
            finally:
                if pipeline is not None:
                    pipeline.close()
                preparation.close()


if __name__ == "__main__":
    unittest.main()
