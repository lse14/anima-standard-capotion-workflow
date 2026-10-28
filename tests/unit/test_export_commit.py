from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_core.annotation_backup import write_backup as write_backup_impl
from anima_core.contracts import JobConfig, SampleRunState
from anima_core.commit_journal import CommitJournal, CommitJournalError, load as load_journal
from anima_core.db import StateDatabase
from anima_core.export_commit import ExportCommitCoordinator, ExportCommitError
from anima_core.export_staging import create_hardlink_staging
from anima_core.job_preflight import JobPreparationService
from anima_core.overlay import OverlayLayout
from anima_core.pipeline import PipelineService
from anima_core.replace_overlay import ReplaceOverlayWriter
from anima_core.replace_provenance import (
    DatasetReplaceProvenance,
    ReplaceProvenanceChange,
    apply_provenance_changes,
    provenance_database_path,
)
from anima_core.scheduler import BoundedScheduler
from anima_core.moss_sidecar import serialize_moss_json


def _ocr_sidecar(relative_image_path: str, *, status: str = "success") -> bytes:
    value: dict[str, object] = {
        "schemaVersion": 1,
        "relativeImagePath": relative_image_path,
        "image": {"width": 2, "height": 2, "sizeBytes": 4, "sha256": "a" * 64},
        "status": status,
        "engine": {"backend": "paddle", "resourceId": "ocr-ppocrv5-server-paddle-v1", "resourceFingerprint": "b" * 64},
        "settings": {"llmMinConfidence": 0.5, "inference": {
            "useDocOrientationClassify": False, "useDocUnwarping": False, "useTextlineOrientation": True,
            "textRecScoreThresh": 0, "textDetLimitSideLen": 1920, "textDetLimitType": "max",
        }},
        "items": [], "error": None,
    }
    if status == "success":
        value["items"] = [{
            "index": 0, "text": "Hello", "confidence": 0.9,
            "polygonPixels": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]],
            "bboxPixels": [0, 0, 1, 1], "bbox": [0, 0, 0.5, 0.5],
            "position": "top-left", "textlineOrientationDegrees": 0, "includedForLlm": True,
        }]
    elif status == "failed":
        value["error"] = {"code": "ocr_inference_failed", "message": "OCR engine failed.", "retriable": True}
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


class ExportCommitTests(unittest.TestCase):
    def test_moss_sidecar_is_schema_checked_and_exported_without_marking_its_image_unlabeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                config = json.loads(str(database.get_job(job_id)["config_json"]))
                config["nl"].update({
                    "enabled": True,
                    "apiEnabled": True,
                    "useImage": True,
                    "promptVersion": "nl-default-prompt-moss-v1",
                    "captionPreset": "moss-general",
                })
                database.connection.execute(
                    "UPDATE jobs SET config_json=? WHERE job_id=?",
                    (json.dumps(config), job_id),
                )
                moss = {
                    "caption": {
                        "long": "A cat rests by a bright window.",
                        "medium": "A cat rests by a window.",
                        "short": "A cat rests.",
                    },
                    "watermark": [],
                    "borders": ["none"],
                }
                sidecar = serialize_moss_json(moss)
                layout.write_moss_sidecar("sample.png", sidecar)
                self._complete_export(database, job_id, layout, format_value="both")

                result = ExportCommitCoordinator(database, layout, job_id=job_id).commit()

                self.assertEqual(0, result.unlabeled)
                self.assertEqual(sidecar, (dataset / "moss_annotations" / "sample.png.moss.json").read_bytes())
            finally:
                database.close()
                service.close()

    def test_commit_journal_reads_v1_and_strict_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            staging = root / ".dataset.anima-stage-job"
            rollback = root / ".dataset.anima-rollback-job"
            backup = root / ".dataset.anima-backups" / "job.zip"
            shared = {
                "jobId": "job",
                "datasetRoot": str(dataset),
                "stagingRoot": str(staging),
                "rollbackRoot": str(rollback),
                "backupZip": str(backup),
            }

            for state in (
                "prepared", "backup_verified", "rollback_created",
                "committed", "rollback_required", "rolled_back",
            ):
                journal = CommitJournal.from_value(
                    {"schemaVersion": 1, "state": state, **shared}, job_id="job",
                )
                self.assertEqual((1, state, None, None), (
                    journal.schema_version, journal.state, journal.phase, journal.updated_at,
                ))

            value = {
                "schemaVersion": 2,
                "state": "declared",
                "phase": "creating_stage",
                "updatedAt": "2026-09-16T00:00:00Z",
                **shared,
            }
            journal = CommitJournal.from_value(value, job_id="job")
            self.assertEqual((2, "declared", "creating_stage", value["updatedAt"]), (
                journal.schema_version, journal.state, journal.phase, journal.updated_at,
            ))
            self.assertEqual(value, journal.to_value())

            invalid_values = (
                {**value, "phase": "unknown"},
                {**value, "state": "committed", "phase": "creating_stage"},
                {key: item for key, item in value.items() if key != "updatedAt"},
                {**value, "unexpected": True},
                {**value, "jobId": "another-job"},
                {**value, "stagingRoot": str(root / "elsewhere" / "stage")},
            )
            for invalid in invalid_values:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(CommitJournalError):
                        CommitJournal.from_value(invalid, job_id="job")

    def test_commit_declares_journal_before_stage_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, _dataset, layout = self._prepared_job(Path(temporary))
            try:
                self._complete_export(database, job_id, layout, format_value="both")
                observed: list[tuple[int, str, str | None, str]] = []

                def fail_stage_creation(*_args, **_kwargs):
                    journal = load_journal(layout)
                    self.assertIsNotNone(journal)
                    assert journal is not None
                    observed.append((
                        journal.schema_version,
                        journal.state,
                        journal.phase,
                        str(database.get_job(job_id)["status"]),
                    ))
                    raise OSError("injected stage creation failure")

                with patch("anima_core.export_commit.create_hardlink_staging", side_effect=fail_stage_creation):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()

                self.assertEqual([(2, "declared", "creating_stage", "committing")], observed)
                journal = load_journal(layout)
                self.assertIsNotNone(journal)
                assert journal is not None
                self.assertEqual((2, "declared", "creating_stage"), (
                    journal.schema_version, journal.state, journal.phase,
                ))
                self.assertEqual("committing", database.get_job(job_id)["status"])
            finally:
                database.close()
                service.close()

    def test_commit_persists_ordered_v2_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, _dataset, layout = self._prepared_job(
                Path(temporary), ocr_enabled=True,
            )
            try:
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png"))
                self._complete_export(database, job_id, layout, format_value="both")
                real_write = __import__("anima_core.export_commit", fromlist=["write_journal"]).write_journal
                observed: list[tuple[int, str, str | None]] = []

                def record_phase(current_layout, journal):
                    observed.append((journal.schema_version, journal.state, journal.phase))
                    return real_write(current_layout, journal)

                with patch("anima_core.export_commit.write_journal", side_effect=record_phase):
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()

                self.assertEqual([
                    (2, "declared", "preparing"),
                    (2, "declared", "creating_stage"),
                    (2, "staging", "writing_annotations"),
                    (2, "staging", "copying_unlabeled"),
                    (2, "staging", "copying_ocr"),
                    (2, "staging", "applying_provenance"),
                    (2, "staged", "verifying_baseline"),
                    (2, "staged", "writing_backup"),
                    (2, "backup_verified", "switching_directories"),
                    (2, "rollback_created", "switching_directories"),
                    (2, "committed", "cleanup"),
                ], observed)
            finally:
                database.close()
                service.close()

    def _prepared_job(self, root: Path, *, format_value: str = "both", extra_images: tuple[str, ...] = (), unlabeled_images: tuple[str, ...] = (), ocr_enabled: bool = False):
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (2, 2), (20, 30, 40)).save(dataset / "sample.png")
        (dataset / "sample.json").write_bytes(b'{"legacy":true}\n')
        (dataset / "sample.txt").write_bytes(b"legacy\n")
        for image_name in extra_images:
            image_path = dataset / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), (50, 60, 70)).save(image_path)
            annotation = Path(image_name).with_suffix("")
            (dataset / annotation).parent.mkdir(parents=True, exist_ok=True)
            (dataset / f"{annotation}.json").write_bytes(b'{"legacy":true}\n')
            (dataset / f"{annotation}.txt").write_bytes(b"legacy\n")
        for image_name in unlabeled_images:
            image_path = dataset / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), (80, 90, 100)).save(image_path)
        config = JobConfig(workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset), recursive=True, schemaVersion=10)
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
        config.classify["overwriteJson"] = True
        config.ocr["enabled"] = ocr_enabled
        config.countReview["enabled"] = False  # type: ignore[index]
        assert config.tokenBudget is not None
        config.tokenBudget["enabled"] = False
        config.export["format"] = format_value
        service = JobPreparationService(root / "state.db")
        job_id = service.preflight(config.to_dict()).jobId
        service.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        database = StateDatabase.open(root / "state.db")
        scheduler = BoundedScheduler(database)
        for module_id in (
            "caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget",
        ):
            scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
        scheduler.start_module(job_id, "export", enabled=True, profile="e621")
        layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
        return database, service, job_id, dataset, layout

    def _complete_export(self, database: StateDatabase, job_id: str, layout: OverlayLayout, *, format_value: str, include_all: bool = True) -> None:
        artifacts = []
        samples = database.page_samples(job_id, limit=500)
        for sample in samples:
            sample_id = int(sample["sample_id"])
            lease = f"lease-{sample_id}"
            json_path, json_digest = layout.write_prepared("export", lease, ".json", b'{\n  "tags": [\n    "verified"\n  ]\n}\n')
            current = [(job_id, sample_id, "json", str(json_path.relative_to(layout.root)).replace("/", "\\"), json_digest)]
            if format_value in {"flat_txt", "both"} and include_all:
                txt_path, txt_digest = layout.write_prepared("export", f"{lease}-txt", ".txt", b"verified.\n")
                current.append((job_id, sample_id, "txt", str(txt_path.relative_to(layout.root)).replace("/", "\\"), txt_digest))
            if format_value == "flat_txt":
                current = [artifact for artifact in current if artifact[2] == "txt"]
            artifacts.extend(current)
        database.connection.executemany(
            "INSERT INTO export_artifacts(job_id,sample_id,kind,relative_path,sha256) VALUES (?,?,?,?,?)", artifacts,
        )
        for sample in samples:
            sample_id = int(sample["sample_id"])
            database.set_sample_state(job_id, sample_id, SampleRunState(
                sampleId=sample_id, currentModuleId="export", status="completed", attempt=1,
            ))
        database.set_module_summary(job_id, "export", status="completed", completed=len(samples), finished=True)

    @staticmethod
    def _staging_residue(dataset: Path) -> list[Path]:
        legacy = [entry for entry in dataset.parent.iterdir() if ".anima-stage-" in entry.name]
        dedicated = list((dataset.parent / ".anima-dataset-tool").glob("*/jobs/*/stage")) if (dataset.parent / ".anima-dataset-tool").is_dir() else []
        return legacy + dedicated

    @staticmethod
    def _rollback_residue(dataset: Path) -> list[Path]:
        legacy = [entry for entry in dataset.parent.iterdir() if ".anima-rollback-" in entry.name]
        dedicated = list((dataset.parent / ".anima-dataset-tool").glob("*/jobs/*/rollback")) if (dataset.parent / ".anima-dataset-tool").is_dir() else []
        return legacy + dedicated

    def _assert_preserved_pre_swap(
        self,
        database: StateDatabase,
        job_id: str,
        dataset: Path,
        layout: OverlayLayout,
        *,
        state: str,
        phase: str,
    ) -> None:
        journal = load_journal(layout)
        self.assertIsNotNone(journal)
        assert journal is not None
        self.assertEqual((2, state, phase), (
            journal.schema_version, journal.state, journal.phase,
        ))
        self.assertEqual("committing", database.get_job(job_id)["status"])
        self.assertEqual([journal.staging_root], self._staging_residue(dataset))
        self.assertEqual([], self._rollback_residue(dataset))

    @staticmethod
    def _business_bytes(dataset: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(dataset)).replace("/", "\\"): path.read_bytes()
            for path in dataset.rglob("*")
            if path.is_file() and path.suffix in {".json", ".txt"} and "ocr_annotations" not in path.parts
        }

    def test_commit_replaces_only_business_annotations_and_keeps_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                source_image_id = (dataset / "sample.png").stat().st_ino
                self._complete_export(database, job_id, layout, format_value="both")
                result = ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(1, result.exported)
                self.assertEqual("succeeded", database.get_job(job_id)["status"])
                self.assertEqual(b'{\n  "tags": [\n    "verified"\n  ]\n}\n', (dataset / "sample.json").read_bytes())
                self.assertEqual(b"verified.\n", (dataset / "sample.txt").read_bytes())
                self.assertEqual(source_image_id, (dataset / "sample.png").stat().st_ino)
                self.assertTrue(result.backupZip.is_file())
                self.assertFalse(layout.root.exists())
                job = database.get_job(job_id)
                self.assertIsNone(job["overlay_root"])
                self.assertIsNone(job["commit_journal_path"])
                if __import__("os").name == "nt":
                    attributes = __import__("ctypes").windll.kernel32.GetFileAttributesW(str(dataset))
                    self.assertFalse(bool(attributes & 0x2), "the committed dataset must not inherit the staging hidden attribute")
            finally:
                database.close()
                service.close()

    def test_commit_uses_partial_success_and_idempotent_unlabeled_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, _dataset, layout = self._prepared_job(
                Path(temporary), unlabeled_images=("nested/no-label.png",),
            )
            try:
                self._complete_export(database, job_id, layout, format_value="both")
                unlabeled = next(
                    row for row in database.page_samples(job_id, limit=500)
                    if row["relative_image_path"] == "nested\\no-label.png"
                )
                database.connection.execute(
                    "DELETE FROM export_artifacts WHERE job_id=? AND sample_id=?",
                    (job_id, int(unlabeled["sample_id"])),
                )
                database.set_sample_state(job_id, int(unlabeled["sample_id"]), SampleRunState(
                    sampleId=int(unlabeled["sample_id"]), currentModuleId="export", status="skipped", attempt=1,
                ))
                database.set_module_summary(job_id, "export", status="completed", completed=1, skipped=1, finished=True)
                setter = database.set_module_diagnostic_count
                with patch.object(database, "increment_module_diagnostic", side_effect=AssertionError("increment is not idempotent")):
                    with patch.object(database, "set_module_diagnostic_count", wraps=setter) as set_count:
                        result = ExportCommitCoordinator(database, layout, job_id=job_id).commit()

                self.assertEqual(1, result.unlabeled)
                self.assertEqual("succeeded_with_issues", database.get_job(job_id)["status"])
                set_count.assert_called_once_with(
                    job_id, "export", "unlabeled_images", severity="warning", count=1,
                )
                self.assertEqual(1, database.module_diagnostic_count(job_id, "export", "unlabeled_images"))
                database.set_module_diagnostic_count(
                    job_id, "export", "unlabeled_images", severity="warning", count=186,
                )
                database.set_module_diagnostic_count(
                    job_id, "export", "unlabeled_images", severity="warning", count=186,
                )
                self.assertEqual(186, database.module_diagnostic_count(job_id, "export", "unlabeled_images"))
            finally:
                database.close()
                service.close()

    def test_ocr_warning_and_conversion_diagnostic_do_not_make_partial_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, _dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                database.increment_module_diagnostic(job_id, "ocr", "ocr_failed", severity="warning", amount=1)
                database.increment_module_diagnostic(job_id, "export", "samples_converted", severity="info", amount=2)
                self._complete_export(database, job_id, layout, format_value="both")
                ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual("succeeded", database.get_job(job_id)["status"])
            finally:
                database.close()
                service.close()

    def test_repair_export_artifact_groups_only_include_repair_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), extra_images=("untargeted.jpg",), format_value="json",
            )
            try:
                samples = database.page_samples(job_id, limit=500)
                target = next(row for row in samples if row["relative_image_path"] == "sample.png")
                untargeted = next(row for row in samples if row["relative_image_path"] == "untargeted.jpg")
                database.connection.execute(
                    "INSERT INTO repair_jobs(repair_job_id,parent_job_id,created_at) VALUES (?,?,?)",
                    (job_id, job_id, "now"),
                )
                database.connection.execute(
                    "INSERT INTO repair_targets(repair_job_id,sample_id,repair_start_module) VALUES (?,?,?)",
                    (job_id, int(target["sample_id"]), "export"),
                )
                prepared, digest = layout.write_prepared("export", "lease", ".json", b'{"ok":true}\n')
                database.connection.execute(
                    "INSERT INTO export_artifacts(job_id,sample_id,kind,relative_path,sha256) VALUES (?,?,?,?,?)",
                    (job_id, int(target["sample_id"]), "json", str(prepared.relative_to(layout.root)).replace("/", "\\"), digest),
                )
                rows = database.page_export_artifact_groups(job_id, limit=500)
                self.assertEqual([int(target["sample_id"])], sorted({int(row["sample_id"]) for row in rows}))
                self.assertNotIn(int(untargeted["sample_id"]), {int(row["sample_id"]) for row in rows})
            finally:
                database.close()
                service.close()

    def test_repair_unlabeled_projection_only_includes_repair_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), extra_images=("untargeted.jpg",), format_value="json",
            )
            staging = dataset.parent / "staging"
            try:
                samples = database.page_samples(job_id, limit=500)
                target = next(row for row in samples if row["relative_image_path"] == "sample.png")
                untargeted = next(row for row in samples if row["relative_image_path"] == "untargeted.jpg")
                database.connection.execute(
                    "INSERT INTO repair_jobs(repair_job_id,parent_job_id,created_at) VALUES (?,?,?)",
                    (job_id, job_id, "now"),
                )
                database.connection.execute(
                    "INSERT INTO repair_targets(repair_job_id,sample_id,repair_start_module) VALUES (?,?,?)",
                    (job_id, int(target["sample_id"]), "export"),
                )
                for row in (target, untargeted):
                    for suffix in (".json", ".txt"):
                        path = dataset / (str(row["annotation_key"]) + suffix)
                        if path.exists():
                            path.unlink()
                staging = create_hardlink_staging(dataset, staging)
                copied, relative_root = ExportCommitCoordinator(
                    database, layout, job_id=job_id,
                )._copy_unlabeled_images_to_staging(staging)
                self.assertEqual(1, copied)
                self.assertTrue((staging / relative_root / "sample.png").exists())
                self.assertFalse((staging / relative_root / "untargeted.jpg").exists())
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
                database.close()
                service.close()

    def test_unlabeled_projection_falls_back_to_copy_when_link_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), format_value="json", unlabeled_images=("nested/no-label.png",),
            )
            staging = dataset.parent / "staging"
            try:
                staging = create_hardlink_staging(dataset, staging)
                expected = (staging / "nested" / "no-label.png").read_bytes()
                with patch("anima_core.export_commit.os.link", side_effect=OSError("link unavailable")), patch(
                    "anima_core.export_commit.shutil.copy2", wraps=shutil.copy2,
                ) as copy2:
                    copied, relative_root = ExportCommitCoordinator(
                        database, layout, job_id=job_id,
                    )._copy_unlabeled_images_to_staging(staging)

                target = staging / relative_root / "nested" / "no-label.png"
                self.assertEqual(1, copied)
                self.assertEqual(expected, target.read_bytes())
                copy2.assert_called_once_with(staging / "nested" / "no-label.png", target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
                database.close()
                service.close()

    def test_unlabeled_projection_fails_when_link_and_copy_both_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), format_value="json", unlabeled_images=("no-label.png",),
            )
            staging = dataset.parent / "staging"
            try:
                staging = create_hardlink_staging(dataset, staging)
                with patch("anima_core.export_commit.os.link", side_effect=OSError("link unavailable")), patch(
                    "anima_core.export_commit.shutil.copy2", side_effect=OSError("copy unavailable"),
                ), self.assertRaisesRegex(ExportCommitError, "could not be staged"):
                    ExportCommitCoordinator(
                        database, layout, job_id=job_id,
                    )._copy_unlabeled_images_to_staging(staging)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
                database.close()
                service.close()

    def test_directory_rename_retries_transient_windows_access_denied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            destination = Path(temporary) / "destination"
            source.mkdir()
            access_error = PermissionError(5, "access denied")
            access_error.winerror = 5
            with patch("anima_core.export_commit.os.replace", side_effect=[access_error, None]) as replace, patch(
                "time.sleep",
            ) as sleep:
                ExportCommitCoordinator._replace_directory_with_retry(source, destination)
            self.assertEqual(2, replace.call_count)
            sleep.assert_called_once()

    def test_commit_copies_unannotated_skips_to_a_task_scoped_parallel_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), format_value="json", unlabeled_images=("nested/no-label.png",),
            )
            try:
                samples = database.page_samples(job_id, limit=500)
                labeled = next(row for row in samples if row["relative_image_path"] == "sample.png")
                unlabeled = next(row for row in samples if row["relative_image_path"] == "nested\\no-label.png")
                lease = "lease-labeled"
                prepared, digest = layout.write_prepared("export", lease, ".json", b'{"tags":["verified"]}\n')
                database.connection.execute(
                    "INSERT INTO export_artifacts(job_id,sample_id,kind,relative_path,sha256) VALUES (?,?,?,?,?)",
                    (job_id, labeled["sample_id"], "json", str(prepared.relative_to(layout.root)).replace("/", "\\"), digest),
                )
                database.set_sample_state(job_id, int(labeled["sample_id"]), SampleRunState(
                    sampleId=int(labeled["sample_id"]), currentModuleId="export", status="completed", attempt=1,
                ))
                database.set_sample_state(job_id, int(unlabeled["sample_id"]), SampleRunState(
                    sampleId=int(unlabeled["sample_id"]), currentModuleId="export", status="skipped", attempt=1,
                ))
                database.set_module_summary(job_id, "export", status="completed", completed=1, skipped=1, finished=True)

                source_bytes = (dataset / "nested" / "no-label.png").read_bytes()
                result = ExportCommitCoordinator(database, layout, job_id=job_id).commit()

                expected = dataset / result.unlabeledFolder / "nested" / "no-label.png"
                self.assertEqual(1, result.unlabeled)
                self.assertEqual(source_bytes, expected.read_bytes())
                self.assertFalse((dataset / result.unlabeledFolder / "sample.png").exists())
                self.assertFalse((dataset / "nested" / "no-label.json").exists())
                self.assertFalse((dataset / "nested" / "no-label.txt").exists())
            finally:
                database.close()
                service.close()

    def test_commit_preserves_existing_txt_and_json_for_a_skipped_failed_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                original_json = (dataset / "sample.json").read_bytes()
                original_txt = (dataset / "sample.txt").read_bytes()
                database.set_sample_state(job_id, 1, SampleRunState(
                    sampleId=1, currentModuleId="export", status="skipped", attempt=1,
                ))
                database.set_module_summary(
                    job_id, "export", status="completed", completed=0, skipped=1, finished=True,
                )

                result = ExportCommitCoordinator(database, layout, job_id=job_id).commit()

                self.assertEqual(0, result.exported)
                self.assertEqual(0, result.unlabeled)
                self.assertEqual(original_json, (dataset / "sample.json").read_bytes())
                self.assertEqual(original_txt, (dataset / "sample.txt").read_bytes())
                self.assertFalse((dataset / result.unlabeledFolder).exists())
            finally:
                database.close()
                service.close()

    def test_incomplete_artifact_index_preserves_owned_pre_swap_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                original = (dataset / "sample.json").read_bytes()
                self._complete_export(database, job_id, layout, format_value="both", include_all=False)
                with self.assertRaises(ExportCommitError):
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(original, (dataset / "sample.json").read_bytes())
                self._assert_preserved_pre_swap(
                    database, job_id, dataset, layout,
                    state="staging", phase="writing_annotations",
                )
            finally:
                database.close()
                service.close()

    def test_second_directory_rename_failure_restores_original_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                original = (dataset / "sample.json").read_bytes()
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="failed"))
                self._complete_export(database, job_id, layout, format_value="both")
                real_replace = __import__("os").replace
                def fail_second_replace(source, target):
                    if (".anima-stage-" in Path(source).name or Path(source).name == "stage") and Path(target) == dataset:
                        raise OSError("injected staging rename failure")
                    return real_replace(source, target)

                with patch("anima_core.export_commit.os.replace", side_effect=fail_second_replace), patch(
                    "anima_core.export_commit.replace_ocr_sidecar", wraps=__import__("anima_core.export_commit", fromlist=["replace_ocr_sidecar"]).replace_ocr_sidecar,
                ) as copied:
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                copied.assert_called_once()
                self.assertEqual(original, (dataset / "sample.json").read_bytes())
                self.assertFalse((dataset / "ocr_annotations" / "sample.png.ocr.json").exists())
                self.assertEqual("failed", database.get_job(job_id)["status"])
                self.assertEqual([], self._staging_residue(dataset))
                PipelineService(Path(temporary) / "state.db", install_root=ROOT / ".runtime-build").recover_pending_commits()
                self.assertEqual("rolled_back", json.loads(layout.commit_journal_path().read_text(encoding="utf-8"))["state"])
                self.assertEqual([], self._staging_residue(dataset))
            finally:
                database.close()
                service.close()

    def test_first_rename_journal_failure_restores_the_complete_old_ocr_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                original = (dataset / "sample.json").read_bytes()
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="failed"))
                self._complete_export(database, job_id, layout, format_value="both")
                real_write = __import__("anima_core.export_commit", fromlist=["write_journal"]).write_journal
                failed = [False]

                def fail_once_after_first_rename(current_layout, journal):
                    if journal.state == "rollback_created" and not failed[0]:
                        failed[0] = True
                        raise OSError("injected first-rename journal failure")
                    return real_write(current_layout, journal)

                with patch("anima_core.export_commit.write_journal", side_effect=fail_once_after_first_rename):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertTrue(failed[0])
                self.assertEqual(original, (dataset / "sample.json").read_bytes())
                self.assertFalse((dataset / "ocr_annotations" / "sample.png.ocr.json").exists())
                self.assertEqual([], self._staging_residue(dataset))
                self.assertEqual([], self._rollback_residue(dataset))
                self.assertEqual("rolled_back", json.loads(layout.commit_journal_path().read_text(encoding="utf-8"))["state"])
            finally:
                database.close()
                service.close()

    def test_second_rename_journal_failure_recovers_the_complete_new_ocr_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                sidecar = _ocr_sidecar("sample.png", status="failed")
                layout.write_ocr_sidecar("sample.png", sidecar)
                self._complete_export(database, job_id, layout, format_value="both")
                real_write = __import__("anima_core.export_commit", fromlist=["write_journal"]).write_journal
                failed = [False]

                def fail_once_after_second_rename(current_layout, journal):
                    if journal.state == "committed" and not failed[0]:
                        failed[0] = True
                        raise OSError("injected second-rename journal failure")
                    return real_write(current_layout, journal)

                with patch("anima_core.export_commit.write_journal", side_effect=fail_once_after_second_rename):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertTrue(failed[0])
                self.assertEqual(b'{\n  "tags": [\n    "verified"\n  ]\n}\n', (dataset / "sample.json").read_bytes())
                self.assertEqual(sidecar, (dataset / "ocr_annotations" / "sample.png.ocr.json").read_bytes())
                self.assertEqual("committed", json.loads(layout.commit_journal_path().read_text(encoding="utf-8"))["state"])
                PipelineService(Path(temporary) / "state.db", install_root=ROOT / ".runtime-build").recover_pending_commits()
                self.assertEqual("committed", json.loads(layout.commit_journal_path().read_text(encoding="utf-8"))["state"])
                self.assertEqual([], self._staging_residue(dataset))
                self.assertEqual([], self._rollback_residue(dataset))
            finally:
                database.close()
                service.close()

    def test_staging_write_failure_preserves_owned_pre_swap_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                original = (dataset / "sample.json").read_bytes()
                self._complete_export(database, job_id, layout, format_value="both")
                with patch("anima_core.export_commit.replace_business_annotation", side_effect=OSError("injected disk full")):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(original, (dataset / "sample.json").read_bytes())
                self._assert_preserved_pre_swap(
                    database, job_id, dataset, layout,
                    state="staging", phase="writing_annotations",
                )
            finally:
                database.close()
                service.close()

    def test_backup_failure_preserves_owned_pre_swap_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                original = (dataset / "sample.json").read_bytes()
                self._complete_export(database, job_id, layout, format_value="both")
                with patch("anima_core.export_commit.write_backup", side_effect=OSError("injected backup full")):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(original, (dataset / "sample.json").read_bytes())
                self._assert_preserved_pre_swap(
                    database, job_id, dataset, layout,
                    state="staged", phase="writing_backup",
                )
            finally:
                database.close()
                service.close()

    def test_failure_after_backup_leaves_a_complete_old_directory_without_ocr_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="success"))
                self._complete_export(database, job_id, layout, format_value="both")
                def fail_after_backup(*args, **kwargs):
                    write_backup_impl(*args, **kwargs)
                    raise OSError("injected post-backup failure")

                with patch("anima_core.export_commit.write_backup", side_effect=fail_after_backup):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertFalse((dataset / "ocr_annotations" / "sample.png.ocr.json").exists())
                self.assertEqual(b'{"legacy":true}\n', (dataset / "sample.json").read_bytes())
                self._assert_preserved_pre_swap(
                    database, job_id, dataset, layout,
                    state="staged", phase="writing_backup",
                )
            finally:
                database.close()
                service.close()

    def test_externally_changed_baseline_annotation_aborts_before_any_directory_switch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                self._complete_export(database, job_id, layout, format_value="both")
                external = b"changed by another program\n"
                (dataset / "sample.txt").write_bytes(external)
                with self.assertRaises(ExportCommitError):
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(external, (dataset / "sample.txt").read_bytes())
                self.assertEqual(b'{"legacy":true}\n', (dataset / "sample.json").read_bytes())
                self._assert_preserved_pre_swap(
                    database, job_id, dataset, layout,
                    state="staged", phase="verifying_baseline",
                )
            finally:
                database.close()
                service.close()

    def test_commit_records_replace_provenance_for_the_final_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                fingerprint = "a" * 64
                self._complete_export(database, job_id, layout, format_value="both")
                ReplaceOverlayWriter(database, layout, job_id).mark_provenance("sample", fingerprint)
                ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                final_digest = hashlib.sha256((dataset / "sample.json").read_bytes()).hexdigest()
                with DatasetReplaceProvenance.open(dataset) as provenance:
                    self.assertTrue(provenance.matches("sample", fingerprint, final_digest))
            finally:
                database.close()
                service.close()

    def test_flat_txt_commit_removes_replace_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), format_value="flat_txt",
            )
            try:
                fingerprint = "b" * 64
                digest = hashlib.sha256((dataset / "sample.json").read_bytes()).hexdigest()
                apply_provenance_changes(dataset, [
                    ReplaceProvenanceChange.upsert("sample", fingerprint, digest),
                ])
                self._complete_export(database, job_id, layout, format_value="flat_txt")
                ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertFalse((dataset / "sample.json").exists())
                with DatasetReplaceProvenance.open(dataset) as provenance:
                    self.assertFalse(provenance.matches("sample", fingerprint, digest))
            finally:
                database.close()
                service.close()

    def test_failed_commit_does_not_change_dataset_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                old_fingerprint = "c" * 64
                old_digest = hashlib.sha256((dataset / "sample.json").read_bytes()).hexdigest()
                apply_provenance_changes(dataset, [
                    ReplaceProvenanceChange.upsert("sample", old_fingerprint, old_digest),
                ])
                database_bytes = provenance_database_path(dataset).read_bytes()
                self._complete_export(database, job_id, layout, format_value="both")
                ReplaceOverlayWriter(database, layout, job_id).mark_provenance("sample", "d" * 64)
                (dataset / "sample.txt").write_bytes(b"changed outside the task\n")
                with self.assertRaises(ExportCommitError):
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(database_bytes, provenance_database_path(dataset).read_bytes())
                with DatasetReplaceProvenance.open(dataset) as provenance:
                    self.assertTrue(provenance.matches("sample", old_fingerprint, old_digest))
            finally:
                database.close()
                service.close()

    def test_missing_hardlink_support_or_free_space_stops_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary))
            try:
                self._complete_export(database, job_id, layout, format_value="both")
                with patch("anima_core.export_commit.os.link", side_effect=OSError("hard links unavailable")):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                usage = __import__("shutil").disk_usage(dataset.parent)
                with patch("anima_core.export_commit.shutil.disk_usage", return_value=usage._replace(free=0)):
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual([], self._staging_residue(dataset))
                self.assertFalse(layout.commit_journal_path().exists())
                self.assertEqual(b'{"legacy":true}\n', (dataset / "sample.json").read_bytes())
            finally:
                database.close()
                service.close()

    def test_commit_preserves_existing_replaces_new_and_keeps_failed_ocr_sidecars_out_of_business_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), extra_images=("new.jpg", "failed.png"), ocr_enabled=True,
            )
            try:
                formal = dataset / "ocr_annotations"
                formal.mkdir()
                (formal / "sample.png.ocr.json").write_bytes(_ocr_sidecar("sample.png", status="no_text"))
                layout.write_ocr_sidecar("new.jpg", _ocr_sidecar("new.jpg", status="success"))
                layout.write_ocr_sidecar("failed.png", _ocr_sidecar("failed.png", status="failed"))
                self._complete_export(database, job_id, layout, format_value="both")
                result = ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(3, result.exported)
                self.assertEqual(_ocr_sidecar("sample.png", status="no_text"), (formal / "sample.png.ocr.json").read_bytes())
                self.assertEqual(_ocr_sidecar("new.jpg", status="success"), (formal / "new.jpg.ocr.json").read_bytes())
                self.assertEqual(_ocr_sidecar("failed.png", status="failed"), (formal / "failed.png.ocr.json").read_bytes())
                self.assertEqual(
                    {"sample.json", "new.json", "failed.json"},
                    {path.name for path in dataset.glob("*.json")},
                )
                self.assertTrue(all(path.read_bytes() == b'{\n  "tags": [\n    "verified"\n  ]\n}\n' for path in dataset.glob("*.json")))
                with __import__("zipfile").ZipFile(result.backupZip) as archive:
                    self.assertFalse(any(name.startswith("ocr_annotations/") for name in archive.namelist()))
            finally:
                database.close()
                service.close()

    def test_invalid_overlay_sidecar_fails_before_any_directory_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                layout.write_ocr_sidecar("sample.png", b"not-json")
                self._complete_export(database, job_id, layout, format_value="both")
                with self.assertRaises(ExportCommitError):
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertFalse(layout.commit_journal_path().exists())
                self.assertEqual(b'{"legacy":true}\n', (dataset / "sample.json").read_bytes())
            finally:
                database.close()
                service.close()

    def test_sidecar_copy_failure_after_staging_and_backup_recovers_to_complete_old_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="failed"))
                self._complete_export(database, job_id, layout, format_value="both")
                copy = getattr(__import__("anima_core.export_commit", fromlist=["replace_ocr_sidecar"]), "replace_ocr_sidecar", None)
                self.assertTrue(callable(copy), "OCR commit copy API is missing")
                with patch("anima_core.export_commit.write_backup", side_effect=OSError("injected backup full")), patch(
                    "anima_core.export_commit.replace_ocr_sidecar", wraps=copy,
                ) as copied:
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                copied.assert_called_once()
                self.assertFalse((dataset / "ocr_annotations" / "sample.png.ocr.json").exists())
                self._assert_preserved_pre_swap(
                    database, job_id, dataset, layout,
                    state="staged", phase="writing_backup",
                )
            finally:
                database.close()
                service.close()

    def test_sidecar_copy_failure_after_staging_does_not_switch_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            try:
                original = (dataset / "sample.json").read_bytes()
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="failed"))
                self._complete_export(database, job_id, layout, format_value="both")
                with patch(
                    "anima_core.export_commit.replace_ocr_sidecar",
                    side_effect=OSError("injected sidecar copy failure"),
                ) as copied:
                    with self.assertRaises(ExportCommitError):
                        ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                copied.assert_called_once()
                self.assertEqual(original, (dataset / "sample.json").read_bytes())
                self.assertFalse((dataset / "ocr_annotations" / "sample.png.ocr.json").exists())
                self._assert_preserved_pre_swap(
                    database, job_id, dataset, layout,
                    state="staging", phase="copying_ocr",
                )
            finally:
                database.close()
                service.close()

    def test_all_export_formats_commit_ocr_sidecars_without_changing_business_artifact_rules(self) -> None:
        for format_value in ("json", "flat_txt", "both"):
            with self.subTest(format=format_value), tempfile.TemporaryDirectory() as temporary:
                database, service, job_id, dataset, layout = self._prepared_job(
                    Path(temporary), format_value=format_value, ocr_enabled=True,
                )
                try:
                    sidecar = _ocr_sidecar("sample.png", status="no_text")
                    layout.write_ocr_sidecar("sample.png", sidecar)
                    self._complete_export(database, job_id, layout, format_value=format_value)
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                    self.assertEqual(sidecar, (dataset / "ocr_annotations" / "sample.png.ocr.json").read_bytes())
                    self.assertEqual(format_value != "flat_txt", (dataset / "sample.json").is_file())
                    self.assertEqual(format_value != "json", (dataset / "sample.txt").is_file())
                finally:
                    database.close()
                    service.close()

    def test_ocr_enabled_and_disabled_commits_have_byte_identical_business_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            disabled_root, enabled_root = root / "disabled", root / "enabled"
            disabled_root.mkdir()
            enabled_root.mkdir()
            disabled_db, disabled_service, disabled_job, disabled_dataset, disabled_layout = self._prepared_job(
                disabled_root, ocr_enabled=False,
            )
            enabled_db, enabled_service, enabled_job, enabled_dataset, enabled_layout = self._prepared_job(
                enabled_root, ocr_enabled=True,
            )
            try:
                self._complete_export(disabled_db, disabled_job, disabled_layout, format_value="both")
                ExportCommitCoordinator(disabled_db, disabled_layout, job_id=disabled_job).commit()
                enabled_layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="success"))
                self._complete_export(enabled_db, enabled_job, enabled_layout, format_value="both")
                ExportCommitCoordinator(enabled_db, enabled_layout, job_id=enabled_job).commit()
                self.assertEqual(self._business_bytes(disabled_dataset), self._business_bytes(enabled_dataset))
                self.assertTrue((enabled_dataset / "ocr_annotations" / "sample.png.ocr.json").is_file())
            finally:
                disabled_db.close()
                disabled_service.close()
                enabled_db.close()
                enabled_service.close()

    def test_ocr_disabled_does_not_commit_a_stale_overlay_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), ocr_enabled=False,
            )
            try:
                self._complete_export(database, job_id, layout, format_value="both")
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="success"))
                ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertFalse((dataset / "ocr_annotations" / "sample.png.ocr.json").exists())
            finally:
                database.close()
                service.close()

    def test_unscoped_overlay_sidecar_is_not_committed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(
                Path(temporary), extra_images=("unscoped.jpg",), ocr_enabled=True,
            )
            try:
                database.connection.execute(
                    "UPDATE samples SET in_processing_scope=0 WHERE job_id=? AND relative_image_path=?",
                    (job_id, "unscoped.jpg"),
                )
                layout.write_ocr_sidecar("sample.png", _ocr_sidecar("sample.png", status="no_text"))
                layout.write_ocr_sidecar("unscoped.jpg", _ocr_sidecar("unscoped.jpg", status="failed"))
                self._complete_export(database, job_id, layout, format_value="both")
                ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertTrue((dataset / "ocr_annotations" / "sample.png.ocr.json").is_file())
                self.assertFalse((dataset / "ocr_annotations" / "unscoped.jpg.ocr.json").exists())
            finally:
                database.close()
                service.close()

    def test_overlay_sidecar_reparse_point_blocks_commit_before_directory_switch(self) -> None:
        if os.name != "nt":
            self.skipTest("the reparse-point fixture is Windows-specific")
        with tempfile.TemporaryDirectory() as temporary:
            database, service, job_id, dataset, layout = self._prepared_job(Path(temporary), ocr_enabled=True)
            sidecar_directory = layout.root / "ocr_annotations"
            try:
                original = (dataset / "sample.json").read_bytes()
                target = Path(temporary) / "outside-sidecars"
                target.mkdir()
                (target / "sample.png.ocr.json").write_bytes(_ocr_sidecar("sample.png", status="success"))
                sidecar_directory.rmdir()
                junction = subprocess.run(
                    ["cmd.exe", "/d", "/c", "mklink", "/J", str(sidecar_directory), str(target)],
                    capture_output=True,
                    check=False,
                    text=True,
                )
                if junction.returncode != 0:
                    self.fail(f"unable to create reparse-point fixture: {junction.stderr or junction.stdout}")
                self._complete_export(database, job_id, layout, format_value="both")
                with self.assertRaises(ExportCommitError):
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                self.assertEqual(original, (dataset / "sample.json").read_bytes())
                self.assertFalse(layout.commit_journal_path().exists())
                self.assertEqual([], self._staging_residue(dataset))
            finally:
                if sidecar_directory.exists():
                    sidecar_directory.rmdir()
                database.close()
                service.close()


if __name__ == "__main__":
    unittest.main()
