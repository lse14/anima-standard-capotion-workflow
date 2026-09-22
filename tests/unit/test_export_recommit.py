from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from PIL import Image

from anima_core.annotation_backup import write_backup
from anima_core.commit_journal import CommitJournal, write as write_commit_journal
from anima_core.contracts import JobConfig, SampleRunState
from anima_core.db import StateDatabase
from anima_core.export_recommit import ExportRecommitError, ExportRecommitService
from anima_core.export_staging import create_hardlink_staging
from anima_core.job_preflight import JobPreparationService
from anima_core.overlay import OverlayLayout
from anima_core.scheduler import BoundedScheduler


class ExportRecommitTests(unittest.TestCase):
    def _interrupted_job(self, root: Path) -> SimpleNamespace:
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (3, 3), "white").save(dataset / "sample.png")
        (dataset / "sample.json").write_bytes(b'{"legacy":true}\n')
        (dataset / "sample.txt").write_bytes(b"legacy\n")
        config = JobConfig(
            schemaVersion=10,
            workMode="in_place",
            overwriteMode="incremental",
            sourceRoot=str(dataset),
        )
        config.caption["enabled"] = False
        config.classify["enabled"] = False
        config.replace["enabled"] = False
        config.ocr["enabled"] = False
        config.nl["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        config.dropout["enabled"] = False
        config.tokenBudget["enabled"] = False  # type: ignore[index]
        config.export["format"] = "both"

        preparation = JobPreparationService(root / "state.db")
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        database = StateDatabase.open(root / "state.db")
        try:
            scheduler = BoundedScheduler(database)
            for module_id in (
                "caption", "classify", "replace", "ocr", "nl",
                "count_review", "dropout", "token_budget",
            ):
                scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
            scheduler.start_module(job_id, "export", enabled=True, profile="e621")
            layout = OverlayLayout.open_existing(
                str(database.get_job(job_id)["overlay_root"]), job_id,
            )
            json_path, json_digest = layout.write_prepared(
                "export", "json-lease", ".json", b'{"tags":["verified"]}\n',
            )
            txt_path, txt_digest = layout.write_prepared(
                "export", "txt-lease", ".txt", b"verified.\n",
            )
            database.connection.executemany(
                "INSERT INTO export_artifacts(job_id,sample_id,kind,relative_path,sha256) VALUES (?,?,?,?,?)",
                (
                    (job_id, 1, "json", str(json_path.relative_to(layout.root)).replace("/", "\\"), json_digest),
                    (job_id, 1, "txt", str(txt_path.relative_to(layout.root)).replace("/", "\\"), txt_digest),
                ),
            )
            database.set_sample_state(job_id, 1, SampleRunState(
                sampleId=1, currentModuleId="export", status="completed", attempt=1,
            ))
            database.set_module_summary(
                job_id, "export", status="completed", completed=1, finished=True,
            )
            database.set_job_status(job_id, "committing", current_module_id="export")
            token = job_id.replace("-", "")[:24]
            staging = create_hardlink_staging(
                dataset, dataset.parent / f".{dataset.name}.anima-stage-{token}",
            )
            nested = staging / "inventory"
            nested.mkdir()
            (nested / "evidence.bin").write_bytes(b"inventory-evidence")
            rollback = dataset.parent / f".{dataset.name}.anima-rollback-{token}"
            backup = dataset.parent / f".{dataset.name}.anima-backups" / f"{job_id}.zip"
            write_commit_journal(layout, CommitJournal.v2(
                job_id, "staged", dataset, staging, rollback, backup,
                phase="writing_backup",
            ))
            database.mark_interrupted(job_id)
        finally:
            database.close()
        return SimpleNamespace(
            preparation=preparation,
            job_id=job_id,
            dataset=dataset,
            layout=layout,
            staging=staging,
            rollback=rollback,
            backup=backup,
            database_path=root / "state.db",
        )

    @staticmethod
    def _expected_inventory(stage: Path) -> tuple[int, int, int]:
        files = [path for path in stage.rglob("*") if path.is_file()]
        directories = [stage, *(path for path in stage.rglob("*") if path.is_dir())]
        return len(files), len(directories), sum(path.stat().st_size for path in files)

    def test_preview_uses_only_the_journal_path_and_returns_stable_complete_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._interrupted_job(Path(temporary))
            try:
                service = ExportRecommitService(fixture.database_path)
                self.assertEqual(["job_id"], list(inspect.signature(service.preview).parameters))
                first = service.preview(fixture.job_id)
                second = service.preview(fixture.job_id)
                expected = self._expected_inventory(fixture.staging)

                self.assertEqual(first["inventoryFingerprint"], second["inventoryFingerprint"])
                self.assertEqual(str(fixture.staging), first["stage"]["path"])
                self.assertEqual(expected, (
                    first["stage"]["fileCount"],
                    first["stage"]["directoryCount"],
                    first["stage"]["apparentBytes"],
                ))
                self.assertEqual((0, 0, False, 0), (
                    first["stage"]["errorCount"], first["stage"]["skippedCount"],
                    first["stage"]["truncated"], first["stage"]["exitCode"],
                ))
                self.assertTrue(first["dataset"]["baselineMatches"])
                self.assertTrue(first["overlay"]["artifactsValid"])
                self.assertEqual("missing", first["backup"]["state"])
                self.assertTrue(first["safeToRecommit"])
            finally:
                fixture.preparation.close()

    def test_confirmation_and_second_scan_conflicts_never_remove_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._interrupted_job(Path(temporary))
            try:
                service = ExportRecommitService(fixture.database_path)
                preview = service.preview(fixture.job_id)
                with self.assertRaisesRegex(ExportRecommitError, "confirmation"):
                    service.recommit(
                        fixture.job_id, confirmed=False,
                        inventory_fingerprint=preview["inventoryFingerprint"],
                    )
                with self.assertRaisesRegex(ExportRecommitError, "fingerprint"):
                    service.recommit(
                        fixture.job_id, confirmed=True,
                        inventory_fingerprint="0" * 64,
                    )
                (fixture.staging / "changed-after-preview.bin").write_bytes(b"changed")
                with self.assertRaisesRegex(ExportRecommitError, "fingerprint"):
                    service.recommit(
                        fixture.job_id, confirmed=True,
                        inventory_fingerprint=preview["inventoryFingerprint"],
                    )
                self.assertTrue((fixture.staging / "changed-after-preview.bin").is_file())
                self.assertTrue(fixture.dataset.is_dir())
            finally:
                fixture.preparation.close()

    def test_dataset_and_overlay_changes_fail_closed_without_removing_stage(self) -> None:
        for changed in ("dataset", "overlay"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                fixture = self._interrupted_job(Path(temporary))
                try:
                    service = ExportRecommitService(fixture.database_path)
                    preview = service.preview(fixture.job_id)
                    if changed == "dataset":
                        (fixture.dataset / "sample.json").write_bytes(b'{"external":true}\n')
                    else:
                        database = StateDatabase.open(fixture.database_path)
                        try:
                            row = database.page_export_artifacts(fixture.job_id, limit=1)[0]
                            prepared = fixture.layout.resolve_prepared(str(row["relative_path"]))
                            prepared.write_bytes(b"changed-overlay")
                        finally:
                            database.close()
                    changed_preview = service.preview(fixture.job_id)
                    self.assertNotEqual(
                        preview["inventoryFingerprint"], changed_preview["inventoryFingerprint"],
                    )
                    self.assertFalse(changed_preview["safeToRecommit"])
                    with self.assertRaises(ExportRecommitError):
                        service.recommit(
                            fixture.job_id, confirmed=True,
                            inventory_fingerprint=changed_preview["inventoryFingerprint"],
                        )
                    self.assertTrue(fixture.staging.is_dir())
                finally:
                    fixture.preparation.close()

    def test_invalid_stage_identity_and_incomplete_inventory_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._interrupted_job(Path(temporary))
            try:
                wrong_stage = fixture.dataset.parent / ".dataset.anima-stage-another-job"
                wrong_stage.mkdir()
                write_commit_journal(fixture.layout, CommitJournal.v2(
                    fixture.job_id, "staged", fixture.dataset, wrong_stage,
                    fixture.rollback, fixture.backup, phase="writing_backup",
                ))
                service = ExportRecommitService(fixture.database_path)
                with self.assertRaisesRegex(ExportRecommitError, "stage path"):
                    service.preview(fixture.job_id)
                self.assertTrue(wrong_stage.is_dir())
                self.assertTrue(fixture.staging.is_dir())
            finally:
                fixture.preparation.close()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._interrupted_job(Path(temporary))
            try:
                with patch("anima_core.export_recommit._is_reparse_stat", return_value=True):
                    reparse = ExportRecommitService(fixture.database_path).preview(fixture.job_id)
                self.assertEqual("reparse_point", reparse["stage"]["type"])
                self.assertFalse(reparse["safeToRecommit"])

                with patch("anima_core.export_recommit.os.scandir", side_effect=PermissionError("denied")):
                    preview = ExportRecommitService(fixture.database_path).preview(fixture.job_id)
                self.assertGreater(preview["stage"]["errorCount"], 0)
                self.assertNotEqual(0, preview["stage"]["exitCode"])
                self.assertFalse(preview["safeToRecommit"])

                truncated = ExportRecommitService(
                    fixture.database_path, max_inventory_entries=1,
                ).preview(fixture.job_id)
                self.assertTrue(truncated["stage"]["truncated"])
                self.assertFalse(truncated["safeToRecommit"])
                self.assertTrue(fixture.staging.is_dir())
            finally:
                fixture.preparation.close()

    def test_confirmed_recommit_reuses_valid_backup_and_rebuilds_invalid_backup(self) -> None:
        for backup_kind in ("valid", "invalid", "partial"):
            with self.subTest(backup=backup_kind), tempfile.TemporaryDirectory() as temporary:
                fixture = self._interrupted_job(Path(temporary))
                try:
                    fixture.backup.parent.mkdir()
                    valid_backup_bytes = None
                    if backup_kind == "valid":
                        database = StateDatabase.open(fixture.database_path)
                        try:
                            write_backup(
                                fixture.dataset,
                                fixture.backup,
                                lambda cursor: database.page_samples(
                                    fixture.job_id, after_sample_id=cursor, limit=500,
                                ),
                            )
                        finally:
                            database.close()
                        valid_backup_bytes = fixture.backup.read_bytes()
                    elif backup_kind == "invalid":
                        fixture.backup.write_bytes(b"not-a-zip")
                    else:
                        fixture.backup.with_suffix(fixture.backup.suffix + ".partial").write_bytes(b"partial")
                    unrelated_stage = fixture.dataset.parent / ".dataset.anima-stage-unrelated"
                    unrelated_stage.mkdir()
                    (unrelated_stage / "keep.txt").write_text("keep", encoding="ascii")
                    service = ExportRecommitService(fixture.database_path)
                    preview = service.preview(fixture.job_id)
                    self.assertEqual(backup_kind, preview["backup"]["state"])
                    with patch.object(
                        StateDatabase,
                        "resolve_repaired_parent_issues",
                        autospec=True,
                        wraps=StateDatabase.resolve_repaired_parent_issues,
                    ) as resolved:
                        result = service.recommit(
                            fixture.job_id, confirmed=True,
                            inventory_fingerprint=preview["inventoryFingerprint"],
                        )
                    self.assertEqual("succeeded", result["status"])
                    self.assertFalse(fixture.staging.exists())
                    self.assertTrue(fixture.backup.is_file())
                    self.assertEqual(b'{"tags":["verified"]}\n', (fixture.dataset / "sample.json").read_bytes())
                    self.assertEqual("keep", (unrelated_stage / "keep.txt").read_text(encoding="ascii"))
                    self.assertFalse(fixture.backup.with_suffix(fixture.backup.suffix + ".partial").exists())
                    if valid_backup_bytes is not None:
                        self.assertEqual(valid_backup_bytes, fixture.backup.read_bytes())
                    resolved.assert_called_once()
                finally:
                    fixture.preparation.close()


if __name__ == "__main__":
    unittest.main()
