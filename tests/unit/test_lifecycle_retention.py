from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from PIL import Image

from anima_core.contracts import JobConfig
from anima_core.db import StateDatabase
from anima_core.job_preflight import JobPreparationService
from anima_core.lifecycle import JobLifecycle, JobLifecycleError
from anima_core.locks import DatasetLockError, release_completed_dataset_lock
from anima_core.overlay import OverlayLayout
from anima_core.path_safety import windows_key
from anima_core.retention import RetentionManager
from anima_core.state_machine import InvalidTransition


def _job(job_id: str, root: Path, *, status: str = "failed", overlay_root: str | None = None) -> dict[str, object]:
    config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(root))
    return {
        "job_id": job_id, "config_schema_version": 2, "config_json": json.dumps(config.to_dict()),
        "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental",
        "source_root": str(root), "output_root": None, "dataset_root": str(root), "dataset_root_key": windows_key(root),
        "manifest_schema_version": 1, "recursive": 0, "sample_count": 0, "manifest_generated_at": None,
        "status": status, "current_module_id": None, "last_event_id": 0, "pinned": 0,
        "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": overlay_root, "commit_journal_path": None,
        "resume_status": None, "created_at": "2026-01-01T00:00:00Z", "started_at": None,
        "cancel_requested_at": None, "finished_at": "2026-01-01T00:00:00Z" if status in {"succeeded", "succeeded_with_issues"} else None,
    }


class LifecycleAndRetentionTests(unittest.TestCase):
    def test_history_lock_release_keeps_claim_when_stale_lock_file_cannot_be_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            lock_path = root / "job-done.lock"
            lock_path.write_text("stale lock", encoding="utf-8")
            database_path = root / "state.db"
            database = StateDatabase.open(database_path)
            try:
                database.insert_job(_job("job-done", dataset, status="succeeded"))
                database.connection.execute(
                    """INSERT INTO dataset_claims(dataset_root,dataset_root_key,job_id,lock_path,acquired_at)
                       VALUES (?,?,?,?,?)""",
                    (str(dataset), windows_key(dataset), "job-done", str(lock_path), "2026-01-01T00:00:00Z"),
                )
            finally:
                database.close()

            with patch.object(Path, "unlink", side_effect=PermissionError("injected stale lock contention")):
                with self.assertRaisesRegex(DatasetLockError, "cannot remove dataset lock"):
                    release_completed_dataset_lock({}, database_path, "job-done")

            database = StateDatabase.open(database_path)
            try:
                self.assertEqual("succeeded", database.get_job("job-done")["status"])
                self.assertEqual(
                    1,
                    database.connection.execute(
                        "SELECT COUNT(*) FROM dataset_claims WHERE job_id='job-done'"
                    ).fetchone()[0],
                )
            finally:
                database.close()
            self.assertTrue(lock_path.is_file())

    def test_partial_success_releases_completed_dataset_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            lock_path = root / "partial.lock"
            lock_path.write_text("stale lock", encoding="ascii")
            database_path = root / "state.db"
            database = StateDatabase.open(database_path)
            try:
                database.insert_job(_job("job-partial", dataset, status="succeeded_with_issues"))
                database.connection.execute(
                    """INSERT INTO dataset_claims(dataset_root,dataset_root_key,job_id,lock_path,acquired_at)
                       VALUES (?,?,?,?,?)""",
                    (str(dataset), windows_key(dataset), "job-partial", str(lock_path), "2026-01-01T00:00:00Z"),
                )
            finally:
                database.close()

            self.assertTrue(release_completed_dataset_lock({}, database_path, "job-partial"))
            database = StateDatabase.open(database_path)
            try:
                self.assertEqual(
                    0,
                    database.connection.execute(
                        "SELECT COUNT(*) FROM dataset_claims WHERE job_id='job-partial'"
                    ).fetchone()[0],
                )
            finally:
                database.close()

    def test_delete_history_removes_only_succeeded_control_record_and_preserves_business_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            image = dataset / "image.png"
            annotation = dataset / "image.json"
            source_txt = dataset / "image.txt"
            export = root / "exports" / "image.json"
            backup = root / ".dataset.anima-backups" / "job-done.zip"
            Image.new("RGB", (3, 3), "white").save(image)
            annotation.write_text('{"count":"solo"}', encoding="utf-8")
            source_txt.write_text("tag", encoding="utf-8")
            export.parent.mkdir()
            export.write_text("exported", encoding="utf-8")
            backup.parent.mkdir()
            backup.write_bytes(b"original backup")
            paths = (image, annotation, source_txt, export, backup)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("job-done", dataset, status="succeeded"))
                result = JobLifecycle(database).delete_history("job-done", confirmed=True)
                self.assertTrue(result.deletedControlRecord)
                with self.assertRaises(KeyError):
                    database.get_job("job-done")
            finally:
                database.close()
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})

    def test_delete_history_blocks_parent_until_repair_child_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("parent", dataset, status="succeeded"))
                database.insert_job(_job("child", dataset, status="succeeded"))
                database.create_repair_link("child", "parent")
                with self.assertRaisesRegex(JobLifecycleError, "请先删除修复子任务"):
                    JobLifecycle(database).delete_history("parent", confirmed=True)
                JobLifecycle(database).delete_history("child", confirmed=True)
                JobLifecycle(database).delete_history("parent", confirmed=True)
            finally:
                database.close()

    def test_delete_history_requires_confirmation_and_a_succeeded_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("job-failed", dataset, status="failed"))
                database.insert_job(_job("job-done", dataset, status="succeeded"))
                database.insert_job(_job("job-partial", dataset, status="succeeded_with_issues"))
                lifecycle = JobLifecycle(database)
                with self.assertRaises(JobLifecycleError):
                    lifecycle.delete_history("job-done", confirmed=False)
                with self.assertRaises(JobLifecycleError):
                    lifecycle.delete_history("job-failed", confirmed=True)
                self.assertEqual("failed", database.get_job("job-failed")["status"])

                lifecycle.delete_history("job-done", confirmed=True)
                lifecycle.delete_history("job-partial", confirmed=True)

                with self.assertRaises(KeyError):
                    database.get_job("job-done")
            finally:
                database.close()

    def test_delete_history_refuses_a_remaining_dataset_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("job-done", dataset, status="succeeded"))
                database.connection.execute(
                    """INSERT INTO dataset_claims(dataset_root,dataset_root_key,job_id,lock_path,acquired_at)
                       VALUES (?,?,?,?,?)""",
                    (str(dataset), windows_key(dataset), "job-done", str(root / "job-done.lock"), "2026-01-01T00:00:00Z"),
                )

                with self.assertRaisesRegex(JobLifecycleError, "dataset lock release"):
                    JobLifecycle(database).delete_history("job-done", confirmed=True)

                self.assertEqual("succeeded", database.get_job("job-done")["status"])
                self.assertEqual(
                    1,
                    database.connection.execute(
                        "SELECT COUNT(*) FROM dataset_claims WHERE job_id='job-done'"
                    ).fetchone()[0],
                )
            finally:
                database.close()

    def test_delete_history_preserves_resolved_overlay_and_blocks_unresolved_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            layout = OverlayLayout.create(dataset, "job-done")
            layout.write_journal({"state": "rollback_created"})
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("job-done", dataset, status="succeeded", overlay_root=str(layout.root)))
                database.connection.execute(
                    "UPDATE jobs SET commit_journal_path=? WHERE job_id=?",
                    (str(layout.commit_journal_path()), "job-done"),
                )
                lifecycle = JobLifecycle(database)
                with self.assertRaises(JobLifecycleError):
                    lifecycle.delete_history("job-done", confirmed=True)
                self.assertTrue(layout.root.exists())
                self.assertEqual("succeeded", database.get_job("job-done")["status"])

                layout.write_journal({"state": "committed"})
                lifecycle.delete_history("job-done", confirmed=True)

                self.assertTrue(layout.root.exists())
                self.assertTrue(layout.commit_journal_path().is_file())
            finally:
                database.close()

    def test_discard_needs_confirmation_and_resolved_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            layout = OverlayLayout.create(dataset, "job-1")
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("job-1", dataset, overlay_root=str(layout.root)))
                lifecycle = JobLifecycle(database)
                with self.assertRaises(InvalidTransition):
                    lifecycle.discard("job-1", confirmed=False)
                layout.write_journal({"state": "rollback_created"})
                with self.assertRaises(JobLifecycleError):
                    lifecycle.discard("job-1", confirmed=True)
                layout.write_journal({"state": "rolled_back"})
                result = lifecycle.discard("job-1", confirmed=True)
                self.assertTrue(result.overlayDeleted)
                self.assertEqual("discarded", database.get_job("job-1")["status"])
            finally:
                database.close()

    def test_discard_and_startup_sweep_release_dataset_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            other = root / "other"
            dataset.mkdir()
            other.mkdir()
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("job-discard", dataset))
                database.insert_job(_job("job-done", other, status="succeeded"))
                partial = root / "partial"
                partial.mkdir()
                database.insert_job(_job("job-partial", partial, status="succeeded_with_issues"))
                failed = root / "failed"
                cancelled = root / "cancelled"
                failed.mkdir()
                cancelled.mkdir()
                database.insert_job(_job("job-failed", failed, status="failed"))
                database.insert_job(_job("job-cancelled", cancelled, status="cancelled_recoverable"))
                for job_id, path in (
                    ("job-discard", dataset),
                    ("job-done", other),
                    ("job-partial", partial),
                    ("job-failed", failed),
                    ("job-cancelled", cancelled),
                ):
                    database.connection.execute(
                        """INSERT INTO dataset_claims(dataset_root,dataset_root_key,job_id,lock_path,acquired_at)
                           VALUES (?,?,?,?,?)""",
                        (str(path), str(path), job_id, str(path) + ".lock", "2026-01-01T00:00:00Z"),
                    )
                # Before the fix a discarded or finished job kept claiming its
                # dataset forever, so no later job could ever acquire it.
                JobLifecycle(database).discard("job-discard", confirmed=True)
                self.assertEqual(
                    ["job-cancelled", "job-done", "job-failed", "job-partial"],
                    [
                        str(row["job_id"])
                        for row in database.connection.execute("SELECT job_id FROM dataset_claims ORDER BY job_id")
                    ],
                )
                self.assertEqual(2, database.clear_stale_dataset_claims())
                self.assertEqual(
                    ["job-cancelled", "job-failed"],
                    [
                        str(row["job_id"])
                        for row in database.connection.execute("SELECT job_id FROM dataset_claims ORDER BY job_id")
                    ],
                )
            finally:
                database.close()

    def test_discard_releases_the_live_dataset_handle_for_immediate_reacquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            try:
                first_job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(first_job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    database.set_job_status(first_job_id, "running", current_module_id="dropout")
                    database.set_job_status(first_job_id, "failed", current_module_id="dropout")
                    JobLifecycle(database).discard(first_job_id, confirmed=True)
                finally:
                    database.close()
                self.assertTrue(preparation.release_lock_for_discard(first_job_id))
                self.assertFalse(preparation.release_lock_for_discard(first_job_id))

                second_job_id = preparation.preflight(config.to_dict()).jobId
                result = preparation.confirm_workspace(second_job_id, confirmed=True, confirmed_rebuild=False)
                self.assertEqual("preparing_workspace", result["status"])
            finally:
                preparation.close()

    def test_history_deletion_retries_a_live_lock_after_file_contention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            database_path = root / "state.db"
            preparation = JobPreparationService(database_path)
            try:
                job_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(database_path)
                try:
                    database.set_job_status(job_id, "running", current_module_id="dropout")
                    database.set_job_status(job_id, "exporting", current_module_id="export")
                    database.set_job_status(job_id, "committing", current_module_id="export")
                    database.set_job_status(job_id, "succeeded", current_module_id="export")
                    lock_path = Path(str(database.connection.execute(
                        "SELECT lock_path FROM dataset_claims WHERE job_id=?", (job_id,)
                    ).fetchone()["lock_path"]))
                finally:
                    database.close()

                with patch.object(Path, "unlink", side_effect=PermissionError("injected live lock contention")):
                    with self.assertRaisesRegex(DatasetLockError, "cannot remove dataset lock"):
                        preparation.release_lock_for_history_deletion(job_id)

                database = StateDatabase.open(database_path)
                try:
                    self.assertEqual("succeeded", database.get_job(job_id)["status"])
                    self.assertEqual(
                        1,
                        database.connection.execute(
                            "SELECT COUNT(*) FROM dataset_claims WHERE job_id=?", (job_id,)
                        ).fetchone()[0],
                    )
                finally:
                    database.close()
                self.assertTrue(lock_path.is_file())

                self.assertTrue(preparation.release_lock_for_history_deletion(job_id))
                self.assertFalse(preparation.release_lock_for_history_deletion(job_id))
                self.assertFalse(lock_path.exists())
                database = StateDatabase.open(database_path)
                try:
                    self.assertEqual(
                        0,
                        database.connection.execute(
                            "SELECT COUNT(*) FROM dataset_claims WHERE job_id=?", (job_id,)
                        ).fetchone()[0],
                    )
                    JobLifecycle(database).delete_history(job_id, confirmed=True)
                    with self.assertRaises(KeyError):
                        database.get_job(job_id)
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_retention_keeps_newest_unpinned_and_never_touches_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            logs = root / "logs"
            logs.mkdir()
            backup = root / "annotation-backups" / "original.zip"
            backup.parent.mkdir()
            backup.write_bytes(b"permanent")
            database = StateDatabase.open(root / "state.db")
            try:
                for index in range(22):
                    job_id = f"job-{index:02d}"
                    database.insert_job(_job(job_id, dataset, status="succeeded_with_issues" if index == 1 else "succeeded"))
                    database.connection.execute(
                        "UPDATE jobs SET finished_at=? WHERE job_id=?", (f"2026-01-01T00:00:{index:02d}Z", job_id)
                    )
                    (logs / f"{job_id}.log").write_text("ordinary log", encoding="utf-8")
                database.set_pinned("job-00", True)
                manager = RetentionManager(database, logs_root=logs)
                candidates = manager.candidates()
                self.assertEqual({"job-01"}, {candidate.jobId for candidate in candidates})
                result = manager.cleanup()
                self.assertEqual({"job-01"}, set(result.deletedJobIds))
                self.assertTrue(backup.exists())
                self.assertEqual("succeeded", database.get_job("job-00")["status"])
            finally:
                database.close()

    def test_retention_protects_unresolved_overlay_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            layout = OverlayLayout.create(dataset, "job-1")
            layout.write_journal({"state": "prepared"})
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job("job-1", dataset, status="succeeded", overlay_root=str(layout.root)))
                database.connection.execute("UPDATE jobs SET commit_journal_path=? WHERE job_id=?", (str(layout.commit_journal_path()), "job-1"))
                manager = RetentionManager(database, keep_successes=0)
                self.assertEqual([], manager.candidates())
                layout.write_journal({"state": "rolled_back"})
                self.assertEqual(["job-1"], [candidate.jobId for candidate in manager.candidates()])
                manager.cleanup()
                with self.assertRaises(KeyError):
                    database.get_job("job-1")
            finally:
                database.close()


if __name__ == "__main__":
    unittest.main()
