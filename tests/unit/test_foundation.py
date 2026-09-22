from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from PIL import Image

import sqlite3

from anima_core.contracts import JobConfig, ProgressEvent, SampleIssue, SampleRecord
from anima_core.db import (
    SCHEMA_SQL,
    SCHEMA_VERSION,
    StateDatabase,
    assert_database_outside_datasets,
    default_state_database_path,
)
from anima_core.db_schema import FINISHED_JOB_STATUSES, NON_INTERRUPTIBLE_JOB_STATUSES, TERMINAL_JOB_STATUSES
from anima_core.manifest import ManifestBuilder, ManifestError
from anima_core.overlay import BaselineView, OverlayError, OverlayLayout, WorkingAnnotationView
from anima_core.path_safety import (
    PathSafetyError,
    canonicalize,
    detect_annotation_collisions,
    atomic_write_bytes,
    replace_with_sharing_retry,
    read_annotation_state,
    safe_relative_path,
    sha256_file,
    windows_compare,
    windows_key,
)
from anima_core.state_machine import InvalidTransition, can_discard, decide_startup_recovery, transition_job, transition_module
from anima_core.workspace import prepare_dataset


def _job_row(job_id: str, dataset_root: Path) -> dict[str, object]:
    config = JobConfig(
        workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset_root), schemaVersion=9,
    )
    return {
        "job_id": job_id,
        "config_schema_version": config.schemaVersion,
        "config_json": json.dumps(config.to_dict(), sort_keys=True),
        "config_hash": config.config_hash,
        "profile": "e621",
        "work_mode": "in_place",
        "overwrite_mode": "incremental",
        "source_root": str(dataset_root),
        "output_root": None,
        "dataset_root": str(dataset_root),
        "dataset_root_key": windows_key(dataset_root),
        "manifest_schema_version": 1,
        "recursive": 0,
        "sample_count": 0,
        "manifest_generated_at": None,
        "status": "ready",
        "current_module_id": None,
        "last_event_id": 0,
        "pinned": 0,
        "api_budget_extra": 0,
        "api_budget_revision": 0,
        "overlay_root": None,
        "commit_journal_path": None,
        "resume_status": None,
        "created_at": "2026-07-23T00:00:00Z",
        "started_at": None,
        "cancel_requested_at": None,
        "finished_at": None,
    }


class FoundationTests(unittest.TestCase):
    def test_partial_success_is_terminal_finished_and_not_recoverable(self) -> None:
        self.assertIn("succeeded_with_issues", TERMINAL_JOB_STATUSES)
        self.assertIn("succeeded_with_issues", NON_INTERRUPTIBLE_JOB_STATUSES)
        self.assertIn("succeeded_with_issues", FINISHED_JOB_STATUSES)
        self.assertEqual("succeeded_with_issues", transition_job("committing", "succeeded_with_issues"))
        self.assertEqual(
            ("succeeded_with_issues", False),
            (decide_startup_recovery("succeeded_with_issues", None).nextStatus,
             decide_startup_recovery("succeeded_with_issues", None).requiresUserConfirmation),
        )

    def test_atomic_write_retries_transient_windows_file_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sharing_error = PermissionError(32, "sharing violation")
            sharing_error.winerror = 32
            with patch(
                "anima_core.path_safety.os.replace",
                side_effect=[sharing_error, None],
            ) as replace:
                atomic_write_bytes(temporary, "sample.txt", b"ok")
            self.assertEqual(2, replace.call_count)

    def test_sharing_retry_is_windows_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tmp"
            destination = root / "destination.txt"
            source.write_bytes(b"payload")
            sharing_error = PermissionError(32, "sharing violation")
            sharing_error.winerror = 32
            with patch("anima_core.path_safety.os.name", "posix"), patch(
                "anima_core.path_safety.os.replace",
                side_effect=[sharing_error, None],
            ) as replace:
                with self.assertRaises(PermissionError):
                    replace_with_sharing_retry(source, destination)
            self.assertEqual(1, replace.call_count)

    def test_sharing_retry_does_not_retry_non_sharing_permission_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tmp"
            destination = root / "destination.txt"
            source.write_bytes(b"payload")
            access_error = PermissionError(5, "access denied")
            access_error.winerror = 5
            with patch(
                "anima_core.path_safety.os.replace",
                side_effect=access_error,
            ) as replace:
                with self.assertRaises(PermissionError):
                    replace_with_sharing_retry(source, destination)
            self.assertEqual(1, replace.call_count)

    def test_sharing_retry_raises_after_five_windows_sharing_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tmp"
            destination = root / "destination.txt"
            source.write_bytes(b"payload")
            sharing_error = PermissionError(32, "sharing violation")
            sharing_error.winerror = 32
            with patch(
                "anima_core.path_safety.os.replace",
                side_effect=[sharing_error] * 5,
            ) as replace:
                with self.assertRaises(PermissionError):
                    replace_with_sharing_retry(source, destination)
            self.assertEqual(5, replace.call_count)

    def test_state_database_connection_can_be_used_by_api_worker_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = StateDatabase.open(Path(temporary) / "state.db")
            try:
                errors: list[BaseException] = []

                def read_from_worker() -> None:
                    try:
                        database.connection.execute("SELECT 1").fetchone()
                    except BaseException as exc:  # pragma: no cover - assertion below reports the failure
                        errors.append(exc)

                worker = threading.Thread(target=read_from_worker)
                worker.start()
                worker.join()
                self.assertEqual([], errors)
            finally:
                database.close()

    def test_fresh_state_database_uses_schema4_without_a_job_profile_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = StateDatabase.open(Path(temporary) / "state.db")
            try:
                self.assertEqual(4, int(database.connection.execute("PRAGMA user_version").fetchone()[0]))
                columns = {
                    str(row["name"])
                    for row in database.connection.execute("PRAGMA table_info(jobs)")
                }
                self.assertNotIn("profile", columns)
            finally:
                database.close()

    def test_existing_schema1_to3_is_rejected_without_writing(self) -> None:
        for version in (1, 2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "state.db"
                connection = sqlite3.connect(path)
                connection.execute("CREATE TABLE legacy_marker(value TEXT NOT NULL)")
                connection.execute("INSERT INTO legacy_marker(value) VALUES ('keep')")
                connection.execute(f"PRAGMA user_version = {version}")
                connection.commit()
                connection.close()
                before = sha256_file(path)
                try:
                    database = StateDatabase.open(path)
                except RuntimeError as exc:
                    self.assertRegex(str(exc), "incompatible|reinitialize|重新初始化")
                else:
                    database.close()
                    self.fail("legacy state database was accepted")
                self.assertEqual(before, sha256_file(path))

    def test_dropout_supports_running_and_configured_skip(self) -> None:
        self.assertEqual("running", transition_module("pending", "running", module_id="dropout"))
        self.assertEqual("skipped", transition_module("pending", "skipped", module_id="dropout"))
        self.assertEqual("skipped_not_available", transition_module("pending", "skipped_not_available", module_id="dropout"))
        with self.assertRaises(InvalidTransition):
            transition_module("pending", "skipped_not_available", module_id="caption")
        self.assertFalse(can_discard("failed", journal_state="rollback_created"))
        self.assertTrue(can_discard("failed", journal_state="rolled_back"))

    def test_path_safety_rejects_escape_and_preserves_blank_definition(self) -> None:
        with self.assertRaises(PathSafetyError):
            safe_relative_path("..\\caption.txt")
        with self.assertRaises(PathSafetyError):
            safe_relative_path("C:\\caption.txt")
        with self.assertRaises(PathSafetyError):
            safe_relative_path("nested\\NUL.txt")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blank = root / "blank.txt"
            blank.write_bytes(b"\xef\xbb\xbf \t\r\n")
            self.assertEqual("missing_or_blank", read_annotation_state(blank))
            invalid = root / "invalid.txt"
            invalid.write_bytes(b"\xff")
            with self.assertRaises(PathSafetyError):
                read_annotation_state(invalid)
            with patch.object(Path, "exists", side_effect=AssertionError("exists() must not inspect ancestors")):
                canonicalize(root / "not-created-yet")
        with self.assertRaises(PathSafetyError):
            detect_annotation_collisions(["Folder\\Image.png", "folder\\image.jpg"])

    def test_database_schema_keyset_issue_upsert_and_leases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-1", root))
                rows = [
                    {
                        "sample_id": index,
                        "relative_image_path": f"{index}.png",
                        "annotation_key": str(index),
                        "source": "e621",
                        "in_processing_scope": True,
                        "image_format": "png",
                        "image_frame_count": 1,
                        "original_txt_state": "missing_or_blank",
                        "original_json_state": "missing_or_blank",
                    }
                    for index in range(1, 4)
                ]
                database.insert_samples("job-1", rows)
                self.assertEqual([1, 2], [row["sample_id"] for row in database.page_samples("job-1", limit=2)])
                database.set_job_status("job-1", "running", current_module_id="caption")
                leases = database.claim_leases(
                    "job-1", "caption", "worker-a", database.get_job("job-1")["config_hash"], limit=2,
                    max_in_flight=64, single_worker=True,
                    lease_id_factory=iter(["lease-1", "lease-2"]).__next__,
                    expires_at="2030-01-01T00:00:00Z",
                )
                self.assertEqual(["lease-1", "lease-2"], [row["lease_id"] for row in leases])
                self.assertEqual(2, database.heartbeat("job-1", "worker-a", "2030-01-01T00:01:00Z", lease_ids=["lease-1", "lease-2"]))
                issue = SampleIssue(
                    issueId="issue-z", jobId="job-1", sampleId=1, relativeImagePath="1.png", moduleId="caption",
                    code="decode_failed", severity="error", blocking=False, retriable=True, message="first", attempt=1,
                )
                database.upsert_issue(issue)
                database.upsert_issue(SampleIssue(**{**issue.__dict__, "message": "second", "attempt": 2}))
                database.upsert_issue(SampleIssue(
                    issueId="issue-a", jobId="job-1", sampleId=2, relativeImagePath="2.png", moduleId="caption",
                    code="decode_failed", severity="error", blocking=False, retriable=True, message="third", attempt=1,
                ))
                first_page = database.page_issues("job-1", limit=1)
                second_page = database.page_issues("job-1", after_sample_id=1, limit=1)
                self.assertEqual([1], [row["sample_id"] for row in first_page])
                self.assertEqual([2], [row["sample_id"] for row in second_page])
                self.assertEqual("second", first_page[0]["message"])
                self.assertEqual(2, first_page[0]["attempt"])
                database.insert_job(_job_row("repair-1", root))
                database.insert_samples("repair-1", rows)
                database.create_repair_link("repair-1", "job-1")
                database.upsert_issue(SampleIssue(
                    issueId="issue-nl", jobId="job-1", sampleId=1, relativeImagePath="1.png", moduleId="nl",
                    code="nl_retry", severity="warning", blocking=False, retriable=True, message="retry nl", attempt=1,
                    repairStartModule="nl",
                ))
                database.upsert_issue(SampleIssue(
                    issueId="issue-classify", jobId="job-1", sampleId=1, relativeImagePath="1.png", moduleId="classify",
                    code="json_retry", severity="error", blocking=True, retriable=True, message="retry classify", attempt=1,
                    repairStartModule="classify",
                ))
                database.upsert_issue(SampleIssue(
                    issueId="issue-info", jobId="job-1", sampleId=3, relativeImagePath="3.png", moduleId="replace",
                    code="not_retry", severity="info", blocking=False, retriable=False, message="info", attempt=1,
                ))
                self.assertEqual([1], database.stage_repair_target_page("repair-1", "job-1", after_sample_id=None, limit=10))
                target = database.connection.execute("SELECT * FROM repair_targets WHERE repair_job_id='repair-1'").fetchone()
                linked = database.connection.execute("SELECT parent_issue_id FROM repair_target_issues WHERE repair_job_id='repair-1' ORDER BY parent_issue_id").fetchall()
                self.assertEqual((1, "classify"), (target["sample_id"], target["repair_start_module"]))
                self.assertEqual(["issue-classify", "issue-nl"], [row["parent_issue_id"] for row in linked])
                self.assertEqual(SCHEMA_VERSION, database.connection.execute("PRAGMA user_version").fetchone()[0])
            finally:
                database.close()

    @staticmethod
    def _write_v1_database(path: Path, job_row: dict[str, object]) -> None:
        """Build a version 1 database: no Export/repair tables, no resume_status."""
        connection = sqlite3.connect(path)
        try:
            connection.create_collation("WIN_ORDINAL_NOCASE", lambda left, right: windows_compare(left, right))
            connection.executescript(SCHEMA_SQL)
            for table in ("repair_target_issues", "repair_targets", "repair_jobs", "export_artifacts"):
                connection.execute(f"DROP TABLE {table}")
            connection.execute("ALTER TABLE jobs DROP COLUMN resume_status")
            columns = [name for name in job_row if name not in {"profile", "resume_status"}]
            connection.execute(
                f"INSERT INTO jobs({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                [job_row[name] for name in columns],
            )
            connection.execute("DELETE FROM schema_migrations")
            connection.execute(
                "INSERT INTO schema_migrations(version,checksum,applied_at) VALUES (1,'v1','2026-01-01T00:00:00Z')"
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
        finally:
            connection.close()

    @unittest.skipIf(sqlite3.sqlite_version_info < (3, 35), "ALTER TABLE DROP COLUMN requires SQLite 3.35")
    def test_schema_v1_is_rejected_without_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "state.db"
            self._write_v1_database(target, _job_row("job-v1", root))
            before = sha256_file(target)
            with self.assertRaisesRegex(RuntimeError, "incompatible|reinitialize"):
                StateDatabase.open(target)
            self.assertEqual(before, sha256_file(target))

    def test_schema_v2_is_rejected_without_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "state.db"
            database = StateDatabase.open(target)
            database.close()
            connection = sqlite3.connect(target)
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
            connection.close()
            before = sha256_file(target)
            with self.assertRaisesRegex(RuntimeError, "incompatible|reinitialize"):
                StateDatabase.open(target)
            self.assertEqual(before, sha256_file(target))

    def test_schema_open_rejects_a_newer_database_without_writing_to_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "state.db"
            connection = sqlite3.connect(target)
            try:
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(RuntimeError, "unsupported database schema version"):
                StateDatabase.open(target)
            connection = sqlite3.connect(target)
            try:
                self.assertEqual(
                    [],
                    list(connection.execute("SELECT name FROM sqlite_master WHERE type='table'")),
                )
            finally:
                connection.close()

    def test_issue_paging_keeps_a_multi_issue_sample_visible_across_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-issues", root))
                database.insert_samples("job-issues", [
                    {
                        "sample_id": index, "relative_image_path": f"{index}.png", "annotation_key": str(index),
                        "source": "e621", "in_processing_scope": True, "image_format": "png", "image_frame_count": 1,
                        "original_txt_state": "missing_or_blank", "original_json_state": "missing_or_blank",
                    }
                    for index in (1, 2)
                ])
                for issue_id, sample_id, code in (
                    ("issue-a", 1, "code_a"), ("issue-b", 1, "code_b"), ("issue-c", 2, "code_c"),
                ):
                    database.upsert_issue(SampleIssue(
                        issueId=issue_id, jobId="job-issues", sampleId=sample_id,
                        relativeImagePath=f"{sample_id}.png", moduleId="classify", code=code,
                        severity="error", blocking=True, retriable=False, message=code, attempt=1,
                    ))
                first = database.page_issues("job-issues", limit=1)
                self.assertEqual(["issue-a"], [row["issue_id"] for row in first])
                # The sample_id-only cursor skipped issue-b permanently.
                self.assertEqual(
                    ["issue-c"],
                    [row["issue_id"] for row in database.page_issues("job-issues", after_sample_id=1, limit=10)],
                )
                self.assertEqual(
                    ["issue-b", "issue-c"],
                    [
                        row["issue_id"]
                        for row in database.page_issues(
                            "job-issues", after_sample_id=1, after_issue_id="issue-a", limit=10,
                        )
                    ],
                )
                self.assertEqual(3, database.count_unresolved_blocking_issues("job-issues"))
                self.assertEqual(0, database.count_unresolved_blocking_issues("job-issues", exclude_module_id="classify"))
            finally:
                database.close()

    def test_failed_sample_issue_is_retained_but_does_not_block_export_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-failed-gate", root))
                database.insert_samples("job-failed-gate", [{
                    "sample_id": 1, "relative_image_path": "1.png", "annotation_key": "1",
                    "source": "e621", "in_processing_scope": True, "image_format": "png", "image_frame_count": 1,
                    "original_txt_state": "missing_or_blank", "original_json_state": "missing_or_blank",
                }])
                database.upsert_issue(SampleIssue(
                    issueId="failed-issue", jobId="job-failed-gate", sampleId=1,
                    relativeImagePath="1.png", moduleId="caption", code="caption_failed",
                    severity="error", blocking=True, retriable=False, message="caption failed", attempt=1,
                ))
                database.connection.execute(
                    "UPDATE sample_state SET current_module_id='caption',status='failed' WHERE job_id=? AND sample_id=1",
                    ("job-failed-gate",),
                )
                self.assertEqual(0, database.count_unresolved_blocking_issues("job-failed-gate"))
                self.assertEqual(0, database.count_unresolved_blocking_issues("job-failed-gate", exclude_module_id="export"))
                self.assertEqual(1, database.connection.execute(
                    "SELECT COUNT(*) FROM issues WHERE job_id=? AND resolved_at IS NULL", ("job-failed-gate",)
                ).fetchone()[0])
            finally:
                database.close()

    def test_manifest_excludes_generated_unlabeled_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            Image.new("RGB", (2, 2), "white").save(source / "source.png")
            generated = source / "_unlabeled" / "job-token" / "nested"
            generated.mkdir(parents=True)
            Image.new("RGB", (2, 2), "black").save(generated / "copy.png")
            records = list(ManifestBuilder(source, recursive=True).iter_records())
            self.assertEqual(["source.png"], [record.relativeImagePath for record in records])

    def test_preflight_projection_counts_split_create_overwrite_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-projection", root))
                database.insert_samples("job-projection", [
                    {
                        "sample_id": index, "relative_image_path": f"{index}.png", "annotation_key": str(index),
                        "source": "e621", "in_processing_scope": index != 3, "image_format": "png",
                        "image_frame_count": 1,
                        "original_txt_state": "nonblank" if index == 1 else "missing_or_blank",
                        "original_json_state": "nonblank" if index == 2 else "missing_or_blank",
                    }
                    for index in (1, 2, 3)
                ])
                projection = database.preflight_projection_counts("job-projection")
                self.assertEqual("both", projection["format"])
                self.assertEqual((2, 1), (projection["inScopeSamples"], projection["retainedSamples"]))
                self.assertEqual((1, 1, 0), (projection["jsonCreate"], projection["jsonOverwrite"], projection["jsonDelete"]))
                self.assertEqual((1, 1, 0), (projection["txtCreate"], projection["txtOverwrite"], projection["txtDelete"]))
                config = json.loads(str(database.get_job("job-projection")["config_json"]))
                config["export"]["format"] = "json"
                database.connection.execute(
                    "UPDATE jobs SET config_json=? WHERE job_id=?",
                    (json.dumps(config), "job-projection"),
                )
                json_only = database.preflight_projection_counts("job-projection")
                self.assertEqual((1, 1, 0), (json_only["jsonCreate"], json_only["jsonOverwrite"], json_only["jsonDelete"]))
                self.assertEqual((0, 0, 1), (json_only["txtCreate"], json_only["txtOverwrite"], json_only["txtDelete"]))
            finally:
                database.close()

    def test_job_status_transitions_are_validated_and_timestamped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-status", root))
                with self.assertRaises(InvalidTransition):
                    # A plain UPDATE used to accept any impossible state.
                    database.set_job_status("job-status", "succeeded")
                database.set_job_status("job-status", "preparing_workspace", current_module_id="workspace")
                started = database.get_job("job-status")["started_at"]
                self.assertIsNotNone(started)
                database.set_job_status("job-status", "running", current_module_id="caption")
                job = database.get_job("job-status")
                self.assertEqual(started, job["started_at"])
                self.assertIsNone(job["cancel_requested_at"])
                self.assertIsNone(job["finished_at"])
                database.set_job_status("job-status", "cancelling")
                self.assertIsNotNone(database.get_job("job-status")["cancel_requested_at"])
                database.set_job_status("job-status", "failed", current_module_id="caption")
                self.assertIsNotNone(database.get_job("job-status")["finished_at"])
            finally:
                database.close()

    def test_startup_recovery_freezes_only_jobs_that_still_need_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = StateDatabase.open(root / "state.db")
            try:
                for job_id, status in (
                    ("job-run", "running"), ("job-cancel", "cancelling"), ("job-done", "succeeded"),
                ):
                    row = _job_row(job_id, root)
                    row["status"] = status
                    database.insert_job(row)
                self.assertEqual(
                    ["job-run"], [str(row["job_id"]) for row in database.page_active_jobs()]
                )
                for job_id in ("job-run", "job-cancel", "job-done"):
                    database.mark_interrupted(job_id)
                self.assertEqual("interrupted", database.get_job("job-run")["status"])
                self.assertEqual("running", database.get_job("job-run")["resume_status"])
                self.assertEqual("cancelling", database.get_job("job-cancel")["status"])
                self.assertEqual("succeeded", database.get_job("job-done")["status"])
                # A second startup pass must not overwrite the resume target.
                database.mark_interrupted("job-run")
                self.assertEqual("running", database.get_job("job-run")["resume_status"])
                self.assertEqual(
                    "cancelling", decide_startup_recovery("cancelling", None).nextStatus
                )
                self.assertFalse(decide_startup_recovery("cancelled_recoverable", None).requiresUserConfirmation)
                pre_swap = decide_startup_recovery("committing", "staged")
                self.assertEqual(("interrupted", True), (
                    pre_swap.nextStatus, pre_swap.requiresUserConfirmation,
                ))
                post_swap = decide_startup_recovery("committing", "rollback_created")
                self.assertEqual(("committing", False), (
                    post_swap.nextStatus, post_swap.requiresUserConfirmation,
                ))
            finally:
                database.close()

    def test_database_location_wal_and_event_ring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            with self.assertRaises(ValueError):
                assert_database_outside_datasets(dataset / "state.db", [dataset])
            expected = default_state_database_path(local_app_data=root / "local")
            self.assertEqual(root / "local" / "AnimaDatasetTool" / "state.db", expected)
            database = StateDatabase.open(expected)
            try:
                self.assertEqual("wal", database.connection.execute("PRAGMA journal_mode").fetchone()[0].lower())
                database.insert_job(_job_row("job-events", dataset))
                rows = [
                    ("job-events", index, "caption", "running", index, 10_001, None, None, 1, "hash", "2026-01-01T00:00:00Z", None)
                    for index in range(1, 10_001)
                ]
                with database.transaction():
                    database.connection.executemany(
                        "INSERT INTO event_ring(job_id,event_id,module_id,status,completed,total,sample_id,issue_code,attempt,config_hash,occurred_at,message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        rows,
                    )
                config_hash = database.get_job("job-events")["config_hash"]
                database.append_event(ProgressEvent("job-events", 10_001, "caption", "running", 10_001, 10_001, config_hash, 1))
                with self.assertRaises(ValueError):
                    database.append_event(ProgressEvent("job-events", 10_000, "caption", "running", 10_000, 10_001, config_hash, 1))
                self.assertEqual(10_000, database.count("event_ring", "job-events"))
                self.assertTrue(database.event_snapshot_required("job-events", 0))
                self.assertFalse(database.event_snapshot_required("job-events", 1))
                first_page = database.event_page("job-events", 1, limit=1000)
                second_page = database.event_page("job-events", 1001, limit=1000)
                self.assertEqual(1000, len(first_page))
                self.assertEqual(2, first_page[0]["event_id"])
                self.assertEqual(1000, len(second_page))
            finally:
                database.close()

    def test_overlay_isolated_views_prepared_commit_and_journal_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "image.json").write_text('{"nl":"baseline"}', encoding="utf-8")
            layout = OverlayLayout.create(dataset, "job-1")
            try:
                baseline = BaselineView(dataset)
                working = WorkingAnnotationView(baseline, layout)
                self.assertEqual(b'{"nl":"baseline"}', working.read("image", ".json"))
                layout.write_annotation("image", ".json", b'{"nl":"overlay"}')
                self.assertEqual(b'{"nl":"overlay"}', working.read("image", ".json"))
                prepared, digest = layout.write_prepared("caption", "lease-1", ".txt", b"caption")
                self.assertTrue(prepared.is_file())
                layout.commit_prepared(os.path.relpath(prepared, layout.root), digest, "image", ".txt")
                self.assertEqual(b"caption", layout.annotation_path("image", ".txt").read_bytes())
                layout.write_journal({"state": "rollback_created"})
                with self.assertRaises(OverlayError):
                    layout.discard()
                layout.write_journal({"state": "rolled_back"})
                layout.discard()
                self.assertFalse(layout.root.exists())
            finally:
                if layout.root.exists():
                    # Only test cleanup; production discard has the journal guard above.
                    import shutil
                    shutil.rmtree(layout.root)

    def test_manifest_scan_into_decodes_once_and_workspace_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (2, 2), "white").save(source / "one.png")
            nested = source / "nested"
            nested.mkdir()
            Image.new("RGB", (2, 2), "black").save(nested / "two.png")
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-1", source))
                manifest = ManifestBuilder(source, recursive=False).scan_into(database, "job-1")
                self.assertEqual(2, manifest.sampleCount)
                records = database.page_samples("job-1", limit=10)
                self.assertEqual([1, 2], [row["sample_id"] for row in records])
                self.assertEqual([1, 0], [row["in_processing_scope"] for row in records])
            finally:
                database.close()
            in_place = prepare_dataset(source, None, "in_place", "job-1")
            self.assertFalse(in_place.copied)
            destination = root / "destination"
            copied = prepare_dataset(source, destination, "full_copy", "job-2")
            self.assertTrue(copied.copied)
            self.assertTrue((destination / "nested" / "two.png").is_file())

    def test_manifest_candidate_count_and_progress_count_images_once_including_bad_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (2, 2), "white").save(source / "one.png")
            Image.new("RGB", (2, 2), "black").save(source / "two.jpg")
            (source / "broken.webp").write_bytes(b"not an image")
            (source / "one.txt").write_text("tags", encoding="utf-8")
            (source / "one.json").write_text('{"nl":"description"}', encoding="utf-8")
            (source / "notes.txt").write_text("not a sample", encoding="utf-8")
            (source / "notes.json").write_text("{}", encoding="utf-8")
            (source / "unsupported.gif").write_bytes(b"not counted")
            database = StateDatabase.open(root / "state.db")
            progress: list[tuple[int, int]] = []
            try:
                database.insert_job(_job_row("job-progress", source))
                builder = ManifestBuilder(source, recursive=False)
                self.assertEqual(3, builder.count_candidate_images())
                manifest = builder.scan_into(
                    database,
                    "job-progress",
                    total_files=3,
                    on_progress=lambda checked, total: progress.append((checked, total)),
                )
                self.assertEqual(3, manifest.sampleCount)
                self.assertEqual([(1, 3), (2, 3), (3, 3)], progress)
                self.assertEqual(1, builder.image_issue_count)
            finally:
                database.close()

    def test_manifest_progress_rejects_an_added_candidate_before_callback_exceeds_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (2, 2), "white").save(source / "one.png")
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-progress-change", source))
                builder = ManifestBuilder(source, recursive=False)
                total = builder.count_candidate_images()
                Image.new("RGB", (2, 2), "black").save(source / "two.png")
                callbacks: list[tuple[int, int]] = []
                with self.assertRaisesRegex(ManifestError, "supported image set changed during scan"):
                    builder.scan_into(
                        database,
                        "job-progress-change",
                        total_files=total,
                        on_progress=lambda checked, expected: callbacks.append((checked, expected)),
                    )
                self.assertEqual([(1, 1)], callbacks)
            finally:
                database.close()

    def test_manifest_ignores_internal_directories_while_full_copy_preserves_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, include_cache in (("without-cache", False), ("with-cache", True)):
                with self.subTest(name=name):
                    source = root / name
                    source.mkdir()
                    Image.new("RGB", (2, 2), "white").save(source / "sample.png")
                    cache_payload = b'{"cache":"must remain unchanged"}\n'
                    if include_cache:
                        cache = source / ".mikazuki-cache"
                        cache.mkdir()
                        Image.new("RGB", (2, 2), "black").save(cache / "cached.png")
                        (cache / "cached.json").write_bytes(cache_payload)
                    metadata = source / ".anima-idg"
                    metadata.mkdir()
                    Image.new("RGB", (2, 2), "black").save(metadata / "internal.png")
                    metadata_payload = b"portable dataset metadata\n"
                    (metadata / "replace-provenance-v1.sqlite3").write_bytes(metadata_payload)
                    records = list(ManifestBuilder(source, recursive=True).iter_records())
                    self.assertEqual(["sample.png"], [record.relativeImagePath for record in records])
                    output = root / f"{name}-output"
                    result = prepare_dataset(source, output, "full_copy", f"copy-{name}")
                    self.assertTrue(result.copied)
                    self.assertEqual(
                        metadata_payload,
                        (output / ".anima-idg" / "replace-provenance-v1.sqlite3").read_bytes(),
                    )
                    if include_cache:
                        self.assertEqual(cache_payload, (output / ".mikazuki-cache" / "cached.json").read_bytes())

    def test_manifest_preflight_checks_formats_signatures_and_multiframe_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (3, 2), "white").save(source / "jpeg-image.jpg")
            Image.new("RGB", (3, 2), "white").save(source / "png-image.png")
            Image.new("RGB", (3, 2), "white").save(source / "webp-image.webp")
            Image.new("RGB", (3, 2), "white").save(source / "bmp-image.bmp")
            records = list(ManifestBuilder(source, recursive=False).iter_records())
            self.assertEqual({"jpeg", "png", "webp", "bmp"}, {record.imageFormat for record in records})
            self.assertEqual(4, len(records))

            # F36: a single unusable image is a per-sample defect, not a batch failure.
            (source / "not-an-image.png").write_bytes(b"not a PNG")
            first = Image.new("RGB", (3, 2), "white")
            second = Image.new("RGB", (3, 2), "black")
            first.save(source / "animated.webp", save_all=True, append_images=[second], duration=10, loop=0)
            Image.new("RGB", (3, 2), "white").save(source / "mislabelled.png")
            (source / "mislabelled.png").write_bytes((source / "jpeg-image.jpg").read_bytes())
            scanned = list(ManifestBuilder(source, recursive=False).iter_scan_records())
            self.assertEqual(7, len(scanned))
            self.assertEqual(
                {
                    "animated.webp": "image_multi_frame",
                    "mislabelled.png": "image_format_mismatch",
                    "not-an-image.png": "image_decode_failed",
                },
                {
                    record.relativeImagePath: defect.code
                    for record, defect in scanned if defect is not None
                },
            )
            self.assertEqual(
                [0, 0, 0],
                [record.imageFrameCount for record, defect in scanned if defect is not None],
            )

    def test_manifest_scan_failure_removes_partial_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (2, 2), "white").save(source / "z-valid.png")
            # An annotationKey collision is still a fatal, batch-level failure.
            Image.new("RGB", (2, 2), "white").save(source / "a-same.png")
            Image.new("RGB", (2, 2), "white").save(source / "a-same.jpg")
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-failed-scan", source))
                with self.assertRaises(ManifestError):
                    ManifestBuilder(source, recursive=False).scan_into(
                        database, "job-failed-scan", batch_size=1
                    )
                self.assertEqual(0, database.count("samples", "job-failed-scan"))
                self.assertEqual(0, database.count("sample_state", "job-failed-scan"))
                job = database.get_job("job-failed-scan")
                self.assertEqual(0, job["sample_count"])
                self.assertIsNone(job["manifest_generated_at"])
            finally:
                database.close()

    def test_dotted_image_names_keep_their_own_annotation_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (2, 2), "white").save(source / "d.va_overwatch.png")
            Image.new("RGB", (2, 2), "black").save(source / "d.png")
            # with_suffix("") + with_suffix(".txt") used to resolve both images
            # to "d.txt", so the dotted sample reported a foreign annotation.
            (source / "d.va_overwatch.txt").write_text("dotted caption", encoding="utf-8")
            records = {
                record.relativeImagePath: record
                for record in ManifestBuilder(source, recursive=False).iter_records()
            }
            self.assertEqual(
                {"d.png", "d.va_overwatch.png"}, set(records),
            )
            self.assertEqual("d.va_overwatch", records["d.va_overwatch.png"].annotationKey)
            self.assertEqual("nonblank", records["d.va_overwatch.png"].originalTxtState)
            self.assertEqual("missing_or_blank", records["d.png"].originalTxtState)
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-dotted", source))
                ManifestBuilder(source, recursive=False).scan_into(database, "job-dotted")
                stored = {
                    str(row["relative_image_path"]): row
                    for row in database.page_samples("job-dotted", limit=10)
                }
                self.assertEqual(
                    sha256_file(source / "d.va_overwatch.txt"),
                    stored["d.va_overwatch.png"]["original_txt_sha256"],
                )
                self.assertIsNone(stored["d.png"]["original_txt_sha256"])
            finally:
                database.close()

    def test_sample_ids_follow_ascending_file_order_inside_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            nested = source / "nested"
            nested.mkdir(parents=True)
            for name in ("c.png", "a.png", "b.png"):
                Image.new("RGB", (2, 2), "white").save(source / name)
                Image.new("RGB", (2, 2), "black").save(nested / name)
            records = list(ManifestBuilder(source, recursive=True).iter_records())
            # Files used to be handed out in descending order inside a directory.
            self.assertEqual(
                ["a.png", "b.png", "c.png", "nested\\a.png", "nested\\b.png", "nested\\c.png"],
                [record.relativeImagePath for record in records],
            )
            self.assertEqual([1, 2, 3, 4, 5, 6], [record.sampleId for record in records])

    def test_unusable_image_is_persisted_as_a_blocking_sample_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (2, 2), "white").save(source / "good.png")
            (source / "broken.png").write_bytes(b"not a PNG")
            database = StateDatabase.open(root / "state.db")
            try:
                database.insert_job(_job_row("job-broken", source))
                builder = ManifestBuilder(source, recursive=False)
                # The whole scan used to abort on the first unusable image.
                manifest = builder.scan_into(database, "job-broken", batch_size=1)
                self.assertEqual(2, manifest.sampleCount)
                self.assertEqual(1, builder.image_issue_count)
                self.assertEqual(2, database.count("samples", "job-broken"))
                issues = database.page_issues("job-broken", limit=10)
                self.assertEqual(1, len(issues))
                self.assertEqual(
                    ("broken.png", "workspace", "image_decode_failed", 1, 0),
                    (
                        str(issues[0]["relative_image_path"]), str(issues[0]["module_id"]),
                        str(issues[0]["code"]), int(issues[0]["blocking"]), int(issues[0]["retriable"]),
                    ),
                )
                self.assertEqual(1, database.count_unresolved_blocking_issues("job-broken"))
            finally:
                database.close()


if __name__ == "__main__":
    unittest.main()
