from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))

from anima_core.contracts import DEFAULT_MODULE_BATCH_SIZE, JobConfig, SampleRunState, pipeline_module_ids
from anima_core.db import StateDatabase
from anima_core.path_safety import windows_key
from anima_core.scheduler import BoundedScheduler, SchedulerError


def _job(
    job_id: str,
    root: Path,
    *,
    module_batch_size: dict[str, int] | None = None,
) -> dict[str, object]:
    config = JobConfig(
        workMode="in_place",
        overwriteMode="incremental",
        sourceRoot=str(root),
        moduleBatchSize={**DEFAULT_MODULE_BATCH_SIZE, **(module_batch_size or {})},
    )
    return {
        "job_id": job_id, "config_schema_version": 10, "config_json": json.dumps(config.to_dict()),
        "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental",
        "source_root": str(root), "output_root": None, "dataset_root": str(root), "dataset_root_key": windows_key(root),
        "manifest_schema_version": 1, "recursive": 0, "sample_count": 0, "manifest_generated_at": None,
        "status": "ready", "current_module_id": None, "last_event_id": 0, "pinned": 0,
        "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None,
        "resume_status": None, "created_at": "2026-07-23T00:00:00Z", "started_at": None,
        "cancel_requested_at": None, "finished_at": None,
    }


def _samples(count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": index, "relative_image_path": f"{index}.png", "annotation_key": str(index), "source": "e621",
            "in_processing_scope": True, "image_format": "png", "image_frame_count": 1,
            "original_txt_state": "missing_or_blank", "original_json_state": "missing_or_blank",
        }
        for index in range(1, count + 1)
    ]


class SchedulerTests(unittest.TestCase):
    def _database(
        self,
        root: Path,
        count: int = 70,
        *,
        module_batch_size: dict[str, int] | None = None,
    ) -> StateDatabase:
        database = StateDatabase.open(root / "state.db")
        database.insert_job(_job(
            "job-1",
            root,
            module_batch_size=module_batch_size,
        ))
        database.insert_samples("job-1", _samples(count))
        return database

    @staticmethod
    def _prime_predecessors(database: StateDatabase, target: str) -> None:
        order = pipeline_module_ids(int(database.get_job("job-1")["config_schema_version"]))
        predecessors = order[:order.index(target)]
        for module_id in predecessors:
            database.initialize_module_summary("job-1", module_id, total=0, status="completed")
        if predecessors:
            current = predecessors[-1]
            database.connection.execute(
                "UPDATE sample_state SET current_module_id=?,status='completed' WHERE job_id='job-1'",
                (current,),
            )
            database.set_job_status("job-1", "running", current_module_id=current)

    def test_caption_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), module_batch_size={"caption": 64})
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=(lambda values=iter(f"lease-{index}" for index in range(1, 200)): next(values)))
                self.assertEqual("running", scheduler.start_module("job-1", "caption", enabled=True, profile="e621"))
                config_hash = database.get_job("job-1")["config_hash"]
                leases = scheduler.claim_batch("job-1", "caption", "worker-1", config_hash)
                self.assertEqual(64, len(leases))
                self.assertEqual(64, len({lease.sampleId for lease in leases}))
                self.assertEqual([], scheduler.claim_batch("job-1", "caption", "worker-1", config_hash))
                with self.assertRaises(SchedulerError):
                    scheduler.claim_batch("job-1", "caption", "worker-2", config_hash)
                with self.assertRaises(SchedulerError):
                    scheduler.claim_batch("job-1", "caption", "worker-1", config_hash, limit=65)
                self.assertEqual(64, scheduler.heartbeat("job-1", "worker-1", [lease.leaseId for lease in leases if lease.leaseId]))
            finally:
                database.close()

    def test_module_order_and_dropout_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database)
                with self.assertRaises(SchedulerError):
                    scheduler.start_module("job-1", "classify", enabled=True, profile="e621")
                self.assertEqual("skipped", scheduler.start_module("job-1", "caption", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "classify", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "replace", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "ocr", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "nl", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "count_review", enabled=False, profile="e621"))
                self.assertEqual("running", scheduler.start_module("job-1", "dropout", enabled=True, profile="e621"))
                lease = scheduler.claim_batch(
                    "job-1", "dropout", "policy-worker", database.get_job("job-1")["config_hash"]
                )
                self.assertEqual(1, len(lease))
            finally:
                database.close()

    def test_prepaused_successor_does_not_block_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=lambda: "caption-lease")
                self.assertEqual("running", scheduler.start_module("job-1", "caption", enabled=True, profile="e621"))
                scheduler.pause_future_module("job-1", "nl", total=1)
                self.assertIsNone(database.module_summary("job-1", "nl")["started_at"])

                caption_lease = scheduler.claim_batch(
                    "job-1", "caption", "caption-worker", str(database.get_job("job-1")["config_hash"])
                )[0]
                scheduler.complete(caption_lease)
                self.assertEqual("completed", scheduler.finish_module("job-1", "caption"))

                self.assertEqual("skipped", scheduler.start_module("job-1", "classify", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "replace", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "ocr", enabled=False, profile="e621"))
                before = database.get_sample_state("job-1", 1)
                self.assertEqual(("caption", "completed", None), (
                    before["current_module_id"], before["status"], before["lease_id"],
                ))

                self.assertEqual("paused", scheduler.start_module("job-1", "nl", enabled=True, profile="e621"))
                job = database.get_job("job-1")
                self.assertEqual(("paused", "nl", "running"), (
                    job["status"], job["current_module_id"], job["resume_status"],
                ))
                after = database.get_sample_state("job-1", 1)
                self.assertEqual(("caption", "completed", None), (
                    after["current_module_id"], after["status"], after["lease_id"],
                ))
                self.assertEqual(
                    [],
                    scheduler.claim_batch("job-1", "nl", "nl-worker", str(job["config_hash"])),
                )
            finally:
                database.close()

    def test_cancel_future_pause_removes_only_a_zero_work_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database)
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                scheduler.pause_future_module("job-1", "nl", total=1)
                scheduler.cancel_future_pause("job-1", "nl")
                with self.assertRaises(KeyError):
                    database.module_summary("job-1", "nl")

                scheduler.pause_future_module("job-1", "nl", total=1)
                database.set_module_summary("job-1", "nl", status="paused", completed=1)
                with self.assertRaisesRegex(ValueError, "zero-work"):
                    scheduler.cancel_future_pause("job-1", "nl")
                self.assertEqual("paused", database.module_summary("job-1", "nl")["status"])
            finally:
                database.close()

    def test_nonpaused_successor_summary_still_blocks_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                database.initialize_module_summary("job-1", "nl", total=1)
                scheduler = BoundedScheduler(database)
                with self.assertRaisesRegex(SchedulerError, "cannot start after nl has been initialized"):
                    scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
            finally:
                database.close()

    def test_disabled_nl_records_not_requested_observations_in_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1001)
            try:
                scheduler = BoundedScheduler(database)
                self.assertEqual("skipped", scheduler.start_module("job-1", "caption", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "classify", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "replace", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "ocr", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("job-1", "nl", enabled=False, profile="e621"))
                rows = database.connection.execute(
                    "SELECT status,not_requested_reason FROM count_observations WHERE job_id='job-1' ORDER BY sample_id"
                ).fetchall()
                self.assertEqual(1001, len(rows))
                self.assertEqual({("not_requested", "nl_disabled")}, {(row["status"], row["not_requested_reason"]) for row in rows})
            finally:
                database.close()

    def test_dropout_queue_uses_frozen_batch_size_and_one_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=20, module_batch_size={"dropout": 6})
            try:
                self._prime_predecessors(database, "dropout")
                lease_ids = iter(f"policy-{index}" for index in range(1, 40))
                scheduler = BoundedScheduler(database, lease_id_factory=lease_ids.__next__)
                self.assertEqual("running", scheduler.start_module("job-1", "dropout", enabled=True, profile="e621"))
                config_hash = database.get_job("job-1")["config_hash"]
                self.assertEqual(6, len(scheduler.claim_batch("job-1", "dropout", "worker-1", config_hash)))
                self.assertEqual([], scheduler.claim_batch("job-1", "dropout", "worker-1", config_hash))
                with self.assertRaises(SchedulerError):
                    scheduler.claim_batch("job-1", "dropout", "worker-2", config_hash)
                with self.assertRaises(SchedulerError):
                    scheduler.claim_batch("job-1", "dropout", "worker-1", config_hash, limit=7)
            finally:
                database.close()

    def test_two_page_modules_enforce_total_resident_capacity(self) -> None:
        for module_id in ("classify", "replace", "export"):
            with self.subTest(module_id=module_id), tempfile.TemporaryDirectory() as temporary:
                database = self._database(Path(temporary), count=1_101, module_batch_size={
                    {"classify": "classify", "replace": "replace", "export": "export"}[module_id]: 500,
                })
                try:
                    self._prime_predecessors(database, module_id)
                    lease_ids = iter(f"{module_id}-{index}" for index in range(1, 1_500))
                    scheduler = BoundedScheduler(database, lease_id_factory=lease_ids.__next__)
                    self.assertEqual("running", scheduler.start_module("job-1", module_id, enabled=True, profile="e621"))
                    config_hash = database.get_job("job-1")["config_hash"]
                    first = scheduler.claim_batch("job-1", module_id, "worker-1", config_hash)
                    second = scheduler.claim_batch("job-1", module_id, "worker-2", config_hash)
                    third = scheduler.claim_batch("job-1", module_id, "worker-3", config_hash)
                    self.assertEqual((500, 500, 0), (len(first), len(second), len(third)))
                finally:
                    database.close()

    def test_nl_queue_uses_frozen_v10_module_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=40, module_batch_size={"nl": 3})
            try:
                self._prime_predecessors(database, "nl")
                lease_ids = iter(f"nl-{index}" for index in range(1, 100))
                scheduler = BoundedScheduler(database, lease_id_factory=lease_ids.__next__)
                scheduler.start_module("job-1", "nl", enabled=True, profile="e621")
                config_hash = database.get_job("job-1")["config_hash"]
                self.assertEqual(3, len(scheduler.claim_batch("job-1", "nl", "worker-1", config_hash)))
                self.assertEqual(3, len(scheduler.claim_batch("job-1", "nl", "worker-2", config_hash)))
                self.assertEqual([], scheduler.claim_batch("job-1", "nl", "worker-3", config_hash))
            finally:
                database.close()

    def test_worker_restart_budget_and_cancellation_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database)
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                self.assertTrue(scheduler.worker_exited("job-1", "caption", abnormal=True))
                self.assertTrue(scheduler.worker_exited("job-1", "caption", abnormal=True))
                self.assertFalse(scheduler.worker_exited("job-1", "caption", abnormal=True))
                self.assertEqual("running", database.get_job("job-1")["status"])
                self.assertEqual("completed_with_issues", database.module_summary("job-1", "caption")["status"])
                state = database.get_sample_state("job-1", 1)
                self.assertEqual(("caption", "failed"), (state["current_module_id"], state["status"]))
                issue = database.page_issues("job-1", limit=10)
                self.assertEqual(1, len(issue))
                self.assertEqual(("caption", "module_worker_unavailable", 0), (
                    issue[0]["module_id"], issue[0]["code"], issue[0]["retriable"],
                ))
                scheduler.begin_cancellation("job-1")
                self.assertEqual(
                    [],
                    scheduler.claim_batch(
                        "job-1", "caption", "worker-after-cancel", database.get_job("job-1")["config_hash"]
                    ),
                )
                scheduler.settle_cancellation("job-1")
                self.assertEqual("cancelled_recoverable", database.get_job("job-1")["status"])
            finally:
                database.close()

    def test_worker_exhaustion_preserves_completed_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=2)
            try:
                scheduler = BoundedScheduler(
                    database, lease_id_factory=iter(("lease-1", "lease-2")).__next__,
                )
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                leases = scheduler.claim_batch(
                    "job-1", "caption", "worker-1", database.get_job("job-1")["config_hash"],
                )
                scheduler.complete(leases[0])

                self.assertTrue(scheduler.worker_exited("job-1", "caption", abnormal=True))
                self.assertTrue(scheduler.worker_exited("job-1", "caption", abnormal=True))
                self.assertFalse(scheduler.worker_exited("job-1", "caption", abnormal=True))

                self.assertEqual("completed", database.get_sample_state("job-1", 1)["status"])
                self.assertEqual("failed", database.get_sample_state("job-1", 2)["status"])
                summary = database.module_summary("job-1", "caption")
                self.assertEqual((1, 1, 1), (
                    summary["completed"], summary["failed"], summary["issue_count"],
                ))
            finally:
                database.close()

    def test_worker_exit_after_cancellation_does_not_consume_restart_budget_or_fail_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database)
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                scheduler.begin_cancellation("job-1")

                self.assertFalse(scheduler.worker_exited("job-1", "caption", abnormal=True))
                self.assertEqual(0, database.module_summary("job-1", "caption")["worker_restart_count"])
                self.assertEqual("pending", database.get_sample_state("job-1", 1)["status"])
            finally:
                database.close()

    def test_begin_cancellation_preserves_the_normalized_resume_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                cases = (
                    ("preparing_workspace", None, "preparing_workspace"),
                    ("running", "caption", "running"),
                    ("paused", "caption", "running"),
                    ("reviewing", "count_review", "reviewing"),
                    ("exporting", "export", "exporting"),
                )
                for status, module_id, expected_resume_status in cases:
                    with self.subTest(status=status):
                        database.connection.execute(
                            "UPDATE jobs SET status=?,current_module_id=?,resume_status=NULL WHERE job_id='job-1'",
                            (status, module_id),
                        )
                        database.begin_cancellation("job-1")
                        self.assertEqual(
                            ("cancelling", expected_resume_status),
                            tuple(database.get_job("job-1")[key] for key in ("status", "resume_status")),
                        )
                        database.begin_cancellation("job-1")
                        self.assertEqual(expected_resume_status, database.get_job("job-1")["resume_status"])
            finally:
                database.close()

    def test_startup_interruption_preserves_a_paused_tasks_resume_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                for module_id, resume_status in (
                    ("caption", "running"),
                    ("count_review", "running"),
                    ("export", "exporting"),
                ):
                    with self.subTest(module_id=module_id):
                        database.connection.execute(
                            "UPDATE jobs SET status='paused',current_module_id=?,resume_status=? WHERE job_id='job-1'",
                            (module_id, resume_status),
                        )
                        database.mark_interrupted("job-1")
                        job = database.get_job("job-1")
                        self.assertEqual(("interrupted", resume_status), (job["status"], job["resume_status"]))
            finally:
                database.close()

    def test_atomic_pause_and_resume_reject_stale_module_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database)
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                database.set_module_summary("job-1", "caption", status="completed", finished=True)
                with self.assertRaisesRegex(ValueError, "state changed"):
                    database.pause_active_module("job-1", "caption", active_status="running")
                self.assertEqual("running", database.get_job("job-1")["status"])
                self.assertEqual("completed", database.module_summary("job-1", "caption")["status"])

                database.connection.execute(
                    "UPDATE module_summary SET status='paused' WHERE job_id='job-1' AND module_id='caption'"
                )
                database.connection.execute(
                    "UPDATE jobs SET status='paused',resume_status='running' WHERE job_id='job-1'"
                )
                database.begin_cancellation("job-1")
                with self.assertRaisesRegex(ValueError, "state changed"):
                    database.resume_paused_module("job-1", "caption", target_status="running")
                self.assertEqual("cancelling", database.get_job("job-1")["status"])
                self.assertEqual("paused", database.module_summary("job-1", "caption")["status"])
            finally:
                database.close()

    def test_active_pause_stamps_zero_work_summary_as_started(self) -> None:
        for module_id, active_status in (
            ("count_review", "running"),
            ("dropout", "running"),
            ("export", "exporting"),
        ):
            with self.subTest(module_id=module_id), tempfile.TemporaryDirectory() as temporary:
                database = self._database(Path(temporary), count=0)
                try:
                    database.initialize_module_summary(module_id=module_id, job_id="job-1", total=0, status="running")
                    database.connection.execute(
                        "UPDATE jobs SET status=?,current_module_id=? WHERE job_id='job-1'",
                        (active_status, module_id),
                    )
                    database.pause_active_module("job-1", module_id, active_status=active_status)
                    active = database.module_summary("job-1", module_id)
                    self.assertEqual("paused", active["status"])
                    self.assertIsNotNone(active["started_at"])
                finally:
                    database.close()

    def test_module_cannot_finish_until_all_work_is_settled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=lambda: "only-lease")
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                with self.assertRaises(SchedulerError):
                    scheduler.finish_module("job-1", "caption")
                config_hash = database.get_job("job-1")["config_hash"]
                lease = scheduler.claim_batch("job-1", "caption", "worker-1", config_hash)[0]
                scheduler.complete(lease)
                self.assertEqual("completed", scheduler.finish_module("job-1", "caption"))
            finally:
                database.close()

    def test_manual_recovery_only_returns_safe_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=3)
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=(lambda values=iter(["lease-a", "lease-b", "lease-c"]): next(values)))
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                config_hash = database.get_job("job-1")["config_hash"]
                first, second, third = scheduler.claim_batch("job-1", "caption", "worker-1", config_hash, limit=3)
                scheduler.stage_prepared(first, relative_path="prepared\\caption\\lease-a.txt", sha256="a" * 64)
                database.set_sample_state(
                    "job-1", third.sampleId,
                    SampleRunState(
                        sampleId=third.sampleId, currentModuleId="caption", status="request_started", attempt=1,
                        leaseId=third.leaseId, workerInstanceId="worker-1", leaseExpiresAt="2030-01-01T00:00:00Z",
                    ),
                )
                database.mark_interrupted("job-1")
                with self.assertRaises(SchedulerError):
                    scheduler.recover(
                        "job-1", confirmed=False, expected_config_hash=config_hash, manifest_schema_version=1,
                        protocol_version="1.0", verify_source_fingerprints=lambda: True, commit_prepared=lambda *_: True,
                    )
                report = scheduler.recover(
                    "job-1", confirmed=True, expected_config_hash=config_hash, manifest_schema_version=1,
                    protocol_version="1.0", verify_source_fingerprints=lambda: True, commit_prepared=lambda *_: True,
                )
                self.assertEqual(1, report.returnedLeases)
                self.assertEqual(1, report.committedPrepared)
                self.assertEqual(1, report.pendingApiDecisions)
                self.assertEqual("completed", database.get_sample_state("job-1", first.sampleId)["status"])
                self.assertEqual("pending", database.get_sample_state("job-1", second.sampleId)["status"])
                self.assertEqual("request_started", database.get_sample_state("job-1", third.sampleId)["status"])
            finally:
                database.close()

    def test_cancelled_recoverable_recovery_requires_confirmation_and_settles_safe_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=2)
            try:
                scheduler = BoundedScheduler(database, lease_id_factory=iter(("lease-prepared", "lease-returned")).__next__)
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                config_hash = str(database.get_job("job-1")["config_hash"])
                prepared, leased = scheduler.claim_batch("job-1", "caption", "worker-1", config_hash, limit=2)
                scheduler.stage_prepared(prepared, relative_path="prepared\\caption\\lease-prepared.txt", sha256="a" * 64)
                database.connection.execute(
                    "UPDATE jobs SET status='cancelled_recoverable',resume_status='running' WHERE job_id='job-1'"
                )
                committed: list[tuple[str, int, str, str]] = []

                def commit_prepared(job_id: str, sample_id: int, relative: str, digest: str) -> bool:
                    committed.append((job_id, sample_id, relative, digest))
                    return True

                with self.assertRaisesRegex(SchedulerError, "manual confirmation is required"):
                    scheduler.recover(
                        "job-1", confirmed=False, expected_config_hash=config_hash, manifest_schema_version=1,
                        protocol_version="1.0", verify_source_fingerprints=lambda: True, commit_prepared=commit_prepared,
                    )
                report = scheduler.recover(
                    "job-1", confirmed=True, expected_config_hash=config_hash, manifest_schema_version=1,
                    protocol_version="1.0", verify_source_fingerprints=lambda: True, commit_prepared=commit_prepared,
                )
                self.assertEqual((1, 1, 0), (report.returnedLeases, report.committedPrepared, report.repeatedPrepared))
                self.assertEqual(
                    [("job-1", prepared.sampleId, "prepared\\caption\\lease-prepared.txt", "a" * 64)],
                    committed,
                )
                self.assertEqual("completed", database.get_sample_state("job-1", prepared.sampleId)["status"])
                self.assertEqual("pending", database.get_sample_state("job-1", leased.sampleId)["status"])
            finally:
                database.close()

    def test_cancelled_recoverable_unknown_api_decision_can_transition_to_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=1)
            try:
                scheduler = BoundedScheduler(database)
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                scheduler.begin_cancellation("job-1")
                scheduler.settle_cancellation("job-1")
                database.set_job_status("job-1", "interrupted", current_module_id="nl", resume_status="running")
                database.set_sample_state(
                    "job-1", 1,
                    SampleRunState(
                        sampleId=1, currentModuleId="nl", status="request_started", attempt=1,
                        leaseId="unknown-api", workerInstanceId="nl-worker", leaseExpiresAt="2030-01-01T00:00:00Z",
                    ),
                )
                self.assertEqual(1, scheduler.confirm_nl_api_outcome_unknown("job-1", confirmed=True))
                self.assertEqual("pending", database.get_sample_state("job-1", 1)["status"])
                job = database.get_job("job-1")
                self.assertEqual(("interrupted", "nl", "running"), (
                    job["status"], job["current_module_id"], job["resume_status"],
                ))
            finally:
                database.close()

    def test_expired_leases_are_reclaimed_before_the_next_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=2)
            try:
                lease_ids = iter(f"expire-{index}" for index in range(1, 10))
                scheduler = BoundedScheduler(database, lease_id_factory=lease_ids.__next__)
                scheduler.start_module("job-1", "caption", enabled=True, profile="e621")
                config_hash = database.get_job("job-1")["config_hash"]
                self.assertEqual(2, len(scheduler.claim_batch("job-1", "caption", "dead-worker", config_hash)))
                database.connection.execute(
                    "UPDATE sample_state SET lease_expires_at='2000-01-01T00:00:00Z' WHERE job_id='job-1'"
                )
                # Without the reclaim the dead worker still owns both leases and
                # the replacement worker is rejected as a second caption worker.
                replacements = scheduler.claim_batch("job-1", "caption", "new-worker", config_hash)
                self.assertEqual({1, 2}, {lease.sampleId for lease in replacements})
                self.assertEqual(
                    "new-worker", database.get_sample_state("job-1", 1)["worker_instance_id"]
                )
            finally:
                database.close()

    def test_export_carries_samples_that_failed_in_earlier_modules_as_skips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=3)
            try:
                self._prime_predecessors(database, "export")
                database.connection.execute(
                    "UPDATE sample_state SET status='failed' WHERE job_id='job-1' AND sample_id=2"
                )
                lease_ids = iter(f"export-{index}" for index in range(1, 10))
                scheduler = BoundedScheduler(database, lease_id_factory=lease_ids.__next__)
                self.assertEqual("running", scheduler.start_module("job-1", "export", enabled=True, profile="e621"))
                summary = database.module_summary("job-1", "export")
                self.assertEqual((3, 1), (int(summary["total"]), int(summary["skipped"])))
                state = database.get_sample_state("job-1", 2)
                self.assertEqual(("export", "skipped"), (state["current_module_id"], state["status"]))
                config_hash = database.get_job("job-1")["config_hash"]
                self.assertEqual(
                    {1, 3},
                    {lease.sampleId for lease in scheduler.claim_batch("job-1", "export", "worker", config_hash)},
                )
                # The commit projection must stop demanding artifacts for a
                # sample Export intentionally skipped.
                self.assertEqual(
                    {1, 3},
                    {int(row["sample_id"]) for row in database.page_export_artifact_groups("job-1")},
                )
            finally:
                database.close()

    def test_repair_modules_only_lease_targets_until_full_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=3)
            try:
                database.insert_job(_job("repair-1", Path(temporary)))
                database.insert_samples("repair-1", _samples(3))
                database.create_repair_link("repair-1", "job-1")
                database.connection.executemany(
                    "INSERT INTO repair_targets(repair_job_id,sample_id,repair_start_module) VALUES (?,?,?)",
                    [("repair-1", 1, "classify"), ("repair-1", 2, "nl"), ("repair-1", 3, "dropout")],
                )
                scheduler = BoundedScheduler(database, lease_id_factory=iter(tuple(f"repair-lease-{index}" for index in range(1, 10))).__next__)
                self.assertEqual("skipped", scheduler.start_module("repair-1", "caption", enabled=False, profile="e621"))
                self.assertEqual(0, database.module_summary("repair-1", "caption")["total"])
                self.assertEqual("running", scheduler.start_module("repair-1", "classify", enabled=True, profile="e621"))
                self.assertEqual(1, database.module_summary("repair-1", "classify")["total"])
                config_hash = database.get_job("repair-1")["config_hash"]
                self.assertEqual([1], [lease.sampleId for lease in scheduler.claim_batch("repair-1", "classify", "worker", config_hash)])
                database.connection.execute("UPDATE sample_state SET current_module_id='classify',status='completed' WHERE job_id='repair-1' AND sample_id=1")
                database.set_module_summary("repair-1", "classify", status="completed", completed=1, finished=True)
                database.set_job_status("repair-1", "running", current_module_id="classify")
                self.assertEqual("skipped", scheduler.start_module("repair-1", "replace", enabled=False, profile="e621"))
                self.assertEqual("skipped", scheduler.start_module("repair-1", "ocr", enabled=False, profile="e621"))
                self.assertEqual("running", scheduler.start_module("repair-1", "nl", enabled=True, profile="e621"))
                self.assertEqual({1, 2}, {lease.sampleId for lease in scheduler.claim_batch("repair-1", "nl", "worker", config_hash)})
                database.connection.execute("UPDATE sample_state SET current_module_id='nl',status='completed' WHERE job_id='repair-1' AND sample_id IN (1,2)")
                database.set_module_summary("repair-1", "nl", status="completed", completed=2, finished=True)
                database.set_job_status("repair-1", "running", current_module_id="nl")
                self.assertEqual("skipped", scheduler.start_module("repair-1", "count_review", enabled=False, profile="e621"))
                self.assertEqual("running", scheduler.start_module("repair-1", "dropout", enabled=True, profile="e621"))
                self.assertEqual(3, database.module_summary("repair-1", "dropout")["total"])
                self.assertEqual({1, 2, 3}, {lease.sampleId for lease in scheduler.claim_batch("repair-1", "dropout", "worker", config_hash)})
                database.connection.execute("UPDATE sample_state SET current_module_id='dropout',status='completed' WHERE job_id='repair-1' AND sample_id IN (1,2,3)")
                database.set_module_summary("repair-1", "dropout", status="completed", completed=3, finished=True)
                database.set_job_status("repair-1", "running", current_module_id="dropout")
                self.assertEqual("skipped", scheduler.start_module("repair-1", "token_budget", enabled=False, profile="e621"))
                self.assertEqual("running", scheduler.start_module("repair-1", "export", enabled=True, profile="e621"))
                self.assertEqual(3, database.module_summary("repair-1", "export")["total"])
                self.assertEqual({1, 2, 3}, {lease.sampleId for lease in scheduler.claim_batch("repair-1", "export", "worker", config_hash)})
            finally:
                database.close()

    def test_v10_ocr_has_an_independent_single_worker_queue_and_disabled_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = self._database(Path(temporary), count=2, module_batch_size={"ocr": 1})
            try:
                lease_ids = iter(("ocr-lease-1", "ocr-lease-2"))
                scheduler = BoundedScheduler(database, lease_id_factory=lease_ids.__next__)
                for module_id in ("caption", "classify", "replace"):
                    self.assertEqual("skipped", scheduler.start_module("job-1", module_id, enabled=False, profile="e621"))

                self.assertEqual("running", scheduler.start_module("job-1", "ocr", enabled=True, profile="e621"))
                config_hash = str(database.get_job("job-1")["config_hash"])
                first = scheduler.claim_batch("job-1", "ocr", "ocr-worker-1", config_hash)
                self.assertEqual([1], [lease.sampleId for lease in first])
                with self.assertRaises(SchedulerError):
                    scheduler.claim_batch("job-1", "ocr", "ocr-worker-2", config_hash)
                with self.assertRaises(SchedulerError):
                    scheduler.claim_batch("job-1", "ocr", "ocr-worker-1", config_hash, limit=2)
                scheduler.complete(first[0])
                second = scheduler.claim_batch("job-1", "ocr", "ocr-worker-1", config_hash)
                self.assertEqual([2], [lease.sampleId for lease in second])
                scheduler.complete(second[0])
                self.assertEqual("completed", scheduler.finish_module("job-1", "ocr"))

                try:
                    nl_status = scheduler.start_module("job-1", "nl", enabled=False, profile="e621")
                except ValueError as exc:
                    self.fail(f"v5 disabled NL must be a normal skipped module: {exc}")
                self.assertEqual("skipped", nl_status)
                summary = database.module_summary("job-1", "nl")
                self.assertEqual(("skipped", 2), (summary["status"], int(summary["skipped"])))
            finally:
                database.close()

    def test_v6_places_token_budget_after_dropout_and_before_export(self) -> None:
        try:
            modules = pipeline_module_ids(6)
        except ValueError as exc:
            self.fail(f"JobConfig v6 route is missing: {exc}")
        self.assertEqual(
            ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"),
            modules,
        )


if __name__ == "__main__":
    unittest.main()
