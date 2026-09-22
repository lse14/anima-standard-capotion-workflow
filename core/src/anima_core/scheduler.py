from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from . import PROTOCOL_VERSION
from .contracts import (
    CURRENT_JOB_CONFIG_SCHEMA_VERSION,
    MODULE_BATCH_SIZE_BOUNDS,
    ModuleId,
    SampleIssue,
    SampleRunState,
    WorkLease,
    pipeline_module_ids,
)
from .db import MAX_PAGE_SIZE, StateDatabase
from .state_machine import transition_module


FINAL_MODULE_STATUSES = frozenset({"completed", "completed_with_issues", "skipped", "skipped_not_available"})
_MODULE_BATCH_CONFIG_KEYS = {
    "caption": "caption",
    "classify": "classify",
    "replace": "replace",
    "ocr": "ocr",
    "nl": "nl",
    "count_review": "countReview",
    "dropout": "dropout",
    "token_budget": "tokenBudget",
    "export": "export",
}


@dataclass(frozen=True)
class ModuleQueueLimits:
    lease_batch_size: int
    max_resident_pages: int
    max_prefetch: int | None = None


MODULE_QUEUE_LIMITS: dict[str, ModuleQueueLimits] = {
    "caption": ModuleQueueLimits(lease_batch_size=64, max_resident_pages=1, max_prefetch=64),
    "classify": ModuleQueueLimits(lease_batch_size=500, max_resident_pages=2),
    "replace": ModuleQueueLimits(lease_batch_size=500, max_resident_pages=2),
    "ocr": ModuleQueueLimits(lease_batch_size=1024, max_resident_pages=1),
    "nl": ModuleQueueLimits(lease_batch_size=16, max_resident_pages=1),
    "count_review": ModuleQueueLimits(lease_batch_size=500, max_resident_pages=1),
    "dropout": ModuleQueueLimits(lease_batch_size=16, max_resident_pages=1, max_prefetch=16),
    "token_budget": ModuleQueueLimits(lease_batch_size=500, max_resident_pages=1),
    "export": ModuleQueueLimits(lease_batch_size=500, max_resident_pages=2),
}


@dataclass(frozen=True)
class RecoveryReport:
    returnedLeases: int
    committedPrepared: int
    repeatedPrepared: int
    pendingApiDecisions: int


class SchedulerError(RuntimeError):
    pass


def _expires_at(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class BoundedScheduler:
    """Control-plane scheduler; it leases IDs only and never holds business payloads."""

    def __init__(self, database: StateDatabase, *, lease_seconds: int = 60, lease_id_factory: Callable[[], str] | None = None) -> None:
        if lease_seconds < 5 or lease_seconds > 3600:
            raise ValueError("lease_seconds must be between 5 and 3600")
        self.database = database
        self.lease_seconds = lease_seconds
        self.lease_id_factory = lease_id_factory or (lambda: uuid.uuid4().hex)

    def nl_queue_limit(self, concurrency: int) -> int:
        if not 1 <= concurrency <= 16:
            raise ValueError("NL concurrency must be between 1 and 16")
        return min(2 * concurrency, 32)

    @staticmethod
    def _require_v10_config(job: object) -> dict[str, object]:
        try:
            config = json.loads(job["config_json"])  # type: ignore[index]
            schema_version = job["config_schema_version"]  # type: ignore[index]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SchedulerError("frozen JobConfig is invalid") from exc
        if not isinstance(config, dict):
            raise SchedulerError("frozen JobConfig is invalid")
        if config.get("schemaVersion") != CURRENT_JOB_CONFIG_SCHEMA_VERSION or schema_version != CURRENT_JOB_CONFIG_SCHEMA_VERSION:
            raise SchedulerError(
                "legacy JobConfig is incompatible; reinitialize the state database and create a new task"
            )
        batches = config.get("moduleBatchSize")
        if not isinstance(batches, dict) or set(batches) != set(MODULE_BATCH_SIZE_BOUNDS):
            raise SchedulerError("frozen JobConfig moduleBatchSize is invalid")
        for key, (minimum, maximum) in MODULE_BATCH_SIZE_BOUNDS.items():
            value = batches[key]
            if type(value) is not int or not minimum <= value <= maximum:
                raise SchedulerError("frozen JobConfig moduleBatchSize is invalid")
        return config

    @classmethod
    def _frozen_module_batch_size(cls, job: object, module_id: ModuleId) -> int:
        """Read the immutable v10 per-round batch size without rewriting it."""
        config = cls._require_v10_config(job)
        batches = config.get("moduleBatchSize")
        key = _MODULE_BATCH_CONFIG_KEYS[module_id]
        bounds = MODULE_BATCH_SIZE_BOUNDS.get(key)
        if not isinstance(batches, dict) or bounds is None:
            raise SchedulerError(f"frozen {module_id} moduleBatchSize is invalid")
        batch_size = batches.get(key)
        if type(batch_size) is not int or not bounds[0] <= batch_size <= bounds[1]:
            raise SchedulerError(f"frozen {module_id} moduleBatchSize is invalid")
        return batch_size

    def _assert_module_order(self, job_id: str, module_id: ModuleId) -> None:
        job = self.database.get_job(job_id)
        try:
            module_order = pipeline_module_ids(int(job["config_schema_version"]))
            index = module_order.index(module_id)
        except ValueError as exc:
            raise SchedulerError(f"module {module_id} is not valid for this task schema") from exc
        statuses = {row["module_id"]: row["status"] for row in self.database.module_summaries(job_id)}
        for predecessor in module_order[:index]:
            if statuses.get(predecessor) not in FINAL_MODULE_STATUSES:
                raise SchedulerError(f"{module_id} cannot start before {predecessor} reaches a final state")
        for successor in module_order[index + 1:]:
            if successor in statuses and statuses[successor] != "paused":
                raise SchedulerError(f"{module_id} cannot start after {successor} has been initialized")

    @staticmethod
    def _is_module_enabled(job: object, module_id: ModuleId) -> bool:
        try:
            config = json.loads(job["config_json"])  # type: ignore[index]
            if module_id == "export":
                return True
            section_name = "countReview" if module_id == "count_review" else "tokenBudget" if module_id == "token_budget" else module_id
            section = config[section_name]
            enabled = section["enabled"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SchedulerError(f"frozen {module_id} configuration is invalid") from exc
        if type(enabled) is not bool:
            raise SchedulerError(f"frozen {module_id} configuration is invalid")
        if module_id != "dropout" or not enabled:
            return enabled
        children = (section.get("artist"), section.get("quality"), section.get("appearanceNl"))
        if any(not isinstance(child, dict) or type(child.get("enabled")) is not bool for child in children):
            raise SchedulerError("frozen dropout subfeature configuration is invalid")
        return any(child["enabled"] for child in children)

    def _assert_future_module(self, job_id: str, module_id: ModuleId) -> tuple[object, str]:
        job = self.database.get_job(job_id)
        try:
            module_order = pipeline_module_ids(int(job["config_schema_version"]))
            current_module_id = job["current_module_id"]
            if not isinstance(current_module_id, str):
                raise ValueError
            if module_order.index(module_id) <= module_order.index(current_module_id):
                raise ValueError
        except ValueError as exc:
            raise SchedulerError(f"{module_id} is not an enabled later successor") from exc
        if not self._is_module_enabled(job, module_id):
            raise SchedulerError(f"{module_id} is not an enabled later successor")
        return job, current_module_id

    def pause_future_module(self, job_id: str, module_id: ModuleId, *, total: int) -> None:
        job, current_module_id = self._assert_future_module(job_id, module_id)
        self.database.pause_future_module(
            job_id, module_id, total=total, current_module_id=current_module_id, active_status=str(job["status"]),
        )

    def cancel_future_pause(self, job_id: str, module_id: ModuleId) -> None:
        job, current_module_id = self._assert_future_module(job_id, module_id)
        self.database.cancel_future_pause(
            job_id, module_id, current_module_id=current_module_id, active_status=str(job["status"]),
        )

    def start_module(
        self,
        job_id: str,
        module_id: ModuleId,
        *,
        enabled: bool,
        profile: str | None = None,
    ) -> str:
        job = self.database.get_job(job_id)
        self._require_v10_config(job)
        allowed_job_states = {"ready", "running", "paused", "reviewing", "exporting"}
        if module_id == "caption":
            allowed_job_states.add("preparing_workspace")
        if job["status"] not in allowed_job_states:
            raise SchedulerError(f"job state {job['status']} cannot start a module")
        self._assert_module_order(job_id, module_id)
        if module_id not in {
            "caption", "classify", "replace", "ocr", "nl", "count_review",
            "dropout", "token_budget", "export",
        }:
            raise SchedulerError(f"unknown module:{module_id}")
        availability = "pending" if enabled or module_id == "export" else "skipped"
        try:
            summary = self.database.module_summary(job_id, module_id)
        except KeyError:
            total = self.database.count_module_samples(job_id, module_id)
            self.database.initialize_module_summary(job_id, module_id, total=total)
            summary = self.database.module_summary(job_id, module_id)
        else:
            total = int(summary["total"])
        if summary["status"] == "paused":
            self.database.start_prepaused_module(
                job_id,
                module_id,
                resume_status="exporting" if module_id == "export" else "running",
            )
            return "paused"
        if summary["status"] == "running":
            expected_job_status = "exporting" if module_id == "export" else "running"
            if job["current_module_id"] == module_id and job["status"] == expected_job_status:
                # Recovery must retain already completed samples rather than reset the module queue.
                return "running"
            raise SchedulerError("only the persisted active module can be resumed")
        if availability == "skipped_not_available":
            transition_module(summary["status"], availability, module_id=module_id)
            self.database.set_module_summary(job_id, module_id, status=availability, skipped=total, finished=True)
            if job["status"] == "preparing_workspace":
                self.database.set_job_status(job_id, "running", current_module_id=module_id)
            return availability
        if availability == "skipped":
            transition_module(summary["status"], availability, module_id=module_id)
            if module_id == "caption":
                cursor: int | None = None
                while True:
                    page = self.database.skip_disabled_caption_page(
                        job_id,
                        after_sample_id=cursor,
                        limit=MAX_PAGE_SIZE,
                    )
                    if not page:
                        break
                    cursor = page[-1]
            elif module_id == "nl":
                cursor = None
                while True:
                    page = self.database.record_nl_disabled_observations_page(
                        job_id, after_sample_id=cursor, limit=500
                    )
                    if not page:
                        break
                    cursor = page[-1]
            self.database.set_module_summary(job_id, module_id, status=availability, skipped=total, finished=True)
            if job["status"] == "preparing_workspace":
                self.database.set_job_status(job_id, "running", current_module_id=module_id)
            return availability
        transition_module(summary["status"], "running", module_id=module_id)
        if module_id == "export":
            # Samples that failed before Export never reach it on their own.
            # Record them as explicit export skips so the module can complete and
            # the commit projection stops expecting artifacts they cannot have.
            cursor: int | None = None
            skipped_failures = 0
            while True:
                page = self.database.skip_failed_samples_for_export_page(
                    job_id, after_sample_id=cursor, limit=MAX_PAGE_SIZE
                )
                if not page:
                    break
                skipped_failures += len(page)
                cursor = page[-1]
            if skipped_failures:
                self.database.increment_module_counts(job_id, "export", skipped=skipped_failures)
        # A page-at-a-time reset is required for all modules after their
        # predecessor. Caption starts from the initial NULL module state.
        if module_id != "caption":
            cursor: int | None = None
            while True:
                page = self.database.reset_next_module_page(job_id, module_id, after_sample_id=cursor, limit=MAX_PAGE_SIZE)
                if not page:
                    break
                cursor = page[-1]
        # The preflight decision is shared by every module, including Export.
        # Export still commits the baseline directory, but preserved samples
        # must not acquire an Export worker lease or hit its Token Budget gate.
        cursor = None
        while True:
            page = self.database.skip_incremental_preserved_page(
                job_id, module_id, after_sample_id=cursor, limit=MAX_PAGE_SIZE,
            )
            if not page:
                break
            cursor = page[-1]
        self.database.set_module_summary(job_id, module_id, status="running")
        self.database.set_job_status(
            job_id, "exporting" if module_id == "export" else "running", current_module_id=module_id
        )
        return "running"

    def reclaim_expired_leases(self, job_id: str) -> int:
        """Return leases whose holder stopped heart-beating before a new claim."""
        return self.database.return_expired_leases(job_id, _expires_at(0))

    def claim_batch(self, job_id: str, module_id: ModuleId, worker_instance_id: str, config_hash: str, *, limit: int | None = None) -> list[WorkLease]:
        job = self.database.get_job(job_id)
        expected_job_status = "exporting" if module_id == "export" else "running"
        if job["status"] != expected_job_status or job["current_module_id"] != module_id:
            return []
        # Every claim loop passes through here, so expired leases are recovered
        # before capacity and single-worker limits are measured.
        self.reclaim_expired_leases(job_id)
        if job["config_hash"] != config_hash:
            raise SchedulerError("config hash does not match immutable job configuration")
        protocol_maximum = MODULE_QUEUE_LIMITS[module_id].lease_batch_size
        configured_limit = self._frozen_module_batch_size(job, module_id)
        actual_limit = configured_limit if limit is None else limit
        if not 1 <= actual_limit <= configured_limit or actual_limit > protocol_maximum:
            raise SchedulerError(f"{module_id} lease limit must be between 1 and {configured_limit}")
        max_in_flight = configured_limit * MODULE_QUEUE_LIMITS[module_id].max_resident_pages
        if module_id == "nl":
            max_in_flight = self.nl_queue_limit(configured_limit)
        try:
            rows = self.database.claim_leases(
                job_id,
                module_id,
                worker_instance_id,
                config_hash,
                limit=actual_limit,
                max_in_flight=max_in_flight,
                single_worker=module_id in {"caption", "ocr", "count_review", "dropout"},
                lease_bound=max(protocol_maximum, max_in_flight),
                lease_id_factory=self.lease_id_factory,
                expires_at=_expires_at(self.lease_seconds),
            )
        except ValueError as exc:
            raise SchedulerError(str(exc)) from exc
        return [
            WorkLease(
                jobId=job_id,
                moduleId=module_id,
                sampleId=int(row["sample_id"]),
                status="leased",
                attempt=int(row["attempt"]),
                configHash=config_hash,
                leaseId=str(row["lease_id"]),
                workerInstanceId=str(row["worker_instance_id"]),
                leaseExpiresAt=str(row["lease_expires_at"]),
            )
            for row in rows
        ]

    def heartbeat(self, job_id: str, worker_instance_id: str, lease_ids: list[str]) -> int:
        return self.database.heartbeat(job_id, worker_instance_id, _expires_at(self.lease_seconds), lease_ids=lease_ids)

    def stage_prepared(self, lease: WorkLease, *, relative_path: str, sha256: str) -> None:
        if not lease.leaseId:
            raise SchedulerError("prepared artifacts require a lease id")
        self.database.stage_prepared_artifact(
            lease.jobId, lease.sampleId, lease_id=lease.leaseId, relative_path=relative_path, sha256=sha256
        )

    def complete(self, lease: WorkLease, *, txt_provenance: str | None = None) -> None:
        if not lease.leaseId:
            raise SchedulerError("completion requires a lease id")
        self.database.complete_leased_sample_and_count(
            lease.jobId,
            lease.moduleId,
            lease.sampleId,
            lease_id=lease.leaseId,
            txt_provenance=txt_provenance,
        )

    def skip_caption(self, lease: WorkLease) -> None:
        if lease.moduleId != "caption" or not lease.leaseId:
            raise SchedulerError("caption skip requires a caption lease id")
        self.database.skip_leased_caption_sample(
            lease.jobId,
            lease.sampleId,
            lease_id=lease.leaseId,
        )

    def fail_with_issue(self, lease: WorkLease, issue: SampleIssue, *, allowed_statuses: tuple[str, ...] = ("leased",)) -> None:
        if not lease.leaseId:
            raise SchedulerError("failure requires a lease id")
        self.database.fail_leased_sample_with_issue(
            lease.jobId,
            lease.moduleId,
            lease.sampleId,
            lease_id=lease.leaseId,
            issue=issue,
            allowed_statuses=allowed_statuses,
        )

    def skip_export_with_issue(self, lease: WorkLease, issue: SampleIssue) -> None:
        if lease.moduleId != "export" or not lease.leaseId:
            raise SchedulerError("Export issue skip requires an Export lease id")
        self.database.skip_leased_sample_with_issue(
            lease.jobId,
            lease.moduleId,
            lease.sampleId,
            lease_id=lease.leaseId,
            issue=issue,
        )

    def release_unstarted(self, lease: WorkLease) -> None:
        if not lease.leaseId:
            raise SchedulerError("returning a lease requires a lease id")
        self.database.return_lease_to_pending(
            lease.jobId,
            lease.sampleId,
            lease_id=lease.leaseId,
        )

    def fail(self, lease: WorkLease, *, retriable: bool) -> None:
        if not lease.leaseId:
            raise SchedulerError("failure requires a lease id")
        if retriable:
            self.database.return_lease_to_pending(lease.jobId, lease.sampleId, lease_id=lease.leaseId)
        else:
            state = self.database.get_sample_state(lease.jobId, lease.sampleId)
            self.database.set_sample_state(
                lease.jobId,
                lease.sampleId,
                SampleRunState(
                    sampleId=lease.sampleId,
                    txtProvenance=state["txt_provenance"],
                    currentModuleId=lease.moduleId,
                    status="failed",
                    attempt=int(state["attempt"]),
                    leaseId=None,
                    workerInstanceId=None,
                    leaseExpiresAt=None,
                ),
            )
            self.database.increment_module_counts(lease.jobId, lease.moduleId, failed=1)

    def finish_module(self, job_id: str, module_id: ModuleId, *, with_issues: bool = False) -> str:
        with self.database.transaction(immediate=True):
            job = self.database.get_job(job_id)
            if job["current_module_id"] != module_id:
                raise SchedulerError("only the current module can be finished")
            if job["status"] in {"paused", "cancelling"}:
                return str(job["status"])
            expected_status = "exporting" if module_id == "export" else "running"
            if job["status"] != expected_status:
                raise SchedulerError("only an active module can be finished")
            if self.database.count_module_unsettled(job_id, module_id):
                raise SchedulerError("module still has pending or in-flight samples")
            summary = self.database.module_summary(job_id, module_id)
            target = "completed_with_issues" if with_issues else "completed"
            transition_module(summary["status"], target, module_id=module_id)
            self.database.set_module_summary(job_id, module_id, status=target, finished=True)
        self.database.checkpoint()
        return target

    def worker_exited(self, job_id: str, module_id: ModuleId, *, abnormal: bool) -> bool:
        """Return whether to restart; otherwise fail only unfinished samples and continue."""
        if not abnormal:
            return False
        job = self.database.get_job(job_id)
        expected_job_status = "exporting" if module_id == "export" else "running"
        if job["status"] != expected_job_status or job["current_module_id"] != module_id:
            return False
        attempts = self.database.increment_worker_restart(job_id, module_id)
        if attempts <= 2:
            return True
        cursor: int | None = None
        failed = 0
        while True:
            try:
                page = self.database.fail_unsettled_module_samples_page(
                    job_id, module_id, after_sample_id=cursor, limit=MAX_PAGE_SIZE,
                )
            except ValueError:
                current = self.database.get_job(job_id)
                if current["status"] != expected_job_status or current["current_module_id"] != module_id:
                    return False
                raise
            if not page:
                break
            failed += len(page)
            cursor = page[-1]
        self.finish_module(
            job_id,
            module_id,
            with_issues=failed > 0 and module_id != "export",
        )
        return False

    def begin_cancellation(self, job_id: str) -> None:
        self.database.begin_cancellation(job_id)

    def settle_cancellation(self, job_id: str, *, commit_succeeded: bool = False) -> None:
        self.database.settle_cancellation(job_id, succeeded=commit_succeeded)

    def confirm_nl_api_outcome_unknown(self, job_id: str, *, confirmed: bool) -> int:
        if not confirmed:
            raise SchedulerError("explicit confirmation is required before repeating unknown API outcomes")
        job = self.database.get_job(job_id)
        if job["current_module_id"] != "nl" or job["status"] not in {"paused", "interrupted"}:
            raise SchedulerError("unknown API outcomes can be confirmed only for a paused or interrupted NL module")
        returned = 0
        while True:
            page = self.database.confirm_nl_unknown_requests(job_id)
            if not page:
                return returned
            self.database.increment_module_diagnostic(job_id, "nl", "nl_api_outcome_unknown_confirmed", severity="warning", amount=page)
            returned += page

    def recover(
        self,
        job_id: str,
        *,
        confirmed: bool,
        expected_config_hash: str,
        manifest_schema_version: int,
        protocol_version: str,
        verify_source_fingerprints: Callable[[], bool],
        commit_prepared: Callable[[str, int, str, str], bool | str],
        commit_response_staged: Callable[[str, int, str], bool] | None = None,
    ) -> RecoveryReport:
        if not confirmed:
            raise SchedulerError("manual confirmation is required before recovery")
        job = self.database.get_job(job_id)
        if job["status"] not in {"interrupted", "cancelled_recoverable"}:
            raise SchedulerError("only interrupted or cancelled recoverable jobs require manual recovery")
        if job["config_hash"] != expected_config_hash or int(job["manifest_schema_version"]) != manifest_schema_version:
            raise SchedulerError("job configuration or manifest schema no longer matches")
        if protocol_version != PROTOCOL_VERSION:
            raise SchedulerError("worker protocol version no longer matches")
        if not verify_source_fingerprints():
            raise SchedulerError("source file fingerprints no longer match the immutable manifest")
        returned = 0
        committed = 0
        repeated = 0
        api_pending = 0
        cursor: int | None = None
        while True:
            page = self.database.recovery_state_page(job_id, after_sample_id=cursor, limit=500)
            if not page:
                break
            for row in page:
                status = row["status"]
                sample_id = int(row["sample_id"])
                if status == "leased":
                    self.database.return_lease_to_pending(job_id, sample_id, lease_id=row["lease_id"])
                    returned += 1
                elif status == "prepared":
                    path = row["prepared_artifact_relative_path"]
                    digest = row["prepared_artifact_sha256"]
                    result = (
                        commit_prepared(job_id, sample_id, path, digest)
                        if isinstance(path, str) and isinstance(digest, str)
                        else False
                    )
                    if result == "settled":
                        committed += 1
                    elif result is True:
                        self.database.complete_leased_sample_and_count(
                            job_id,
                            str(row["current_module_id"]),
                            sample_id,
                            lease_id=row["lease_id"],
                            allowed_statuses=("prepared",),
                            txt_provenance="module1_written" if row["current_module_id"] == "caption" else None,
                        )
                        committed += 1
                    else:
                        self.database.return_lease_to_pending(job_id, sample_id, lease_id=row["lease_id"])
                        repeated += 1
                elif status == "request_started":
                    # The API could have charged the request. It remains for a
                    # user decision and is never blindly resent during restart.
                    api_pending += 1
                elif status == "response_staged":
                    if commit_response_staged is not None and isinstance(row["lease_id"], str) and commit_response_staged(job_id, sample_id, row["lease_id"]):
                        self.database.complete_leased_sample_and_count(
                            job_id, str(row["current_module_id"]), sample_id, lease_id=row["lease_id"],
                            allowed_statuses=("prepared", "response_staged"),
                        )
                        committed += 1
                    else:
                        api_pending += 1
            cursor = int(page[-1]["sample_id"])
        return RecoveryReport(returned, committed, repeated, api_pending)
