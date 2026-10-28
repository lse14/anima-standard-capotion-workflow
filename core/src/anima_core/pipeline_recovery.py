"""Recovery ownership for the pipeline service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Literal

from . import PROTOCOL_VERSION
from .contracts import SampleIssue
from .commit_journal import CommitJournal, load as load_commit_journal
from .caption_overlay import CaptionOverlayWriter
from .classify_overlay import ClassifyOverlayWriter
from .count_review_overlay import CountReviewOverlayWriter
from .db import StateDatabase
from .db_scheduler import _complete_leased_sample_with_issue
from .export_commit import ExportCommitCoordinator
from .export_summary import UNLABELED_IMAGES_CODE
from .job_preflight import config_from_dict
from .nl_overlay import NlOverlayWriter
from .ocr_sidecar import OcrSidecarError, parse_ocr_sidecar
from .overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from .path_safety import PathSafetyError, assert_no_reparse_tree, file_fingerprint, safe_relative_path, sha256_file
from .pipeline_dispatch import PipelineError, _RUNTIMES
from .replace_overlay import ReplaceOverlayWriter
from .token_budget_overlay import TokenBudgetOverlayWriter
from .token_budget_review import TokenBudgetReviewService
from .retention import RetentionManager
from .scheduler import BoundedScheduler


ExportRecoveryKind = Literal["manual_recommit", "automatic_switch_recovery", "resolved"]


def classify_export_recovery(journal: CommitJournal) -> ExportRecoveryKind:
    """Classify directory recovery without changing any filesystem state."""
    if journal.state in {"prepared", "declared", "staging", "staged", "backup_verified"}:
        # The dataset is renamed to the rollback path one statement before the
        # journal records `rollback_created`. When a crash lands in that window the
        # dataset is absent and its full old copy sits in the rollback directory, so
        # the journal's own `recover()` can put it back; asking the user for a
        # recommit there would refuse because a rollback directory exists.
        if not journal.dataset_root.exists() and journal.rollback_root.exists():
            return "automatic_switch_recovery"
        return "manual_recommit"
    if journal.state in {"rollback_created", "rollback_required"}:
        return "automatic_switch_recovery"
    if journal.state in {"committed", "rolled_back"}:
        return "resolved"
    raise PipelineError("commit journal has no supported recovery classification")


def _recover_ocr_prepared(
    database: StateDatabase, layout: OverlayLayout, job_id: str, sample_id: int,
    prepared_relative_path: str, expected_sha256: str,
) -> bool | str:
    row = database.get_sample_with_state(job_id, sample_id)
    lease_id = row["lease_id"]
    if (
        row["current_module_id"] != "ocr" or row["status"] != "prepared"
        or not isinstance(lease_id, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or safe_relative_path(prepared_relative_path) != f"prepared\\ocr\\{lease_id}.json"
        or row["prepared_artifact_relative_path"] != prepared_relative_path
        or row["prepared_artifact_sha256"] != expected_sha256
    ):
        raise PipelineError("OCR recovery metadata is invalid")
    target = layout.ocr_sidecar_path(str(row["relative_image_path"]))
    prepared = layout.resolve_prepared(prepared_relative_path)
    if target.is_file():
        if sha256_file(target) != expected_sha256:
            raise PipelineError("OCR committed sidecar digest does not match prepared state")
        payload = target.read_bytes()
        if prepared.is_file() and sha256_file(prepared) == expected_sha256:
            prepared.unlink()
    else:
        if not prepared.is_file() or sha256_file(prepared) != expected_sha256:
            return False
        payload = prepared.read_bytes()
    try:
        sidecar = parse_ocr_sidecar(
            payload, expected_relative_image_path=str(row["relative_image_path"]),
        )
    except OcrSidecarError as exc:
        raise PipelineError("OCR prepared sidecar is invalid") from exc
    if sidecar.image.sizeBytes != int(row["image_size"]):
        raise PipelineError("OCR prepared sidecar image identity is invalid")
    if not target.is_file():
        committed = layout.commit_ocr_prepared(
            prepared_relative_path, expected_sha256, str(row["relative_image_path"]),
        )
        if sha256_file(committed) != expected_sha256:
            raise PipelineError("OCR prepared sidecar commit digest is invalid")
    if sidecar.status != "failed":
        return True
    if sidecar.error is None:
        raise PipelineError("OCR failed prepared sidecar has no error")
    issue = SampleIssue(
        issueId=hashlib.sha256(
            f"{job_id}\0{sample_id}\0ocr\0{sidecar.error.code}".encode("utf-8")
        ).hexdigest(),
        jobId=job_id,
        sampleId=sample_id,
        relativeImagePath=str(row["relative_image_path"]),
        moduleId="ocr",
        code=sidecar.error.code,
        severity="warning",
        blocking=False,
        retriable=sidecar.error.retriable,
        repairStartModule="ocr",
        message=sidecar.error.message[:1024],
        attempt=int(row["attempt"]),
    )
    _complete_leased_sample_with_issue(
        database,
        job_id,
        "ocr",
        sample_id,
        lease_id=lease_id,
        issue=issue,
        allowed_statuses=("prepared",),
    )
    return "settled"


class PipelineRecoveryMixin:
    """Startup, commit, fingerprint, prepared artifact, and explicit recovery."""

    @staticmethod
    def _recovery_target_status(current_module_id: object) -> str:
        if current_module_id in {None, "workspace"}:
            return "preparing_workspace"
        if current_module_id in {"count_review", "token_budget"}:
            return "reviewing"
        if current_module_id == "export":
            return "exporting"
        if current_module_id in _RUNTIMES:
            return "running"
        raise PipelineError("interrupted job has no recoverable current module")

    def startup_recovery(self) -> dict[str, int]:
        """The single backend startup entry point before any job may be resumed.

        Freezes every job an earlier process left mid-flight, sweeps dataset
        claims those jobs can no longer own, resolves persisted commit journals
        and finally prunes retained successes.
        """
        database = StateDatabase.open(self.database_path)
        try:
            interrupted = 0
            cursor: str | None = None
            while True:
                page = database.page_active_jobs(after_job_id=cursor, limit=200)
                if not page:
                    break
                for row in page:
                    job_id_value = str(row["job_id"])
                    if str(row["status"]) == "cancelling":
                        # No worker survives a restart, so the leases it held can never
                        # be released and the job could never finish cancelling.
                        database.abandon_cancellation_work(job_id_value)
                        if not database.count_in_flight(job_id_value):
                            try:
                                # Nothing is left to drain, so finish the cancellation
                                # instead of parking the job in a state no action accepts.
                                database.settle_cancellation(job_id_value)
                            except ValueError:
                                pass
                    database.mark_interrupted(job_id_value)
                    interrupted += 1
                cursor = str(page[-1]["job_id"])
            claims = database.clear_stale_dataset_claims()
        finally:
            database.close()
        self.recover_pending_commits()
        database = StateDatabase.open(self.database_path)
        try:
            retention = RetentionManager(database).cleanup()
        finally:
            database.close()
        return {
            "interruptedJobs": interrupted,
            "clearedDatasetClaims": claims,
            "deletedJobs": len(retention.deletedJobIds),
            "deletedOverlays": retention.deletedOverlays,
        }

    def recover_pending_commits(self) -> None:
        """Resolve journaled directory switches before workers may be started."""
        database = StateDatabase.open(self.database_path)
        try:
            cursor: str | None = None
            while True:
                page = database.page_overlay_jobs(after_job_id=cursor, limit=200)
                if not page:
                    return
                for row in page:
                    job_id = str(row["job_id"])
                    cursor = job_id
                    if not Path(str(row["overlay_root"])).exists():
                        # A discarded or already pruned overlay is a resolved
                        # terminal state; only the stale pointer is left.
                        database.clear_workspace_metadata(job_id)
                        continue
                    layout = OverlayLayout.open_existing(
                        str(row["overlay_root"]), job_id, allow_missing_dataset=True,
                    )
                    journal = load_commit_journal(layout)
                    if journal is None:
                        continue
                    recovery_kind = classify_export_recovery(journal)
                    if recovery_kind == "manual_recommit":
                        database.mark_interrupted(
                            job_id, allow_failed=str(row["status"]) == "failed",
                        )
                        continue
                    result = ExportCommitCoordinator.recover(layout)
                    # Startup recovery may already have frozen the job, so the
                    # module it was interrupted in decides the outcome.
                    status = str(row["status"])
                    if status == "interrupted" and row["resume_status"]:
                        status = str(row["resume_status"])
                    if result == "committed" and status == "committing":
                        terminal_status = "succeeded_with_issues" if database.module_diagnostic_count(
                            job_id, "export", UNLABELED_IMAGES_CODE,
                        ) > 0 else "succeeded"
                        database.set_job_status(job_id, terminal_status, current_module_id="export")
                    elif result == "rolled_back" and status in {"exporting", "committing", "cancelling"}:
                        database.set_job_status(job_id, "failed", current_module_id="export")
        finally:
            database.close()

    @staticmethod
    def _verify_source_fingerprints(database: StateDatabase, job_id: str) -> bool:
        """Compare immutable manifest image fingerprints without materializing the manifest."""
        job = database.get_job(job_id)
        source = Path(str(job["source_root"]))
        try:
            assert_no_reparse_tree(source)
        except (OSError, PathSafetyError):
            return False
        cursor: int | None = None
        while True:
            page = database.page_samples(job_id, after_sample_id=cursor, limit=500)
            if not page:
                return True
            for row in page:
                try:
                    relative = safe_relative_path(str(row["relative_image_path"]))
                    actual = file_fingerprint(source / Path(relative.replace("\\", os.sep)))
                except (OSError, PathSafetyError):
                    return False
                if (
                    actual["file_id"] != row["image_file_id"]
                    or actual["size"] != row["image_size"]
                    or actual["mtime_ns"] != row["image_mtime_ns"]
                ):
                    return False
            cursor = int(page[-1]["sample_id"])

    @staticmethod
    def _recover_policy_prepared(
        database: StateDatabase, layout: OverlayLayout, job_id: str, sample_id: int,
        prepared_relative_path: str, expected_sha256: str,
    ) -> bool:
        row = database.get_sample_with_state(job_id, sample_id)
        lease_id = row["lease_id"]
        if (
            row["current_module_id"] != "dropout" or row["status"] != "prepared"
            or not isinstance(lease_id, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or safe_relative_path(prepared_relative_path) != f"prepared\\dropout\\{lease_id}.json"
            or row["prepared_artifact_relative_path"] != prepared_relative_path
            or row["prepared_artifact_sha256"] != expected_sha256
        ):
            raise PipelineError("policy recovery metadata is invalid")
        target = layout.annotation_path(str(row["annotation_key"]), ".json")
        prepared = layout.resolve_prepared(prepared_relative_path)
        if target.is_file() and sha256_file(target) == expected_sha256:
            if prepared.is_file() and sha256_file(prepared) == expected_sha256:
                prepared.unlink()
            return True
        if not prepared.is_file() or sha256_file(prepared) != expected_sha256:
            return False
        committed = layout.commit_prepared(prepared_relative_path, expected_sha256, str(row["annotation_key"]), ".json")
        return sha256_file(committed) == expected_sha256

    def recover_job(self, job_id: str, *, confirmed: bool) -> dict[str, object]:
        """Run the explicit recovery protocol and restart only a safe current module."""
        with self._lock:
            if job_id in self._threads:
                raise PipelineError("job pipeline is already running")
            database = StateDatabase.open(self.database_path)
            thread: threading.Thread | None = None
            try:
                job = database.get_job(job_id)
                try:
                    config = config_from_dict(json.loads(str(job["config_json"])))
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    raise PipelineError("frozen JobConfig cannot be recovered") from exc
                if config.config_hash != job["config_hash"]:
                    raise PipelineError("frozen JobConfig hash no longer matches")
                module_id = job["current_module_id"]
                resume_status = job["resume_status"]
                if not isinstance(job["overlay_root"], str) or not job["overlay_root"]:
                    raise PipelineError("interrupted job has no annotation overlay")
                layout = OverlayLayout.open_existing(str(job["overlay_root"]), job_id)
                if module_id == "export" and job["status"] == "interrupted":
                    summary = database.module_summary(job_id, "export")
                    journal = load_commit_journal(layout)
                    if (
                        summary["status"] == "completed"
                        and journal is not None
                        and classify_export_recovery(journal) == "manual_recommit"
                    ):
                        return {
                            "jobId": job_id,
                            "returnedLeases": 0,
                            "committedPrepared": 0,
                            "repeatedPrepared": 0,
                            "pendingApiDecisions": 0,
                            "started": False,
                            "status": "interrupted",
                            "recoveryKind": "export_recommit",
                        }
                target_status = self._recovery_target_status(module_id)
                if not isinstance(resume_status, str):
                    raise PipelineError("interrupted job has no recoverable worker state")
                # A module that finished with issues moves the job to `reviewing`
                # before the crash, so that persisted status is a legal resume target
                # for every module and not only the review-owned ones.
                allowed_resume_statuses = (
                    {"running", "reviewing"}
                    if module_id in {"count_review", "token_budget"}
                    else {target_status, "reviewing"}
                )
                if resume_status not in allowed_resume_statuses:
                    raise PipelineError("persisted recovery target is incompatible with current module")
                view = WorkingAnnotationView(BaselineView(Path(str(job["dataset_root"]))), layout)
                if module_id == "token_budget":
                    # Review-owned unprepared leases must return to the failed review
                    # state before generic recovery would otherwise make them pending.
                    TokenBudgetReviewService(database, job_id)._recover_leased_applies()

                def commit_prepared(_: str, sample_id: int, relative: str, digest: str) -> bool | str:
                    if module_id == "caption":
                        return CaptionOverlayWriter.open_for_job(database, job_id).recover_prepared(job_id, sample_id, relative, digest)
                    if module_id == "classify":
                        return ClassifyOverlayWriter.open_for_job(database, job_id).recover_prepared(job_id, sample_id, relative, digest)
                    if module_id == "replace":
                        return ReplaceOverlayWriter(database, layout, job_id).recover_prepared(job_id, sample_id, relative, digest)
                    if module_id == "ocr":
                        return _recover_ocr_prepared(database, layout, job_id, sample_id, relative, digest)
                    if module_id == "nl":
                        return NlOverlayWriter(database, layout, view, job_id).recover_prepared(job_id, sample_id, relative, digest)
                    if module_id == "count_review":
                        return CountReviewOverlayWriter(database, layout, view, job_id).recover_prepared(job_id, sample_id, relative, digest)
                    if module_id == "dropout":
                        return self._recover_policy_prepared(database, layout, job_id, sample_id, relative, digest)
                    if module_id == "token_budget":
                        state = database.get_sample_with_state(job_id, sample_id)
                        if state["worker_instance_id"] == "token-budget-review":
                            return TokenBudgetReviewService(database, job_id)._recover_prepared_apply(
                                job,
                                layout,
                                sample_id=sample_id,
                                lease_id=str(state["lease_id"]),
                                prepared_relative_path=relative,
                                expected_sha256=digest,
                            )
                        return TokenBudgetOverlayWriter(database, layout, view, job_id).recover_prepared(job_id, sample_id, relative, digest)
                    return False

                nl_writer = NlOverlayWriter(database, layout, view, job_id) if module_id == "nl" else None
                report = BoundedScheduler(database).recover(
                    job_id, confirmed=confirmed, expected_config_hash=config.config_hash,
                    manifest_schema_version=1, protocol_version=PROTOCOL_VERSION,
                    verify_source_fingerprints=lambda: self._verify_source_fingerprints(database, job_id),
                    commit_prepared=commit_prepared,
                    commit_response_staged=(nl_writer.commit_staged if nl_writer is not None else None),
                )
                response: dict[str, object] = {
                    "jobId": job_id, "returnedLeases": report.returnedLeases,
                    "committedPrepared": report.committedPrepared, "repeatedPrepared": report.repeatedPrepared,
                    "pendingApiDecisions": report.pendingApiDecisions, "started": False,
                }
                if report.pendingApiDecisions:
                    if job["status"] == "cancelled_recoverable":
                        database.set_job_status(
                            job_id,
                            "interrupted",
                            current_module_id=module_id if isinstance(module_id, str) else None,
                            resume_status=resume_status,
                        )
                    response["status"] = "interrupted"
                    return response
                if job["status"] == "cancelled_recoverable":
                    database.clear_cancellation_metadata(job_id)
                database.set_job_status(
                    job_id,
                    target_status,
                    current_module_id=module_id if isinstance(module_id, str) else None,
                    resume_status=None,
                )
                response["status"] = target_status
                if target_status == "reviewing":
                    return response
                thread = threading.Thread(
                    target=self._thread_main, args=(job_id, target_status != "preparing_workspace"),
                    daemon=True, name=f"anima-{job_id[:12]}",
                )
                self._threads[job_id] = thread
                response["started"] = True
            finally:
                database.close()
            assert thread is not None
            try:
                thread.start()
            except Exception:
                self._threads.pop(job_id, None)
                database = StateDatabase.open(self.database_path)
                try:
                    database.set_job_status(
                        job_id,
                        "interrupted",
                        current_module_id=module_id if isinstance(module_id, str) else None,
                        resume_status=target_status,
                    )
                finally:
                    database.close()
                raise
            return response
