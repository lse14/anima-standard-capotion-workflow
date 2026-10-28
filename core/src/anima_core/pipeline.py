"""Serialized core orchestration for the worker modules already wired in core."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Callable

from .contracts import (
    NL_MOSS_PROMPT_VERSION,
    SampleIssue,
    job_config_supports_ocr_device,
    job_config_supports_token_budget,
    pipeline_module_ids,
    sha256_json,
)
from .count_review_service import CountReviewService
from .credentials import CredentialStoreError, DpapiCredentialStore
from .db import StateDatabase
from .db_primitives import repair_target_clause
from .export_commit import ExportCommitCoordinator
from .job_preflight import JobPreflightError, config_from_dict
from .launcher import WorkerLaunchError, WorkerLauncher
from .nl_profiles import NlApiProfileStore
from .nl_runner import NlApiCredentials
from .overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from .pipeline_dispatch import PipelineDispatchMixin, PipelineError, _RUNTIMES
from .pipeline_recovery import PipelineRecoveryMixin
from .resource_catalog import ResourceCatalog, default_resource_library_root
from .scheduler import FINAL_MODULE_STATUSES, BoundedScheduler, SchedulerError
from .stdio_transport import StdioJsonlTransportError
from .token_budget_overlay import TokenBudgetOverlayError, TokenBudgetOverlayWriter


def default_install_root() -> Path:
    """Locate only the distributed runtime root, never a Python installation."""
    configured = os.environ.get("ANIMA_INSTALL_ROOT")
    candidates = ([Path(configured)] if configured else []) + list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "manifests" / "runtimes" / "core.json").is_file():
            return candidate
        if (candidate / ".runtime-build" / "manifests" / "runtimes" / "core.json").is_file():
            return candidate / ".runtime-build"
    raise PipelineError("distributed embedded runtime manifests are unavailable")


def _frozen_v10_config(job: object) -> dict[str, object]:
    try:
        raw_config = json.loads(str(job["config_json"]))  # type: ignore[index]
        config_hash = job["config_hash"]  # type: ignore[index]
        schema_version = job["config_schema_version"]  # type: ignore[index]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PipelineError("frozen JobConfig is invalid JSON") from exc
    if not isinstance(raw_config, dict) or not isinstance(config_hash, str) or sha256_json(raw_config) != config_hash:
        raise PipelineError("frozen JobConfig hash does not match task record")
    try:
        config = config_from_dict(raw_config)
    except (JobPreflightError, TypeError, ValueError) as exc:
        raise PipelineError(str(exc)) from exc
    if config.config_hash != config_hash:
        raise PipelineError("frozen JobConfig hash does not match task record")
    if config.schemaVersion != schema_version:
        raise PipelineError("frozen JobConfig schema version does not match the task record")
    return raw_config


def _forced_cuda_start_gate(service: "PipelineService", job_id: str, job: object) -> None:
    config = _frozen_v10_config(job)
    if not job_config_supports_ocr_device(config.get("schemaVersion")):
        return
    ocr = config.get("ocr")
    if not isinstance(ocr, dict) or ocr.get("enabled") is not True or ocr.get("device") != "cuda":
        return
    try:
        overlay_root = job["overlay_root"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise PipelineError("prepared job has no annotation overlay") from exc
    if not isinstance(overlay_root, str) or not overlay_root:
        raise PipelineError("prepared job has no annotation overlay")
    binding_path = OverlayLayout.open_existing(overlay_root, job_id).resource_path("ocr-runtime-binding-v1.json")
    if binding_path.exists():
        return
    try:
        launch, _ = service._resolve_ocr_runtime("ocr-paddle-gpu")
        service._probe_ocr_gpu_runtime(launch, service.install_root)
    except Exception as exc:
        raise PipelineError(
            "The OCR CUDA runtime is unavailable or incompatible with this GPU. Choose Auto or CPU."
        ) from exc


def _is_unstarted_paused_summary(summary: object) -> bool:
    try:
        return (
            summary["status"] == "paused"  # type: ignore[index]
            and int(summary["completed"]) == 0  # type: ignore[index]
            and int(summary["failed"]) == 0  # type: ignore[index]
            and int(summary["skipped"]) == 0  # type: ignore[index]
            and int(summary["issue_count"]) == 0  # type: ignore[index]
            and int(summary["worker_restart_count"]) == 0  # type: ignore[index]
            and summary["started_at"] is None  # type: ignore[index]
            and summary["finished_at"] is None  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError):
        return False


class PipelineService(PipelineRecoveryMixin, PipelineDispatchMixin):
    """Owns background threads, while individual runners own bounded module work."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        install_root: str | Path | None = None,
        profile_store: NlApiProfileStore | None = None,
        credential_store: DpapiCredentialStore | None = None,
        resource_catalog: ResourceCatalog | None = None,
        launcher_factory: Callable[[Path], WorkerLauncher] | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.install_root = Path(install_root) if install_root is not None else default_install_root()
        self.resource_catalog = resource_catalog or ResourceCatalog(default_resource_library_root(self.install_root))
        self.resource_root = self.resource_catalog.root
        self.profile_store = profile_store or NlApiProfileStore()
        self.credential_store = credential_store or DpapiCredentialStore()
        self.launcher_factory = launcher_factory or (
            lambda root: WorkerLauncher.from_install_root(root, resource_root=self.resource_root)
        )
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._api_config_lock = threading.Lock()

    def update_nl_api_config(self, job_id: str, payload: dict[str, object]) -> dict[str, object]:
        """Queue a connection-only API revision for the next NL batch.

        The immutable JobConfig remains untouched.  A worker that has already
        started its current batch continues with its existing hello revision;
        the queued revision is exposed to the control plane immediately.
        """
        with self._api_config_lock:
            database = StateDatabase.open(self.database_path)
            try:
                job = database.get_job(job_id)
                if str(job["status"]) not in {"running", "paused", "reviewing", "preparing_workspace"}:
                    raise PipelineError("NL API settings can only change before or during an active task")
                current = database.get_nl_api_runtime_override(job_id)
                task_revision = database.get_nl_api_task_revision(job_id)
                expected = payload.get("expectedTaskRevision")
                if type(expected) is not int or expected != task_revision:
                    raise PipelineError("api_task_revision_conflict")
                profile_id = payload.get("profileId")
                profile = next((item for item in self.profile_store.load_all() if item.profileId == profile_id), None)
                if profile is None:
                    raise PipelineError("NL API profile does not exist")
                expected_profile = payload.get("expectedProfileRevision")
                if type(expected_profile) is not int or profile.revision != expected_profile:
                    raise PipelineError("profile_revision_conflict")
                profile_snapshot = self.profile_store.snapshot_document()
                updated = None
                try:
                    with database.transaction(immediate=True):
                        updated = self.profile_store.save(
                            type(profile)(
                                profile.profileId, str(payload["endpoint"]), str(payload["model"]),
                                payload.get("backupModel") if isinstance(payload.get("backupModel"), str) else None,
                                str(payload["credentialRef"]), "", dict(payload.get("policy", {})),
                                profile.name, str(payload.get("protocol", profile.protocol)),
                                str(payload.get("providerLabel", profile.providerLabel)), str(payload.get("jsonMode", profile.jsonMode)),
                                profile.revision, profile.updatedAt,
                            ),
                            expected_revision=profile.revision,
                        )
                        next_revision = task_revision + 1
                        runtime = {
                            "revision": updated.revision, "protocol": updated.protocol, "providerLabel": updated.providerLabel,
                            "endpoint": updated.endpoint, "model": updated.model, "backupModel": updated.backupModel,
                            "credentialRef": updated.apiCredentialRef, "jsonMode": updated.jsonMode,
                            "policy": updated.policy, "appliedAt": updated.updatedAt,
                        }
                        database.set_nl_api_runtime_override(
                            job_id, {"taskRevision": next_revision, "runtime": runtime},
                        )
                except Exception:
                    if updated is not None:
                        try:
                            self.profile_store.restore_document(profile_snapshot)
                        except Exception as rollback_error:
                            raise PipelineError("NL API profile rollback failed") from rollback_error
                    raise
                return {"taskRevision": next_revision, "profileRevision": updated.revision, "apiRuntime": runtime, "effective": "next_batch"}
            finally:
                database.close()

    def pending_nl_api_credentials(self, job_id: str) -> NlApiCredentials | None:
        database = StateDatabase.open(self.database_path)
        try:
            pending = database.get_nl_api_runtime_override(job_id)
        finally:
            database.close()
        runtime = pending.get("runtime") if isinstance(pending, dict) else None
        if not isinstance(runtime, dict):
            return None
        try:
            secret = self.credential_store.load(str(runtime["credentialRef"]))
            return NlApiCredentials(
                str(runtime["endpoint"]), str(runtime["model"]), secret,
                runtime.get("backupModel") if isinstance(runtime.get("backupModel"), str) else None,
                str(runtime.get("protocol", "openai_chat_completions")),
                str(runtime.get("providerLabel", "gemini-relay")),
                str(runtime.get("jsonMode", "prompt_only")), int(runtime["revision"]),
                dict(runtime.get("policy", {})) if isinstance(runtime.get("policy"), dict) else None,
            )
        except (CredentialStoreError, KeyError, TypeError, ValueError):
            return None

    def mark_nl_api_runtime_applied(self, job_id: str, credentials: NlApiCredentials) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            pending = database.get_nl_api_runtime_override(job_id)
            runtime = pending.get("runtime") if isinstance(pending, dict) else None
            if isinstance(runtime, dict) and int(runtime.get("revision", 0)) == credentials.revision:
                database.apply_nl_api_runtime_override(job_id, runtime)
        finally:
            database.close()

    def start(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._threads:
                raise PipelineError("job pipeline is already running")
            database = StateDatabase.open(self.database_path)
            try:
                job = database.get_job(job_id)
                if job["status"] != "preparing_workspace":
                    raise PipelineError("only a prepared workspace can start its module pipeline")
                _forced_cuda_start_gate(self, job_id, job)
            finally:
                database.close()
            thread = threading.Thread(target=self._thread_main, args=(job_id, False), daemon=True, name=f"anima-{job_id[:12]}")
            self._threads[job_id] = thread
            try:
                thread.start()
            except Exception:
                self._threads.pop(job_id, None)
                raise

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._threads

    @staticmethod
    def _active_module_status(module_id: object) -> str:
        if module_id == "export":
            return "exporting"
        if module_id == "count_review" or module_id in _RUNTIMES:
            return "running"
        raise PipelineError("only an active worker module can change pause state")

    def pause(self, job_id: str) -> bool:
        """Persist a cooperative pause for the active worker module."""
        with self._lock:
            database = StateDatabase.open(self.database_path)
            try:
                job = database.get_job(job_id)
                if job_id not in self._threads:
                    raise PipelineError("only a running worker module can pause")
                module_id = job["current_module_id"]
                active_status = self._active_module_status(module_id)
                if job["status"] != active_status:
                    raise PipelineError("only a running worker module can pause")
                try:
                    database.pause_active_module(job_id, str(module_id), active_status=active_status)
                except ValueError as exc:
                    raise PipelineError(str(exc)) from exc
            finally:
                database.close()
            return True

    def pause_module(self, job_id: str, module_id: str) -> str:
        """Pause the active module or persist an enabled later-module pre-pause."""
        with self._lock:
            database = StateDatabase.open(self.database_path)
            try:
                job = database.get_job(job_id)
                try:
                    module_order = pipeline_module_ids(int(job["config_schema_version"]))
                    current_module_id = str(job["current_module_id"])
                    target_index = module_order.index(module_id)
                    current_index = module_order.index(current_module_id)
                except (TypeError, ValueError) as exc:
                    raise PipelineError("task has no controllable current module") from exc
                if target_index < current_index:
                    raise PipelineError("only an unfinished module can be paused")
                if target_index == current_index:
                    if job_id not in self._threads:
                        raise PipelineError("only a running worker module can pause")
                    active_status = self._active_module_status(module_id)
                    if job["status"] != active_status:
                        raise PipelineError("only a running worker module can pause")
                    try:
                        database.pause_active_module(job_id, module_id, active_status=active_status)
                    except ValueError as exc:
                        raise PipelineError(str(exc)) from exc
                    return "paused"
                if job["status"] not in {"running", "exporting"}:
                    raise PipelineError("only a running task can pre-pause a later module")
                try:
                    BoundedScheduler(database).pause_future_module(
                        job_id, module_id, total=database.count_module_samples(job_id, module_id),
                    )
                except (SchedulerError, ValueError) as exc:
                    raise PipelineError(str(exc)) from exc
                return "paused"
            finally:
                database.close()

    def resume_module(self, job_id: str, module_id: str) -> str:
        """Cancel a later pre-pause or resume the exact current module."""
        resume_current = False
        with self._lock:
            database = StateDatabase.open(self.database_path)
            try:
                job = database.get_job(job_id)
                try:
                    module_order = pipeline_module_ids(int(job["config_schema_version"]))
                    current_module_id = str(job["current_module_id"])
                    target_index = module_order.index(module_id)
                    current_index = module_order.index(current_module_id)
                except (TypeError, ValueError) as exc:
                    raise PipelineError("task has no controllable current module") from exc
                if target_index < current_index:
                    raise PipelineError("only an unfinished module can be resumed")
                if target_index > current_index:
                    if job["status"] not in {"running", "exporting", "paused", "reviewing"}:
                        raise PipelineError("only a running task can resume a later module")
                    try:
                        BoundedScheduler(database).cancel_future_pause(job_id, module_id)
                    except (SchedulerError, ValueError) as exc:
                        raise PipelineError(str(exc)) from exc
                    return "pending"
                resume_current = True
            finally:
                database.close()
        if resume_current:
            self.resume(job_id)
            return "running"
        raise AssertionError("module resume did not select a control path")

    def resume(self, job_id: str) -> bool:
        """Resume a paused current module, starting a thread only when needed."""
        with self._lock:
            if job_id in self._threads:
                raise PipelineError("task pipeline is still settling; retry resume")
            resuming_token_budget_review = False
            resuming_prepaused_module = False
            prepaused_total: int | None = None
            resume_target_status: str | None = None
            database = StateDatabase.open(self.database_path)
            try:
                job = database.get_job(job_id)
                _frozen_v10_config(job)
                module_id = job["current_module_id"]
                if job["status"] == "reviewing" and module_id == "token_budget":
                    resuming_token_budget_review = True
                    config = _frozen_v10_config(job)
                    if not job_config_supports_token_budget(config.get("schemaVersion")):
                        raise PipelineError("only a Token Budget-capable review can continue to Export")
                    if database.count_unresolved_blocking_issues(job_id):
                        return False
                    # The gate pages issues/samples and reads overlay files.  It must
                    # run before taking SQLite's write lock so cancellation remains
                    # available while this potentially slow verification runs.
                    if not self._token_budget_export_gate(database, job_id, config):
                        return False
                    with database.transaction(immediate=True):
                        current = database.get_job(job_id)
                        if current["status"] in {"cancelling", "cancelled_recoverable"}:
                            return False
                        if current["status"] != "reviewing" or current["current_module_id"] != "token_budget":
                            raise PipelineError("Token Budget review state changed before resume")
                        database.set_job_status(job_id, "exporting", current_module_id="export", resume_status=None)
                    thread = threading.Thread(
                        target=self._thread_main, args=(job_id, True), daemon=True, name=f"anima-{job_id[:12]}",
                    )
                    self._threads[job_id] = thread
                else:
                    resume_target_status = self._active_module_status(module_id)
                    if job["status"] != "paused" or job["resume_status"] != resume_target_status:
                        raise PipelineError("only a paused worker module can resume")
                    try:
                        summary = database.module_summary(job_id, str(module_id))
                        if _is_unstarted_paused_summary(summary):
                            prepaused_total = int(summary["total"])
                            database.resume_prepaused_module(
                                job_id, str(module_id), target_status=resume_target_status, total=prepaused_total,
                            )
                            resuming_prepaused_module = True
                        else:
                            database.resume_paused_module(
                                job_id, str(module_id), target_status=resume_target_status
                            )
                    except ValueError as exc:
                        raise PipelineError(str(exc)) from exc
                    thread = threading.Thread(
                        target=self._thread_main, args=(job_id, True), daemon=True, name=f"anima-{job_id[:12]}",
                    )
                    self._threads[job_id] = thread
            finally:
                database.close()
            try:
                thread.start()
            except Exception:
                self._threads.pop(job_id, None)
                database = StateDatabase.open(self.database_path)
                try:
                    if resuming_token_budget_review:
                        with database.transaction(immediate=True):
                            current = database.get_job(job_id)
                            if current["status"] == "exporting" and current["current_module_id"] == "export":
                                database.set_job_status(job_id, "reviewing", current_module_id="token_budget", resume_status=None)
                    elif resuming_prepaused_module:
                        assert resume_target_status is not None and prepaused_total is not None
                        try:
                            database.restore_prepaused_module(
                                job_id, str(module_id), active_status=resume_target_status, total=prepaused_total,
                            )
                        except (KeyError, ValueError):
                            pass
                    else:
                        assert resume_target_status is not None
                        try:
                            database.pause_active_module(
                                job_id, str(module_id), active_status=resume_target_status
                            )
                        except (KeyError, ValueError):
                            pass
                except Exception:
                    # Preserve the thread start failure even if its best-effort
                    # rollback cannot observe a compatible persisted state.
                    pass
                finally:
                    database.close()
                raise
            return True

    def confirm_count_review(self, job_id: str, *, confirmed: bool) -> bool:
        """Open the confirmed application phase and continue from the Core module."""
        with self._lock:
            settling_thread = self._threads.get(job_id)
        if settling_thread is not None:
            database = StateDatabase.open(self.database_path)
            try:
                settling_job = database.get_job(job_id)
            finally:
                database.close()
            if settling_job["status"] == "reviewing" and settling_job["current_module_id"] == "count_review":
                # CountReviewRunner persists the review state before _thread_main
                # unregisters itself.  Wait outside _lock, because the thread's
                # finally block needs the same lock to finish unregistering.
                settling_thread.join(timeout=1.0)
        with self._lock:
            database = StateDatabase.open(self.database_path)
            thread: threading.Thread | None = None
            try:
                job = database.get_job(job_id)
                _frozen_v10_config(job)
                if job_id in self._threads:
                    if job["status"] == "running" and job["current_module_id"] == "count_review":
                        return False
                    raise PipelineError("task pipeline is still settling its Count Review state")
                CountReviewService(database, job_id).confirm(
                    confirmed=confirmed,
                    expected_config_hash=str(job["config_hash"]),
                )
                current = database.get_job(job_id)
                if current["status"] != "running" or current["current_module_id"] != "count_review":
                    return False
                thread = threading.Thread(
                    target=self._thread_main,
                    args=(job_id, True),
                    daemon=True,
                    name=f"anima-{job_id[:12]}",
                )
                self._threads[job_id] = thread
            finally:
                database.close()
            assert thread is not None
            try:
                thread.start()
            except Exception:
                self._threads.pop(job_id, None)
                database = StateDatabase.open(self.database_path)
                try:
                    current = database.get_job(job_id)
                    if current["status"] == "running" and current["current_module_id"] == "count_review":
                        database.set_job_status(job_id, "reviewing", current_module_id="count_review")
                finally:
                    database.close()
                raise
            return True

    def close(self) -> None:
        """Wait briefly; persistent locks and recovery state deliberately remain intact."""
        with self._lock:
            threads = tuple(self._threads.values())
        for thread in threads:
            thread.join(timeout=5)

    def shutdown(self) -> None:
        """Request recoverable cancellation before the local control plane exits."""
        with self._lock:
            active = tuple(self._threads.items())
        if not active:
            return
        database = StateDatabase.open(self.database_path)
        try:
            for job_id, _ in active:
                try:
                    database.begin_cancellation(job_id)
                    if not database.count_in_flight(job_id):
                        database.settle_cancellation(job_id)
                except ValueError:
                    continue
        finally:
            database.close()
        # Do not end the process while an embedded worker is still writing an
        # overlay artifact. The launcher has a bounded wait and never force-kills.
        for _, thread in active:
            thread.join()

    def _thread_main(self, job_id: str, resume_current: bool = False) -> None:
        try:
            self._run(job_id, resume_current=resume_current)
        finally:
            with self._lock:
                self._threads.pop(job_id, None)

    def _run_module_with_restarts(
        self, database: StateDatabase, scheduler: BoundedScheduler, job_id: str, module_id: str, config: dict[str, object],
    ) -> str:
        """Restart an abnormally exited embedded worker at most twice."""
        while True:
            try:
                return self._run_active_module(database, scheduler, job_id, module_id, config)
            except (StdioJsonlTransportError, WorkerLaunchError):
                if database.module_summary(job_id, module_id)["status"] != "running":
                    raise
                if scheduler.worker_exited(job_id, module_id, abnormal=True):
                    database.return_module_leases(job_id, module_id)
                    continue
                job = database.get_job(job_id)
                if job["status"] in {"paused", "reviewing", "cancelling", "cancelled_recoverable"}:
                    return str(job["status"])
                return str(database.module_summary(job_id, module_id)["status"])

    @staticmethod
    def _token_budget_export_gate(database: StateDatabase, job_id: str, config: dict[str, object]) -> bool:
        section = config.get("tokenBudget")
        if not job_config_supports_token_budget(config.get("schemaVersion")) or not isinstance(section, dict) or section.get("enabled") is not True:
            return True
        max_tokens = section.get("maxTokens")
        caption_format = config.get("captionFormat")
        job = database.get_job(job_id)
        if type(max_tokens) is not int or max_tokens < 1 or not isinstance(caption_format, dict) or not isinstance(job["overlay_root"], str):
            raise PipelineError("frozen Token Budget export gate configuration is invalid")
        layout = OverlayLayout.open_existing(str(job["overlay_root"]), job_id)
        writer = TokenBudgetOverlayWriter(database, layout, WorkingAnnotationView(BaselineView(layout.dataset_root), layout), job_id)
        failed = False
        blocking_samples: set[int] = set()
        issue_after_sample_id: int | None = None
        issue_after_issue_id: str | None = None
        while True:
            issues = database.page_issues(
                job_id,
                after_sample_id=issue_after_sample_id,
                after_issue_id=issue_after_issue_id,
                limit=500,
            )
            if not issues:
                break
            for issue in issues:
                if (
                    issue["module_id"] == "token_budget"
                    and issue["resolved_at"] is None
                    and bool(issue["blocking"])
                ):
                    sample_id = int(issue["sample_id"])
                    sample_state = database.get_sample_state(job_id, sample_id)
                    if sample_state["status"] not in {"failed", "skipped"}:
                        blocking_samples.add(sample_id)
            issue_after_sample_id = int(issues[-1]["sample_id"])
            issue_after_issue_id = str(issues[-1]["issue_id"])
        cursor: int | None = None
        target_clause, target_parameters = repair_target_clause(
            database.connection, job_id, "export",
        )
        while True:
            cursor_clause = "" if cursor is None else " AND s.sample_id>?"
            parameters: list[object] = [job_id]
            if cursor is not None:
                parameters.append(cursor)
            parameters.extend(target_parameters)
            parameters.append(500)
            page = list(database.connection.execute(
                """SELECT s.* FROM samples AS s
                   WHERE s.job_id=? AND s.in_processing_scope=1"""
                + cursor_clause + target_clause + " ORDER BY s.sample_id LIMIT ?",
                parameters,
            ))
            if not page:
                break
            cursor = int(page[-1]["sample_id"])
            for row in page:
                if not bool(row["in_processing_scope"]):
                    continue
                sample_id = int(row["sample_id"])
                sample_state = database.get_sample_state(job_id, sample_id)
                if sample_state["status"] in {"failed", "skipped"}:
                    continue
                if sample_id in blocking_samples:
                    failed = True
                    continue
                try:
                    writer.record_for_export(sample_id=sample_id, annotation_key=str(row["annotation_key"]), caption_format=caption_format, max_tokens=max_tokens)
                except TokenBudgetOverlayError:
                    failed = True
                    database.upsert_issue(SampleIssue(
                        issueId=hashlib.sha256(f"{job_id}\0{sample_id}\0token_budget\0token_budget_export_gate_failed".encode("utf-8")).hexdigest(),
                        jobId=job_id, sampleId=sample_id, relativeImagePath=str(row["relative_image_path"]),
                        moduleId="token_budget", code="token_budget_export_gate_failed", severity="error", blocking=True,
                        retriable=True, message="Token Budget verification failed before Export", attempt=0,
                        repairStartModule="token_budget",
                    ))
        if failed:
            with database.transaction(immediate=True):
                current = database.get_job(job_id)
                if current["status"] == "running" and current["current_module_id"] == "token_budget":
                    database.set_job_status(job_id, "reviewing", current_module_id="token_budget")
                elif current["status"] == "exporting" and current["current_module_id"] == "export":
                    database.set_job_status(job_id, "reviewing", current_module_id="token_budget")
            return False
        return True

    def _run(self, job_id: str, *, resume_current: bool = False) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            job = database.get_job(job_id)
            config = _frozen_v10_config(job)
            scheduler = BoundedScheduler(database)
            try:
                modules = pipeline_module_ids(int(job["config_schema_version"]))
            except ValueError as exc:
                raise PipelineError(str(exc)) from exc
            if resume_current:
                try:
                    modules = modules[modules.index(str(job["current_module_id"])):]
                except ValueError as exc:
                    raise PipelineError("interrupted job has no recoverable current module") from exc
            for module_id in modules:
                if module_id == "export" and database.count_unresolved_blocking_issues(job_id, exclude_module_id="export"):
                    # Export never bypasses unresolved issues from upstream modules.
                    current = database.get_job(job_id)["current_module_id"]
                    database.set_job_status(job_id, "reviewing", current_module_id=str(current or "dropout"))
                    return
                if module_id == "export" and not self._token_budget_export_gate(database, job_id, config):
                    return
                nl_config = config.get("nl")
                moss_nl = (
                    isinstance(nl_config, dict)
                    and nl_config.get("enabled") is True
                    and nl_config.get("promptVersion") == NL_MOSS_PROMPT_VERSION
                )
                section = config.get("countReview" if module_id == "count_review" else "tokenBudget" if module_id == "token_budget" else module_id)
                if module_id == "export":
                    if not isinstance(section, dict) or section.get("format") not in {"json", "flat_txt", "both"}:
                        raise PipelineError("frozen export configuration is invalid")
                    enabled = True
                else:
                    if not isinstance(section, dict) or type(section.get("enabled")) is not bool:
                        raise PipelineError(f"frozen {module_id} configuration is invalid")
                    enabled = section["enabled"]
                    if module_id == "dropout":
                        children = (section.get("artist"), section.get("quality"), section.get("appearanceNl"))
                        if any(not isinstance(child, dict) or type(child.get("enabled")) is not bool for child in children):
                            raise PipelineError("frozen dropout subfeature configuration is invalid")
                        # MOSS sidecars are standalone captions and must not be changed by NL dropout.
                        enabled = enabled and not moss_nl and any(child["enabled"] for child in children)
                state = scheduler.start_module(job_id, module_id, enabled=enabled)
                if state == "paused":
                    return
                if state in FINAL_MODULE_STATUSES:
                    # Resume after a crash that landed between finish_module() and the
                    # next module: this module is already done, so move on instead of
                    # failing the whole task. Export still has to commit its artifacts.
                    finished = state
                elif state == "running":
                    summary = database.module_summary(job_id, module_id)
                    if module_id != "export" and int(summary["total"]) == 0:
                        finished = scheduler.finish_module(job_id, module_id)
                    else:
                        finished = self._run_module_with_restarts(database, scheduler, job_id, module_id, config)
                else:
                    finished = state
                if finished in {"paused", "reviewing", "cancelling"}:
                    if finished == "cancelling" and not database.count_in_flight(job_id):
                        database.settle_cancellation(job_id)
                    return
                if module_id == "export":
                    if finished == "completed_with_issues":
                        database.set_job_status(job_id, "reviewing", current_module_id="export")
                        return
                    if finished != "completed":
                        raise PipelineError(f"Export module did not complete: {finished}")
                    job = database.get_job(job_id)
                    if job["status"] != "exporting":
                        database.set_job_status(job_id, "exporting", current_module_id="export")
                    layout = OverlayLayout.open_existing(str(job["overlay_root"]), job_id)
                    ExportCommitCoordinator(database, layout, job_id=job_id).commit()
                    database.resolve_repaired_parent_issues(job_id)
                    return
        except Exception:
            job = database.get_job(job_id)
            if job["status"] not in {"cancelled_recoverable", "failed", "succeeded", "succeeded_with_issues"}:
                database.set_job_status(job_id, "failed", current_module_id=str(job["current_module_id"] or "workspace"))
            raise
        finally:
            database.close()
