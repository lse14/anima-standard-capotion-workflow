from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query

from .annotation_restore import AnnotationRestoreCoordinator, AnnotationRestoreError
from .api_context import ControlPlaneContext, bad_request, conflict, not_found
from .api_models import (
    _ConfirmBody,
    _ExportRecommitBody,
    _PinBody,
    _PreflightBody,
    _WorkspaceBody,
    parse_create_job_body,
)
from .commit_journal import CommitJournalError, load as load_journal
from .contracts import job_config_supports_ocr_device, pipeline_module_ids
from .dataset_workspace import DatasetWorkspace
from .db import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, StateDatabase
from .export_recommit import ExportRecommitError, ExportRecommitService
from .export_summary import build_export_summary
from .job_preflight import JobPreflightError
from .lifecycle import JobLifecycle, JobLifecycleError
from .locks import DatasetClaimConflict, DatasetLockError
from .nl_runner import pending_api_decisions
from .ocr_runtime_binding import OcrExecutionError, read_runtime_binding
from .overlay import OverlayError, OverlayLayout
from .path_safety import PathSafetyError
from .pipeline import PipelineError
from .repair import RepairPreparationError
from .scheduler import SchedulerError


def _ocr_runtime_snapshot(job: object, config: object) -> dict[str, object] | None:
    if not isinstance(job, dict) or not isinstance(config, dict) or not job_config_supports_ocr_device(config.get("schemaVersion")):
        return None
    ocr = config.get("ocr")
    if not isinstance(ocr, dict) or ocr.get("enabled") is not True:
        return None
    requested = ocr.get("device") if ocr.get("device") in {"auto", "cuda", "cpu"} else None
    empty = {
        "runtimeId": None,
        "gpuName": None,
        "totalVramBytes": None,
        "requestedDevice": requested,
        "observedDevice": None,
        "recommended": None,
        "effective": None,
        "startupReason": None,
    }
    if isinstance(job, dict):
        raw_evidence = job.get("runtime_evidence_json")
        if isinstance(raw_evidence, str) and raw_evidence:
            try:
                evidence = json.loads(raw_evidence)
            except (TypeError, json.JSONDecodeError):
                evidence = None
            persisted = evidence.get("ocr") if isinstance(evidence, dict) else None
            if isinstance(persisted, dict):
                return persisted
    overlay_root = job.get("overlay_root")
    if not isinstance(overlay_root, str) or not overlay_root:
        return {"availability": "pending", **empty}
    try:
        layout = OverlayLayout.open_existing(overlay_root, str(job["job_id"]))
        path = layout.resource_path("ocr-runtime-binding-v1.json")
        if not path.is_file():
            return {"availability": "pending", **empty}
        binding = read_runtime_binding(path)
    except (KeyError, OverlayError, OcrExecutionError, OSError, ValueError):
        return {"availability": "unavailable", **empty, "startupReason": "binding_invalid"}
    return {
        "availability": "available",
        "runtimeId": binding.runtimeId,
        "gpuName": binding.gpuName,
        "totalVramBytes": binding.totalVramBytes,
        "requestedDevice": binding.requestedDevice,
        "observedDevice": binding.observedDevice,
        "recommended": {
            "textDetLimitSideLen": binding.recommended.textDetLimitSideLen,
            "textBatchSize": binding.recommended.textBatchSize,
        },
        "effective": {
            "textDetLimitSideLen": binding.effectiveTextDetLimitSideLen,
            "textBatchSize": binding.effectiveTextBatchSize,
        },
        "startupReason": binding.startupReason,
    }


def _dropout_runtime_snapshot(job: object) -> dict[str, object] | None:
    if not isinstance(job, dict):
        return None
    raw_evidence = job.get("runtime_evidence_json")
    if not isinstance(raw_evidence, str) or not raw_evidence:
        return None
    try:
        evidence = json.loads(raw_evidence)
    except (TypeError, json.JSONDecodeError):
        return None
    value = evidence.get("dropout") if isinstance(evidence, dict) else None
    return value if isinstance(value, dict) else None


def _nl_api_runtime_snapshot(job: object, config: object) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Expose the active and queued connection-only revisions without secrets."""
    frozen: dict[str, object] | None = None
    if isinstance(config, dict) and isinstance(config.get("nl"), dict):
        value = config["nl"].get("apiRuntime")
        if isinstance(value, dict):
            frozen = dict(value)
    active = frozen
    pending: dict[str, object] | None = None
    if isinstance(job, dict) and isinstance(job.get("runtime_evidence_json"), str):
        try:
            evidence = json.loads(str(job["runtime_evidence_json"]))
        except (TypeError, json.JSONDecodeError):
            evidence = None
        if isinstance(evidence, dict):
            persisted = evidence.get("_nlApiRuntime")
            queued = evidence.get("_nlApiOverride")
            if isinstance(persisted, dict):
                active = dict(persisted)
            if isinstance(queued, dict) and isinstance(queued.get("runtime"), dict):
                pending = dict(queued["runtime"])
    return active, pending


def _nl_api_task_revision(job: object) -> int:
    """Expose the monotonic task-local API revision without credential data."""
    if not isinstance(job, dict) or not isinstance(job.get("runtime_evidence_json"), str):
        return 0
    try:
        evidence = json.loads(str(job["runtime_evidence_json"]))
    except (TypeError, json.JSONDecodeError):
        return 0
    if not isinstance(evidence, dict):
        return 0
    pending = evidence.get("_nlApiOverride")
    if isinstance(pending, dict) and type(pending.get("taskRevision")) is int:
        return int(pending["taskRevision"])
    value = evidence.get("_nlApiTaskRevision")
    return int(value) if type(value) is int and value >= 0 else 0


def _preflight_progress_response(database: StateDatabase, job_id: str) -> dict[str, object]:
    """Map the durable preflight row to the public polling payload."""
    progress = database.get_preflight_progress(job_id)
    job = database.get_job(job_id)
    raw_summary = progress.get("summary_json")
    summary: dict[str, object] | None = None
    if raw_summary:
        try:
            parsed = json.loads(str(raw_summary))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("preflight summary is invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("preflight summary must be an object")
        summary = parsed
    return {
        "jobId": job_id,
        "status": str(progress["status"]),
        "checkedFiles": int(progress["checked_files"]),
        "totalFiles": int(progress["total_files"]),
        "error": progress["error"],
        "summary": summary,
        "updatedAt": progress["updated_at"],
        # Preflight progress is intentionally independent from the event ring.
        "lastEventId": int(job["last_event_id"]),
    }


def _annotation_backup_path(job_id: str, dataset: Path) -> Path | None:
    """Locate the ZIP an Export commit recorded, for both workspace layouts.

    Export writes into the dataset's workspace beside the dataset
    (``DatasetWorkspace.backup_zip``) for new overlays and into the legacy
    ``.<dataset>.anima-backups`` sibling for older ones, so both are probed.
    """
    candidates: list[Path] = []
    try:
        candidates.append(DatasetWorkspace.for_dataset(dataset, must_exist=False).backup_zip(job_id))
    except (OSError, ValueError):
        pass
    candidates.append(dataset.parent / f".{dataset.name}.anima-backups" / f"{job_id}.zip")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_jobs_router(context: ControlPlaneContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/jobs")
    def list_jobs(
        afterCreatedAt: str | None = Query(default=None, max_length=64),
        afterJobId: str | None = Query(default=None, max_length=128),
        limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    ) -> dict[str, object]:
        """F40: bounded newest-first task list so a refreshed page can find its task again."""
        if (afterCreatedAt is None) != (afterJobId is None):
            raise bad_request(ValueError("both task-list cursor parts are required"))
        database = StateDatabase.open(context.database_path)
        try:
            predicate = "" if afterCreatedAt is None else " WHERE (created_at<? OR (created_at=? AND job_id<?))"
            cursor: list[object] = [] if afterCreatedAt is None else [afterCreatedAt, afterCreatedAt, afterJobId]
            rows = list(database.connection.execute(
                "SELECT job_id,status,current_module_id,dataset_root,sample_count,pinned,created_at,finished_at"
                f" FROM jobs{predicate}{' AND ' if predicate else ' WHERE '}NOT EXISTS "
                "(SELECT 1 FROM repair_jobs WHERE repair_jobs.repair_job_id=jobs.job_id) "
                "ORDER BY created_at DESC,job_id DESC LIMIT ?", [*cursor, limit],
            ))
            jobs = [{
                "jobId": row["job_id"], "status": row["status"], "currentModuleId": row["current_module_id"],
                "datasetRoot": row["dataset_root"], "sampleCount": row["sample_count"],
                "pinned": bool(row["pinned"]), "createdAt": row["created_at"], "finishedAt": row["finished_at"],
            } for row in rows]
            exhausted = len(jobs) < limit
            return {
                "jobs": jobs,
                "nextAfterCreatedAt": None if exhausted else jobs[-1]["createdAt"],
                "nextAfterJobId": None if exhausted else jobs[-1]["jobId"],
            }
        finally:
            database.close()

    @router.get("/api/jobs/{job_id}")
    def job_snapshot(
        job_id: str,
        afterEventId: int = Query(default=0, ge=0),
        issueAfterSampleId: int = Query(default=0, ge=0),
        issueAfterIssueId: str | None = Query(default=None, max_length=128),
        limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    ) -> dict[str, object]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                job = database.get_job(job_id)
            except KeyError as exc:
                raise not_found(exc) from exc
            events = database.event_page(job_id, afterEventId, limit)
            issues = database.page_issues(
                job_id, after_sample_id=issueAfterSampleId or None, after_issue_id=issueAfterIssueId, limit=limit,
            )
            parent_job_id = database.repair_parent_job_id(job_id)
            repair_children = [{
                "jobId": row["job_id"], "status": row["status"], "currentModuleId": row["current_module_id"],
                "sampleCount": row["sample_count"], "targetCount": row["target_count"],
                "createdAt": row["created_at"], "finishedAt": row["finished_at"],
            } for row in database.repair_children(job_id)]
            export_summary = None
            repair_preview = None
            ocr_runtime = None
            dropout_runtime = None
            try:
                config = json.loads(str(job["config_json"]))
                export_config = config.get("export") if isinstance(config, dict) else None
                if not isinstance(export_config, dict) or not isinstance(export_config.get("format"), str):
                    raise ValueError("frozen export configuration is invalid")
                try:
                    export_module = database.module_summary(job_id, "export")
                except KeyError:
                    export_module = None
                journal = None
                if isinstance(job["overlay_root"], str):
                    try:
                        journal = load_journal(OverlayLayout.open_existing(str(job["overlay_root"]), job_id))
                    except (OverlayError, PathSafetyError):
                        # A committed or discarded task no longer owns its overlay; that is a
                        # normal terminal state, not a control-plane failure.
                        journal = None
                export_summary = build_export_summary(
                    job_id=job_id, format_value=export_config["format"], job_status=job["status"],
                    module_summary=export_module, journal=journal,
                    diagnostics=database.module_diagnostics(job_id, "export"),
                ).to_dict()
                if job["status"] != "succeeded_with_issues":
                    target_count, nl_upper_bound = database.repair_candidate_summary(job_id)
                    nl_config = config.get("nl") if isinstance(config, dict) else None
                    api_enabled = isinstance(nl_config, dict) and nl_config.get("enabled") is True and nl_config.get("apiEnabled") is True
                    repair_preview = {
                        "eligibleTargetCount": target_count,
                        "estimatedApiRequests": nl_upper_bound if api_enabled else 0,
                    }
                ocr_runtime = _ocr_runtime_snapshot(dict(job), config)
                dropout_runtime = _dropout_runtime_snapshot(dict(job))
                api_runtime, api_runtime_pending = _nl_api_runtime_snapshot(dict(job), config)
            except (CommitJournalError, OSError, UnicodeError, ValueError) as exc:
                raise bad_request(exc) from exc
            return {
                "job": {
                    "jobId": job["job_id"], "status": job["status"], "currentModuleId": job["current_module_id"],
                    "configSchemaVersion": job["config_schema_version"],
                    "parentJobId": parent_job_id,
                    "lastEventId": job["last_event_id"], "apiBudgetExtra": job["api_budget_extra"],
                    "apiBudgetRevision": job["api_budget_revision"], "pinned": bool(job["pinned"]),
                    # F41: the frozen JobState fields (ROADMAP.md:1233-1246).
                    "configHash": job["config_hash"], "manifestSchemaVersion": job["manifest_schema_version"],
                    "createdAt": job["created_at"], "startedAt": job["started_at"],
                    "cancelRequestedAt": job["cancel_requested_at"], "finishedAt": job["finished_at"],
                    "frozenConfig": config,
                    "apiRuntime": api_runtime,
                    "apiRuntimePending": api_runtime_pending,
                    "apiConfigRevision": _nl_api_task_revision(dict(job)),
                },
                "moduleOrder": list(pipeline_module_ids(int(job["config_schema_version"]))),
                "modules": [dict(row) for row in database.module_summaries(job_id)],
                "diagnostics": [dict(row) for row in database.module_diagnostics(job_id, "nl")],
                "captionDiagnostics": [dict(row) for row in database.module_diagnostics(job_id, "caption")],
                "ocrDiagnostics": [
                    *[dict(row) for row in database.module_diagnostics(job_id, "ocr")],
                    *[
                        dict(row)
                        for row in database.module_diagnostics(job_id, "nl")
                        if row["code"] == "nl_ocr_context_omitted_too_large"
                    ],
                ],
                "ocrRuntime": ocr_runtime,
                "dropoutRuntime": dropout_runtime,
                "events": [dict(row) for row in events],
                "issues": [dict(row) for row in issues],
                "exportSummary": export_summary,
                "repairPreview": repair_preview,
                "repairChildren": repair_children,
                "nlPendingApiDecisions": pending_api_decisions(database, job_id),
                "nextAfterEventId": int(events[-1]["event_id"]) if events else afterEventId,
                "nextIssueAfterSampleId": int(issues[-1]["sample_id"]) if issues else issueAfterSampleId,
                "nextIssueAfterIssueId": str(issues[-1]["issue_id"]) if issues else issueAfterIssueId,
                "snapshotRequired": database.event_snapshot_required(job_id, afterEventId),
            }
        finally:
            database.close()

    @router.post("/api/jobs/preflight")
    def preflight_job(body: _PreflightBody) -> dict[str, object]:
        try:
            request = parse_create_job_body(body.model_dump())
            summary = context.preparation_service.preflight(request.config, ocr_execution=request.ocrExecution)
        except (JobPreflightError, ValueError) as exc:
            raise bad_request(exc) from exc
        return {
            "jobId": summary.jobId, "sampleCount": summary.sampleCount, "inScopeCount": summary.inScopeCount,
            "outOfScopeCount": summary.outOfScopeCount, "nonblankTxtCount": summary.nonblankTxtCount,
            "nonblankJsonCount": summary.nonblankJsonCount, "configHash": summary.configHash,
            "replaceIndex": summary.replaceIndex, "resources": summary.resources,
            # F39: everything ROADMAP.md:1224 requires the preflight page to display.
            "blankTxtCount": summary.blankTxtCount, "blankJsonCount": summary.blankJsonCount,
            "annotationKeyCollisionCount": summary.annotationKeyCollisionCount, "imageIssueCount": summary.imageIssueCount,
            "projection": summary.projection, "estimate": summary.estimate, "api": summary.api,
        }

    @router.post("/api/jobs/preflight/async")
    def start_async_preflight(body: _PreflightBody) -> dict[str, object]:
        """Queue preflight work and return a durable, event-independent cursor."""
        try:
            request = parse_create_job_body(body.model_dump())
            job_id = context.preparation_service.start_preflight(
                request.config, ocr_execution=request.ocrExecution,
            )
            database = StateDatabase.open(context.database_path)
            try:
                progress = database.get_preflight_progress(job_id)
                job = database.get_job(job_id)
                # The POST acknowledges queueing, even when a tiny/empty dataset
                # finishes before the HTTP response is serialized.  Clients then
                # use GET to observe the authoritative status.
                return {
                    "jobId": job_id,
                    "status": "queued",
                    "checkedFiles": 0,
                    "totalFiles": int(progress["total_files"]),
                    "error": None,
                    "summary": None,
                    "updatedAt": progress["updated_at"],
                    "lastEventId": int(job["last_event_id"]),
                }
            finally:
                database.close()
        except (JobPreflightError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.get("/api/jobs/preflight/{job_id}/progress")
    def get_async_preflight_progress(job_id: str) -> dict[str, object]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                return _preflight_progress_response(database, job_id)
            except KeyError as exc:
                raise not_found(exc) from exc
            except ValueError as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/preflight/{job_id}/cancel")
    def cancel_async_preflight(job_id: str) -> dict[str, object]:
        try:
            cancelled = context.preparation_service.cancel_preflight(job_id)
            return {"jobId": job_id, "cancelled": cancelled}
        except KeyError as exc:
            raise not_found(exc) from exc
        except JobPreflightError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/confirm-workspace")
    def confirm_workspace(job_id: str, body: _WorkspaceBody) -> dict[str, object]:
        try:
            return context.preparation_service.confirm_workspace(
                job_id,
                confirmed=body.confirmed,
                confirmed_rebuild=body.confirmedRebuild,
            )
        except KeyError as exc:
            raise not_found(exc) from exc
        except DatasetClaimConflict as exc:
            raise conflict(ValueError(
                f"Dataset is claimed by task {exc.claiming_job_id}. Select it under Recent tasks: "
                "Recover keeps its progress and continues to hold the dataset; "
                "Discard deletes its overlay and releases the dataset."
            )) from exc
        except DatasetLockError as exc:
            raise conflict(exc) from exc
        except JobPreflightError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/start")
    def start_pipeline(job_id: str) -> dict[str, object]:
        try:
            context.pipeline_service.start(job_id)
            return {"jobId": job_id, "started": True}
        except KeyError as exc:
            raise not_found(exc) from exc
        except PipelineError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/pause")
    def pause_pipeline(job_id: str) -> dict[str, str]:
        try:
            context.pipeline_service.pause(job_id)
            return {"status": "paused"}
        except KeyError as exc:
            raise not_found(exc) from exc
        except PipelineError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/resume")
    def resume_pipeline(job_id: str) -> dict[str, str]:
        try:
            context.pipeline_service.resume(job_id)
            return {"status": "running"}
        except KeyError as exc:
            raise not_found(exc) from exc
        except PipelineError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/modules/{module_id}/pause")
    def pause_module(job_id: str, module_id: str) -> dict[str, object]:
        try:
            context.pipeline_service.pause_module(job_id, module_id)
            return job_snapshot(job_id, afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=DEFAULT_PAGE_SIZE)
        except KeyError as exc:
            raise not_found(exc) from exc
        except PipelineError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/modules/{module_id}/resume")
    def resume_module(job_id: str, module_id: str) -> dict[str, object]:
        try:
            context.pipeline_service.resume_module(job_id, module_id)
            return job_snapshot(job_id, afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=DEFAULT_PAGE_SIZE)
        except KeyError as exc:
            raise not_found(exc) from exc
        except PipelineError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/repair")
    def repair_job(job_id: str) -> dict[str, object]:
        released_parent_lock: str | None = None
        try:
            database = StateDatabase.open(context.database_path)
            try:
                parent = database.get_job(job_id)
                target_count, nl_upper_bound = database.repair_candidate_summary(job_id)
                config = json.loads(str(parent["config_json"]))
                nl = config.get("nl") if isinstance(config, dict) else None
                estimated_requests = nl_upper_bound if isinstance(nl, dict) and nl.get("enabled") is True and nl.get("apiEnabled") is True else 0
            finally:
                database.close()
            released_parent_lock = context.release_lock_for_repair(job_id)
            result = context.repair_service.prepare(job_id)
            context.pipeline_service.start(result.repairJobId)
            return {
                "jobId": result.repairJobId, "parentJobId": result.parentJobId,
                "targetCount": target_count, "preparedTargetCount": result.targetCount,
                "estimatedApiRequests": estimated_requests, "started": True,
            }
        except KeyError as exc:
            raise not_found(exc) from exc
        except (JobPreflightError, RepairPreparationError, PipelineError, SchedulerError, ValueError) as exc:
            if released_parent_lock:
                context.restore_lock_after_repair_failure(job_id, released_parent_lock)
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/recover")
    def recover_job(job_id: str, body: _ConfirmBody) -> dict[str, object]:
        try:
            return context.pipeline_service.recover_job(job_id, confirmed=body.confirmed)
        except KeyError as exc:
            raise not_found(exc) from exc
        except (PipelineError, SchedulerError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.get("/api/jobs/{job_id}/export/recommit")
    def preview_export_recommit(job_id: str) -> dict[str, object]:
        try:
            return ExportRecommitService(context.database_path).preview(job_id)
        except KeyError as exc:
            raise not_found(exc) from exc
        except (ExportRecommitError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/export/recommit")
    def recommit_export(job_id: str, body: _ExportRecommitBody) -> dict[str, object]:
        try:
            return ExportRecommitService(context.database_path).recommit(
                job_id,
                confirmed=body.confirmed,
                inventory_fingerprint=body.inventoryFingerprint,
            )
        except KeyError as exc:
            raise not_found(exc) from exc
        except (ExportRecommitError, PipelineError, ValueError) as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, str]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                database.begin_cancellation(job_id)
                if not database.count_in_flight(job_id):
                    database.settle_cancellation(job_id)
                return {"status": str(database.get_job(job_id)["status"])}
            except KeyError as exc:
                raise not_found(exc) from exc
            except ValueError as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.put("/api/jobs/{job_id}/pin")
    def set_job_pin(job_id: str, body: _PinBody) -> dict[str, bool]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                database.set_pinned(job_id, body.pinned)
                return {"pinned": body.pinned}
            except KeyError as exc:
                raise not_found(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/{job_id}/discard")
    def discard_job(job_id: str, body: _ConfirmBody) -> dict[str, object]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                job = database.get_job(job_id)
                lifecycle = JobLifecycle(database)
                if body.confirmed:
                    lifecycle.ensure_delete_allowed(job_id)
                if body.confirmed and job["status"] == "discarded":
                    context.preparation_service.release_lock_for_discard(job_id)
                    database.delete_job_control_record(job_id)
                    return {"jobId": job_id, "overlayDeleted": False}
                result = lifecycle.discard(job_id, confirmed=body.confirmed)
                context.preparation_service.release_lock_for_discard(job_id)
                database.delete_job_control_record(job_id)
                return {"jobId": result.jobId, "overlayDeleted": result.overlayDeleted}
            except KeyError as exc:
                raise not_found(exc) from exc
            except (JobLifecycleError, JobPreflightError, ValueError) as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/{job_id}/delete-history")
    def delete_completed_history(job_id: str, body: _ConfirmBody) -> dict[str, object]:
        database = StateDatabase.open(context.database_path)
        try:
            try:
                lifecycle = JobLifecycle(database)
                lifecycle.ensure_history_delete_allowed(job_id, confirmed=body.confirmed)
                if database.repair_parent_job_id(job_id) is None:
                    context.preparation_service.release_lock_for_history_deletion(job_id)
                else:
                    context.repair_service.release_lock_for_history_deletion(job_id)
                result = lifecycle.delete_history(job_id, confirmed=body.confirmed)
                return {
                    "jobId": result.jobId,
                    "deletedControlRecord": result.deletedControlRecord,
                    "preserved": {"dataset": True, "exports": True, "backup": True},
                }
            except KeyError as exc:
                raise not_found(exc) from exc
            except (JobLifecycleError, JobPreflightError, DatasetLockError, ValueError) as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/{job_id}/restore-original-annotations")
    def restore_original_annotations(job_id: str, body: _ConfirmBody) -> dict[str, object]:
        if not body.confirmed:
            raise bad_request(ValueError("annotation restoration requires explicit confirmation"))
        database = StateDatabase.open(context.database_path)
        try:
            try:
                job = database.get_job(job_id)
            except KeyError as exc:
                raise not_found(exc) from exc
            dataset = Path(str(job["dataset_root"]))
            backup = _annotation_backup_path(job_id, dataset)
            if job["status"] not in {"succeeded", "succeeded_with_issues"} or backup is None:
                raise bad_request(ValueError("only a completed task with its annotation backup can restore originals"))
            # A committed task no longer owns an overlay, so restoration carries its own
            # one-shot overlay for the commit journal (ROADMAP.md:1056).
            try:
                layout = OverlayLayout.create(dataset, f"restore-{job_id}")
            except (OverlayError, OSError, ValueError) as exc:
                raise bad_request(exc) from exc
            try:
                result = AnnotationRestoreCoordinator(layout).restore(backup)
            except (AnnotationRestoreError, OSError, ValueError) as exc:
                raise bad_request(exc) from exc
            try:
                layout.discard()
            except (OverlayError, OSError):
                pass  # The switch already committed; a retained journal is diagnostic only.
            return {"jobId": job_id, "restored": result.restored, "backupZip": str(result.backupZip)}
        finally:
            database.close()

    return router
