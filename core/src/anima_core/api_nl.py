from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query

from .api_context import ControlPlaneContext, bad_request, conflict, not_found
from .api_models import (
    _AmountBody,
    _ConfirmBody,
    _NlManualRetryBatchBody,
    _NlManualRetryBody,
    _NlManualWriteBody,
    _NlModelDiscoveryBody,
    _NlPromptPresetBody,
    _ProfileCopyBody,
    _NlTestMessageBody,
    _NlApiConfigBody,
    _ProfileBody,
    _SecretBody,
)
from .credentials import CredentialStoreError, DpapiCredentialStore
from .db import StateDatabase
from .nl_profiles import NlApiProfile, NlProfileError, default_nl_prompt
from .nl_prompt_presets import (
    NlPromptPreset,
    PromptPresetConflictError,
    PromptPresetNotFoundError,
    PromptPresetValidationError,
)
from .nl_manual_review import NlManualWriteService
from .repair import RepairPreparationError
from .pipeline_dispatch import PipelineError
from .scheduler import BoundedScheduler, SchedulerError


def _profile_response(profile: NlApiProfile, *, credentials: DpapiCredentialStore) -> dict[str, object]:
    # Loading verifies current-user availability without exposing the secret.
    try:
        credentials.load(profile.apiCredentialRef)
        has_credential = True
    except CredentialStoreError:
        has_credential = False
    return {
        "profileId": profile.profileId,
        "name": profile.name or profile.profileId,
        "protocol": profile.protocol,
        "providerLabel": profile.providerLabel,
        "endpoint": profile.endpoint,
        "model": profile.model,
        "backupModel": profile.backupModel,
        "credentialRef": profile.apiCredentialRef,
        # Legacy aliases remain response-only so existing local clients can
        # migrate without reintroducing prompt data into the profile file.
        "apiCredentialRef": profile.apiCredentialRef,
        "jsonMode": profile.jsonMode,
        "policy": profile.policy,
        "apiPolicy": profile.policy,
        "revision": profile.revision,
        "updatedAt": profile.updatedAt,
        "systemPrompt": "",
        "hasCredential": has_credential,
    }


def _preset_response(preset: NlPromptPreset, *, detail: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "presetId": preset.presetId,
        "name": preset.name,
        "type": preset.type,
        "builtIn": preset.builtIn,
        "sha256": preset.sha256,
        "sizeBytes": preset.sizeBytes,
    }
    if detail:
        # Keep basePrompt in the response for older local clients; both names
        # resolve to the same saved prompt source.
        result["promptText"] = preset.promptText
        result["basePrompt"] = preset.promptText
    return result


def _resolved_diagnostic_key(context: ControlPlaneContext, body: _NlModelDiscoveryBody | _NlTestMessageBody) -> str | None:
    transient = body.apiKey.get_secret_value().strip() if body.apiKey is not None else ""
    if transient:
        return transient
    try:
        return context.credential_store.load(body.apiCredentialRef)
    except CredentialStoreError:
        return None


def _credential_unavailable(*, test_message: bool) -> dict[str, object]:
    if test_message:
        return {
            "ok": False,
            "latencyMs": 0,
            "actualModel": None,
            "replyText": None,
            "usage": None,
            "errorCode": "credential_unavailable",
            "errorReason": "credential unavailable",
        }
    return {
        "ok": False,
        "latencyMs": 0,
        "models": [],
        "errorCode": "credential_unavailable",
        "errorReason": "credential unavailable",
    }


def build_nl_router(context: ControlPlaneContext) -> APIRouter:
    router = APIRouter()

    @router.get("/api/nl/default-prompt")
    def nl_default_prompt(
        promptVersion: str | None = None,
    ) -> dict[str, object]:
        """F25: the single source of truth behind the editor default and 'restore default'."""
        try:
            return default_nl_prompt(prompt_version=promptVersion or "nl-default-prompt-v2")
        except NlProfileError as exc:
            raise bad_request(exc) from exc

    @router.get("/api/nl/prompt-presets")
    def list_prompt_presets(includeMoss: bool = Query(default=False)) -> dict[str, object]:
        try:
            return {"presets": list(context.prompt_preset_store.list_summaries(include_moss=includeMoss))}
        except PromptPresetValidationError as exc:
            raise bad_request(exc) from exc

    @router.get("/api/nl/prompt-presets/{preset_id}")
    def get_prompt_preset(preset_id: str) -> dict[str, object]:
        try:
            return _preset_response(context.prompt_preset_store.get(preset_id), detail=True)
        except PromptPresetNotFoundError as exc:
            raise not_found(exc) from exc
        except PromptPresetValidationError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/nl/prompt-presets")
    def create_prompt_preset(body: _NlPromptPresetBody) -> dict[str, object]:
        try:
            return _preset_response(
                context.prompt_preset_store.create(
                    name=body.name,
                    preset_type=body.type,
                    prompt_text=body.promptText,
                    base_prompt=body.basePrompt,
                ),
                detail=True,
            )
        except PromptPresetValidationError as exc:
            raise bad_request(exc) from exc

    @router.put("/api/nl/prompt-presets/{preset_id}")
    def update_prompt_preset(preset_id: str, body: _NlPromptPresetBody) -> dict[str, object]:
        try:
            return _preset_response(
                context.prompt_preset_store.update(
                    preset_id,
                    name=body.name,
                    preset_type=body.type,
                    prompt_text=body.promptText,
                    base_prompt=body.basePrompt,
                ),
                detail=True,
            )
        except PromptPresetConflictError as exc:
            raise conflict(exc) from exc
        except PromptPresetNotFoundError as exc:
            raise not_found(exc) from exc
        except PromptPresetValidationError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/nl/prompt-presets/{preset_id}/reset")
    def reset_prompt_preset(preset_id: str) -> dict[str, object]:
        try:
            return _preset_response(context.prompt_preset_store.reset(preset_id), detail=True)
        except PromptPresetConflictError as exc:
            raise conflict(exc) from exc
        except PromptPresetNotFoundError as exc:
            raise not_found(exc) from exc
        except PromptPresetValidationError as exc:
            raise bad_request(exc) from exc

    @router.delete("/api/nl/prompt-presets/{preset_id}")
    def delete_prompt_preset(preset_id: str) -> dict[str, bool]:
        try:
            context.prompt_preset_store.delete(preset_id)
        except PromptPresetConflictError as exc:
            raise conflict(exc) from exc
        except PromptPresetNotFoundError as exc:
            raise not_found(exc) from exc
        except PromptPresetValidationError as exc:
            raise bad_request(exc) from exc
        return {"deleted": True}

    @router.post("/api/nl/diagnostics/models")
    def discover_nl_models(body: _NlModelDiscoveryBody) -> dict[str, object]:
        api_key = _resolved_diagnostic_key(context, body)
        if api_key is None:
            return _credential_unavailable(test_message=False)
        return context.nl_diagnostic_client.discover_models(endpoint=body.endpoint, api_key=api_key)

    @router.post("/api/nl/diagnostics/test-message")
    def test_nl_message(body: _NlTestMessageBody) -> dict[str, object]:
        api_key = _resolved_diagnostic_key(context, body)
        if api_key is None:
            return _credential_unavailable(test_message=True)
        return context.nl_diagnostic_client.test_message(
            endpoint=body.endpoint,
            model=body.model,
            api_key=api_key,
            base_prompt=body.basePrompt,
        )

    @router.get("/api/nl/profiles")
    def list_nl_profiles() -> dict[str, object]:
        try:
            profiles = context.profile_store.load_all()
        except NlProfileError as exc:
            raise bad_request(exc) from exc
        return {
            "defaultProfileId": context.profile_store.default_profile_id(),
            "profiles": [_profile_response(profile, credentials=context.credential_store) for profile in profiles],
        }

    @router.put("/api/nl/profiles/{profile_id}")
    def save_nl_profile(profile_id: str, body: _ProfileBody) -> dict[str, object]:
        try:
            raw = body.model_dump(exclude_none=True)
            credential = raw.get("credentialRef") or raw.get("apiCredentialRef")
            policy = raw.get("policy") if raw.get("policy") is not None else raw.get("apiPolicy", {})
            # Normalize legacy aliases before NlApiProfile.from_dict().  The
            # Pydantic body keeps compatibility defaults, but those defaults
            # must not make a schema-2 request look like a legacy profile.
            payload = {
                key: value for key, value in raw.items()
                if key not in {"apiCredentialRef", "apiPolicy", "systemPrompt"}
            }
            payload.update({"credentialRef": credential, "policy": policy})
            expected = payload.pop("revision", None)
            profile = NlApiProfile.from_dict({"profileId": profile_id, **payload})
            saved = context.profile_store.save(profile, expected_revision=expected)
            return _profile_response(saved, credentials=context.credential_store)
        except NlProfileError as exc:
            if str(exc) == "profile_revision_conflict":
                raise conflict(exc) from exc
            raise bad_request(exc) from exc

    @router.post("/api/nl/profiles/{profile_id}/default")
    def set_default_nl_profile(profile_id: str, body: _ProfileBody | None = None) -> dict[str, object]:
        expected = body.revision if body is not None else None
        try:
            context.profile_store.set_default(profile_id, expected_revision=expected)
            return {"defaultProfileId": profile_id}
        except NlProfileError as exc:
            if str(exc) == "profile_revision_conflict":
                raise conflict(exc) from exc
            raise bad_request(exc) from exc

    @router.post("/api/nl/profiles/{profile_id}/copy")
    def copy_nl_profile(profile_id: str, body: _ProfileCopyBody) -> dict[str, object]:
        try:
            source = next(item for item in context.profile_store.load_all() if item.profileId == profile_id)
            copied = NlApiProfile(
                body.profileId, source.endpoint, source.model, source.backupModel,
                source.apiCredentialRef, "", source.policy,
                body.name or f"{source.name or source.profileId} copy", source.protocol,
                source.providerLabel, source.jsonMode, 1, "",
            )
            saved = context.profile_store.save(copied)
            return _profile_response(saved, credentials=context.credential_store)
        except StopIteration as exc:
            raise not_found(exc) from exc
        except NlProfileError as exc:
            raise bad_request(exc) from exc

    @router.delete("/api/nl/profiles/{profile_id}")
    def delete_nl_profile(profile_id: str) -> dict[str, bool]:
        try:
            database = StateDatabase.open(context.database_path)
            try:
                profiles = {item.profileId: item for item in context.profile_store.load_all()}
                if profile_id in profiles and len(profiles) > 1 and context.profile_store.default_profile_id() != profile_id:
                    rows = database.connection.execute(
                        "SELECT job_id,status FROM jobs WHERE status NOT IN ('succeeded','succeeded_with_issues','discarded','cancelled_recoverable') "
                        "AND instr(config_json, ?) > 0 ORDER BY job_id LIMIT 50",
                        (f'\"apiProfileId\": \"{profile_id}\"',),
                    ).fetchall()
                    if rows:
                        detail = {
                            "code": "profile_in_use", "profileId": profile_id,
                            "tasks": [{"jobId": str(row["job_id"]), "status": str(row["status"])} for row in rows],
                        }
                        raise RuntimeError(json.dumps(detail, separators=(",", ":")))
            finally:
                database.close()
            context.profile_store.delete(profile_id)
        except NlProfileError as exc:
            if str(exc) in {"profile_last_one", "profile_default_in_use"}:
                raise conflict(exc) from exc
            raise bad_request(exc) from exc
        except RuntimeError as exc:
            raise conflict(exc) from exc
        return {"deleted": True}

    @router.put("/api/nl/credentials/{reference}")
    def save_nl_credential(reference: str, body: _SecretBody) -> dict[str, bool]:
        try:
            context.credential_store.save(reference, body.secret)
        except CredentialStoreError as exc:
            raise bad_request(exc) from exc
        return {"stored": True}

    @router.delete("/api/nl/credentials/{reference}")
    def delete_nl_credential(reference: str) -> dict[str, bool]:
        try:
            context.credential_store.delete(reference)
        except CredentialStoreError as exc:
            raise bad_request(exc) from exc
        return {"deleted": True}

    @router.put("/api/jobs/{job_id}/nl/api-config")
    def update_running_nl_api_config(job_id: str, body: _NlApiConfigBody) -> dict[str, object]:
        try:
            return context.pipeline_service.update_nl_api_config(job_id, body.model_dump())
        except KeyError as exc:
            raise not_found(exc) from exc
        except PipelineError as exc:
            if str(exc) in {"api_task_revision_conflict", "profile_revision_conflict"}:
                raise conflict(exc) from exc
            raise bad_request(exc) from exc

    def nl_control(job_id: str, action: str, body: _AmountBody | _ConfirmBody | None = None) -> dict[str, Any]:
        if action in {"pause", "resume"}:
            try:
                status = (
                    context.pipeline_service.pause_module(job_id, "nl")
                    if action == "pause"
                    else context.pipeline_service.resume_module(job_id, "nl")
                )
                return {"status": status}
            except KeyError as exc:
                raise not_found(exc) from exc
            except PipelineError as exc:
                raise bad_request(exc) from exc
        database = StateDatabase.open(context.database_path)
        try:
            scheduler = BoundedScheduler(database)
            try:
                if action == "budget":
                    assert isinstance(body, _AmountBody)
                    revision = database.add_api_budget_extra(job_id, body.amount)
                    return {"apiBudgetExtra": int(database.get_job(job_id)["api_budget_extra"]), "apiBudgetRevision": revision}
                assert isinstance(body, _ConfirmBody)
                return {"requeued": scheduler.confirm_nl_api_outcome_unknown(job_id, confirmed=body.confirmed)}
            except (PipelineError, SchedulerError, ValueError) as exc:
                raise bad_request(exc) from exc
        finally:
            database.close()

    @router.post("/api/jobs/{job_id}/nl/pause")
    def pause_nl(job_id: str) -> dict[str, Any]:
        return nl_control(job_id, "pause")

    @router.post("/api/jobs/{job_id}/nl/resume")
    def resume_nl(job_id: str) -> dict[str, Any]:
        return nl_control(job_id, "resume")

    @router.post("/api/jobs/{job_id}/nl/api-budget")
    def add_nl_budget(job_id: str, body: _AmountBody) -> dict[str, Any]:
        return nl_control(job_id, "budget", body)

    @router.post("/api/jobs/{job_id}/nl/confirm-api-outcomes")
    def confirm_nl_outcomes(job_id: str, body: _ConfirmBody) -> dict[str, Any]:
        return nl_control(job_id, "confirm", body)

    @router.post("/api/jobs/{job_id}/nl/manual-retry")
    def manual_nl_retry(job_id: str, body: _NlManualRetryBody) -> dict[str, Any]:
        if not body.confirmed:
            raise bad_request(ValueError("manual NL retry requires explicit confirmation"))
        released_parent_lock: str | None = None
        try:
            released_parent_lock = context.release_lock_for_repair(job_id)
            result = context.repair_service.prepare_manual_nl(
                job_id, sample_id=body.sampleId, issue_id=body.issueId, confirmed=body.confirmed,
            )
            context.pipeline_service.start(result.repairJobId)
            return {
                "jobId": result.repairJobId,
                "parentJobId": result.parentJobId,
                "sampleId": body.sampleId,
                "issueId": body.issueId,
                "targetCount": result.targetCount,
                "started": True,
            }
        except KeyError as exc:
            raise not_found(exc) from exc
        except (RepairPreparationError, PipelineError, SchedulerError, ValueError) as exc:
            if released_parent_lock:
                context.restore_lock_after_repair_failure(job_id, released_parent_lock)
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/nl/manual-retry-batch")
    def manual_nl_retry_batch(job_id: str, body: _NlManualRetryBatchBody) -> dict[str, Any]:
        if not body.confirmed:
            raise bad_request(ValueError("manual NL retry requires explicit confirmation"))
        released_parent_lock: str | None = None
        try:
            released_parent_lock = context.release_lock_for_repair(job_id)
            result = context.repair_service.prepare_manual_nl_batch(
                job_id, issue_ids=body.issueIds, confirmed=body.confirmed,
            )
            context.pipeline_service.start(result.repairJobId)
            return {
                "jobId": result.repairJobId,
                "parentJobId": result.parentJobId,
                "issueIds": body.issueIds,
                "targetCount": result.targetCount,
                "started": True,
            }
        except KeyError as exc:
            raise not_found(exc) from exc
        except (RepairPreparationError, PipelineError, SchedulerError, ValueError) as exc:
            if released_parent_lock:
                context.restore_lock_after_repair_failure(job_id, released_parent_lock)
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/nl/manual-write")
    def manual_nl_write(job_id: str, body: _NlManualWriteBody) -> dict[str, Any]:
        if not body.confirmed:
            raise bad_request(ValueError("manual NL write requires explicit confirmation"))
        try:
            NlManualWriteService.validate_text(body.nl)
        except ValueError as exc:
            raise bad_request(exc) from exc
        released_parent_lock: str | None = None
        try:
            released_parent_lock = context.release_lock_for_repair(job_id)
            result = context.repair_service.prepare_manual_nl(
                job_id, sample_id=body.sampleId, issue_id=body.issueId, confirmed=body.confirmed, for_manual_write=True,
            )
            database = StateDatabase.open(context.database_path)
            try:
                target = database.connection.execute(
                    "SELECT sample_id FROM repair_targets WHERE repair_job_id=?", (result.repairJobId,),
                ).fetchone()
                if target is None:
                    raise RepairPreparationError("manual NL write repair target is missing")
                NlManualWriteService(database, result.repairJobId).seed(
                    sample_id=int(target["sample_id"]), issue_id=body.issueId, nl=body.nl, confirmed=body.confirmed,
                )
            finally:
                database.close()
            context.pipeline_service.start(result.repairJobId)
            return {
                "jobId": result.repairJobId,
                "parentJobId": result.parentJobId,
                "sampleId": body.sampleId,
                "issueId": body.issueId,
                "targetCount": result.targetCount,
                "started": True,
            }
        except KeyError as exc:
            raise not_found(exc) from exc
        except (RepairPreparationError, PipelineError, SchedulerError, ValueError) as exc:
            if released_parent_lock:
                context.restore_lock_after_repair_failure(job_id, released_parent_lock)
            raise bad_request(exc) from exc

    return router
