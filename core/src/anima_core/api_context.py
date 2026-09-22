from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import HTTPException

from .credentials import DpapiCredentialStore
from .job_preflight import JobPreparationService
from .nl_profiles import NlApiProfileStore
from .nl_diagnostics import NlDiagnosticClient
from .nl_prompt_presets import NlPromptPresetStore
from .native_path_picker import NativePathPicker
from .pipeline import PipelineService
from .repair import RepairPreparationService
from .resource_catalog import ResourceCatalog
from .device_recommendation import DeviceRecommendationService


@dataclass(frozen=True)
class ControlPlaneContext:
    database_path: Path
    profile_store: NlApiProfileStore
    credential_store: DpapiCredentialStore
    prompt_preset_store: NlPromptPresetStore
    nl_diagnostic_client: NlDiagnosticClient
    preparation_service: JobPreparationService
    pipeline_service: PipelineService
    repair_service: RepairPreparationService
    resource_catalog: ResourceCatalog
    device_recommendation_service: DeviceRecommendationService
    shutdown_token: str | None
    shutdown_callback: Callable[[], None] | None
    native_path_picker: NativePathPicker

    def release_lock_for_repair(self, job_id: str) -> str | None:
        if self.repair_service.release_lock_for_repair(job_id):
            return "repair"
        if self.preparation_service.release_lock_for_repair(job_id):
            return "preparation"
        return None

    def restore_lock_after_repair_failure(self, job_id: str, owner: str | None) -> None:
        if owner == "repair":
            self.repair_service.restore_lock_after_repair_failure(job_id)
        elif owner == "preparation":
            self.preparation_service.restore_lock_after_repair_failure(job_id)


def bad_request(error: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


def not_found(error: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


def conflict(error: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(error))
