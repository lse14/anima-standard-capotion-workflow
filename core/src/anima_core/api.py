"""Local FastAPI control plane composition facade."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api_application import build_application_router
from .api_context import ControlPlaneContext, bad_request as _bad_request, conflict as _conflict, not_found as _not_found
from .api_count_review import build_count_review_router
from .api_token_budget import build_token_budget_router
from .api_jobs import build_jobs_router
from .api_models import (
    _AmountBody,
    _ConfirmBody,
    _CountReviewBatchBody,
    _CountReviewBatchItem,
    _CountReviewDecisionBody,
    _NlDiagnosticCredentialsBody,
    _NlManualRetryBatchBody,
    _NlManualRetryBody,
    _NlManualWriteBody,
    _NlModelDiscoveryBody,
    _NlPromptPresetBody,
    _NlTestMessageBody,
    _PinBody,
    _PreflightBody,
    _ProfileBody,
    _SecretBody,
    _SelectPathBody,
    _ShutdownBody,
    _TokenBudgetApplyBody,
    _TokenBudgetRecountBody,
    _TokenBudgetRewriteShortBody,
    _WorkspaceBody,
)
from .api_nl import _profile_response, build_nl_router
from .credentials import DpapiCredentialStore
from .db import default_state_database_path
from .job_preflight import JobPreparationService
from .nl_profiles import NlApiProfileStore
from .nl_diagnostics import NlDiagnosticClient
from .nl_prompt_presets import NlPromptPresetStore
from .native_path_picker import NativePathPicker
from .pipeline import PipelineService
from .repair import RepairPreparationService
from .resource_catalog import ResourceCatalog, default_resource_library_root
from .device_recommendation import DeviceRecommendationService


def build_control_app(
    *,
    database_path: Path | str | None = None,
    profile_store: NlApiProfileStore | None = None,
    credential_store: DpapiCredentialStore | None = None,
    prompt_preset_store: NlPromptPresetStore | None = None,
    nl_diagnostic_client: NlDiagnosticClient | None = None,
    preparation_service: JobPreparationService | None = None,
    pipeline_service: PipelineService | None = None,
    repair_service: RepairPreparationService | None = None,
    resource_catalog: ResourceCatalog | None = None,
    static_root: Path | str | None = None,
    shutdown_token: str | None = None,
    shutdown_callback: Callable[[], None] | None = None,
    native_path_picker: NativePathPicker | None = None,
    device_recommendation_service: DeviceRecommendationService | None = None,
) -> FastAPI:
    """Create a localhost control-plane application from explicit stable stores."""
    database_path = Path(database_path) if database_path is not None else default_state_database_path()
    profile_store = profile_store or NlApiProfileStore()
    credential_store = credential_store or DpapiCredentialStore()
    prompt_preset_store = prompt_preset_store or NlPromptPresetStore()
    nl_diagnostic_client = nl_diagnostic_client or NlDiagnosticClient()
    service_catalogs = [
        candidate
        for candidate in (
            getattr(preparation_service, "resource_catalog", None),
            getattr(pipeline_service, "resource_catalog", None),
        )
        if isinstance(candidate, ResourceCatalog)
    ]
    resource_catalog = resource_catalog or (service_catalogs[0] if service_catalogs else None)
    resource_catalog = resource_catalog or ResourceCatalog(default_resource_library_root())
    if any(candidate.root != resource_catalog.root for candidate in service_catalogs):
        raise ValueError("control-plane services must share one resource catalog")
    pipeline_service = pipeline_service or PipelineService(
        database_path,
        profile_store=profile_store,
        credential_store=credential_store,
        resource_catalog=resource_catalog,
    )
    preparation_service = preparation_service or JobPreparationService(
        database_path,
        resource_catalog=resource_catalog,
        install_root=getattr(pipeline_service, "install_root", None),
        profile_store=profile_store,
    )
    repair_service = repair_service or RepairPreparationService(database_path)
    native_path_picker = native_path_picker or NativePathPicker()
    device_recommendation_service = device_recommendation_service or DeviceRecommendationService()
    pipeline_service.startup_recovery()
    context = ControlPlaneContext(
        database_path=database_path,
        profile_store=profile_store,
        credential_store=credential_store,
        prompt_preset_store=prompt_preset_store,
        nl_diagnostic_client=nl_diagnostic_client,
        preparation_service=preparation_service,
        pipeline_service=pipeline_service,
        repair_service=repair_service,
        resource_catalog=resource_catalog,
        device_recommendation_service=device_recommendation_service,
        shutdown_token=shutdown_token,
        shutdown_callback=shutdown_callback,
        native_path_picker=native_path_picker,
    )
    app = FastAPI(title="Anima Dataset Tool", version="0.1.0")
    app.include_router(build_application_router(context))
    app.include_router(build_jobs_router(context))
    app.include_router(build_token_budget_router(context))
    app.include_router(build_count_review_router(context))
    app.include_router(build_nl_router(context))
    if static_root is not None:
        root = Path(static_root)
        if not (root / "index.html").is_file():
            raise ValueError("static frontend root does not contain index.html")
        # Register after API routes so the SPA cannot shadow the control plane.
        app.mount("/", StaticFiles(directory=str(root), html=True), name="frontend")
    return app
