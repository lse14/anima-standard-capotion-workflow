from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from pydantic import ValidationError
from pathlib import Path

from .api_context import ControlPlaneContext, bad_request, not_found
from .api_models import _ShutdownBody, parse_select_path_body
from .db import StateDatabase
from .native_path_picker import NativePathPickerBusyError, NativePathPickerUnavailableError
from .launcher import WorkerLaunchError, WorkerLauncher
from .pipeline import PipelineError
from .resource_catalog import ResourceCatalogError
from .scheduler import SchedulerError


def build_application_router(context: ControlPlaneContext) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    @router.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "protocolVersion": "1.0"}

    @router.get("/api/resources")
    def list_resources() -> dict[str, object]:
        try:
            snapshot = context.resource_catalog.scan()
            payload = snapshot.api_dict()
            if payload.get("resources"):
                install_root = getattr(context.pipeline_service, "install_root", None)
                policy = {"status": "missing", "runtimeId": "policy"}
                if install_root is not None:
                    try:
                        WorkerLauncher.from_install_root(
                            install_root,
                            resource_root=context.resource_catalog.root,
                        ).resolve("policy", expected_owner="policy", verify_interpreter=False)
                    except (WorkerLaunchError, OSError):
                        manifest = Path(install_root) / "manifests" / "runtimes" / "policy.json"
                        policy["status"] = "invalid" if manifest.exists() else "missing"
                    else:
                        policy["status"] = "available"
                quality = {"status": "missing", "resourceId": "lse14-scorer-5k-v1"}
                try:
                    snapshot.package("dropout-model", "lse14-scorer-5k-v1", verify_hashes=True)
                except ResourceCatalogError:
                    quality_paths = [item.relative_path.casefold() for item in snapshot.invalid]
                    quality["status"] = (
                        "invalid"
                        if any("dropout-models\\lse14-scorer-5k-v1" in path for path in quality_paths)
                        else "missing"
                    )
                else:
                    quality["status"] = "available"
                payload["capabilities"] = {"policy": policy, "qualityStack": quality}
            return payload
        except ResourceCatalogError as exc:
            raise bad_request(exc) from exc

    @router.get("/api/application/device-recommendations")
    def device_recommendations(rpm: int = 60) -> dict[str, object]:
        if type(rpm) is not int or not 1 <= rpm <= 100_000:
            raise bad_request(ValueError("rpm must be between 1 and 100000"))
        try:
            return context.device_recommendation_service.recommend(rpm=rpm)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise bad_request(exc) from exc

    def policy_control(job_id: str, action: str) -> dict[str, str]:
        try:
            status = (
                context.pipeline_service.pause_module(job_id, "dropout")
                if action == "pause"
                else context.pipeline_service.resume_module(job_id, "dropout")
            )
            return {"status": status}
        except KeyError as exc:
            raise not_found(exc) from exc
        except PipelineError as exc:
            raise bad_request(exc) from exc

    @router.post("/api/jobs/{job_id}/policy/pause")
    def pause_policy(job_id: str) -> dict[str, str]:
        return policy_control(job_id, "pause")

    @router.post("/api/jobs/{job_id}/policy/resume")
    def resume_policy(job_id: str) -> dict[str, str]:
        return policy_control(job_id, "resume")

    @router.post("/api/application/select-path")
    def select_path(body: object = Body(None)) -> dict[str, object]:
        try:
            request = parse_select_path_body(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="invalid_path_picker_request") from exc
        try:
            selected = context.native_path_picker.select(request.purpose, request.current_path)
        except NativePathPickerBusyError as exc:
            raise HTTPException(status_code=409, detail="path_picker_busy") from exc
        except NativePathPickerUnavailableError as exc:
            raise HTTPException(status_code=503, detail="path_picker_unavailable") from exc
        return {"cancelled": selected is None, "path": selected}

    @router.post("/api/application/shutdown")
    def shutdown_application(body: _ShutdownBody) -> dict[str, bool]:
        if (
            context.shutdown_callback is None
            or not isinstance(context.shutdown_token, str)
            or len(context.shutdown_token) < 32
            or body.token != context.shutdown_token
        ):
            raise HTTPException(status_code=404, detail="route not found")
        context.shutdown_callback()
        return {"accepted": True}

    return router
