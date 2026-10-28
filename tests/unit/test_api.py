from __future__ import annotations

import importlib
import inspect
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from fastapi import HTTPException
from fastapi.routing import APIRoute
from pydantic import SecretStr
from starlette.routing import Mount

from anima_core.api import (
    _AmountBody, _ConfirmBody, _NlManualRetryBody, _NlManualWriteBody, _PinBody, _PreflightBody, _ProfileBody, _SecretBody, _ShutdownBody,
    _TokenBudgetApplyBody, _TokenBudgetRecountBody, _TokenBudgetRewriteShortBody, _WorkspaceBody,
    build_control_app,
)
from anima_core.api_token_budget import _IsolatedShortRewriter
from anima_core.api_models import _NlApiConfigBody
from anima_core.token_budget_review import TokenBudgetReviewConflictError, TokenBudgetReviewError
from anima_core.commit_journal import CommitJournal, write as write_journal
from anima_core.job_preflight import JobPreflightError, JobPreparationService
from anima_core.contracts import JobConfig, ProgressEvent, SampleIssue, pipeline_module_ids, sha256_json
from anima_core.credentials import DpapiCredentialStore
from anima_core.db import StateDatabase
from anima_core.nl_profiles import NlApiProfile, NlApiProfileStore
from anima_core.nl_prompt_presets import NlPromptPresetStore
from anima_core.native_path_picker import NativePathPickerBusyError, NativePathPickerUnavailableError
from anima_core.path_safety import windows_key
from anima_core.pipeline import PipelineError, PipelineService
from anima_core.scheduler import BoundedScheduler
from anima_core.annotation_backup import write_backup
from anima_core.overlay import OverlayLayout
from anima_core.ocr_runtime_binding import OcrRuntimeBindingV1, write_runtime_binding
from anima_core.worker_protocol import ProtocolEnvelopeV1
from anima_core.device_recommendation import DeviceRecommendationService, GpuFacts


EXPECTED_BUILD_PARAMETERS = (
    "database_path", "profile_store", "credential_store", "prompt_preset_store", "nl_diagnostic_client", "preparation_service",
    "pipeline_service", "repair_service", "resource_catalog", "static_root",
    "shutdown_token", "shutdown_callback", "native_path_picker", "device_recommendation_service",
)

EXPECTED_CONTROL_ROUTES = {
    ("GET", "/health"),
    ("GET", "/api/health"),
    ("GET", "/api/resources"),
    ("GET", "/api/application/device-recommendations"),
    ("GET", "/api/jobs"),
    ("GET", "/api/jobs/{job_id}"),
    ("POST", "/api/jobs/preflight"),
    ("POST", "/api/jobs/preflight/async"),
    ("GET", "/api/jobs/preflight/{job_id}/progress"),
    ("POST", "/api/jobs/preflight/{job_id}/cancel"),
    ("POST", "/api/jobs/{job_id}/confirm-workspace"),
    ("POST", "/api/jobs/{job_id}/start"),
    ("POST", "/api/jobs/{job_id}/pause"),
    ("POST", "/api/jobs/{job_id}/resume"),
    ("POST", "/api/jobs/{job_id}/modules/{module_id}/pause"),
    ("POST", "/api/jobs/{job_id}/modules/{module_id}/resume"),
    ("GET", "/api/jobs/{job_id}/token-budget/reviews"),
    ("POST", "/api/jobs/{job_id}/token-budget/recount"),
    ("POST", "/api/jobs/{job_id}/token-budget/rewrite-short"),
    ("POST", "/api/jobs/{job_id}/token-budget/apply"),
    ("GET", "/api/jobs/{job_id}/count-review"),
    ("GET", "/api/jobs/{job_id}/count-review/{sample_id}/image"),
    ("PUT", "/api/jobs/{job_id}/count-review/{sample_id}"),
    ("POST", "/api/jobs/{job_id}/count-review/batch"),
    ("POST", "/api/jobs/{job_id}/count-review/confirm"),
    ("POST", "/api/jobs/{job_id}/repair"),
    ("POST", "/api/jobs/{job_id}/recover"),
    ("POST", "/api/jobs/{job_id}/cancel"),
    ("PUT", "/api/jobs/{job_id}/pin"),
    ("POST", "/api/jobs/{job_id}/discard"),
    ("POST", "/api/jobs/{job_id}/delete-history"),
    ("POST", "/api/jobs/{job_id}/restore-original-annotations"),
    ("GET", "/api/jobs/{job_id}/export/recommit"),
    ("POST", "/api/jobs/{job_id}/export/recommit"),
    ("GET", "/api/nl/default-prompt"),
    ("GET", "/api/nl/prompt-presets"),
    ("GET", "/api/nl/prompt-presets/{preset_id}"),
    ("POST", "/api/nl/prompt-presets"),
    ("PUT", "/api/nl/prompt-presets/{preset_id}"),
    ("POST", "/api/nl/prompt-presets/{preset_id}/reset"),
    ("DELETE", "/api/nl/prompt-presets/{preset_id}"),
    ("POST", "/api/nl/diagnostics/models"),
    ("POST", "/api/nl/diagnostics/test-message"),
    ("GET", "/api/nl/profiles"),
    ("PUT", "/api/nl/profiles/{profile_id}"),
    ("POST", "/api/nl/profiles/{profile_id}/default"),
    ("POST", "/api/nl/profiles/{profile_id}/copy"),
    ("DELETE", "/api/nl/profiles/{profile_id}"),
    ("PUT", "/api/nl/credentials/{reference}"),
    ("DELETE", "/api/nl/credentials/{reference}"),
    ("POST", "/api/jobs/{job_id}/nl/pause"),
    ("POST", "/api/jobs/{job_id}/nl/resume"),
    ("PUT", "/api/jobs/{job_id}/nl/api-config"),
    ("POST", "/api/jobs/{job_id}/nl/api-budget"),
    ("POST", "/api/jobs/{job_id}/nl/confirm-api-outcomes"),
    ("POST", "/api/jobs/{job_id}/nl/manual-retry"),
    ("POST", "/api/jobs/{job_id}/nl/manual-retry-batch"),
    ("POST", "/api/jobs/{job_id}/nl/manual-write"),
    ("POST", "/api/jobs/{job_id}/policy/pause"),
    ("POST", "/api/jobs/{job_id}/policy/resume"),
    ("POST", "/api/application/select-path"),
    ("POST", "/api/application/shutdown"),
}


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


class FakeNlDiagnosticClient:
    def __init__(self) -> None:
        self.model_calls: list[dict[str, str]] = []
        self.message_calls: list[dict[str, str]] = []

    def discover_models(self, *, endpoint: str, api_key: str) -> dict[str, object]:
        self.model_calls.append({"endpoint": endpoint, "api_key": api_key})
        return {"ok": True, "latencyMs": 12, "models": ["alpha", "Beta"], "errorCode": None, "errorReason": None}

    def test_message(self, *, endpoint: str, model: str, api_key: str, base_prompt: str) -> dict[str, object]:
        self.message_calls.append({"endpoint": endpoint, "model": model, "api_key": api_key, "base_prompt": base_prompt})
        return {"ok": True, "latencyMs": 34, "actualModel": model, "replyText": "Connected.", "usage": {"promptTokens": 3, "completionTokens": 4, "totalTokens": 7}, "errorCode": None, "errorReason": None}


class FakeNativePathPicker:
    def __init__(self, *, result: str | None = r"E:\\picked", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def select(self, purpose: str, current_path: str | None) -> str | None:
        self.calls.append((purpose, current_path))
        if self.error is not None:
            raise self.error
        return self.result


class ControlPlaneApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database_path = root / "state.db"
        self.profiles = NlApiProfileStore(root / "profiles.json")
        self.credentials = DpapiCredentialStore(root / "credentials")
        database = StateDatabase.open(self.database_path)
        dataset = root / "dataset"
        dataset.mkdir()
        config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        database.insert_job({"job_id": "job-api", "config_schema_version": config.schemaVersion, "config_json": json.dumps(config.to_dict()), "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental", "source_root": str(dataset), "output_root": None, "dataset_root": str(dataset), "dataset_root_key": windows_key(dataset), "manifest_schema_version": 1, "recursive": 0, "sample_count": 1, "manifest_generated_at": "2026-07-24T00:00:00Z", "status": "ready", "current_module_id": None, "last_event_id": 0, "pinned": 0, "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None, "resume_status": None, "created_at": "2026-07-24T00:00:00Z", "started_at": None, "cancel_requested_at": None, "finished_at": None})
        database.insert_samples("job-api", [{"sample_id": 1, "relative_image_path": "image.png", "annotation_key": "image", "source": "e621", "in_processing_scope": True, "image_format": "png", "image_frame_count": 1, "original_txt_state": "missing_or_blank", "original_json_state": "missing_or_blank", "image_file_id": "volume:1", "image_size": 1, "image_mtime_ns": 1}])
        database.close()
        self.preparation = JobPreparationService(self.database_path)
        self.pipeline = PipelineService(
            self.database_path,
            install_root=Path(r"E:\Desktop\Anima idg标准标注处理\.runtime-build"),
            profile_store=self.profiles,
            credential_store=self.credentials,
            resource_catalog=self.preparation.resource_catalog,
        )
        # Startup recovery freezes every job an earlier process left mid-flight, so the app is
        # built while the fixture job is still `ready` and only then started.
        self.app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=self.pipeline,
        )
        database = StateDatabase.open(self.database_path)
        scheduler = BoundedScheduler(database)
        for module in ("caption", "classify", "replace"):
            scheduler.start_module("job-api", module, enabled=False, profile="e621")
        scheduler.start_module("job-api", "ocr", enabled=False, profile="e621")
        scheduler.start_module("job-api", "nl", enabled=True, profile="e621")
        database.close()

    def _wait_for_pipeline_completion(self, job_id: str, expected_status: str) -> list[object]:
        deadline = time.monotonic() + 10.0
        while True:
            database = StateDatabase.open(self.database_path)
            try:
                job = database.get_job(job_id)
                summaries = database.module_summaries(job_id)
                status = str(job["status"])
            finally:
                database.close()
            if status == expected_status and not self.pipeline.is_running(job_id):
                return summaries
            if status in {"failed", "cancelled", "cancelled_recoverable"}:
                self.fail(f"pipeline reached unexpected terminal status: {status}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.fail(f"pipeline did not reach {expected_status} after its worker thread exited")
            time.sleep(min(0.02, remaining))

    def test_job_list_and_snapshot_do_not_expose_task_profile(self) -> None:
        list_jobs = _endpoint(self.app, "/api/jobs", "GET")
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")
        listed = list_jobs(afterCreatedAt=None, afterJobId=None, limit=200)
        self.assertNotIn("profile", listed["jobs"][0])
        current = snapshot(
            "job-api", afterEventId=0, issueAfterSampleId=0,
            issueAfterIssueId=None, limit=200,
        )
        self.assertNotIn("profile", current["job"])

    def test_device_recommendation_endpoint_reports_injected_facts_without_mutating_tasks(self) -> None:
        service = DeviceRecommendationService(
            baseline_path=Path(self.temporary.name) / "missing-baseline.json",
            cpu_probe=lambda: (6, 12),
            cuda_probe=lambda: GpuFacts(True, "fixture-gpu", 24 * 1024**3, 12 * 1024**3, "cuda-runtime"),
            nvidia_smi_probe=lambda: None,
        )
        app = build_control_app(
            database_path=self.database_path,
            profile_store=self.profiles,
            credential_store=self.credentials,
            preparation_service=self.preparation,
            pipeline_service=self.pipeline,
            device_recommendation_service=service,
        )
        endpoint = _endpoint(app, "/api/application/device-recommendations", "GET")
        result = endpoint(rpm=2)
        self.assertEqual((6, 12), (result["cpuPhysicalCores"], result["cpuLogicalCores"]))
        self.assertEqual("cuda-runtime", result["gpu"]["probeSource"])
        self.assertEqual(2, result["moduleBatchSize"]["nl"])
        self.assertTrue(all(value == 1 for key, value in result["moduleBatchSize"].items() if key != "nl"))

    def tearDown(self) -> None:
        self.pipeline.close()
        self.preparation.close()
        self.temporary.cleanup()

    def test_build_signature_and_route_inventory_are_frozen(self) -> None:
        self.assertEqual(EXPECTED_BUILD_PARAMETERS, tuple(inspect.signature(build_control_app).parameters))
        actual = {
            (method, route.path)
            for route in self.app.routes
            if isinstance(route, APIRoute)
            for method in route.methods
        }
        self.assertEqual(EXPECTED_CONTROL_ROUTES, actual)
        self.assertEqual(
            len(actual),
            sum(len(route.methods) for route in self.app.routes if isinstance(route, APIRoute)),
        )

    def test_v7_create_request_accepts_envelope_and_legacy_config_without_changing_canonical_bytes(self) -> None:
        models = importlib.import_module("anima_core.api_models")
        parser = getattr(models, "parse_create_job_body", None)
        self.assertIsNotNone(parser, "Task 3.4 requires create-job envelope normalization")
        if parser is None:
            return
        config = JobConfig(
            profile="e621", workMode="in_place", overwriteMode="incremental",
            sourceRoot=str(Path(self.temporary.name) / "dataset"), schemaVersion=7,
        )
        before = config.to_dict()
        for body in (
            before,
            {
                "config": before,
                "ocrExecution": {
                    "textDetLimitSideLen": {"mode": "manual", "value": 2304},
                    "textBatchSize": {"mode": "manual", "value": 2},
                },
            },
        ):
            with self.subTest(envelope="config" in body):
                parsed = parser(body)
                self.assertEqual(before, parsed.config)
                self.assertEqual(config.config_hash, sha256_json(parsed.config))
        legacy = parser(before)
        self.assertEqual({"mode": "auto", "value": None}, legacy.ocrExecution.to_dict()["textDetLimitSideLen"])
        self.assertEqual({"mode": "auto", "value": None}, legacy.ocrExecution.to_dict()["textBatchSize"])

    def test_snapshot_exposes_only_compact_bound_ocr_runtime_status(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            dataset = Path(self.temporary.name) / "dataset"
            layout = OverlayLayout.create(dataset, "job-api")
            binding = OcrRuntimeBindingV1.from_dict({
                "schemaVersion": 1,
                "requested": {
                    "device": "auto",
                    "textDetLimitSideLen": {"mode": "auto", "value": None},
                    "textBatchSize": {"mode": "auto", "value": None},
                },
                "recommended": {
                    "source": "gpu_vram_table", "totalVramBytes": 24 * 1024 ** 3,
                    "textDetLimitSideLen": 2560, "textBatchSize": 4,
                },
                "effective": {"textDetLimitSideLen": 2560, "textBatchSize": 4},
                "runtime": {
                    "runtimeId": "ocr-paddle-gpu", "runtimeFingerprint": "a" * 64,
                    "observedDevice": "cuda", "paddleVersion": "3.2.2", "compiledWithCuda": True,
                    "cudaVersion": "12.6", "gpuName": "NVIDIA Test GPU",
                },
                "resourceFingerprint": "b" * 64,
                "startupReason": None,
            })
            write_runtime_binding(layout.resource_path("ocr-runtime-binding-v1.json"), binding)
        finally:
            database.close()
        for schema_version in (7, 8, 9):
            with self.subTest(schema_version=schema_version):
                config = JobConfig(
                    profile="e621", workMode="in_place", overwriteMode="incremental",
                    sourceRoot=str(dataset), schemaVersion=schema_version,
                )
                config.ocr.update({"enabled": True, "device": "auto"})
                frozen = config.to_dict()
                database = StateDatabase.open(self.database_path)
                try:
                    database.connection.execute(
                        "UPDATE jobs SET config_schema_version=?,config_json=?,config_hash=?,overlay_root=? WHERE job_id=?",
                        (schema_version, json.dumps(frozen), sha256_json(frozen), str(layout.root), "job-api"),
                    )
                finally:
                    database.close()
                snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")(
                    "job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=100,
                )
                self.assertEqual(
                    {
                        "availability": "available", "runtimeId": "ocr-paddle-gpu",
                        "gpuName": "NVIDIA Test GPU", "totalVramBytes": 24 * 1024 ** 3,
                        "requestedDevice": "auto", "observedDevice": "cuda",
                        "recommended": {"textDetLimitSideLen": 2560, "textBatchSize": 4},
                        "effective": {"textDetLimitSideLen": 2560, "textBatchSize": 4},
                        "startupReason": None,
                    },
                    snapshot.get("ocrRuntime"),
                )

    def test_snapshot_keeps_persisted_ocr_runtime_after_overlay_cleanup(self) -> None:
        config = JobConfig(
            profile="e621", workMode="in_place", overwriteMode="incremental",
            sourceRoot=str(Path(self.temporary.name) / "dataset"), schemaVersion=9,
        ).to_dict()
        config["ocr"]["enabled"] = True
        persisted = {
            "availability": "available",
            "runtimeId": "ocr-paddle-gpu",
            "gpuName": "NVIDIA Persisted GPU",
            "totalVramBytes": 24 * 1024 ** 3,
            "requestedDevice": "cuda",
            "observedDevice": "cuda",
            "recommended": {"textDetLimitSideLen": 2560, "textBatchSize": 4},
            "effective": {"textDetLimitSideLen": 2560, "textBatchSize": 4},
            "startupReason": None,
        }
        dropout = {
            "availability": "available",
            "runtimeId": "policy",
            "qualityEnabled": True,
            "device": "cuda",
            "modelLoadCount": 1,
            "resourceFingerprint": "c" * 64,
        }
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute(
                "UPDATE jobs SET config_schema_version=?,config_json=?,config_hash=?,overlay_root=NULL,runtime_evidence_json=? WHERE job_id=?",
                (9, json.dumps(config), sha256_json(config), json.dumps({"ocr": persisted, "dropout": dropout}), "job-api"),
            )
        finally:
            database.close()
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")(
            "job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=100,
        )
        self.assertEqual(persisted, snapshot["ocrRuntime"])
        self.assertEqual(dropout, snapshot["dropoutRuntime"])

    def test_profile_and_credential_responses_never_echo_secret(self) -> None:
        save_credential = _endpoint(self.app, "/api/nl/credentials/{reference}", "PUT")
        save_profile = _endpoint(self.app, "/api/nl/profiles/{profile_id}", "PUT")
        profiles = _endpoint(self.app, "/api/nl/profiles", "GET")
        self.assertEqual({"stored": True}, save_credential("key-a", _SecretBody(secret="top-secret")))
        with self.assertRaises(HTTPException) as rejected:
            save_credential("nl-profile:default", _SecretBody(secret="top-secret"))
        self.assertEqual(400, rejected.exception.status_code)
        body = _ProfileBody(endpoint="https://example.test/v1", model="main", backupModel=None, apiCredentialRef="key-a", systemPrompt="describe", apiPolicy={"maxRequestsPerMinute": 60})
        saved = save_profile("profile-a", body)
        self.assertTrue(saved["hasCredential"])
        # Schema-2 clients use credentialRef/policy and must not be treated as
        # legacy merely because the compatibility body supplies empty defaults.
        updated = save_profile(
            "profile-a",
            _ProfileBody(
                endpoint="https://example.test/v2", model="main-v2", credentialRef="key-a",
                revision=saved["revision"], policy={"maxRequestsPerMinute": 30},
            ),
        )
        self.assertEqual(("https://example.test/v2", "main-v2", "key-a"), (updated["endpoint"], updated["model"], updated["credentialRef"]))
        response = profiles()
        self.assertNotIn("top-secret", json.dumps(response))
        self.assertTrue(response["profiles"][0]["hasCredential"])

    def test_running_api_config_queues_and_exposes_monotonic_task_revision(self) -> None:
        self.credentials.save("runtime-key", "runtime-secret")
        saved = self.profiles.save(NlApiProfile(
            "runtime-profile", "https://old.example/v1", "old-model", None,
            "runtime-key", "", {}, "Runtime", "openai_chat_completions", "gemini-relay", "prompt_only", 1, "",
        ))
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute("UPDATE jobs SET status='running',current_module_id='nl' WHERE job_id='job-api'")
        finally:
            database.close()
        update = _endpoint(self.app, "/api/jobs/{job_id}/nl/api-config", "PUT")
        first = update("job-api", _NlApiConfigBody(
            expectedTaskRevision=0, expectedProfileRevision=saved.revision, profileId="runtime-profile",
            endpoint="https://new.example/v1", model="new-model", backupModel=None,
            credentialRef="runtime-key", policy={},
        ))
        self.assertEqual(1, first["taskRevision"])
        database = StateDatabase.open(self.database_path)
        try:
            runtime = first["apiRuntime"]
            database.apply_nl_api_runtime_override("job-api", runtime)
        finally:
            database.close()
        refreshed = self.profiles.load_all()[0]
        second = update("job-api", _NlApiConfigBody(
            expectedTaskRevision=1, expectedProfileRevision=refreshed.revision, profileId="runtime-profile",
            endpoint="https://newer.example/v1", model="newer-model", backupModel=None,
            credentialRef="runtime-key", policy={},
        ))
        self.assertEqual(2, second["taskRevision"])

    def test_prompt_presets_and_diagnostics_are_isolated_from_profiles_and_credentials(self) -> None:
        parameters = inspect.signature(build_control_app).parameters
        self.assertTrue({"prompt_preset_store", "nl_diagnostic_client"} <= set(parameters))
        models = importlib.import_module("anima_core.api_models")
        body_names = ("_NlPromptPresetBody", "_NlModelDiscoveryBody", "_NlTestMessageBody")
        for name in body_names:
            self.assertTrue(hasattr(models, name), name)
        preset_body = getattr(models, "_NlPromptPresetBody", None)
        discovery_body = getattr(models, "_NlModelDiscoveryBody", None)
        message_body = getattr(models, "_NlTestMessageBody", None)
        if not all((preset_body, discovery_body, message_body)):
            return
        prompt_store = NlPromptPresetStore(Path(self.temporary.name) / "prompt-presets.json")
        diagnostic = FakeNlDiagnosticClient()
        app = build_control_app(
            database_path=self.database_path,
            profile_store=self.profiles,
            credential_store=self.credentials,
            prompt_preset_store=prompt_store,
            nl_diagnostic_client=diagnostic,
            preparation_service=self.preparation,
        )
        list_presets = _endpoint(app, "/api/nl/prompt-presets", "GET")
        get_preset = _endpoint(app, "/api/nl/prompt-presets/{preset_id}", "GET")
        create_preset = _endpoint(app, "/api/nl/prompt-presets", "POST")
        update_preset = _endpoint(app, "/api/nl/prompt-presets/{preset_id}", "PUT")
        delete_preset = _endpoint(app, "/api/nl/prompt-presets/{preset_id}", "DELETE")
        discover = _endpoint(app, "/api/nl/diagnostics/models", "POST")
        test_message = _endpoint(app, "/api/nl/diagnostics/test-message", "POST")

        built_in = list_presets()["presets"]
        self.assertEqual(("general", "style", "character"), tuple(item["type"] for item in built_in[:3]))
        self.assertTrue(all(item["builtIn"] for item in built_in[:3]))
        self.assertIn("observable", get_preset(built_in[0]["presetId"])["promptText"])
        moss_built_in = list_presets(includeMoss=True)["presets"]
        self.assertEqual(
            ("moss-general", "moss-style", "moss-character"),
            tuple(item["type"] for item in moss_built_in if str(item["type"]).startswith("moss-")),
        )
        self.assertIn("watermark", get_preset("builtin:nl-preset-moss-v1-general")["promptText"])
        created = create_preset(preset_body(name="Custom", type="style", promptText="Base A"))
        self.assertFalse(created["builtIn"])
        updated = update_preset(created["presetId"], preset_body(name="Renamed", type="character", promptText="Base B"))
        self.assertEqual(("Renamed", "character", "Base B"), (updated["name"], updated["type"], updated["promptText"]))
        changed = update_preset(built_in[0]["presetId"], preset_body(name="General", type="general", promptText="Local override"))
        self.assertEqual("Local override", changed["promptText"])
        reset_preset = _endpoint(app, "/api/nl/prompt-presets/{preset_id}/reset", "POST")
        self.assertNotEqual("Local override", reset_preset(built_in[0]["presetId"])["promptText"])
        with self.assertRaises(HTTPException) as built_in_conflict:
            delete_preset(built_in[0]["presetId"])
        self.assertEqual(409, built_in_conflict.exception.status_code)
        self.assertEqual({"deleted": True}, delete_preset(created["presetId"]))
        with self.assertRaises(HTTPException) as missing:
            get_preset(created["presetId"])
        self.assertEqual(404, missing.exception.status_code)
        with self.assertRaises(Exception):
            preset_body.model_validate({"name": "No", "basePrompt": "No", "extra": True})

        self.credentials.save("saved-ref", "dpapi-saved-key")
        transient = discover(discovery_body(endpoint="https://example.test/v1", apiCredentialRef="saved-ref", apiKey=SecretStr("transient-key")))
        self.assertEqual("transient-key", diagnostic.model_calls[-1]["api_key"])
        self.assertNotIn("transient-key", json.dumps(transient))
        saved = discover(discovery_body(endpoint="https://example.test/v1", apiCredentialRef="saved-ref", apiKey=SecretStr("  ")))
        self.assertEqual("dpapi-saved-key", diagnostic.model_calls[-1]["api_key"])
        self.assertTrue(saved["ok"])
        feedback = test_message(message_body(endpoint="https://example.test/v1", apiCredentialRef="saved-ref", apiKey=SecretStr("transient-key"), model="unsaved-model", basePrompt="Unsaved base"))
        self.assertEqual(
            {"endpoint": "https://example.test/v1", "model": "unsaved-model", "api_key": "transient-key", "base_prompt": "Unsaved base"},
            diagnostic.message_calls[-1],
        )
        self.assertNotIn("transient-key", json.dumps(feedback))
        before_missing = len(diagnostic.model_calls)
        unavailable = discover(discovery_body(endpoint="https://example.test/v1", apiCredentialRef="missing-ref", apiKey=None))
        self.assertEqual((False, "credential_unavailable", before_missing), (unavailable["ok"], unavailable["errorCode"], len(diagnostic.model_calls)))
        self.assertEqual((), self.profiles.load_all())

    def test_resource_endpoint_reuses_the_preparation_service_catalog(self) -> None:
        catalog = self.preparation.resource_catalog
        original_scan = catalog.scan
        marker = {"schemaVersion": 1, "defaults": {"marker": "shared"}, "resources": [], "invalidResources": []}
        catalog.scan = lambda: SimpleNamespace(api_dict=lambda: marker)  # type: ignore[method-assign]
        try:
            self.assertEqual(marker, _endpoint(self.app, "/api/resources", "GET")())
        finally:
            catalog.scan = original_scan  # type: ignore[method-assign]

    def test_snapshot_exposes_caption_raw_e621_conversion_count(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            database.increment_module_diagnostic(
                "job-api", "caption", "e621_raw_json_converted", severity="info", amount=2,
            )
        finally:
            database.close()
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")(
            "job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200,
        )
        diagnostics = snapshot["captionDiagnostics"]
        self.assertEqual(1, len(diagnostics))
        self.assertEqual(
            ("e621_raw_json_converted", "info", 2),
            (diagnostics[0]["code"], diagnostics[0]["severity"], diagnostics[0]["count"]),
        )

    def test_snapshot_exposes_ocr_diagnostics_from_the_existing_module_counter(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            for code, amount in (
                ("ocr_total", 4), ("ocr_new", 2), ("ocr_reused", 2),
                ("ocr_success", 1), ("ocr_no_text", 2), ("ocr_failed", 1),
                ("ocr_text_items", 7), ("ocr_included_for_llm", 5),
            ):
                database.increment_module_diagnostic("job-api", "ocr", code, severity="info", amount=amount)
            database.increment_module_diagnostic(
                "job-api", "nl", "nl_ocr_context_omitted_too_large", severity="warning", amount=1,
            )
        finally:
            database.close()

        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")(
            "job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200,
        )
        self.assertIn("ocrDiagnostics", snapshot)
        self.assertEqual(
            {
                "ocr_total": 4, "ocr_new": 2, "ocr_reused": 2, "ocr_success": 1,
                "ocr_no_text": 2, "ocr_failed": 1, "ocr_text_items": 7,
                "ocr_included_for_llm": 5, "nl_ocr_context_omitted_too_large": 1,
            },
            {item["code"]: item["count"] for item in snapshot["ocrDiagnostics"]},
        )

    def test_snapshot_budget_pause_resume_and_confirm_are_control_plane_only(self) -> None:
        class ResumePipeline:
            def __init__(self) -> None:
                self.resumed: list[str] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def resume(self, job_id: str) -> bool:
                self.resumed.append(job_id)
                return True

            def pause_module(self, _job_id: str, _module_id: str) -> str:
                return "paused"

            def resume_module(self, job_id: str, _module_id: str) -> str:
                self.resume(job_id)
                return "running"

        pipeline = ResumePipeline()
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline,  # type: ignore[arg-type]
        )
        snapshot = _endpoint(app, "/api/jobs/{job_id}", "GET")
        pause = _endpoint(app, "/api/jobs/{job_id}/nl/pause", "POST")
        resume = _endpoint(app, "/api/jobs/{job_id}/nl/resume", "POST")
        budget = _endpoint(app, "/api/jobs/{job_id}/nl/api-budget", "POST")
        confirm = _endpoint(app, "/api/jobs/{job_id}/nl/confirm-api-outcomes", "POST")
        initial = snapshot("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200)
        self.assertEqual(("running", "nl"), (initial["job"]["status"], initial["job"]["currentModuleId"]))
        self.assertEqual({"status": "paused"}, pause("job-api"))
        self.assertEqual({"apiBudgetExtra": 5, "apiBudgetRevision": 1}, budget("job-api", _AmountBody(amount=5)))
        with self.assertRaises(HTTPException):
            confirm("job-api", _ConfirmBody(confirmed=False))
        self.assertEqual({"status": "running"}, resume("job-api"))
        self.assertEqual(["job-api"], pipeline.resumed)
        after = snapshot("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200)
        self.assertEqual((5, 1), (after["job"]["apiBudgetExtra"], after["job"]["apiBudgetRevision"]))

    def test_nl_resume_maps_pipeline_conflict_to_bad_request(self) -> None:
        class SettlingPipeline:
            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def resume(self, _job_id: str) -> bool:
                raise PipelineError("task pipeline is still settling; retry resume")

            def resume_module(self, job_id: str, _module_id: str) -> str:
                self.resume(job_id)
                raise AssertionError("unreachable")

        database = StateDatabase.open(self.database_path)
        try:
            database.set_module_summary("job-api", "nl", status="paused")
            database.set_job_status("job-api", "paused", current_module_id="nl", resume_status="running")
        finally:
            database.close()
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=SettlingPipeline(),  # type: ignore[arg-type]
        )
        resume = _endpoint(app, "/api/jobs/{job_id}/nl/resume", "POST")
        with self.assertRaises(HTTPException) as raised:
            resume("job-api")
        self.assertEqual((400, "task pipeline is still settling; retry resume"), (
            raised.exception.status_code, raised.exception.detail,
        ))

    def test_job_pause_and_resume_forward_to_the_pipeline(self) -> None:
        class JobPipeline:
            def __init__(self) -> None:
                self.paused: list[str] = []
                self.resumed: list[str] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def pause(self, job_id: str) -> bool:
                self.paused.append(job_id)
                return True

            def resume(self, job_id: str) -> bool:
                self.resumed.append(job_id)
                return True

        pipeline = JobPipeline()
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline,  # type: ignore[arg-type]
        )
        pause = _endpoint(app, "/api/jobs/{job_id}/pause", "POST")
        resume = _endpoint(app, "/api/jobs/{job_id}/resume", "POST")
        self.assertEqual({"status": "paused"}, pause("job-api"))
        self.assertEqual({"status": "running"}, resume("job-api"))
        self.assertEqual(["job-api"], pipeline.paused)
        self.assertEqual(["job-api"], pipeline.resumed)

    def test_job_pause_and_resume_map_pipeline_and_missing_job_errors(self) -> None:
        class FailingPipeline:
            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def pause(self, _job_id: str) -> bool:
                raise PipelineError("task cannot be paused")

            def resume(self, _job_id: str) -> bool:
                raise KeyError("missing-job")

        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=FailingPipeline(),  # type: ignore[arg-type]
        )
        pause = _endpoint(app, "/api/jobs/{job_id}/pause", "POST")
        resume = _endpoint(app, "/api/jobs/{job_id}/resume", "POST")
        with self.assertRaises(HTTPException) as pause_error:
            pause("job-api")
        self.assertEqual((400, "task cannot be paused"), (pause_error.exception.status_code, pause_error.exception.detail))
        with self.assertRaises(HTTPException) as missing_job:
            resume("missing-job")
        self.assertEqual(404, missing_job.exception.status_code)

    def test_module_lifecycle_routes_return_the_updated_snapshot_and_legacy_routes_delegate(self) -> None:
        class ModulePipeline:
            def __init__(self, database_path: Path) -> None:
                self.database_path = database_path
                self.calls: list[tuple[str, str, str]] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def pause_module(self, job_id: str, module_id: str) -> str:
                self.calls.append(("pause", job_id, module_id))
                database = StateDatabase.open(self.database_path)
                try:
                    database.initialize_module_summary(job_id, module_id, total=1, status="paused")
                finally:
                    database.close()
                return "paused"

            def resume_module(self, job_id: str, module_id: str) -> str:
                self.calls.append(("resume", job_id, module_id))
                database = StateDatabase.open(self.database_path)
                try:
                    database.connection.execute(
                        "DELETE FROM module_summary WHERE job_id=? AND module_id=?", (job_id, module_id)
                    )
                finally:
                    database.close()
                return "running"

        pipeline = ModulePipeline(self.database_path)
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline,  # type: ignore[arg-type]
        )
        pause = _endpoint(app, "/api/jobs/{job_id}/modules/{module_id}/pause", "POST")
        resume = _endpoint(app, "/api/jobs/{job_id}/modules/{module_id}/resume", "POST")
        pause_nl = _endpoint(app, "/api/jobs/{job_id}/nl/pause", "POST")
        resume_policy = _endpoint(app, "/api/jobs/{job_id}/policy/resume", "POST")

        paused = pause("job-api", "dropout")
        self.assertEqual("paused", next(item["status"] for item in paused["modules"] if item["module_id"] == "dropout"))
        resumed = resume("job-api", "dropout")
        self.assertNotIn("dropout", [item["module_id"] for item in resumed["modules"]])
        pause_nl("job-api")
        resume_policy("job-api")
        self.assertEqual(
            [("pause", "job-api", "dropout"), ("resume", "job-api", "dropout"), ("pause", "job-api", "nl"), ("resume", "job-api", "dropout")],
            pipeline.calls,
        )

    def test_reviewing_token_budget_uses_a_safe_existing_resume_entry(self) -> None:
        """Apply may only reopen Export through a public PipelineService entry."""
        database = StateDatabase.open(self.database_path)
        try:
            dataset = Path(self.temporary.name) / "dataset"
            config = JobConfig(workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset), schemaVersion=10)
            config.nl["systemPrompt"] = "Describe the visible image."
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.tokenBudget.update({  # type: ignore[union-attr]
                "resourceManifestRelativePath": "tokenizers\\anima\\resource.json",
                "resourceFingerprint": "a" * 64,
                "contextLimit": 40960,
            })
            database.connection.execute(
                "UPDATE jobs SET config_schema_version=?,config_json=?,config_hash=? WHERE job_id=?",
                (config.schemaVersion, json.dumps(config.to_dict()), config.config_hash, "job-api"),
            )
            for module_id in pipeline_module_ids(config.schemaVersion):
                if module_id == "export":
                    continue
                database.initialize_module_summary("job-api", module_id, total=1)
                database.set_module_summary(
                    "job-api", module_id,
                    status="completed_with_issues" if module_id == "token_budget" else "completed",
                    finished=True,
                )
            database.set_job_status("job-api", "reviewing", current_module_id="token_budget")
        finally:
            database.close()
        pipeline = PipelineService(self.database_path)
        try:
            pipeline._token_budget_export_gate = lambda *_: True  # type: ignore[method-assign]
            pipeline._thread_main = lambda *_: None  # type: ignore[method-assign]
            self.assertTrue(pipeline.resume("job-api"))
        finally:
            pipeline.close()

    def test_snapshot_exposes_versioned_module_order_without_display_numbers(self) -> None:
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")
        arguments = {
            "afterEventId": 0,
            "issueAfterSampleId": 0,
            "issueAfterIssueId": None,
            "limit": 200,
        }
        current = snapshot("job-api", **arguments)
        self.assertEqual(10, current["job"]["configSchemaVersion"])
        self.assertNotIn("profile", current["job"])
        self.assertEqual(
            [
                "caption", "classify", "replace", "ocr", "nl", "count_review",
                "dropout", "token_budget", "export",
            ],
            current["moduleOrder"],
        )

    def test_policy_pause_and_resume_forward_to_the_pipeline(self) -> None:
        class ResumePipeline:
            def __init__(self) -> None:
                self.resumed: list[str] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def resume(self, job_id: str) -> bool:
                self.resumed.append(job_id)
                return True

            def pause_module(self, _job_id: str, _module_id: str) -> str:
                return "paused"

            def resume_module(self, job_id: str, _module_id: str) -> str:
                self.resume(job_id)
                return "running"

        database = StateDatabase.open(self.database_path)
        try:
            database.initialize_module_summary("job-api", "dropout", total=1, status="running")
            database.set_job_status("job-api", "running", current_module_id="dropout")
        finally:
            database.close()
        pipeline = ResumePipeline()
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline,  # type: ignore[arg-type]
        )
        pause = _endpoint(app, "/api/jobs/{job_id}/policy/pause", "POST")
        resume = _endpoint(app, "/api/jobs/{job_id}/policy/resume", "POST")
        self.assertEqual({"status": "paused"}, pause("job-api"))
        self.assertEqual({"status": "running"}, resume("job-api"))
        self.assertEqual(["job-api"], pipeline.resumed)

    def test_policy_pause_maps_an_atomic_state_conflict_to_bad_request(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            database.initialize_module_summary("job-api", "dropout", total=0, status="running")
            database.set_job_status("job-api", "running", current_module_id="dropout")
        finally:
            database.close()
        pipeline = PipelineService(self.database_path, install_root=ROOT / ".runtime-build")
        pipeline._threads["job-api"] = SimpleNamespace()  # type: ignore[assignment]
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline,
        )
        database = StateDatabase.open(self.database_path)
        try:
            database.set_job_status("job-api", "running", current_module_id="dropout")
        finally:
            database.close()
        pause = _endpoint(app, "/api/jobs/{job_id}/policy/pause", "POST")
        pause_active_module = StateDatabase.pause_active_module

        def finish_before_pause(database: StateDatabase, job_id: str, module_id: str, *, active_status: str) -> None:
            concurrent = StateDatabase.open(self.database_path)
            try:
                BoundedScheduler(concurrent).finish_module(job_id, "dropout")
            finally:
                concurrent.close()
            pause_active_module(database, job_id, module_id, active_status=active_status)

        try:
            with patch.object(StateDatabase, "pause_active_module", new=finish_before_pause):
                with self.assertRaises(HTTPException) as raised:
                    pause("job-api")
            self.assertEqual((400, "active module state changed before pause"), (
                raised.exception.status_code, raised.exception.detail,
            ))
        finally:
            pipeline._threads.pop("job-api", None)
            pipeline.close()

    def test_token_budget_routes_use_the_review_service_and_public_resume_only(self) -> None:
        class ResumePipeline:
            def __init__(self) -> None:
                self.resumed: list[str] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def resume(self, job_id: str) -> bool:
                self.resumed.append(job_id)
                return False

        class ReviewService:
            def page(self, *, after_sample_id, limit):
                return {"items": [], "targetCount": 0, "nextAfterSampleId": None}

            def recount(self, sample_id: int, *, expected_version: int, annotation: dict[str, object]):
                if sample_id == 400:
                    raise TokenBudgetReviewError("Token Budget overflow review is unavailable")
                return {"sampleId": sample_id, "version": expected_version + 1, "annotation": annotation}

            def rewrite_short(self, *, sample_ids, expected_versions):
                return {"operationId": "a" * 64, "sampleIds": sample_ids, "proposals": []}

            def apply(self, sample_id: int, *, expected_version: int):
                if expected_version == 99:
                    raise TokenBudgetReviewConflictError("version conflict")
                return {"sampleId": sample_id, "status": "within_budget"}

        pipeline = ResumePipeline()
        app = build_control_app(
            database_path=self.database_path,
            profile_store=self.profiles,
            credential_store=self.credentials,
            preparation_service=self.preparation,
            pipeline_service=pipeline,  # type: ignore[arg-type]
        )
        service = ReviewService()
        with patch("anima_core.api_token_budget._service", return_value=service):
            reviews = _endpoint(app, "/api/jobs/{job_id}/token-budget/reviews", "GET")
            recount = _endpoint(app, "/api/jobs/{job_id}/token-budget/recount", "POST")
            rewrite = _endpoint(app, "/api/jobs/{job_id}/token-budget/rewrite-short", "POST")
            apply = _endpoint(app, "/api/jobs/{job_id}/token-budget/apply", "POST")
            self.assertEqual(0, reviews("job-api", afterSampleId=0, limit=1)["targetCount"])
            self.assertEqual(2, recount("job-api", _TokenBudgetRecountBody(sampleId=1, expectedVersion=1, annotation={})) ["proposal"]["version"])
            with self.assertRaises(HTTPException) as bad_recount:
                recount("job-api", _TokenBudgetRecountBody(sampleId=400, expectedVersion=1, annotation={}))
            self.assertEqual([1], rewrite("job-api", _TokenBudgetRewriteShortBody(sampleIds=[1], expectedVersions={"1": 1})) ["sampleIds"])
            response = apply("job-api", _TokenBudgetApplyBody(sampleId=1, expectedVersion=2))
            self.assertEqual(("within_budget", False), (response["record"]["status"], response["exportStarted"]))
            with self.assertRaises(HTTPException) as raised:
                apply("job-api", _TokenBudgetApplyBody(sampleId=1, expectedVersion=99))
        self.assertEqual(409, raised.exception.status_code)
        self.assertEqual(400, bad_recount.exception.status_code)
        self.assertEqual(["job-api"], pipeline.resumed)

    def test_token_budget_apply_keeps_the_durable_record_when_public_resume_cannot_start(self) -> None:
        class ResumePipeline:
            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def resume(self, job_id: str) -> bool:
                raise PipelineError("token budget blockers remain")

        class ReviewService:
            def apply(self, sample_id: int, *, expected_version: int) -> dict[str, object]:
                return {"sampleId": sample_id, "status": "within_budget"}

        app = build_control_app(
            database_path=self.database_path,
            profile_store=self.profiles,
            credential_store=self.credentials,
            preparation_service=self.preparation,
            pipeline_service=ResumePipeline(),  # type: ignore[arg-type]
        )
        with patch("anima_core.api_token_budget._service", return_value=ReviewService()):
            apply = _endpoint(app, "/api/jobs/{job_id}/token-budget/apply", "POST")
            response = apply("job-api", _TokenBudgetApplyBody(sampleId=1, expectedVersion=1))
        self.assertEqual(("within_budget", False), (response["record"]["status"], response["exportStarted"]))

    def test_short_rewriter_rejects_a_hello_with_forged_runtime_identity_before_processing(self) -> None:
        class FakeTransport:
            requests: list[ProtocolEnvelopeV1] = []

            def __init__(self, _: object) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def exchange(self, request: ProtocolEnvelopeV1) -> ProtocolEnvelopeV1:
                self.requests.append(request)
                return ProtocolEnvelopeV1(
                    "1.0", "response", "nl-forged", "forged-runtime", "nl", "hello",
                    {"schemaVersion": 1, "payloadType": "nl_hello_result", "ready": True, "concurrency": 3},
                    replyTo=request.messageId, jobId="job-api", configHash="a" * 64,
                )

        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute(
                "UPDATE jobs SET config_json=?,config_hash=? WHERE job_id='job-api'",
                (json.dumps({"nl": {"apiProfileId": "profile-review", "systemPrompt": "", "apiPolicy": {"maxHttpAttempts": 1}}}), "a" * 64),
            )
            profile = SimpleNamespace(
                profileId="profile-review", endpoint="https://loopback.invalid", model="test", backupModel=None,
                apiCredentialRef="review-credential",
            )
            context = SimpleNamespace(
                profile_store=SimpleNamespace(load_all=lambda: [profile]),
                credential_store=SimpleNamespace(load=lambda _: "secret"),
                pipeline_service=SimpleNamespace(
                    install_root=Path(self.temporary.name),
                    launcher_factory=lambda _: SimpleNamespace(spawn=lambda *_args, **_kwargs: object()),
                ),
            )
            rewriter = _IsolatedShortRewriter(context, database, "job-api")
            item = {
                "schemaVersion": 1, "sampleId": 1, "leaseId": "review-rewrite-1-1",
                "relativeImagePath": "image.png", "imagePath": None, "captionPreset": "general",
                "lengthTier": "short", "primaryCharacterName": None, "userSupplement": "",
                "jsonContext": "{}", "ocrContext": None, "currentNl": "",
            }
            with patch("anima_core.api_token_budget.StdioJsonlTransport", FakeTransport):
                with self.assertRaisesRegex(RuntimeError, "NL short rewrite failed"):
                    rewriter(item)
        finally:
            database.close()
        self.assertEqual(["hello"], [request.method for request in FakeTransport.requests])

    def test_recover_route_forwards_explicit_confirmation_to_pipeline_service(self) -> None:
        class RecoveryPipeline:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def recover_job(self, job_id: str, *, confirmed: bool) -> dict[str, object]:
                self.calls.append((job_id, confirmed))
                return {"jobId": job_id, "started": False, "status": "interrupted", "pendingApiDecisions": 1}

        pipeline = RecoveryPipeline()
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline,  # type: ignore[arg-type]
        )
        recover = _endpoint(app, "/api/jobs/{job_id}/recover", "POST")
        self.assertEqual("interrupted", recover("job-api", _ConfirmBody(confirmed=True))["status"])
        self.assertEqual([("job-api", True)], pipeline.calls)

    def test_recover_route_forwards_confirmation_for_the_selected_cancelled_task(self) -> None:
        class RecoveryPipeline:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def recover_job(self, job_id: str, *, confirmed: bool) -> dict[str, object]:
                self.calls.append((job_id, confirmed))
                return {"jobId": job_id, "started": True, "status": "running", "pendingApiDecisions": 0}

        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute(
                "UPDATE jobs SET status='cancelled_recoverable',current_module_id='nl',resume_status='running' WHERE job_id='job-api'"
            )
        finally:
            database.close()
        pipeline = RecoveryPipeline()
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline,  # type: ignore[arg-type]
        )
        recover = _endpoint(app, "/api/jobs/{job_id}/recover", "POST")
        self.assertEqual("running", recover("job-api", _ConfirmBody(confirmed=True))["status"])
        self.assertEqual([("job-api", True)], pipeline.calls)

    def test_repair_route_starts_the_new_task_and_snapshot_exposes_aggregate_preview(self) -> None:
        class RepairService:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def release_lock_for_repair(self, _job_id: str) -> bool:
                return False

            def restore_lock_after_repair_failure(self, _job_id: str) -> None:
                return None

            def prepare(self, job_id: str):
                self.calls.append(job_id)
                from anima_core.repair import RepairPreparationResult
                return RepairPreparationResult("repair-api", job_id, 3, "dataset", "overlay")

        class RepairPipeline:
            def __init__(self) -> None:
                self.started: list[str] = []

            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def start(self, job_id: str) -> None:
                self.started.append(job_id)

        repair = RepairService()
        pipeline = RepairPipeline()
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, pipeline_service=pipeline, repair_service=repair,  # type: ignore[arg-type]
        )
        repair_route = _endpoint(app, "/api/jobs/{job_id}/repair", "POST")
        response = repair_route("job-api")
        self.assertEqual(("repair-api", "job-api", 0, True), (response["jobId"], response["parentJobId"], response["estimatedApiRequests"], response["started"]))
        self.assertEqual((["job-api"], ["repair-api"]), (repair.calls, pipeline.started))
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200)
        self.assertEqual({"eligibleTargetCount": 0, "estimatedApiRequests": 0}, snapshot["repairPreview"])

    def test_preflight_and_workspace_confirmation_are_separate_calls(self) -> None:
        preflight = _endpoint(self.app, "/api/jobs/preflight", "POST")
        confirm = _endpoint(self.app, "/api/jobs/{job_id}/confirm-workspace", "POST")
        config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(Path(self.temporary.name) / "dataset"))
        config.nl["systemPrompt"] = "describe"
        result = preflight(_PreflightBody(config=config.to_dict()))
        self.assertEqual(0, result["sampleCount"])
        with self.assertRaises(HTTPException):
            confirm(result["jobId"], _WorkspaceBody(confirmed=False, confirmedRebuild=False))
        workspace = confirm(result["jobId"], _WorkspaceBody(confirmed=True, confirmedRebuild=False))
        self.assertEqual("preparing_workspace", workspace["status"])

    def test_async_preflight_endpoint_exposes_independent_progress_and_summary(self) -> None:
        start = _endpoint(self.app, "/api/jobs/preflight/async", "POST")
        progress = _endpoint(self.app, "/api/jobs/preflight/{job_id}/progress", "GET")
        config = JobConfig(
            profile="e621", workMode="in_place", overwriteMode="incremental",
            sourceRoot=str(Path(self.temporary.name) / "dataset"),
        )
        config.nl["systemPrompt"] = "describe"
        result = start(_PreflightBody(config=config.to_dict()))
        self.assertEqual("queued", result["status"])
        self.assertEqual(0, int(result["lastEventId"]))
        deadline = time.monotonic() + 10
        while True:
            current = progress(result["jobId"])
            if current["status"] in {"completed", "failed"}:
                break
            if time.monotonic() >= deadline:
                self.fail("async preflight endpoint did not finish")
            time.sleep(0.01)
        self.assertEqual("completed", current["status"])
        self.assertEqual((0, 0), (current["checkedFiles"], current["totalFiles"]))
        self.assertIsInstance(current["summary"], dict)

    def test_async_preflight_cancel_endpoint_delegates_to_preparation_service(self) -> None:
        cancel = _endpoint(self.app, "/api/jobs/preflight/{job_id}/cancel", "POST")
        with patch.object(self.preparation, "cancel_preflight", return_value=True) as requested:
            result = cancel("preflight-job",)
        self.assertEqual({"jobId": "preflight-job", "cancelled": True}, result)
        requested.assert_called_once_with("preflight-job")

    def test_confirm_workspace_maps_dataset_claim_conflict_to_actionable_409(self) -> None:
        dataset = Path(self.temporary.name) / "dataset"
        database = StateDatabase.open(self.database_path)
        try:
            owner = dict(database.get_job("job-api"))
            database.insert_job({
                **owner,
                "job_id": "3bc585",
                "status": "ready",
                "current_module_id": None,
                "overlay_root": None,
                "resume_status": None,
                "started_at": None,
                "cancel_requested_at": None,
                "finished_at": None,
            })
            overlay = OverlayLayout.create(dataset, "3bc585")
            database.set_workspace_metadata(
                "3bc585",
                dataset_root=str(dataset),
                dataset_root_key=windows_key(dataset),
                overlay_root=str(overlay.root),
            )
            database.set_job_status("3bc585", "preparing_workspace", current_module_id="workspace")
            database.set_job_status("3bc585", "interrupted", current_module_id="workspace")
            database.connection.execute(
                """INSERT INTO dataset_claims(dataset_root,dataset_root_key,job_id,lock_path,acquired_at)
                   VALUES (?,?,?,?,?)""",
                (str(dataset), windows_key(dataset), "3bc585", str(Path(self.temporary.name) / ".restart.lock"), "2026-08-17T00:00:00Z"),
            )
        finally:
            database.close()

        preflight = _endpoint(self.app, "/api/jobs/preflight", "POST")
        confirm = _endpoint(self.app, "/api/jobs/{job_id}/confirm-workspace", "POST")
        config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
        config.nl["systemPrompt"] = "describe"
        contender = preflight(_PreflightBody(config=config.to_dict()))
        try:
            confirm(contender["jobId"], _WorkspaceBody(confirmed=True, confirmedRebuild=False))
        except Exception as error:
            rejected = error
        else:
            rejected = None
        self.assertIsInstance(rejected, HTTPException)
        assert isinstance(rejected, HTTPException)
        self.assertEqual(
            (
                409,
                "Dataset is claimed by task 3bc585. Select it under Recent tasks: Recover keeps its progress and continues to hold the dataset; Discard deletes its overlay and releases the dataset.",
            ),
            (rejected.status_code, rejected.detail),
        )

    def test_start_preserves_the_forced_cuda_compatibility_error_detail(self) -> None:
        class FailingPipeline:
            def startup_recovery(self) -> dict[str, int]:
                return {"interruptedJobs": 0, "clearedDatasetClaims": 0, "deletedJobs": 0, "deletedOverlays": 0}

            def start(self, _job_id: str) -> None:
                raise PipelineError(
                    "The OCR CUDA runtime is unavailable or incompatible with this GPU. Choose Auto or CPU."
                )

        app = build_control_app(
            database_path=self.database_path,
            profile_store=self.profiles,
            credential_store=self.credentials,
            preparation_service=self.preparation,
            pipeline_service=FailingPipeline(),  # type: ignore[arg-type]
        )
        start = _endpoint(app, "/api/jobs/{job_id}/start", "POST")
        with self.assertRaises(HTTPException) as rejected:
            start("job-api")
        self.assertEqual(
            (400, "The OCR CUDA runtime is unavailable or incompatible with this GPU. Choose Auto or CPU."),
            (rejected.exception.status_code, rejected.exception.detail),
        )

    def test_start_requires_confirmed_workspace_and_runs_mandatory_export(self) -> None:
        preflight = _endpoint(self.app, "/api/jobs/preflight", "POST")
        confirm = _endpoint(self.app, "/api/jobs/{job_id}/confirm-workspace", "POST")
        start = _endpoint(self.app, "/api/jobs/{job_id}/start", "POST")
        config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(Path(self.temporary.name) / "dataset"))
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        job_id = preflight(_PreflightBody(config=config.to_dict()))["jobId"]
        with self.assertRaises(HTTPException):
            start(job_id)
        confirm(job_id, _WorkspaceBody(confirmed=True, confirmedRebuild=False))
        self.assertEqual({"jobId": job_id, "started": True}, start(job_id))
        summaries = self._wait_for_pipeline_completion(job_id, "succeeded")
        self.assertEqual(9, len(summaries))

    def test_restore_original_annotations_requires_confirmation_and_uses_task_backup(self) -> None:
        restore = _endpoint(self.app, "/api/jobs/{job_id}/restore-original-annotations", "POST")
        dataset = Path(self.temporary.name) / "dataset"
        original_json, original_txt = b'{"tags":["original"]}\n', b"original\n"
        (dataset / "image.json").write_bytes(original_json)
        (dataset / "image.txt").write_bytes(original_txt)
        backup_dir = dataset.parent / ".dataset.anima-backups"
        backup_dir.mkdir()
        write_backup(dataset, backup_dir / "job-api.zip", lambda cursor: [{"sample_id": 1, "annotation_key": "image", "in_processing_scope": True}] if cursor is None else [])
        (dataset / "image.json").write_bytes(b'{"tags":["new"]}\n')
        (dataset / "image.txt").write_bytes(b"new\n")
        database = StateDatabase.open(self.database_path)
        try:
            # A committed task keeps no overlay (F10), so restoration must not depend on one.
            database.connection.execute("UPDATE jobs SET overlay_root=NULL,dataset_root=?,status='succeeded' WHERE job_id='job-api'", (str(dataset),))
        finally:
            database.close()
        with self.assertRaises(HTTPException):
            restore("job-api", _ConfirmBody(confirmed=False))
        response = restore("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(("job-api", 2), (response["jobId"], response["restored"]))
        self.assertEqual(original_json, (dataset / "image.json").read_bytes())
        self.assertEqual(original_txt, (dataset / "image.txt").read_bytes())
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute("UPDATE jobs SET status='succeeded_with_issues' WHERE job_id='job-api'")
        finally:
            database.close()
        partial_response = restore("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(("job-api", 2), (partial_response["jobId"], partial_response["restored"]))

    def test_partial_success_snapshot_does_not_expose_repair_preview(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute("UPDATE jobs SET status='succeeded_with_issues' WHERE job_id='job-api'")
        finally:
            database.close()
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")(
            "job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200,
        )
        self.assertIsNone(snapshot["repairPreview"])

    def test_snapshot_exposes_bounded_export_summary_without_issue_payload(self) -> None:
        dataset = Path(self.temporary.name) / "dataset"
        layout = OverlayLayout.create(dataset, "job-api")
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute("UPDATE jobs SET overlay_root=?, status='exporting', current_module_id='export' WHERE job_id='job-api'", (str(layout.root),))
            database.connection.execute("INSERT INTO module_summary(job_id,module_id,status,completed,failed,skipped,issue_count) VALUES('job-api','export','completed',12,3,4,3)")
            write_journal(layout, CommitJournal("job-api", "backup_verified", dataset, dataset.parent / ".dataset.anima-stage-jobapi", dataset.parent / ".dataset.anima-rollback-jobapi", dataset.parent / ".dataset.anima-backups" / "job-api.zip"))
        finally:
            database.close()
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200)
        self.assertEqual({"format":"both","commitStatus":"backup_verified","scanned":19,"valid":12,"invalid":3,"exported":0,"skipped":4,"issueCount":3,"issuesPageEndpoint":"/api/jobs/job-api?issueAfterSampleId=0","convertedSamples":0,"conversions":{},"unlabeled":0,"unlabeledFolder":None}, snapshot["exportSummary"])

    def test_snapshot_uses_event_cursor_and_keyset_paged_issues_with_attempts(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            config_hash = str(database.get_job("job-api")["config_hash"])
            for event_id in (1, 2, 3):
                database.append_event(ProgressEvent("job-api", event_id, "nl", "running", event_id, 3, config_hash, event_id))
            # The production ring retains a bounded suffix. Simulate that truncation directly.
            database.connection.execute("DELETE FROM event_ring WHERE job_id='job-api' AND event_id=1")
            for sample_id, attempt in ((1, 1), (2, 2), (3, 3)):
                database.upsert_issue(SampleIssue(
                    issueId=f"issue-{sample_id}", jobId="job-api", sampleId=sample_id,
                    relativeImagePath=f"{sample_id}.png", moduleId="nl", code="api_retry",
                    severity="error", blocking=True, retriable=True, message="retry", attempt=attempt,
                    repairStartModule="nl",
                ))
        finally:
            database.close()

        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")
        first = snapshot("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=2)
        self.assertTrue(first["snapshotRequired"])
        self.assertEqual([2, 3], [event["event_id"] for event in first["events"]])
        self.assertEqual([(1, 1), (2, 2)], [(issue["sample_id"], issue["attempt"]) for issue in first["issues"]])
        self.assertEqual(2, first["nextIssueAfterSampleId"])
        second = snapshot("job-api", afterEventId=first["nextAfterEventId"], issueAfterSampleId=first["nextIssueAfterSampleId"], issueAfterIssueId=first["nextIssueAfterIssueId"], limit=2)
        self.assertFalse(second["snapshotRequired"])
        self.assertEqual([(3, 3)], [(issue["sample_id"], issue["attempt"]) for issue in second["issues"]])

    def test_pin_cancel_and_discard_use_persisted_lifecycle_guards(self) -> None:
        pin = _endpoint(self.app, "/api/jobs/{job_id}/pin", "PUT")
        cancel = _endpoint(self.app, "/api/jobs/{job_id}/cancel", "POST")
        discard = _endpoint(self.app, "/api/jobs/{job_id}/discard", "POST")
        released: list[str] = []

        def release_lock(job_id: str) -> bool:
            released.append(job_id)
            return False

        self.preparation.release_lock_for_discard = release_lock  # type: ignore[method-assign]
        self.assertEqual({"pinned": True}, pin("job-api", _PinBody(pinned=True)))
        self.assertEqual({"status": "cancelled_recoverable"}, cancel("job-api"))
        with self.assertRaises(HTTPException):
            discard("job-api", _ConfirmBody(confirmed=False))
        self.assertEqual({"jobId": "job-api", "overlayDeleted": False}, discard("job-api", _ConfirmBody(confirmed=True)))
        self.assertEqual(["job-api"], released)

    def test_discard_retry_finishes_live_lock_release_after_persisted_discard(self) -> None:
        discard = _endpoint(self.app, "/api/jobs/{job_id}/discard", "POST")
        cancel = _endpoint(self.app, "/api/jobs/{job_id}/cancel", "POST")
        attempts: list[str] = []

        def release_lock(job_id: str) -> bool:
            attempts.append(job_id)
            if len(attempts) == 1:
                raise JobPreflightError("injected live lock release failure")
            return True

        self.preparation.release_lock_for_discard = release_lock  # type: ignore[method-assign]
        self.assertEqual({"status": "cancelled_recoverable"}, cancel("job-api"))
        with self.assertRaises(HTTPException) as first:
            discard("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(400, first.exception.status_code)
        database = StateDatabase.open(self.database_path)
        try:
            self.assertEqual("discarded", database.get_job("job-api")["status"])
        finally:
            database.close()

        self.assertEqual(
            {"jobId": "job-api", "overlayDeleted": False},
            discard("job-api", _ConfirmBody(confirmed=True)),
        )
        self.assertEqual(["job-api", "job-api"], attempts)
        database = StateDatabase.open(self.database_path)
        try:
            with self.assertRaises(KeyError):
                database.get_job("job-api")
        finally:
            database.close()
        with self.assertRaises(HTTPException) as missing:
            discard("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(404, missing.exception.status_code)

    def test_discard_deletes_record_after_releasing_claim_and_lock(self) -> None:
        discard = _endpoint(self.app, "/api/jobs/{job_id}/discard", "POST")
        cancel = _endpoint(self.app, "/api/jobs/{job_id}/cancel", "POST")
        released: list[str] = []
        self.preparation.release_lock_for_discard = lambda job_id: released.append(job_id) or False  # type: ignore[method-assign]
        self.assertEqual({"status": "cancelled_recoverable"}, cancel("job-api"))

        self.assertEqual(
            {"jobId": "job-api", "overlayDeleted": False},
            discard("job-api", _ConfirmBody(confirmed=True)),
        )
        self.assertEqual(["job-api"], released)
        database = StateDatabase.open(self.database_path)
        try:
            with self.assertRaises(KeyError):
                database.get_job("job-api")
        finally:
            database.close()

    def test_delete_history_releases_completed_lock_before_removing_control_record(self) -> None:
        delete_history = _endpoint(self.app, "/api/jobs/{job_id}/delete-history", "POST")
        released: list[str] = []
        self.preparation.release_lock_for_history_deletion = lambda job_id: released.append(job_id) or False  # type: ignore[method-assign]
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute(
                "UPDATE jobs SET status='succeeded', finished_at='2026-07-24T00:01:00Z' WHERE job_id='job-api'"
            )
        finally:
            database.close()

        self.assertEqual(
            {
                "jobId": "job-api",
                "deletedControlRecord": True,
                "preserved": {"dataset": True, "exports": True, "backup": True},
            },
            delete_history("job-api", _ConfirmBody(confirmed=True)),
        )
        self.assertEqual(["job-api"], released)
        database = StateDatabase.open(self.database_path)
        try:
            with self.assertRaises(KeyError):
                database.get_job("job-api")
        finally:
            database.close()

    def test_delete_history_keeps_the_control_record_when_confirmation_or_lock_release_fails(self) -> None:
        delete_history = _endpoint(self.app, "/api/jobs/{job_id}/delete-history", "POST")
        released: list[str] = []
        self.preparation.release_lock_for_history_deletion = lambda job_id: released.append(job_id) or (_ for _ in ()).throw(  # type: ignore[method-assign]
            JobPreflightError("injected completed lock release failure")
        )
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute(
                "UPDATE jobs SET status='succeeded', finished_at='2026-07-24T00:01:00Z' WHERE job_id='job-api'"
            )
        finally:
            database.close()

        with self.assertRaises(HTTPException) as unconfirmed:
            delete_history("job-api", _ConfirmBody(confirmed=False))
        self.assertEqual(400, unconfirmed.exception.status_code)
        self.assertEqual([], released)
        with self.assertRaises(HTTPException) as failed_release:
            delete_history("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(400, failed_release.exception.status_code)
        self.assertEqual(["job-api"], released)
        database = StateDatabase.open(self.database_path)
        try:
            self.assertEqual("succeeded", database.get_job("job-api")["status"])
        finally:
            database.close()

    def test_delete_history_rejects_parent_when_repair_child_exists_without_releasing_locks(self) -> None:
        delete_history = _endpoint(self.app, "/api/jobs/{job_id}/delete-history", "POST")
        released: list[str] = []
        self.preparation.release_lock_for_history_deletion = lambda job_id: released.append(job_id) or False  # type: ignore[method-assign]
        database = StateDatabase.open(self.database_path)
        try:
            template = dict(database.get_job("job-api"))
            database.insert_job({
                **template,
                "job_id": "repair-api",
                "status": "succeeded",
                "finished_at": "2026-07-24T00:01:00Z",
            })
            database.create_repair_link("repair-api", "job-api")
            database.connection.execute(
                "UPDATE jobs SET status='succeeded', finished_at='2026-07-24T00:01:00Z' WHERE job_id='job-api'"
            )
        finally:
            database.close()

        with self.assertRaises(HTTPException) as rejected:
            delete_history("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(400, rejected.exception.status_code)
        self.assertIn("请先删除修复子任务", str(rejected.exception.detail))
        self.assertEqual([], released)
        database = StateDatabase.open(self.database_path)
        try:
            self.assertEqual("succeeded", database.get_job("job-api")["status"])
            self.assertEqual("succeeded", database.get_job("repair-api")["status"])
            self.assertTrue(database.has_repair_children("job-api"))
        finally:
            database.close()

    def test_delete_history_for_repair_child_releases_only_repair_service_lock(self) -> None:
        repair_service = importlib.import_module("anima_core.repair").RepairPreparationService(self.database_path)
        app = build_control_app(
            database_path=self.database_path,
            profile_store=self.profiles,
            credential_store=self.credentials,
            preparation_service=self.preparation,
            pipeline_service=self.pipeline,
            repair_service=repair_service,
        )
        delete_history = _endpoint(app, "/api/jobs/{job_id}/delete-history", "POST")
        preparation_releases: list[str] = []
        repair_releases: list[str] = []
        self.preparation.release_lock_for_history_deletion = lambda job_id: preparation_releases.append(job_id) or False  # type: ignore[method-assign]
        repair_service.release_lock_for_history_deletion = lambda job_id: repair_releases.append(job_id) or False  # type: ignore[method-assign]
        database = StateDatabase.open(self.database_path)
        try:
            template = dict(database.get_job("job-api"))
            database.insert_job({
                **template,
                "job_id": "repair-api",
                "status": "succeeded",
                "finished_at": "2026-07-24T00:01:00Z",
            })
            database.create_repair_link("repair-api", "job-api")
        finally:
            database.close()

        self.assertEqual(
            {
                "jobId": "repair-api",
                "deletedControlRecord": True,
                "preserved": {"dataset": True, "exports": True, "backup": True},
            },
            delete_history("repair-api", _ConfirmBody(confirmed=True)),
        )
        self.assertEqual([], preparation_releases)
        self.assertEqual(["repair-api"], repair_releases)
        database = StateDatabase.open(self.database_path)
        try:
            with self.assertRaises(KeyError):
                database.get_job("repair-api")
            self.assertEqual("job-api", database.get_job("job-api")["job_id"])
            self.assertFalse(database.has_repair_children("job-api"))
        finally:
            database.close()

    def test_parent_discard_is_rejected_when_repair_child_exists(self) -> None:
        discard = _endpoint(self.app, "/api/jobs/{job_id}/discard", "POST")
        database = StateDatabase.open(self.database_path)
        try:
            template = dict(database.get_job("job-api"))
            database.insert_job({**template, "job_id": "repair-api", "status": "cancelled_recoverable"})
            database.create_repair_link("repair-api", "job-api")
        finally:
            database.close()

        with self.assertRaises(HTTPException) as rejected:
            discard("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(400, rejected.exception.status_code)
        self.assertIn("\u8bf7\u5148\u5220\u9664\u4fee\u590d\u5b50\u4efb\u52a1", str(rejected.exception.detail))
        database = StateDatabase.open(self.database_path)
        try:
            self.assertEqual("running", database.get_job("job-api")["status"])
            self.assertTrue(database.has_repair_children("job-api"))
        finally:
            database.close()

    def test_discard_rechecks_repair_children_before_side_effects(self) -> None:
        discard = _endpoint(self.app, "/api/jobs/{job_id}/discard", "POST")
        cancel = _endpoint(self.app, "/api/jobs/{job_id}/cancel", "POST")
        lifecycle_module = importlib.import_module("anima_core.lifecycle")
        original_discard = lifecycle_module.JobLifecycle.discard
        self.assertEqual({"status": "cancelled_recoverable"}, cancel("job-api"))

        def add_repair_child_before_discard(lifecycle, job_id: str, *, confirmed: bool):
            database = StateDatabase.open(self.database_path)
            try:
                template = dict(database.get_job(job_id))
                database.insert_job({**template, "job_id": "repair-api", "status": "cancelled_recoverable"})
                database.create_repair_link("repair-api", job_id)
            finally:
                database.close()
            return original_discard(lifecycle, job_id, confirmed=confirmed)

        with patch("anima_core.api_jobs.JobLifecycle.discard", new=add_repair_child_before_discard):
            with self.assertRaises(HTTPException) as rejected:
                discard("job-api", _ConfirmBody(confirmed=True))
        self.assertEqual(400, rejected.exception.status_code)
        self.assertIn("\u8bf7\u5148\u5220\u9664\u4fee\u590d\u5b50\u4efb\u52a1", str(rejected.exception.detail))
        database = StateDatabase.open(self.database_path)
        try:
            self.assertEqual("cancelled_recoverable", database.get_job("job-api")["status"])
            self.assertTrue(database.has_repair_children("job-api"))
        finally:
            database.close()

    def test_repair_link_rejects_a_discarded_parent(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            template = dict(database.get_job("job-api"))
            database.connection.execute("UPDATE jobs SET status='discarded' WHERE job_id='job-api'")
            database.insert_job({**template, "job_id": "repair-api", "status": "cancelled_recoverable"})
            with self.assertRaises(ValueError):
                database.create_repair_link("repair-api", "job-api")
        finally:
            database.close()

    def test_repair_child_can_be_discarded_without_deleting_parent(self) -> None:
        discard = _endpoint(self.app, "/api/jobs/{job_id}/discard", "POST")
        database = StateDatabase.open(self.database_path)
        try:
            template = dict(database.get_job("job-api"))
            database.insert_job({**template, "job_id": "repair-api", "status": "cancelled_recoverable"})
            database.create_repair_link("repair-api", "job-api")
        finally:
            database.close()
        self.assertEqual(
            {"jobId": "repair-api", "overlayDeleted": False},
            discard("repair-api", _ConfirmBody(confirmed=True)),
        )
        database = StateDatabase.open(self.database_path)
        try:
            with self.assertRaises(KeyError):
                database.get_job("repair-api")
            self.assertEqual("running", database.get_job("job-api")["status"])
            self.assertFalse(database.has_repair_children("job-api"))
        finally:
            database.close()

    def test_startup_freezes_jobs_an_earlier_process_left_running(self) -> None:
        # F15: the control plane must run the whole startup recovery sequence, not only
        # `recover_pending_commits`, which never marked an abandoned job as interrupted.
        build_control_app(
            database_path=self.database_path, profile_store=self.profiles,
            credential_store=self.credentials, preparation_service=self.preparation,
        )
        database = StateDatabase.open(self.database_path)
        try:
            job = database.get_job("job-api")
            self.assertEqual(("interrupted", "running"), (job["status"], job["resume_status"]))
        finally:
            database.close()

    def test_snapshot_exposes_frozen_job_state_fields_and_pending_api_decisions(self) -> None:
        # F41 / F28: ROADMAP.md:1233-1246 JobState fields plus the possibly-billed request count.
        database = StateDatabase.open(self.database_path)
        try:
            config_hash = str(database.get_job("job-api")["config_hash"])
            database.connection.execute(
                "UPDATE sample_state SET current_module_id='nl',status='request_started' WHERE job_id='job-api'"
            )
        finally:
            database.close()
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200)
        self.assertEqual(1, snapshot["nlPendingApiDecisions"])
        job = snapshot["job"]
        self.assertEqual((config_hash, 1, "2026-07-24T00:00:00Z"), (job["configHash"], job["manifestSchemaVersion"], job["createdAt"]))
        self.assertIsNotNone(job["startedAt"])
        self.assertEqual((None, None), (job["cancelRequestedAt"], job["finishedAt"]))

    def test_snapshot_pages_issues_by_the_composite_sample_and_issue_cursor(self) -> None:
        # F33: two issues on one sample used to lose the second one at a page boundary.
        database = StateDatabase.open(self.database_path)
        try:
            for issue_id, code in (("issue-a", "api_retry"), ("issue-b", "api_timeout")):
                database.upsert_issue(SampleIssue(
                    issueId=issue_id, jobId="job-api", sampleId=1, relativeImagePath="image.png",
                    moduleId="nl", code=code, severity="error", blocking=True, retriable=True,
                    message="retry", attempt=1, repairStartModule="nl",
                ))
        finally:
            database.close()
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")
        first = snapshot("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=1)
        self.assertEqual([("issue-a", 1)], [(issue["issue_id"], issue["sample_id"]) for issue in first["issues"]])
        self.assertEqual((1, "issue-a"), (first["nextIssueAfterSampleId"], first["nextIssueAfterIssueId"]))
        second = snapshot(
            "job-api", afterEventId=0, issueAfterSampleId=first["nextIssueAfterSampleId"],
            issueAfterIssueId=first["nextIssueAfterIssueId"], limit=1,
        )
        self.assertEqual(["issue-b"], [issue["issue_id"] for issue in second["issues"]])

    def test_snapshot_treats_a_released_overlay_as_a_normal_terminal_state(self) -> None:
        # F10: a succeeded task whose overlay was discarded must not turn the snapshot into a 400.
        database = StateDatabase.open(self.database_path)
        try:
            database.connection.execute(
                "UPDATE jobs SET overlay_root=?,status='succeeded' WHERE job_id='job-api'",
                (str(Path(self.temporary.name) / "gone" / ".dataset.anima-overlay-job-api"),),
            )
        finally:
            database.close()
        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=200)
        self.assertEqual("succeeded", snapshot["job"]["status"])
        self.assertEqual("committed", snapshot["exportSummary"]["commitStatus"])

    def test_job_list_pages_newest_first_by_a_composite_cursor(self) -> None:
        # F40: without a task list a refreshed page can only retype a 32 hex-digit jobId.
        database = StateDatabase.open(self.database_path)
        try:
            template = dict(database.get_job("job-api"))
            for job_id, created_at in (("job-older", "2026-07-23T00:00:00Z"), ("job-newer", "2026-07-25T00:00:00Z")):
                database.insert_job({**template, "job_id": job_id, "created_at": created_at, "status": "succeeded"})
            database.insert_job({**template, "job_id": "repair-older", "created_at": "2026-07-22T00:00:00Z", "status": "cancelled_recoverable"})
            database.create_repair_link("repair-older", "job-api")
        finally:
            database.close()
        listing = _endpoint(self.app, "/api/jobs", "GET")
        first = listing(afterCreatedAt=None, afterJobId=None, limit=2)
        self.assertEqual(["job-newer", "job-api"], [job["jobId"] for job in first["jobs"]])
        self.assertTrue(all("profile" not in job for job in first["jobs"]))
        self.assertEqual(("2026-07-24T00:00:00Z", "job-api"), (first["nextAfterCreatedAt"], first["nextAfterJobId"]))
        second = listing(afterCreatedAt=first["nextAfterCreatedAt"], afterJobId=first["nextAfterJobId"], limit=2)
        self.assertEqual(["job-older"], [job["jobId"] for job in second["jobs"]])
        self.assertEqual((None, None), (second["nextAfterCreatedAt"], second["nextAfterJobId"]))
        with self.assertRaises(HTTPException):
            listing(afterCreatedAt="2026-07-24T00:00:00Z", afterJobId=None, limit=2)

    def test_job_hierarchy_hides_repair_children_from_list_and_exposes_snapshot_links(self) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            template = dict(database.get_job("job-api"))
            database.insert_job({
                **template,
                "job_id": "repair-api",
                "created_at": "2026-07-25T00:00:00Z",
                "status": "cancelled_recoverable",
            })
            database.create_repair_link("repair-api", "job-api")
        finally:
            database.close()

        listing = _endpoint(self.app, "/api/jobs", "GET")
        page = listing(afterCreatedAt=None, afterJobId=None, limit=20)
        self.assertEqual(["job-api"], [job["jobId"] for job in page["jobs"]])

        snapshot = _endpoint(self.app, "/api/jobs/{job_id}", "GET")
        parent = snapshot("job-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=50)
        self.assertEqual("repair-api", parent["repairChildren"][0]["jobId"])
        self.assertEqual(0, parent["repairChildren"][0]["targetCount"])
        child = snapshot("repair-api", afterEventId=0, issueAfterSampleId=0, issueAfterIssueId=None, limit=50)
        self.assertEqual("job-api", child["job"]["parentJobId"])

    def test_preflight_response_exposes_every_frozen_preflight_page_field(self) -> None:
        # F39: ROADMAP.md:1224 requires blank counts, projection, estimates and API bounds.
        preflight = _endpoint(self.app, "/api/jobs/preflight", "POST")
        config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(Path(self.temporary.name) / "dataset"))
        config.nl["systemPrompt"] = "describe"
        result = preflight(_PreflightBody(config=config.to_dict()))
        self.assertLessEqual(
            {"blankTxtCount", "blankJsonCount", "annotationKeyCollisionCount", "imageIssueCount", "projection", "estimate", "api"},
            set(result),
        )
        self.assertEqual("both", result["projection"]["format"])
        self.assertIn("incrementalWriteBytes", result["estimate"])
        self.assertIn("httpAttemptBudget", result["api"])

    def test_default_nl_prompt_endpoint_serves_the_version_frozen_resource(self) -> None:
        # F25: the editor default and "restore default" have exactly one source of truth.
        prompt = _endpoint(self.app, "/api/nl/default-prompt", "GET")()
        self.assertEqual("nl-default-prompt-v2", prompt["promptVersion"])
        self.assertEqual(64, len(str(prompt["sha256"])))
        self.assertIn("120-180+ words", str(prompt["systemPrompt"]))

    def test_default_nl_prompt_endpoint_keeps_v2_default_and_allows_explicit_v3(self) -> None:
        endpoint = _endpoint(self.app, "/api/nl/default-prompt", "GET")
        self.assertIn("promptVersion", inspect.signature(endpoint).parameters)
        v2 = endpoint()
        v3 = endpoint(promptVersion="nl-default-prompt-v3")
        self.assertEqual("nl-default-prompt-v2", v2["promptVersion"])
        self.assertEqual("nl-default-prompt-v3", v3["promptVersion"])
        self.assertNotIn("120-180+ words", str(v3["systemPrompt"]))

    def test_default_nl_prompt_endpoint_exposes_each_v4_task_preset(self) -> None:
        endpoint = _endpoint(self.app, "/api/nl/default-prompt", "GET")
        for preset in ("general", "style", "character"):
            version = f"nl-default-prompt-v4-{preset}"
            expected = (ROOT / "packaging" / "resources" / f"{version}.txt").read_text(encoding="utf-8").replace("\r\n", "\n").strip()
            result = endpoint(promptVersion=version)
            self.assertEqual(version, result["promptVersion"])
            self.assertEqual(expected, result["systemPrompt"])
            self.assertEqual(64, len(str(result["sha256"])))

    def test_select_path_route_forwards_each_allowed_purpose_and_returns_selected_path(self) -> None:
        picker = FakeNativePathPicker(result=r"E:\\picked\\result")
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, native_path_picker=picker,  # type: ignore[arg-type]
        )
        endpoint = _endpoint(app, "/api/application/select-path", "POST")

        expected_calls = [
            ("source_dataset", r"E:\\typed\\source"),
            ("output_dataset", r"E:\\typed\\output"),
            ("replacement_csv", r"E:\\typed\\rules.csv"),
            ("classification_resource_json", r"E:\\typed\\resource.json"),
        ]
        for purpose, current_path in expected_calls:
            self.assertEqual(
                {"cancelled": False, "path": r"E:\\picked\\result"},
                endpoint({"purpose": purpose, "currentPath": current_path}),
            )
        self.assertEqual(expected_calls, picker.calls)

    def test_select_path_route_reports_cancellation_without_substituting_a_value(self) -> None:
        picker = FakeNativePathPicker(result=None)
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, native_path_picker=picker,  # type: ignore[arg-type]
        )
        endpoint = _endpoint(app, "/api/application/select-path", "POST")

        self.assertEqual(
            {"cancelled": True, "path": None},
            endpoint({"purpose": "output_dataset", "currentPath": r"E:\\typed\\output"}),
        )
        self.assertEqual([("output_dataset", r"E:\\typed\\output")], picker.calls)

    def test_select_path_route_rejects_invalid_or_oversized_bodies_with_400(self) -> None:
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, native_path_picker=FakeNativePathPicker(),  # type: ignore[arg-type]
        )
        endpoint = _endpoint(app, "/api/application/select-path", "POST")

        for body in (
            {},
            {"purpose": "shell", "currentPath": None},
            {"purpose": "source_dataset", "currentPath": None, "extra": True},
            {"purpose": "source_dataset", "currentPath": "x" * 32_768},
        ):
            with self.assertRaises(HTTPException) as rejected:
                endpoint(body)
            self.assertEqual((400, "invalid_path_picker_request"), (rejected.exception.status_code, rejected.exception.detail))

    def test_select_path_route_maps_busy_and_unavailable_without_leaking_native_errors(self) -> None:
        for error, expected in (
            (NativePathPickerBusyError("dialog already open"), (409, "path_picker_busy")),
            (NativePathPickerUnavailableError("raw Tcl failure"), (503, "path_picker_unavailable")),
        ):
            app = build_control_app(
                database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
                preparation_service=self.preparation, native_path_picker=FakeNativePathPicker(error=error),  # type: ignore[arg-type]
            )
            endpoint = _endpoint(app, "/api/application/select-path", "POST")
            with self.assertRaises(HTTPException) as rejected:
                endpoint({"purpose": "source_dataset", "currentPath": r"E:\\typed"})
            self.assertEqual(expected, (rejected.exception.status_code, rejected.exception.detail))

    def test_desktop_shutdown_is_token_gated_and_static_files_follow_api_routes(self) -> None:
        static_root = Path(self.temporary.name) / "frontend"
        static_root.mkdir()
        (static_root / "index.html").write_text("<!doctype html><title>Anima</title>", encoding="utf-8")
        callbacks: list[str] = []
        app = build_control_app(
            database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
            preparation_service=self.preparation, static_root=static_root, shutdown_token="a" * 64,
            shutdown_callback=lambda: callbacks.append("called"),
        )
        shutdown = _endpoint(app, "/api/application/shutdown", "POST")
        with self.assertRaises(HTTPException) as rejected:
            shutdown(_ShutdownBody(token="b" * 64))
        self.assertEqual(404, rejected.exception.status_code)
        self.assertEqual({"accepted": True}, shutdown(_ShutdownBody(token="a" * 64)))
        self.assertEqual(["called"], callbacks)
        mount = next(route for route in app.routes if isinstance(route, Mount))
        self.assertEqual("", mount.path)
        self.assertIs(app.routes[-1], mount)
        with self.assertRaisesRegex(ValueError, "index.html"):
            build_control_app(
                database_path=self.database_path, profile_store=self.profiles, credential_store=self.credentials,
                preparation_service=self.preparation, static_root=Path(self.temporary.name) / "missing",
            )


if __name__ == "__main__":
    unittest.main()
