from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_core.db import StateDatabase, migrate
from anima_core.pipeline import PipelineError, PipelineService
from anima_core.resource_catalog import ResourceCatalog, ResourcePackage

import anima_core.db_count_review  # noqa: F401
import anima_core.db_jobs  # noqa: F401
import anima_core.db_nl_export  # noqa: F401
import anima_core.db_primitives  # noqa: F401
import anima_core.db_scheduler  # noqa: F401
import anima_core.db_schema  # noqa: F401
import anima_core.pipeline_dispatch  # noqa: F401
import anima_core.pipeline_recovery  # noqa: F401
import anima_core.resource_catalog_package  # noqa: F401
import anima_core.resource_catalog_validation  # noqa: F401


STATE_DATABASE_API = frozenset({
    "_count_limit", "_insert_count_observation_consistent", "_insert_sample_batch", "_limit",
    "_repair_target_clause", "_resolve_count_review_decision_in_transaction", "add_api_budget_extra",
    "append_event", "begin_cancellation", "checkpoint", "claim_leases", "clear_cancellation_metadata", "clear_manifest_rows",
    "clear_stale_dataset_claims", "clear_workspace_metadata", "close", "complete_leased_sample",
    "complete_leased_sample_and_count", "confirm_count_review", "confirm_nl_unknown_requests", "count",
    "count_current_review_decisions", "count_current_review_targets", "count_in_flight",
    "count_module_samples", "count_module_unsettled", "count_processing_samples", "count_repair_targets",
    "count_unresolved_blocking_issues", "create_repair_link", "delete_job_control_record", "delete_staged_nl",
    "create_preflight_progress", "update_preflight_progress", "get_preflight_progress",
    "event_page", "event_snapshot_required", "fail_leased_sample_with_issue", "get_count_evidence",
    "get_count_observation", "get_count_review_decision", "get_job", "get_leased_sample", "get_runtime_evidence", "has_repair_children",
    "get_sample_state", "get_sample_with_state", "heartbeat", "increment_module_counts",
    "increment_module_diagnostic", "increment_worker_restart", "inherit_repair_count_evidence",
    "initialize_module_summary", "insert_initial_count_review_decisions", "insert_job", "insert_samples",
    "mark_interrupted", "mark_nl_request_started", "module_diagnostic_count", "module_diagnostics",
    "module_summaries", "module_summary", "open", "open_default", "page_active_jobs",
    "page_count_evidence", "page_count_observations", "page_count_review_decisions",
    "page_count_review_inputs", "page_count_review_items", "page_export_artifact_groups",
    "page_export_artifacts", "page_issues", "page_overlay_jobs", "page_samples",
    "pause_active_module",
    "preflight_projection_counts", "record_count_observation_not_requested", "record_migration",
    "record_nl_disabled_observations_page", "record_nl_outcome", "recovery_state_page",
    "release_dataset_claim", "repair_candidate_summary", "repair_children", "repair_parent_cursor", "repair_parent_job_id",
    "reset_next_module_page", "resolve_issue", "resolve_repaired_parent_issues", "return_expired_leases",
    "return_lease_to_pending", "return_module_leases", "return_unsubmitted_nl_request", "set_job_status",
    "set_module_diagnostic_count", "set_module_summary", "set_pinned", "set_runtime_evidence", "set_sample_state",
    "set_workspace_metadata", "settle_cancellation", "skip_disabled_caption_page",
    "skip_failed_samples_for_export_page", "skip_leased_caption_sample", "stage_classify_prepared_artifact",
    "stage_export_artifacts", "stage_nl_response", "stage_prepared_artifact", "stage_repair_target_page",
    "cancel_future_pause", "pause_future_module", "restore_prepaused_module", "resume_paused_module",
    "resume_prepaused_module", "staged_nl", "start_prepaused_module", "transaction", "update_count_review_decision", "update_count_review_decisions",
    "update_preflight_config", "upsert_issue",
})

PIPELINE_SERVICE_API = frozenset({
    "_active_module_status", "_nl_credentials", "_ocr_binding_path", "_probe_ocr_gpu_runtime", "_recover_policy_prepared", "_recovery_target_status", "_resolve_ocr_runtime", "_run", "_run_active_module",
    "_run_module_with_restarts", "_selected_catalog_package", "_selected_classify_resource", "_selected_resource",
    "_spawn_transport", "_thread_main",
    "_spawn_ocr_transport", "_select_ocr_runtime", "_token_budget_export_gate", "_verify_source_fingerprints", "close", "confirm_count_review", "is_running", "pause", "pause_module", "recover_job",
    "recover_pending_commits", "resume", "resume_module", "shutdown", "start", "startup_recovery",
})

RESOURCE_CATALOG_API = frozenset({"_defaults", "scan"})

RESOURCE_PACKAGE_API = frozenset({
    "_load_tokenizer", "adjustable_categories", "api_dict", "context_limit", "default_thresholds",
    "entrypoint", "excluded_categories", "load", "official_model_id", "revision",
    "resolve_relative", "root_relative_path", "tokenizer_family", "verify_files",
})

RESOURCE_VALIDATION_FUNCTIONS = frozenset({
    "_category_list", "_distribution", "_https_url", "_localized", "_metadata",
    "_positive", "_relative", "_sha256", "_text",
})

API_REQUEST_MODELS = frozenset({
    "_AmountBody", "_ConfirmBody", "_CountReviewBatchBody", "_CountReviewBatchItem",
    "_CountReviewDecisionBody", "_NlDiagnosticCredentialsBody", "_NlManualRetryBatchBody", "_NlManualRetryBody", "_NlManualWriteBody", "_NlModelDiscoveryBody",
    "_NlPromptPresetBody", "_NlTestMessageBody", "_PinBody", "_PreflightBody", "_ProfileBody", "_SelectPathBody",
    "_SecretBody", "_ShutdownBody", "_TokenBudgetApplyBody", "_TokenBudgetRecountBody",
    "_TokenBudgetRewriteShortBody", "_WorkspaceBody",
})
API_MODEL_CLASSES = API_REQUEST_MODELS | frozenset({
    "CreateJobRequest",
    "DeviceRecommendationGpu",
    "DeviceRecommendationResponse",
    "SelectPathRequest",
})

API_APPLICATION_ROUTES = frozenset({
    ("GET", "/health"),
    ("GET", "/api/health"),
    ("GET", "/api/resources"),
    ("GET", "/api/application/device-recommendations"),
    ("POST", "/api/jobs/{job_id}/policy/pause"),
    ("POST", "/api/jobs/{job_id}/policy/resume"),
    ("POST", "/api/application/select-path"),
    ("POST", "/api/application/shutdown"),
})

API_COUNT_REVIEW_ROUTES = frozenset({
    ("GET", "/api/jobs/{job_id}/count-review"),
    ("GET", "/api/jobs/{job_id}/count-review/{sample_id}/image"),
    ("PUT", "/api/jobs/{job_id}/count-review/{sample_id}"),
    ("POST", "/api/jobs/{job_id}/count-review/batch"),
    ("POST", "/api/jobs/{job_id}/count-review/confirm"),
})

API_JOB_ROUTES = frozenset({
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
    ("POST", "/api/jobs/{job_id}/repair"),
    ("POST", "/api/jobs/{job_id}/recover"),
    ("POST", "/api/jobs/{job_id}/cancel"),
    ("PUT", "/api/jobs/{job_id}/pin"),
    ("POST", "/api/jobs/{job_id}/discard"),
    ("POST", "/api/jobs/{job_id}/delete-history"),
    ("POST", "/api/jobs/{job_id}/restore-original-annotations"),
})

API_NL_ROUTES = frozenset({
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
    ("DELETE", "/api/nl/profiles/{profile_id}"),
    ("PUT", "/api/nl/credentials/{reference}"),
    ("DELETE", "/api/nl/credentials/{reference}"),
    ("POST", "/api/jobs/{job_id}/nl/pause"),
    ("POST", "/api/jobs/{job_id}/nl/resume"),
    ("POST", "/api/jobs/{job_id}/nl/api-budget"),
    ("POST", "/api/jobs/{job_id}/nl/confirm-api-outcomes"),
    ("POST", "/api/jobs/{job_id}/nl/manual-retry"),
    ("POST", "/api/jobs/{job_id}/nl/manual-retry-batch"),
    ("POST", "/api/jobs/{job_id}/nl/manual-write"),
})

API_TOKEN_BUDGET_ROUTES = frozenset({
    ("GET", "/api/jobs/{job_id}/token-budget/reviews"),
    ("POST", "/api/jobs/{job_id}/token-budget/recount"),
    ("POST", "/api/jobs/{job_id}/token-budget/rewrite-short"),
    ("POST", "/api/jobs/{job_id}/token-budget/apply"),
})

API_ROUTER_OWNERS = {
    "api_application.py": API_APPLICATION_ROUTES,
    "api_jobs.py": API_JOB_ROUTES,
    "api_count_review.py": API_COUNT_REVIEW_ROUTES,
    "api_nl.py": API_NL_ROUTES,
    "api_token_budget.py": API_TOKEN_BUDGET_ROUTES,
}

EXPECTED_API_ROUTES = frozenset().union(*API_ROUTER_OWNERS.values())


def _router_routes(tree: ast.Module) -> set[tuple[str, str]]:
    methods = {"get": "GET", "post": "POST", "put": "PUT", "delete": "DELETE"}
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                not isinstance(decorator, ast.Call)
                or not isinstance(decorator.func, ast.Attribute)
                or decorator.func.attr not in methods
                or not decorator.args
                or not isinstance(decorator.args[0], ast.Constant)
                or not isinstance(decorator.args[0].value, str)
            ):
                continue
            routes.add((methods[decorator.func.attr], decorator.args[0].value))
    return routes


class CoreModuleDecompositionTests(unittest.TestCase):
    def test_stable_facades_expose_the_existing_public_types(self) -> None:
        self.assertEqual("anima_core.db", StateDatabase.__module__)
        self.assertEqual("anima_core.pipeline", PipelineService.__module__)
        self.assertTrue(callable(migrate))
        self.assertTrue(issubclass(PipelineError, RuntimeError))
        self.assertTrue(hasattr(ResourceCatalog, "scan"))
        self.assertTrue(hasattr(ResourcePackage, "load"))

    def test_state_database_method_surface_is_unchanged(self) -> None:
        actual = {name for name in dir(StateDatabase) if not name.startswith("__")}
        self.assertEqual(STATE_DATABASE_API, actual)

    def test_pipeline_service_method_surface_is_unchanged(self) -> None:
        actual = {name for name in dir(PipelineService) if not name.startswith("__")}
        self.assertEqual(PIPELINE_SERVICE_API, actual)

    def test_resource_catalog_method_surfaces_are_unchanged(self) -> None:
        catalog_actual = {name for name in dir(ResourceCatalog) if not name.startswith("__")}
        package_actual = {name for name in dir(ResourcePackage) if not name.startswith("__")}
        self.assertEqual(RESOURCE_CATALOG_API, catalog_actual)
        self.assertEqual(RESOURCE_PACKAGE_API, package_actual)

    def test_resource_catalog_implementation_has_the_planned_owners(self) -> None:
        source_root = ROOT / "core" / "src" / "anima_core"
        facade_tree = ast.parse((source_root / "resource_catalog.py").read_text(encoding="utf-8"))
        package_tree = ast.parse((source_root / "resource_catalog_package.py").read_text(encoding="utf-8"))
        validation_tree = ast.parse((source_root / "resource_catalog_validation.py").read_text(encoding="utf-8"))

        facade_names = {
            node.name for node in facade_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        package_classes = {
            node.name for node in package_tree.body
            if isinstance(node, ast.ClassDef)
        }
        validation_functions = {
            node.name for node in validation_tree.body
            if isinstance(node, ast.FunctionDef)
        }

        self.assertEqual({"ResourceFile", "ResourcePackage"}, package_classes)
        self.assertTrue(RESOURCE_VALIDATION_FUNCTIONS <= validation_functions)
        self.assertTrue({"ResourceFile", "ResourcePackage"}.isdisjoint(facade_names))
        self.assertTrue(RESOURCE_VALIDATION_FUNCTIONS.isdisjoint(facade_names))
        self.assertEqual("anima_core.resource_catalog_package", ResourcePackage.__module__)

    def test_database_mixins_do_not_call_methods_owned_by_other_mixins(self) -> None:
        modules = {
            "db_jobs.py": "JobDatabaseMixin",
            "db_scheduler.py": "SchedulerDatabaseMixin",
            "db_count_review.py": "CountReviewDatabaseMixin",
            "db_nl_export.py": "NlExportDatabaseMixin",
        }
        methods_by_module: dict[str, set[str]] = {}
        trees: dict[str, ast.Module] = {}
        for filename, class_name in modules.items():
            tree = ast.parse((ROOT / "core" / "src" / "anima_core" / filename).read_text(encoding="utf-8"))
            trees[filename] = tree
            class_node = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            methods_by_module[filename] = {
                node.name for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

        violations: list[str] = []
        for filename, tree in trees.items():
            foreign_methods = set().union(*(
                methods for owner, methods in methods_by_module.items()
                if owner != filename
            )) - methods_by_module[filename]
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                    if node.func.attr in foreign_methods:
                        violations.append(f"{filename}:{node.lineno}:self.{node.func.attr}")
        self.assertEqual([], violations)

    def test_api_models_and_context_have_explicit_owners_and_stable_api_reexports(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("anima_core.api_models"))
        self.assertIsNotNone(importlib.util.find_spec("anima_core.api_context"))

        source_root = ROOT / "core" / "src" / "anima_core"
        models_tree = ast.parse((source_root / "api_models.py").read_text(encoding="utf-8"))
        context_tree = ast.parse((source_root / "api_context.py").read_text(encoding="utf-8"))
        model_classes = {
            node.name for node in models_tree.body
            if isinstance(node, ast.ClassDef)
        }
        context_classes = {
            node.name for node in context_tree.body
            if isinstance(node, ast.ClassDef)
        }
        context_functions = {
            node.name for node in context_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(API_MODEL_CLASSES, model_classes)
        self.assertEqual({"ControlPlaneContext"}, context_classes)
        self.assertEqual({"bad_request", "not_found", "conflict"}, context_functions)

        api = importlib.import_module("anima_core.api")
        models = importlib.import_module("anima_core.api_models")
        context = importlib.import_module("anima_core.api_context")
        for name in API_REQUEST_MODELS:
            self.assertIs(getattr(api, name), getattr(models, name))
        self.assertEqual("anima_core.api_models", models.CreateJobRequest.__module__)
        self.assertTrue(models.CreateJobRequest.__dataclass_params__.frozen)
        self.assertFalse(hasattr(api, "CreateJobRequest"))
        self.assertIs(api._bad_request, context.bad_request)
        self.assertIs(api._not_found, context.not_found)
        self.assertIs(api._conflict, context.conflict)
        self.assertTrue(context.ControlPlaneContext.__dataclass_params__.frozen)

    def test_api_application_router_owns_only_application_routes(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("anima_core.api_application"))
        source = ROOT / "core" / "src" / "anima_core" / "api_application.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        factories = {
            node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")
        }
        self.assertEqual({"build_application_router"}, factories)
        self.assertEqual(API_APPLICATION_ROUTES, _router_routes(tree))

    def test_api_count_review_router_owns_only_count_review_routes(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("anima_core.api_count_review"))
        source = ROOT / "core" / "src" / "anima_core" / "api_count_review.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        factories = {
            node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")
        }
        self.assertEqual({"build_count_review_router"}, factories)
        self.assertEqual(API_COUNT_REVIEW_ROUTES, _router_routes(tree))

    def test_api_jobs_router_owns_only_job_routes(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("anima_core.api_jobs"))
        source = ROOT / "core" / "src" / "anima_core" / "api_jobs.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        factories = {
            node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")
        }
        self.assertEqual({"build_jobs_router"}, factories)
        self.assertEqual(API_JOB_ROUTES, _router_routes(tree))

    def test_api_nl_router_owns_only_nl_routes(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("anima_core.api_nl"))
        source = ROOT / "core" / "src" / "anima_core" / "api_nl.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        factories = {
            node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")
        }
        self.assertEqual({"build_nl_router"}, factories)
        self.assertEqual(API_NL_ROUTES, _router_routes(tree))

    def test_api_token_budget_router_owns_only_token_budget_routes(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("anima_core.api_token_budget"))
        source = ROOT / "core" / "src" / "anima_core" / "api_token_budget.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        factories = {
            node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")
        }
        self.assertEqual({"build_token_budget_router"}, factories)
        self.assertEqual(API_TOKEN_BUDGET_ROUTES, _router_routes(tree))

    def test_api_route_owners_are_disjoint_and_the_facade_defines_no_routes(self) -> None:
        source_root = ROOT / "core" / "src" / "anima_core"
        facade_tree = ast.parse((source_root / "api.py").read_text(encoding="utf-8"))
        facade_functions = {
            node.name for node in facade_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual({"build_control_app"}, facade_functions)
        self.assertEqual(set(), _router_routes(facade_tree))

        actual_routes: set[tuple[str, str]] = set()
        router_modules = {filename.removesuffix(".py") for filename in API_ROUTER_OWNERS}
        for filename, expected_routes in API_ROUTER_OWNERS.items():
            tree = ast.parse((source_root / filename).read_text(encoding="utf-8"))
            actual = _router_routes(tree)
            self.assertEqual(expected_routes, actual, filename)
            self.assertTrue(actual.isdisjoint(actual_routes), filename)
            actual_routes.update(actual)
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module in router_modules:
                    self.fail(f"{filename} imports another router: {node.module}")
                if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    self.fail(f"{filename} defines mutable module state at line {node.lineno}")
        self.assertEqual(EXPECTED_API_ROUTES, actual_routes)


if __name__ == "__main__":
    unittest.main()
