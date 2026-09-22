from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from PIL import Image

from anima_caption_format import flat_txt_sha256
from anima_caption_format.normalizer import CaptionDisplayPolicy
from anima_core.classify_overlay import serialize_annotation_json
from anima_core.contracts import JobConfig
from anima_core.db import StateDatabase
from anima_core.job_preflight import JobPreparationService
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.scheduler import BoundedScheduler
from anima_core.token_budget_runner import TokenBudgetRunner, TokenBudgetRunnerError
from anima_core.worker_protocol import ProtocolEnvelopeV1
import test_job_preflight as _preflight_support


class TokenBudgetRunnerContractTests(unittest.TestCase):
    def test_token_budget_runner_is_available_for_untrusted_worker_outcomes(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("anima_core.token_budget_runner"),
            "Task 2.5 requires a Core-owned runner before worker results can be applied",
        )

    def _runner(
        self, root: Path, status: str, *, schema_version: int = 10,
    ) -> tuple[StateDatabase, JobPreparationService, str, TokenBudgetRunner]:
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (2, 2), "white").save(dataset / "image.png")
        annotation = {"quality": [], "count": "solo", "character": "", "series": "", "artist": "", "appearance": [], "tags": ["ok"], "environment": [], "nl": ""}
        (dataset / "image.json").write_bytes(serialize_annotation_json(annotation))
        config = JobConfig(schemaVersion=schema_version, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
        # This fixture exercises Token Budget itself; explicitly opt into
        # rebuilding the existing JSON so incremental preservation does not
        # short-circuit the worker before its safety assertions run.
        config.classify["overwriteJson"] = True
        config.countReview["enabled"] = False  # type: ignore[index]
        catalog, _ = _preflight_support._write_test_resource_library(root, include_ocr=False)
        _preflight_support.JobPreflightTests._write_tokenizer_resource(catalog.root, context_limit=512)
        config.caption["resourceId"] = "tagger-default"
        config.classify["resourceId"] = "classify-default"
        config.replace["resourceId"] = "replace-default"
        config.dropout["quality"]["resourceId"] = "dropout-default"
        preparation = JobPreparationService(root / "state.db", resource_catalog=catalog)
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        database = StateDatabase.open(root / "state.db")
        scheduler = BoundedScheduler(database, lease_id_factory=lambda: "lease-1")
        for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout"):
            scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
        scheduler.start_module(job_id, "token_budget", enabled=True, profile="e621")
        layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)

        class Transport:
            def exchange(self, request: ProtocolEnvelopeV1) -> ProtocolEnvelopeV1:
                if request.method == "hello":
                    payload = {"schemaVersion": 1, "payloadType": "token_budget_hello_result", "ready": True, "resourceFingerprint": json.loads(str(database.get_job(job_id)["config_json"]))["tokenBudget"]["resourceFingerprint"], "contextLimit": json.loads(str(database.get_job(job_id)["config_json"]))["tokenBudget"]["contextLimit"]}
                    method = "hello"
                else:
                    item = request.payload["items"][0]
                    if status == "duplicate":
                        outcomes = [{"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": item["sampleId"], "leaseId": item["leaseId"], "status": "failed", "code": "token_budget_input_invalid"}] * 2
                    elif status == "overflow":
                        outcomes = [{"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": item["sampleId"], "leaseId": item["leaseId"], "status": "overflow", "originalTokens": 600, "finalTokens": 600, "removed": {"quality": [], "environment": [], "tags": ["ok"], "appearance": []}}]
                    else:
                        policy = CaptionDisplayPolicy.from_mapping(request.payload["captionFormat"])
                        outcomes = [{"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": item["sampleId"], "leaseId": item["leaseId"], "status": "within_budget", "originalTokens": 1, "finalTokens": 1, "removed": {"quality": [], "environment": [], "tags": [], "appearance": []}, "annotation": item["annotation"], "flatTextSha256": flat_txt_sha256(item["annotation"], policy)}]
                    if status == "cancel":
                        database.set_job_status(job_id, "cancelling", current_module_id="token_budget")
                    if status == "stale":
                        database.return_lease_to_pending(job_id, int(item["sampleId"]), lease_id=str(item["leaseId"]))
                        BoundedScheduler(database, lease_id_factory=lambda: "lease-2").claim_batch(
                            job_id, "token_budget", "replacement", str(database.get_job(job_id)["config_hash"])
                        )
                    payload = {"schemaVersion": 1, "payloadType": "token_budget_process_result", "outcomes": outcomes}
                    method = "result"
                return ProtocolEnvelopeV1("1.0", "response", "reply-" + request.messageId, "token-budget", "token-budget", method, payload, replyTo=request.messageId, jobId=request.jobId, configHash=request.configHash)

        runner = TokenBudgetRunner(database, scheduler, Transport(), WorkingAnnotationView(BaselineView(dataset), layout), job_id=job_id, worker_instance_id="worker")
        return database, preparation, job_id, runner

    def test_v10_overflow_is_settled_once_as_a_blocking_issue_without_annotation(self) -> None:
        for schema_version in (10,):
            with self.subTest(schema_version=schema_version), tempfile.TemporaryDirectory() as temporary:
                database, preparation, job_id, runner = self._runner(
                    Path(temporary), "overflow", schema_version=schema_version,
                )
                try:
                    self.assertEqual("completed_with_issues", runner.run())
                    state = database.get_sample_state(job_id, 1)
                    issue = database.page_issues(job_id, limit=10)[0]
                    self.assertEqual("failed", state["status"])
                    self.assertEqual(("token_budget", "token_budget_overflow", "token_budget"), (issue["module_id"], issue["code"], issue["repair_start_module"]))
                    review_path = Path(str(database.get_job(job_id)["overlay_root"])) / "resources" / "token-budget" / "reviews" / "1.json"
                    self.assertTrue(review_path.is_file())
                    review = json.loads(review_path.read_text(encoding="utf-8"))
                    token_budget = json.loads(str(database.get_job(job_id)["config_json"]))["tokenBudget"]
                    self.assertEqual((token_budget["resourceId"], token_budget["resourceFingerprint"]), (review["resourceId"], review["resourceFingerprint"]))
                    self.assertNotIn("annotation", review)
                    self.assertNotIn("nl", review)
                finally:
                    database.close()
                    preparation.close()

    def test_v10_runner_sends_hello_and_processes_a_within_budget_sample(self) -> None:
        for schema_version in (10,):
            with self.subTest(schema_version=schema_version), tempfile.TemporaryDirectory() as temporary:
                database, preparation, _job_id, runner = self._runner(
                    Path(temporary), "within_budget", schema_version=schema_version,
                )
                try:
                    self.assertEqual("completed", runner.run())
                    self.assertTrue(runner._hello_done)
                finally:
                    database.close()
                    preparation.close()

    def test_batch_identity_violation_applies_zero_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, runner = self._runner(Path(temporary), "duplicate")
            try:
                with self.assertRaises(TokenBudgetRunnerError):
                    runner.run()
                self.assertEqual("failed", database.get_job(job_id)["status"])
                self.assertEqual("pending", database.get_sample_state(job_id, 1)["status"])
            finally:
                database.close()
                preparation.close()

    def test_cancellation_after_worker_response_returns_unprepared_lease_without_writing_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, runner = self._runner(Path(temporary), "cancel")
            try:
                overlay = Path(str(database.get_job(job_id)["overlay_root"]))
                self.assertEqual("cancelling", runner.run())
                self.assertEqual("pending", database.get_sample_state(job_id, 1)["status"])
                self.assertFalse((overlay / "annotations" / "image.json").exists())
                self.assertFalse((overlay / "resources" / "token-budget" / "records" / "1.json").exists())
            finally:
                database.close()
                preparation.close()

    def test_stale_worker_response_is_rejected_without_touching_the_replacement_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, runner = self._runner(Path(temporary), "stale")
            try:
                overlay = Path(str(database.get_job(job_id)["overlay_root"]))
                with self.assertRaisesRegex(TokenBudgetRunnerError, "stale"):
                    runner.run()
                state = database.get_sample_state(job_id, 1)
                self.assertEqual(("leased", "lease-2", "replacement"), (state["status"], state["lease_id"], state["worker_instance_id"]))
                self.assertFalse((overlay / "annotations" / "image.json").exists())
                self.assertFalse((overlay / "resources" / "token-budget" / "records" / "1.json").exists())
            finally:
                database.close()
                preparation.close()
