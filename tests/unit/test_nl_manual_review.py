from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_core.classify_overlay import serialize_annotation_json
from anima_core.api_models import _NlManualRetryBatchBody, _NlManualRetryBody, _NlManualWriteBody
from anima_core.contracts import JobConfig, SampleIssue
from anima_core.db import StateDatabase
from anima_core.job_preflight import JobPreparationService
from anima_core.repair import RepairPreparationService
from anima_core.scheduler import BoundedScheduler
from anima_core.pipeline import PipelineService
from anima_core.pipeline_dispatch import NlApiCredentials


class NlManualReviewTests(unittest.TestCase):
    def test_manual_api_bodies_require_confirmation_and_exact_selector(self) -> None:
        with self.assertRaises(Exception):
            _NlManualRetryBody.model_validate({"confirmed": True})
        with self.assertRaises(Exception):
            _NlManualRetryBody.model_validate({"sampleId": 1, "issueId": "x", "confirmed": True})
        with self.assertRaises(Exception):
            _NlManualWriteBody.model_validate({"sampleId": 1, "issueId": "x", "nl": "caption", "confirmed": True})
        retry = _NlManualRetryBody(issueId="x", confirmed=True)
        write = _NlManualWriteBody(issueId="x", nl="caption", confirmed=True)
        batch = _NlManualRetryBatchBody(issueIds=["x", "y"], confirmed=True)
        self.assertEqual((None, "x", True), (retry.sampleId, retry.issueId, write.confirmed))
        self.assertEqual((['x', 'y'], True), (batch.issueIds, batch.confirmed))
        with self.assertRaises(Exception):
            _NlManualRetryBatchBody(issueIds=["x", "x"], confirmed=True)

    def _fixture(
        self,
        root: Path,
        *,
        sample_count: int = 1,
        release_parent_lock: bool = True,
        ocr_enabled: bool = False,
    ) -> tuple[JobPreparationService, RepairPreparationService, StateDatabase, str]:
        dataset = root / "dataset"
        dataset.mkdir()
        for sample_id in range(1, sample_count + 1):
            name = "image" if sample_id == 1 else f"image-{sample_id}"
            Image.new("RGB", (2, 2), "white").save(dataset / f"{name}.png")
            (dataset / f"{name}.json").write_bytes(serialize_annotation_json({"nl": "", "tags": []}))
        config = JobConfig(workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset), schemaVersion=10)
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
        config.ocr["enabled"] = ocr_enabled
        config.countReview["enabled"] = False  # type: ignore[index]
        assert config.tokenBudget is not None
        config.tokenBudget["enabled"] = False
        config.nl.update({"apiEnabled": True, "useImage": True, "reuseOriginalNl": False, "systemPrompt": "describe"})
        preparation = JobPreparationService(root / "state.db")
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        database = StateDatabase.open(root / "state.db")
        samples = database.page_samples(job_id, limit=sample_count)
        scheduler = BoundedScheduler(database)
        for module in ("caption", "classify", "replace", "ocr"):
            scheduler.start_module(job_id, module, enabled=False, profile="e621")
        scheduler.start_module(job_id, "nl", enabled=True, profile="e621")
        database.set_job_status(job_id, "reviewing", current_module_id="nl")
        for sample in samples:
            sample_id = int(sample["sample_id"])
            database.connection.execute(
                "UPDATE sample_state SET current_module_id='nl',status='failed',attempt=1 WHERE job_id=? AND sample_id=?",
                (job_id, sample_id),
            )
            database.upsert_issue(SampleIssue(
                issueId=f"nl-manual-{sample_id}", jobId=job_id, sampleId=sample_id,
                relativeImagePath=str(sample["relative_image_path"]), moduleId="nl", code="nl_auth_failed", severity="error",
                blocking=True, retriable=False, repairStartModule=None, message="manual review", attempt=1,
            ))
        if release_parent_lock:
            preparation.release_lock_for_repair(job_id)
        return preparation, RepairPreparationService(root / "state.db"), database, job_id

    def test_manual_retry_requires_confirmation_and_exact_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preparation, repair, database, job_id = self._fixture(Path(temporary))
            try:
                with self.assertRaisesRegex(Exception, "confirmation"):
                    repair.prepare_manual_nl(job_id, sample_id=1, confirmed=False)
                with self.assertRaisesRegex(Exception, "exactly one"):
                    repair.prepare_manual_nl(job_id, sample_id=1, issue_id="nl-manual-1", confirmed=True)
            finally:
                database.close()
                repair.close()
                preparation.close()

    def test_manual_retry_targets_nonretriable_issue_without_changing_generic_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preparation, repair, database, job_id = self._fixture(Path(temporary))
            try:
                database.connection.execute(
                    "UPDATE jobs SET status='succeeded',current_module_id='export' WHERE job_id=?", (job_id,),
                )
                result = repair.prepare_manual_nl(job_id, issue_id="nl-manual-1", confirmed=True)
                target = database.connection.execute(
                    "SELECT sample_id,repair_start_module FROM repair_targets WHERE repair_job_id=?",
                    (result.repairJobId,),
                ).fetchone()
                self.assertEqual((1, "nl"), (target["sample_id"], target["repair_start_module"]))
                self.assertEqual(0, database.repair_candidate_summary(job_id)[0])
            finally:
                database.close()
                repair.close()
                preparation.close()

    def test_manual_retry_batch_creates_one_child_for_multiple_issues_after_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preparation, repair, database, job_id = self._fixture(
                Path(temporary), sample_count=2, release_parent_lock=False,
            )
            try:
                database.connection.execute(
                    "UPDATE jobs SET status='succeeded',current_module_id='export' WHERE job_id=?", (job_id,),
                )
                self.assertTrue(preparation.release_lock_for_repair(job_id))

                result = repair.prepare_manual_nl_batch(
                    job_id,
                    issue_ids=("nl-manual-1", "nl-manual-2"),
                    confirmed=True,
                )

                targets = list(database.connection.execute(
                    "SELECT sample_id,repair_start_module FROM repair_targets WHERE repair_job_id=? ORDER BY sample_id",
                    (result.repairJobId,),
                ))
                linked_issues = [row["parent_issue_id"] for row in database.connection.execute(
                    "SELECT parent_issue_id FROM repair_target_issues WHERE repair_job_id=? ORDER BY parent_issue_id",
                    (result.repairJobId,),
                )]
                self.assertEqual(2, result.targetCount)
                self.assertEqual([(1, "nl"), (2, "nl")], [(row["sample_id"], row["repair_start_module"]) for row in targets])
                self.assertEqual(["nl-manual-1", "nl-manual-2"], linked_issues)
            finally:
                database.close()
                repair.close()
                preparation.close()

    def test_manual_retry_validates_image_config_and_api_budget_before_creating_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preparation, repair, database, job_id = self._fixture(Path(temporary))
            try:
                database.set_module_diagnostic_count(job_id, "nl", "nl_http_attempts", severity="info", count=10_000)
                with self.assertRaisesRegex(Exception, "budget"):
                    repair.prepare_manual_nl(job_id, issue_id="nl-manual-1", confirmed=True)
                self.assertEqual(1, database.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
            finally:
                database.close()
                repair.close()
                preparation.close()

    def test_manual_write_can_prepare_a_child_when_the_image_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, repair, database, job_id = self._fixture(root)
            try:
                (root / "dataset" / "image.png").unlink()
                result = repair.prepare_manual_nl(
                    job_id, issue_id="nl-manual-1", confirmed=True, for_manual_write=True,
                )
                self.assertEqual(1, result.targetCount)
            finally:
                database.close()
                repair.close()
                preparation.close()

    def test_manual_write_rejects_oversize_text_and_records_audit_after_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preparation, repair, database, job_id = self._fixture(Path(temporary))
            try:
                result = repair.prepare_manual_nl(job_id, issue_id="nl-manual-1", confirmed=True)
                from anima_core.nl_manual_review import NlManualWriteService

                service = NlManualWriteService(database, result.repairJobId)
                with self.assertRaisesRegex(Exception, "16 KiB"):
                    service.seed(sample_id=1, issue_id="nl-manual-1", nl="x" * 16_385, confirmed=True)
                with self.assertRaisesRegex(Exception, "blank"):
                    service.seed(sample_id=1, issue_id="nl-manual-1", nl=" \t", confirmed=True)
                result = service.seed(sample_id=1, issue_id="nl-manual-1", nl="manual caption", confirmed=True)
                self.assertEqual(1, result["sampleId"])
                job = database.get_job(result["jobId"])
                overlay_json = Path(str(job["overlay_root"])) / "annotations" / "image.json"
                self.assertEqual("manual caption", json.loads(overlay_json.read_text(encoding="utf-8"))["nl"])
                issue = database.connection.execute("SELECT resolved_at FROM issues WHERE issue_id=?", ("nl-manual-1",)).fetchone()
                self.assertIsNone(issue["resolved_at"])
                state = database.get_sample_state(result["jobId"], 1)
                self.assertEqual(("nl", "completed"), (state["current_module_id"], state["status"]))
                event = database.connection.execute("SELECT message,sample_id,issue_code FROM event_ring WHERE job_id=? ORDER BY event_id DESC LIMIT 1", (result["jobId"],)).fetchone()
                self.assertEqual((1, "nl_auth_failed"), (event["sample_id"], event["issue_code"]))
                self.assertIn("sha256", str(event["message"]))
            finally:
                database.close()
                repair.close()
                preparation.close()

    def test_manual_write_child_runs_downstream_and_only_then_resolves_parent_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, repair, database, job_id = self._fixture(root)
            try:
                result = repair.prepare_manual_nl(job_id, issue_id="nl-manual-1", confirmed=True)
                from anima_core.nl_manual_review import NlManualWriteService

                NlManualWriteService(database, result.repairJobId).seed(
                    sample_id=1, issue_id="nl-manual-1", nl="manual caption", confirmed=True,
                )
                pipeline = PipelineService(root / "state.db")
                with patch.object(pipeline, "_nl_credentials", return_value=NlApiCredentials("https://example.test", "model", "secret", None)):
                    pipeline.start(result.repairJobId)
                    for _ in range(100):
                        if not pipeline.is_running(result.repairJobId):
                            break
                        time.sleep(0.02)
                    pipeline.close()
                child = database.get_job(result.repairJobId)
                self.assertEqual("succeeded", child["status"])
                parent_issue = database.connection.execute(
                    "SELECT resolved_at FROM issues WHERE issue_id='nl-manual-1'",
                ).fetchone()
                self.assertIsNotNone(parent_issue["resolved_at"])
            finally:
                database.close()
                repair.close()
                preparation.close()

    def test_manual_nl_child_with_no_ocr_targets_does_not_initialize_ocr_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preparation, repair, database, parent_id = self._fixture(root, ocr_enabled=True)
            pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
            try:
                result = repair.prepare_manual_nl(parent_id, issue_id="nl-manual-1", confirmed=True)
                calls: list[str] = []

                def run_module(_database, scheduler, job_id, module_id, _config):
                    calls.append(module_id)
                    if module_id == "nl":
                        return "paused"
                    return scheduler.finish_module(job_id, module_id)

                pipeline._run_active_module = run_module  # type: ignore[method-assign]
                pipeline._run(result.repairJobId)

                self.assertNotIn("ocr", calls)
                self.assertIn("nl", calls)
                self.assertEqual(0, database.module_summary(result.repairJobId, "ocr")["total"])
            finally:
                pipeline.close()
                database.close()
                repair.close()
                preparation.close()


if __name__ == "__main__":
    unittest.main()
