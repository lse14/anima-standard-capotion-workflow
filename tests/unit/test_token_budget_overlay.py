from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from PIL import Image

from anima_caption_format import flat_txt_sha256
from anima_caption_format.normalizer import CaptionDisplayPolicy
from anima_core.classify_overlay import serialize_annotation_json
from anima_core.contracts import JobConfig
from anima_core.db import StateDatabase
from anima_core.job_preflight import JobPreparationService
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.scheduler import BoundedScheduler
from anima_core.token_budget_overlay import TokenBudgetOverlayWriter
from anima_core.token_budget_protocol import validate_token_budget_outcome
from anima_core.worker_protocol import PROTOCOL_VERSION


class TokenBudgetOverlayContractTests(unittest.TestCase):
    def test_token_budget_overlay_writer_is_available_for_atomic_prepared_records(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("anima_core.token_budget_overlay"),
            "Task 2.5 requires a Core-owned prepared-record overlay writer",
        )

    @staticmethod
    def _prepared(root: Path) -> tuple[StateDatabase, JobPreparationService, str, OverlayLayout, WorkingAnnotationView, object]:
        dataset = root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (2, 2), "white").save(dataset / "image.png")
        annotation = {"quality": [], "count": "solo", "character": "", "series": "", "artist": "", "appearance": ["white hair"], "tags": ["smile"], "environment": [], "nl": ""}
        (dataset / "image.json").write_bytes(serialize_annotation_json(annotation))
        config = JobConfig(schemaVersion=10, workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        # Incremental mode freezes existing annotations as preserved, which
        # skips the sample past Token Budget; this contract needs a leased sample.
        config.classify["overwriteJson"] = True
        config.export["format"] = "json"
        preparation = JobPreparationService(root / "state.db")
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        database = StateDatabase.open(root / "state.db")
        scheduler = BoundedScheduler(database, lease_id_factory=lambda: "lease-1")
        for module_id in ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout"):
            scheduler.start_module(job_id, module_id, enabled=False, profile="e621")
        scheduler.start_module(job_id, "token_budget", enabled=True, profile="e621")
        lease = scheduler.claim_batch(job_id, "token_budget", "worker", str(database.get_job(job_id)["config_hash"]))[0]
        layout = OverlayLayout.open_existing(str(database.get_job(job_id)["overlay_root"]), job_id)
        view = WorkingAnnotationView(BaselineView(dataset), layout)
        return database, preparation, job_id, layout, view, lease

    def test_prepared_record_precedes_annotation_commit_and_recovers_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, layout, view, lease = self._prepared(Path(temporary))
            try:
                row = database.get_leased_sample(job_id, "token_budget", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id="worker")
                config = json.loads(str(database.get_job(job_id)["config_json"]))
                policy = CaptionDisplayPolicy.from_mapping(config["captionFormat"])
                annotation = json.loads(view.read(str(row["annotation_key"]), ".json") or b"null")
                outcome = validate_token_budget_outcome({"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": lease.sampleId, "leaseId": lease.leaseId, "status": "within_budget", "originalTokens": 1, "finalTokens": 1, "removed": {"quality": [], "environment": [], "tags": [], "appearance": []}, "annotation": annotation, "flatTextSha256": flat_txt_sha256(annotation, policy)}, expected_sample_id=lease.sampleId, expected_lease_id=str(lease.leaseId), original_annotation=annotation, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                writer = TokenBudgetOverlayWriter(database, layout, view, job_id)
                record = writer.prepare_and_commit(sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]), outcome=outcome, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                state = database.get_sample_state(job_id, lease.sampleId)
                self.assertEqual("prepared", state["status"])
                self.assertTrue(layout.resource_path(f"token-budget\\records\\{lease.sampleId}.json").is_file())
                self.assertEqual(record.flat_text_sha256, writer.record_for_export(sample_id=lease.sampleId, annotation_key=str(row["annotation_key"]), caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"])).flat_text_sha256)
                prepared_path = str(state["prepared_artifact_relative_path"])
                self.assertTrue(writer.recover_prepared(job_id, lease.sampleId, prepared_path, str(state["prepared_artifact_sha256"])))
                BoundedScheduler(database).complete(lease)
                self.assertEqual("completed", database.get_sample_state(job_id, lease.sampleId)["status"])
            finally:
                database.close()
                preparation.close()

    def test_export_record_rejects_annotation_hash_mismatch_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, layout, view, lease = self._prepared(Path(temporary))
            try:
                row = database.get_leased_sample(job_id, "token_budget", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id="worker")
                config = json.loads(str(database.get_job(job_id)["config_json"]))
                policy = CaptionDisplayPolicy.from_mapping(config["captionFormat"])
                annotation = json.loads(view.read(str(row["annotation_key"]), ".json") or b"null")
                result = validate_token_budget_outcome({"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": lease.sampleId, "leaseId": lease.leaseId, "status": "within_budget", "originalTokens": 1, "finalTokens": 1, "removed": {"quality": [], "environment": [], "tags": [], "appearance": []}, "annotation": annotation, "flatTextSha256": flat_txt_sha256(annotation, policy)}, expected_sample_id=lease.sampleId, expected_lease_id=str(lease.leaseId), original_annotation=annotation, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                writer = TokenBudgetOverlayWriter(database, layout, view, job_id)
                writer.prepare_and_commit(sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]), outcome=result, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                layout.write_annotation(str(row["annotation_key"]), ".json", serialize_annotation_json({**annotation, "tags": ["changed"]}))
                with self.assertRaises(Exception):
                    writer.record_for_export(sample_id=lease.sampleId, annotation_key=str(row["annotation_key"]), caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
            finally:
                database.close()
                preparation.close()

    def test_recovery_returns_to_pending_when_prepared_record_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, layout, view, lease = self._prepared(Path(temporary))
            try:
                row = database.get_leased_sample(job_id, "token_budget", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id="worker")
                config = json.loads(str(database.get_job(job_id)["config_json"]))
                annotation = json.loads(view.read(str(row["annotation_key"]), ".json") or b"null")
                policy = CaptionDisplayPolicy.from_mapping(config["captionFormat"])
                outcome = validate_token_budget_outcome({"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": lease.sampleId, "leaseId": lease.leaseId, "status": "within_budget", "originalTokens": 1, "finalTokens": 1, "removed": {"quality": [], "environment": [], "tags": [], "appearance": []}, "annotation": annotation, "flatTextSha256": flat_txt_sha256(annotation, policy)}, expected_sample_id=lease.sampleId, expected_lease_id=str(lease.leaseId), original_annotation=annotation, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                writer = TokenBudgetOverlayWriter(database, layout, view, job_id)
                writer.prepare_and_commit(sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]), outcome=outcome, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                committed = layout.annotation_path(str(row["annotation_key"]), ".json").read_bytes()
                layout.resource_path(f"token-budget\\prepared\\{lease.leaseId}.json").unlink()
                database.mark_interrupted(job_id)
                report = BoundedScheduler(database).recover(
                    job_id, confirmed=True, expected_config_hash=str(database.get_job(job_id)["config_hash"]),
                    manifest_schema_version=1, protocol_version=PROTOCOL_VERSION,
                    verify_source_fingerprints=lambda: True, commit_prepared=writer.recover_prepared,
                )
                self.assertEqual((0, 0, 1), (report.returnedLeases, report.committedPrepared, report.repeatedPrepared))
                self.assertEqual("pending", database.get_sample_state(job_id, lease.sampleId)["status"])
                self.assertEqual(committed, layout.annotation_path(str(row["annotation_key"]), ".json").read_bytes())
            finally:
                database.close()
                preparation.close()

    def test_recovery_commits_prepared_annotation_after_crash_before_overlay_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, layout, view, lease = self._prepared(Path(temporary))
            try:
                row = database.get_leased_sample(job_id, "token_budget", lease.sampleId, lease_id=str(lease.leaseId), worker_instance_id="worker")
                config = json.loads(str(database.get_job(job_id)["config_json"]))
                annotation = json.loads(view.read(str(row["annotation_key"]), ".json") or b"null")
                policy = CaptionDisplayPolicy.from_mapping(config["captionFormat"])
                outcome = validate_token_budget_outcome({"schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": lease.sampleId, "leaseId": lease.leaseId, "status": "within_budget", "originalTokens": 1, "finalTokens": 1, "removed": {"quality": [], "environment": [], "tags": [], "appearance": []}, "annotation": annotation, "flatTextSha256": flat_txt_sha256(annotation, policy)}, expected_sample_id=lease.sampleId, expected_lease_id=str(lease.leaseId), original_annotation=annotation, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                writer = TokenBudgetOverlayWriter(database, layout, view, job_id)
                with patch.object(OverlayLayout, "commit_prepared", side_effect=OSError("simulated interruption")):
                    with self.assertRaises(OSError):
                        writer.prepare_and_commit(sample_id=lease.sampleId, lease_id=str(lease.leaseId), annotation_key=str(row["annotation_key"]), outcome=outcome, caption_format=config["captionFormat"], max_tokens=int(config["tokenBudget"]["maxTokens"]))
                self.assertEqual("prepared", database.get_sample_state(job_id, lease.sampleId)["status"])
                self.assertFalse(layout.annotation_path(str(row["annotation_key"]), ".json").exists())
                database.mark_interrupted(job_id)
                report = BoundedScheduler(database).recover(
                    job_id, confirmed=True, expected_config_hash=str(database.get_job(job_id)["config_hash"]),
                    manifest_schema_version=1, protocol_version=PROTOCOL_VERSION,
                    verify_source_fingerprints=lambda: True, commit_prepared=writer.recover_prepared,
                )
                self.assertEqual((1, 0), (report.committedPrepared, report.repeatedPrepared))
                self.assertEqual("completed", database.get_sample_state(job_id, lease.sampleId)["status"])
                self.assertEqual(serialize_annotation_json(annotation), layout.annotation_path(str(row["annotation_key"]), ".json").read_bytes())
            finally:
                database.close()
                preparation.close()
