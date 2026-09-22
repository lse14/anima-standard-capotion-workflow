"""Acceptance of dataset-level preservation, including the final Export gate."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core/src"))
sys.path.insert(0, str(ROOT / "tests/unit"))

from PIL import Image
from anima_core.contracts import JobConfig
from anima_core.db import StateDatabase
from anima_core.export_commit import ExportCommitCoordinator
from anima_core.export_runner import ExportRunner
from anima_core.job_preflight import JobPreparationService
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.pipeline import PipelineService
from anima_core.scheduler import BoundedScheduler
from test_job_preflight import _write_test_resource_library, JobPreflightTests


ANNOTATION = {"quality": [], "count": "solo", "character": "", "series": "", "artist": "", "appearance": [], "tags": ["cat"], "environment": [], "nl": "A cat sits."}


class NoWorker:
    def exchange(self, request):
        raise AssertionError("a preserved image reached a worker")


class IncrementalAcceptanceTests(unittest.TestCase):
    def prepare(self, root, annotations, *, override=None, full_copy=False):
        dataset = root / "dataset"
        dataset.mkdir()
        for name, siblings in annotations.items():
            Image.new("RGB", (2, 2)).save(dataset / (name + ".png"))
            for suffix, data in siblings.items():
                (dataset / (name + suffix)).write_bytes(data)
        config = JobConfig(sourceRoot=str(dataset), workMode="full_copy" if full_copy else "in_place", outputRoot=str(root / "output") if full_copy else None, overwriteMode="incremental")
        for section in (config.caption, config.classify, config.replace, config.ocr, config.nl, config.dropout, config.countReview):
            section["enabled"] = False
        config.tokenBudget["enabled"] = True
        config.export["format"] = "both"
        if override == "txt":
            config.caption["overwriteTxt"] = True
        elif override:
            config.classify[override] = True
        catalog, _ = _write_test_resource_library(root, include_ocr=False)
        JobPreflightTests._write_tokenizer_resource(catalog.root, context_limit=512)
        config.caption["resourceId"] = "tagger-default"
        config.classify["resourceId"] = "classify-default"
        config.replace["resourceId"] = "replace-default"
        config.dropout["quality"]["resourceId"] = "dropout-default"
        prep = JobPreparationService(root / "state.db", resource_catalog=catalog)
        summary = prep.preflight(config.to_dict())
        prep.confirm_workspace(summary.jobId, confirmed=True, confirmed_rebuild=False)
        db = StateDatabase.open(root / "state.db")
        return prep, db, summary

    def test_presence_matrix_and_explicit_overwrite_priority(self):
        for override in (None, "overwriteJson", "overwriteCount", "txt"):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temp:
                annotation = json.dumps(ANNOTATION).encode()
                prep, db, summary = self.prepare(Path(temp), {
                    "json.only": {".json": annotation},
                    "txt": {".txt": b"cat"},
                    "both": {".json": annotation, ".txt": b"cat"},
                    "blank": {".json": b" \r\n", ".txt": b"\xef\xbb\xbf  "},
                    "missing": {},
                    "invalid": {".json": b"{broken"},
                }, override=override)
                rows = {r["annotation_key"]: r for r in db.page_samples(summary.jobId, limit=20)}
                self.assertFalse(rows["blank"]["incremental_preserve"])
                self.assertFalse(rows["missing"]["incremental_preserve"])
                for name in ("json.only", "txt", "both"):
                    self.assertEqual(override is None, bool(rows[name]["incremental_preserve"]), name)
                if override is None:
                    self.assertTrue(rows["invalid"]["incremental_preserve"])
                    self.assertTrue(any(i["code"] == "invalid_existing_annotation" for i in db.page_issues(summary.jobId, limit=20)))
                db.close()
                prep.close()

    def test_preserved_bytes_reach_commit_without_token_or_export_worker(self):
        for full_copy in (False, True):
            with self.subTest(full_copy=full_copy), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                annotations = {"a": {".json": json.dumps(ANNOTATION, indent=3).encode(), ".txt": b"  original caption\r\n"}, "b": {".txt": b"TXT only"}}
                prep, db, summary = self.prepare(root, annotations, full_copy=full_copy)
                job = summary.jobId
                scheduler = BoundedScheduler(db)
                for module in ("caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget"):
                    scheduler.start_module(job, module, enabled=True, profile="e621")
                    self.assertEqual([], scheduler.claim_batch(job, module, "test", str(db.get_job(job)["config_hash"])))
                    scheduler.finish_module(job, module)
                config = json.loads(str(db.get_job(job)["config_json"]))
                self.assertTrue(PipelineService._token_budget_export_gate(db, job, config))
                scheduler.start_module(job, "export", enabled=True, profile="e621")
                layout = OverlayLayout.open_existing(str(db.get_job(job)["overlay_root"]), job)
                view = WorkingAnnotationView(BaselineView(layout.dataset_root), layout)
                self.assertEqual("completed", ExportRunner(db, scheduler, NoWorker(), view, job_id=job, worker_instance_id="test").run())
                self.assertTrue(PipelineService._token_budget_export_gate(db, job, config))
                result = ExportCommitCoordinator(db, layout, job_id=job).commit()
                self.assertEqual(0, result.unlabeled)
                for name, siblings in annotations.items():
                    for suffix, data in siblings.items():
                        self.assertEqual(data, (layout.dataset_root / (name + suffix)).read_bytes())
                self.assertIn(".anima-dataset-tool", str(result.backupZip))
                db.close()
                prep.close()


if __name__ == "__main__":
    unittest.main()
