from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from PIL import Image

from anima_core.contracts import JobConfig, SampleIssue
from anima_core.classify_protocol import ClassifyCountDecisionV1
from anima_core.count_review_protocol import CountEvidenceV1
from anima_core.db import StateDatabase
from anima_core.job_preflight import JobPreparationService
from anima_core.pipeline import PipelineService
from anima_core.repair import RepairPreparationService
from anima_core.classify_overlay import serialize_annotation_json


def _count_evidence(value: str = "solo") -> CountEvidenceV1:
    return CountEvidenceV1.from_decision(ClassifyCountDecisionV1(
        value=value,
        baseValue=value,
        selectedSource="wiki_tags",
        originalRaw=None,
        originalNormalized=None,
        wikiValue=value,
        matchedTags=(value,),
        conflict=False,
        issueCodes=(),
        warnings=(),
        appliedLowerBounds=(),
    ))


class RepairPreparationTests(unittest.TestCase):
    def test_repair_inherits_parent_annotations_for_target_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            config = JobConfig(workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.ocr["enabled"] = config.nl["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.dropout["enabled"] = False
            assert config.tokenBudget is not None
            config.tokenBudget["enabled"] = False
            preparation = JobPreparationService(root / "state.db")
            repair = RepairPreparationService(root / "state.db")
            try:
                parent_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
                database = StateDatabase.open(root / "state.db")
                try:
                    sample = database.page_samples(parent_id, limit=1)[0]
                    parent = database.get_job(parent_id)
                    parent_layout = Path(str(parent["overlay_root"]))
                    expected = serialize_annotation_json({
                        "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                        "appearance": [], "tags": ["smile"], "environment": [], "nl": "A person smiles.",
                    })
                    (parent_layout / "annotations").mkdir(parents=True, exist_ok=True)
                    (parent_layout / "annotations" / "target.json").write_bytes(expected)
                    database.set_job_status(parent_id, "running", current_module_id="caption")
                    database.set_job_status(parent_id, "reviewing", current_module_id="export")
                    database.upsert_issue(SampleIssue(
                        issueId="repair-annotation", jobId=parent_id, sampleId=int(sample["sample_id"]),
                        relativeImagePath="target.png", moduleId="caption", code="retry_json",
                        severity="error", blocking=False, retriable=True, repairStartModule="caption",
                        message="retry", attempt=1,
                    ))
                finally:
                    database.close()
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                child_annotation = Path(result.overlayRoot) / "annotations" / "target.json"
                self.assertEqual(expected, child_annotation.read_bytes())
            finally:
                repair.close()
                preparation.close()

    def test_profileless_repair_inherits_frozen_custom_classification_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            external_package = root / "external" / "classification"
            shutil.copytree(
                ROOT / "resource-library" / "classification-indexes" / "e621-classify-20260724-v1",
                external_package,
            )

            config = JobConfig(
                workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset),
            )
            config.caption["enabled"] = False
            config.classify.update({
                "enabled": True,
                "indexMode": "custom",
                "customResourcePath": str(external_package / "resource.json"),
            })
            config.classify.pop("resourceId", None)
            config.replace["enabled"] = False
            config.ocr["enabled"] = False
            config.nl["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.dropout["enabled"] = False
            config.dropout["quality"]["enabled"] = False
            assert config.tokenBudget is not None
            config.tokenBudget["enabled"] = False

            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                parent = database.get_job(parent_id)
                frozen = json.loads(str(parent["config_json"]))
                manifest_relative = str(frozen["classify"]["resourceManifestRelativePath"])
                fingerprint = str(frozen["classify"]["resourceFingerprint"])
                self.assertNotIn("customResourcePath", frozen["classify"])
                sample = database.page_samples(parent_id, limit=1)[0]
                database.set_job_status(parent_id, "running", current_module_id="classify")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                database.upsert_issue(SampleIssue(
                    issueId="custom-classify-repair", jobId=parent_id,
                    sampleId=int(sample["sample_id"]), relativeImagePath="target.png",
                    moduleId="classify", code="retry_json", severity="error",
                    blocking=True, retriable=True, repairStartModule="classify",
                    message="retry", attempt=1,
                ))
            finally:
                database.close()

            shutil.rmtree(external_package.parent)
            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    child = database.get_job(result.repairJobId)
                    self.assertEqual(frozen, json.loads(str(child["config_json"])))
                    child_manifest = Path(result.overlayRoot) / "resources" / Path(
                        manifest_relative.replace("\\", "/")
                    )
                    self.assertTrue(child_manifest.is_file())
                    self.assertIn(fingerprint, manifest_relative)
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_dropout_repair_runs_through_export_and_closes_the_parent_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            artist = dataset / "1_Artist"
            artist.mkdir(parents=True)
            Image.new("RGB", (3, 3), "white").save(artist / "image.png")
            (artist / "image.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": ["white hair"], "tags": ["smile"], "environment": [], "nl": "A person smiles.",
            }))
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset), recursive=True)
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.dropout["enabled"] = True
            config.dropout["quality"]["enabled"] = False
            config.dropout["appearanceNl"]["enabled"] = False
            assert config.tokenBudget is not None
            config.tokenBudget["enabled"] = False
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                sample = database.page_samples(parent_id, limit=1)[0]
                database.set_job_status(parent_id, "running", current_module_id="dropout")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                database.upsert_issue(SampleIssue(
                    issueId="repair-dropout", jobId=parent_id, sampleId=int(sample["sample_id"]),
                    relativeImagePath=str(sample["relative_image_path"]), moduleId="dropout", code="policy_retry",
                    severity="error", blocking=True, retriable=True, repairStartModule="dropout", message="retry", attempt=1,
                ))
            finally:
                database.close()
            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                repair_job_id = repair.prepare(parent_id).repairJobId
                pipeline = PipelineService(root / "state.db", install_root=ROOT / ".runtime-build")
                try:
                    pipeline.start(repair_job_id)
                    for _ in range(500):
                        if not pipeline.is_running(repair_job_id):
                            break
                        time.sleep(0.01)
                    self.assertFalse(pipeline.is_running(repair_job_id))
                finally:
                    pipeline.close()
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual("succeeded", database.get_job(repair_job_id)["status"])
                    self.assertEqual("completed", database.module_summary(repair_job_id, "dropout")["status"])
                    self.assertEqual("completed", database.module_summary(repair_job_id, "export")["status"])
                    issue = database.connection.execute("SELECT resolved_at FROM issues WHERE issue_id='repair-dropout'").fetchone()
                    self.assertIsNotNone(issue["resolved_at"])
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_repair_rebuilds_current_manifest_and_fresh_overlay_without_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            Image.new("RGB", (3, 3), "black").save(dataset / "untargeted.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.nl["systemPrompt"] = "describe the visible image"
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(parent_id, "running", current_module_id="dropout")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                target = next(row for row in database.page_samples(parent_id, limit=10) if row["relative_image_path"] == "target.png")
                database.upsert_issue(SampleIssue(
                    issueId="retry-target", jobId=parent_id, sampleId=int(target["sample_id"]),
                    relativeImagePath="target.png", moduleId="classify", code="retry_json", severity="error",
                    blocking=True, retriable=True, repairStartModule="classify", message="retry", attempt=1,
                ))
            finally:
                database.close()

            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                self.assertEqual(parent_id, result.parentJobId)
                self.assertEqual(1, result.targetCount)
                self.assertEqual(dataset, Path(result.datasetRoot))
                self.assertTrue(Path(result.overlayRoot).is_dir())
                self.assertFalse((root / "dataset-copy").exists())
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(parent_id, database.repair_parent_job_id(result.repairJobId))
                    self.assertEqual(2, database.count_processing_samples(result.repairJobId))
                    self.assertEqual(1, database.count_module_samples(result.repairJobId, "export"))
                    target_row = database.connection.execute(
                        "SELECT sample_id,repair_start_module FROM repair_targets WHERE repair_job_id=?",
                        (result.repairJobId,),
                    ).fetchone()
                    self.assertEqual("classify", target_row["repair_start_module"])
                    mapped = database.get_sample_with_state(result.repairJobId, int(target_row["sample_id"]))
                    self.assertEqual("target.png", mapped["relative_image_path"])
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_repair_reuses_the_parent_manifest_instead_of_rescanning_the_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.nl["systemPrompt"] = "describe the visible image"
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(parent_id, "running", current_module_id="dropout")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                target = database.page_samples(parent_id, limit=10)[0]
                database.upsert_issue(SampleIssue(
                    issueId="retry-target", jobId=parent_id, sampleId=int(target["sample_id"]),
                    relativeImagePath="target.png", moduleId="classify", code="retry_json", severity="error",
                    blocking=True, retriable=True, repairStartModule="classify", message="retry", attempt=1,
                ))
            finally:
                database.close()
            # A full re-scan would pick this file up; the frozen parent rows do not.
            Image.new("RGB", (3, 3), "black").save(dataset / "added-later.png")
            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(1, database.count_processing_samples(result.repairJobId))
                    self.assertEqual(
                        ["target.png"],
                        [str(row["relative_image_path"]) for row in database.page_samples(result.repairJobId, limit=10)],
                    )
                    job = database.get_job(result.repairJobId)
                    self.assertEqual(1, int(job["sample_count"]))
                    self.assertIsNotNone(job["manifest_generated_at"])
                    self.assertNotEqual("1970-01-01T00:00:00Z", str(job["created_at"]))
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_successful_repair_closes_fixed_parent_issue_and_updates_remaining_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = StateDatabase.open(Path(temporary) / "state.db")
            try:
                for job_id in ("parent", "repair"):
                    config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=temporary)
                    database.insert_job({
                        "job_id": job_id, "config_schema_version": config.schemaVersion, "config_json": __import__("json").dumps(config.to_dict()),
                        "config_hash": config.config_hash, "profile": "e621", "work_mode": "in_place", "overwrite_mode": "incremental",
                        "source_root": temporary, "output_root": None, "dataset_root": temporary, "dataset_root_key": temporary,
                        "manifest_schema_version": 1, "recursive": 0, "sample_count": 0, "manifest_generated_at": None,
                        "status": "reviewing", "current_module_id": "export", "last_event_id": 0, "pinned": 0,
                        "api_budget_extra": 0, "api_budget_revision": 0, "overlay_root": None, "commit_journal_path": None,
                        "resume_status": None, "created_at": "2026-07-24T00:00:00Z", "started_at": None,
                        "cancel_requested_at": None, "finished_at": None,
                    })
                    database.insert_samples(job_id, [{
                        "sample_id": 1, "relative_image_path": "image.png", "annotation_key": "image", "source": "e621",
                        "in_processing_scope": True, "image_format": "png", "image_frame_count": 1,
                        "original_txt_state": "missing_or_blank", "original_json_state": "missing_or_blank",
                    }])
                database.create_repair_link("repair", "parent")
                database.connection.execute("INSERT INTO repair_targets(repair_job_id,sample_id,repair_start_module) VALUES ('repair',1,'classify')")
                for issue_id, code in (("fixed", "fixed_code"), ("remaining", "remaining_code")):
                    database.upsert_issue(SampleIssue(
                        issueId=issue_id, jobId="parent", sampleId=1, relativeImagePath="image.png", moduleId="classify",
                        code=code, severity="error", blocking=True, retriable=True, repairStartModule="classify",
                        message="old", attempt=2,
                    ))
                    database.connection.execute("INSERT INTO repair_target_issues(repair_job_id,sample_id,parent_issue_id) VALUES ('repair',1,?)", (issue_id,))
                database.upsert_issue(SampleIssue(
                    issueId="new", jobId="repair", sampleId=1, relativeImagePath="image.png", moduleId="classify",
                    code="remaining_code", severity="warning", blocking=False, retriable=True, repairStartModule="classify",
                    message="new", attempt=1,
                ))
                self.assertEqual(1, database.resolve_repaired_parent_issues("repair"))
                fixed = database.connection.execute("SELECT resolved_at FROM issues WHERE issue_id='fixed'").fetchone()
                remaining = database.connection.execute("SELECT severity,message,attempt,resolved_at FROM issues WHERE issue_id='remaining'").fetchone()
                self.assertIsNotNone(fixed["resolved_at"])
                self.assertEqual(("warning", "new", 3, None), tuple(remaining))
            finally:
                database.close()

    def test_v3_repair_inherits_only_classify_evidence_required_after_classify(self) -> None:
        starts = ("caption", "classify", "replace", "nl", "dropout", "export")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            for index, start in enumerate(starts, start=1):
                Image.new("RGB", (3, 3), "white").save(dataset / f"{index}-{start}.png")
                (dataset / f"{index}-{start}.json").write_bytes(serialize_annotation_json({
                    "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                    "appearance": [], "tags": ["smile"], "environment": [], "nl": "A visible subject.",
                }))
            config = JobConfig(
                profile="e621", workMode="in_place", overwriteMode="incremental",
                sourceRoot=str(dataset),
            )
            config.nl["systemPrompt"] = "describe the visible image"
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            evidence = _count_evidence()
            database = StateDatabase.open(root / "state.db")
            try:
                database.set_job_status(parent_id, "running", current_module_id="dropout")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                samples = database.page_samples(parent_id, limit=20)
                by_start = {
                    str(row["relative_image_path"]).split("-", 1)[1].removesuffix(".png"): row
                    for row in samples
                }
                now = "2026-07-26T00:00:00Z"
                for start in starts:
                    sample = by_start[start]
                    sample_id = int(sample["sample_id"])
                    database.upsert_issue(SampleIssue(
                        issueId=f"repair-{start}", jobId=parent_id, sampleId=sample_id,
                        relativeImagePath=str(sample["relative_image_path"]), moduleId=start,  # type: ignore[arg-type]
                        code=f"retry_{start}", severity="error", blocking=True, retriable=True,
                        repairStartModule=start, message="retry", attempt=1,  # type: ignore[arg-type]
                    ))
                    database.connection.execute(
                        """INSERT INTO count_evidence(
                               job_id,sample_id,schema_version,value,decision_json,
                               review_warning_codes_json,created_at,updated_at
                           ) VALUES (?,?,?,?,?,?,?,?)""",
                        (
                            parent_id, sample_id, 1, evidence.value, evidence.decision_json,
                            evidence.review_warning_codes_json, now, now,
                        ),
                    )
                parent = database.get_job(parent_id)
            finally:
                database.close()

            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    child = database.get_job(result.repairJobId)
                    self.assertEqual(
                        (parent["config_schema_version"], parent["config_json"], parent["config_hash"]),
                        (child["config_schema_version"], child["config_json"], child["config_hash"]),
                    )
                    target_rows = list(database.connection.execute(
                        """SELECT rt.sample_id,rt.repair_start_module,s.relative_image_path
                             FROM repair_targets AS rt JOIN samples AS s
                               ON s.job_id=rt.repair_job_id AND s.sample_id=rt.sample_id
                            WHERE rt.repair_job_id=? ORDER BY rt.sample_id""",
                        (result.repairJobId,),
                    ))
                    self.assertEqual(set(starts), {str(row["repair_start_module"]) for row in target_rows})
                    inherited_paths = {
                        str(row["relative_image_path"])
                        for row in database.connection.execute(
                            """SELECT s.relative_image_path FROM count_evidence AS e JOIN samples AS s
                                 ON s.job_id=e.job_id AND s.sample_id=e.sample_id WHERE e.job_id=?""",
                            (result.repairJobId,),
                        )
                    }
                    self.assertEqual({"3-replace.png", "4-nl.png"}, inherited_paths)
                    self.assertEqual(0, database.connection.execute(
                        "SELECT COUNT(*) FROM count_observations WHERE job_id=?", (result.repairJobId,)
                    ).fetchone()[0])
                    self.assertEqual(0, database.connection.execute(
                        "SELECT COUNT(*) FROM count_review_decisions WHERE job_id=?", (result.repairJobId,)
                    ).fetchone()[0])
                    self.assertEqual(4, database.count_module_samples(result.repairJobId, "count_review"))
                    self.assertEqual(5, database.count_module_samples(result.repairJobId, "dropout"))
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_v3_repair_blocks_missing_inherited_evidence_before_pipeline_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            config = JobConfig(profile="e621", workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.nl["systemPrompt"] = "describe the visible image"
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                sample = database.page_samples(parent_id, limit=1)[0]
                database.set_job_status(parent_id, "running", current_module_id="dropout")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                database.upsert_issue(SampleIssue(
                    issueId="missing-evidence", jobId=parent_id, sampleId=int(sample["sample_id"]),
                    relativeImagePath="target.png", moduleId="nl", code="retry_nl", severity="error",
                    blocking=True, retriable=True, repairStartModule="nl", message="retry", attempt=1,
                ))
            finally:
                database.close()
            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                with self.assertRaisesRegex(Exception, "evidence is missing or invalid"):
                    repair.prepare(parent_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual([parent_id], [str(row["job_id"]) for row in database.connection.execute("SELECT job_id FROM jobs")])
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_v10_repair_preserves_frozen_version_and_inherits_count_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            config = JobConfig(
                workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset),
            )
            config.nl["systemPrompt"] = "describe the visible image"
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                sample = database.page_samples(parent_id, limit=1)[0]
                evidence = _count_evidence()
                now = "2026-08-18T00:00:00Z"
                database.connection.execute(
                    """INSERT INTO count_evidence(
                           job_id,sample_id,schema_version,value,decision_json,
                           review_warning_codes_json,created_at,updated_at
                       ) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        parent_id, int(sample["sample_id"]), 1, evidence.value,
                        evidence.decision_json, evidence.review_warning_codes_json,
                        now, now,
                    ),
                )
                database.set_job_status(parent_id, "running", current_module_id="dropout")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                database.upsert_issue(SampleIssue(
                    issueId="legacy-repair", jobId=parent_id, sampleId=int(sample["sample_id"]),
                    relativeImagePath="target.png", moduleId="ocr", code="retry_ocr", severity="error",
                    blocking=True, retriable=True, repairStartModule="ocr", message="retry", attempt=1,
                ))
                parent = database.get_job(parent_id)
            finally:
                database.close()
            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    child = database.get_job(result.repairJobId)
                    self.assertEqual(
                        (10, parent["config_json"], parent["config_hash"]),
                        (child["config_schema_version"], child["config_json"], child["config_hash"]),
                    )
                    self.assertEqual(1, database.connection.execute(
                        "SELECT COUNT(*) FROM count_evidence WHERE job_id=?",
                        (result.repairJobId,),
                    ).fetchone()[0])
                    for table in ("count_observations", "count_review_decisions"):
                        self.assertEqual(0, database.connection.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE job_id=?", (result.repairJobId,)
                        ).fetchone()[0])
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_v10_ocr_repair_targets_retriable_inference_and_legacy_oversize_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "retry.png")
            Image.new("RGB", (3, 3), "black").save(dataset / "oversize.png")
            config = JobConfig(
                workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset),
            )
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = False
            config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            config.ocr["enabled"] = True
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                samples = {str(row["relative_image_path"]): row for row in database.page_samples(parent_id, limit=10)}
                database.set_job_status(parent_id, "running", current_module_id="ocr")
                database.set_job_status(parent_id, "reviewing", current_module_id="export")
                database.upsert_issue(SampleIssue(
                    issueId="ocr-retry", jobId=parent_id, sampleId=int(samples["retry.png"]["sample_id"]),
                    relativeImagePath="retry.png", moduleId="ocr", code="ocr_inference_failed",
                    severity="warning", blocking=False, retriable=True, repairStartModule="ocr",
                    message="OCR inference failed for this image.", attempt=1,
                ))
                database.upsert_issue(SampleIssue(
                    issueId="ocr-too-large", jobId=parent_id, sampleId=int(samples["oversize.png"]["sample_id"]),
                    relativeImagePath="oversize.png", moduleId="ocr", code="ocr_image_too_large",
                    severity="warning", blocking=False, retriable=False, repairStartModule="ocr",
                    message="OCR image dimensions exceed the first-release safety limit.", attempt=1,
                ))
            finally:
                database.close()

            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                try:
                    result = repair.prepare(parent_id)
                except Exception as exc:
                    self.fail(f"retriable OCR inference issue must create an OCR-only repair target: {exc}")
                self.assertEqual(2, result.targetCount)
                database = StateDatabase.open(root / "state.db")
                try:
                    targets = database.connection.execute(
                        """SELECT repair_targets.sample_id,repair_targets.repair_start_module,
                                  samples.relative_image_path
                             FROM repair_targets
                             JOIN samples ON samples.job_id=repair_targets.repair_job_id
                                AND samples.sample_id=repair_targets.sample_id
                            WHERE repair_job_id=? ORDER BY samples.relative_image_path""",
                        (result.repairJobId,),
                    ).fetchall()
                    self.assertEqual(
                        [("ocr", "oversize.png"), ("ocr", "retry.png")],
                        [(row["repair_start_module"], row["relative_image_path"]) for row in targets],
                    )
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()

    def test_v10_token_budget_issue_repairs_from_token_budget_without_replaying_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "target.png")
            (dataset / "target.json").write_bytes(serialize_annotation_json({
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["tag"], "environment": [], "nl": "",
            }))
            config = JobConfig(workMode="in_place", overwriteMode="incremental", sourceRoot=str(dataset))
            config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.ocr["enabled"] = config.nl["enabled"] = config.dropout["enabled"] = False
            config.countReview["enabled"] = False  # type: ignore[index]
            preparation = JobPreparationService(root / "state.db")
            parent_id = preparation.preflight(config.to_dict()).jobId
            preparation.confirm_workspace(parent_id, confirmed=True, confirmed_rebuild=False)
            database = StateDatabase.open(root / "state.db")
            try:
                sample = database.page_samples(parent_id, limit=1)[0]
                database.set_job_status(parent_id, "running", current_module_id="token_budget")
                database.set_job_status(parent_id, "reviewing", current_module_id="token_budget")
                database.upsert_issue(SampleIssue(
                    issueId="token-budget-repair", jobId=parent_id, sampleId=int(sample["sample_id"]),
                    relativeImagePath="target.png", moduleId="token_budget", code="token_budget_overflow",
                    severity="error", blocking=True, retriable=True, repairStartModule="token_budget",
                    message="Token Budget exceeded the frozen maximum", attempt=1,
                ))
            finally:
                database.close()
            repair = RepairPreparationService(root / "state.db")
            try:
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    target = database.connection.execute(
                        "SELECT repair_start_module FROM repair_targets WHERE repair_job_id=?", (result.repairJobId,)
                    ).fetchone()
                    self.assertEqual("token_budget", target["repair_start_module"])
                    self.assertEqual(0, database.count_module_samples(result.repairJobId, "dropout"))
                    self.assertEqual(1, database.count_module_samples(result.repairJobId, "token_budget"))
                    self.assertEqual(1, database.count_module_samples(result.repairJobId, "export"))
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()


if __name__ == "__main__":
    unittest.main()
