from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from PIL import Image

from anima_core.classify_overlay import ClassifyOverlayWriter
from anima_core.classify_protocol import (
    ClassifyCountDecisionV1,
    ClassifyHelloResultV1,
    ClassifyProjectionV1,
    ClassifyResultV1,
    ClassifyWorkItemV1,
)
from anima_core.classify_runner import ClassifyRunner
from anima_core.contracts import JobConfig, SampleIssue
from anima_core.db import StateDatabase
from anima_core.export_runner import ExportRunner
from anima_core.job_preflight import JobPreflightError, JobPreparationService
from anima_core.nl_runner import NlRunner
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.pipeline import PipelineService
from anima_core.policy_runner import PolicyRunner
from anima_core.repair import RepairPreparationService
from anima_core.resource_catalog import (
    ResourceCatalog,
    ResourceCatalogSnapshot,
    ResourcePackage,
)
from anima_core.scheduler import BoundedScheduler
from anima_core.worker_protocol import ProtocolEnvelopeV1
from test_danbooru_resource_builder import _build


@dataclass(frozen=True)
class _FixtureDropoutPackage:
    library_root: Path
    package_root: Path
    manifest_relative_path: str = r"dropout-models\fixture-dropout\resource.json"
    schema_version: int = 1
    kind: str = "dropout-model"
    resource_id: str = "fixture-dropout"
    resource_version: str = "fixture-v1"
    profile: str = "e621"
    fingerprint: str = "d" * 64
    metadata: dict[str, object] | None = None

    def verify_files(self, *, verify_hashes: bool) -> None:
        del verify_hashes


@dataclass(frozen=True)
class _FixtureReplacementPackage:
    library_root: Path
    package_root: Path
    manifest_relative_path: str = r"replacement-indexes\fixture-replacement\resource.json"
    schema_version: int = 1
    kind: str = "replacement-index"
    resource_id: str = "fixture-replacement"
    resource_version: str = "fixture-v1"
    profile: str = "e621"
    fingerprint: str = "f" * 64
    metadata: dict[str, object] | None = None

    def verify_files(self, *, verify_hashes: bool) -> None:
        del verify_hashes


class _StaticCatalog:
    def __init__(self, root: Path, snapshot: ResourceCatalogSnapshot) -> None:
        self.root = root
        self._snapshot = snapshot

    def scan(self, *, include_tokenizers: bool = True) -> ResourceCatalogSnapshot:
        del include_tokenizers
        return self._snapshot


def _fixture_catalog(root: Path) -> tuple[_StaticCatalog, ResourcePackage, ResourcePackage]:
    _build(root)
    library = root / "resource-library"
    taggers = [
        ResourcePackage.load(library, path, "tagging-model")
        for path in sorted((library / "tagging-models").glob("*/resource.json"))
    ]
    classification = ResourcePackage.load(
        library,
        next((library / "classification-indexes").glob("*/resource.json")),
        "classification-index",
    )
    cl = next(item for item in taggers if item.runtime_format == "cl-tagger-v2-onnx-v1")
    replacement = _FixtureReplacementPackage(
        library,
        library / "replacement-indexes" / "fixture-replacement",
        metadata={"ruleCount": 0},
    )
    dropout = _FixtureDropoutPackage(library, library / "dropout-models" / "fixture-dropout")
    snapshot = ResourceCatalogSnapshot(
        defaults_schema_version=3,
        defaults={
            "replacementIndex": replacement.resource_id,
            "taggingModel": cl.resource_id,
            "classificationIndex": classification.resource_id,
            "dropoutModel": dropout.resource_id,
        },
        packages=(*taggers, classification, replacement, dropout),  # type: ignore[arg-type]
        invalid=(),
    )
    return _StaticCatalog(library, snapshot), cl, classification


def _config(
    dataset: Path,
    cl: ResourcePackage,
    classification: ResourcePackage,
    *,
    active: bool,
) -> JobConfig:
    config = JobConfig(
        workMode="in_place",
        overwriteMode="incremental",
        sourceRoot=str(dataset),
        schemaVersion=10,
    )
    config.caption.clear()
    config.caption.update({
        "enabled": active,
        "thresholdMode": "per_category",
        "categoryThresholds": {"general": 0.61, "character": 0.52, "copyright": 0.55},
        "overwriteTxt": False,
        "resourceId": cl.resource_id,
        "inputTxtMode": "tag",
        "taggerFallbackOnMissingTxt": True,
    })
    config.classify.update({
        "enabled": active,
        "resourceId": classification.resource_id,
    })
    config.replace.clear()
    config.replace.update({"enabled": False, "indexMode": "bundled"})
    config.ocr["enabled"] = False
    config.nl.update({"enabled": False, "apiEnabled": False})
    assert config.countReview is not None
    config.countReview.update({"enabled": False, "protocolVersion": "count-review-v1"})
    config.dropout["enabled"] = active
    config.dropout["artist"]["enabled"] = False
    config.dropout["quality"]["enabled"] = False
    config.dropout["quality"]["resourceId"] = "fixture-dropout"
    assert config.tokenBudget is not None
    config.tokenBudget["enabled"] = False
    return config


class _DanbooruClassifyTransport:
    def __init__(self, package: ResourcePackage) -> None:
        self.package = package
        self.hello_requests = 0
        self.process_requests = 0
        self.hello_profile: str | None = None

    @staticmethod
    def _response(
        request: ProtocolEnvelopeV1,
        method: str,
        payload: dict[str, object],
    ) -> ProtocolEnvelopeV1:
        return ProtocolEnvelopeV1(
            protocolVersion="1.0",
            kind="response",
            messageId=f"reply-{request.messageId}",
            runtimeId="classify-e621",
            owner="classify",
            method=method,
            payload=payload,
            replyTo=request.messageId,
            jobId=request.jobId,
            configHash=request.configHash,
        )

    def exchange(self, request: ProtocolEnvelopeV1) -> ProtocolEnvelopeV1:
        if request.method == "hello":
            self.hello_requests += 1
            self.hello_profile = str(request.payload["profile"])
            return self._response(request, "hello", ClassifyHelloResultV1(
                executable=r"C:\fixture\python.exe",
                resourceFingerprint=self.package.fingerprint,
                entryCount=int(self.package.metadata["dictionaryEntryCount"]),
                wikiDataSourceId=str(self.package.metadata["wikiDataSourceId"]),
            ).to_dict())
        self.process_requests += 1
        item = ClassifyWorkItemV1.from_dict(request.payload["items"][0])
        decision = ClassifyCountDecisionV1(
            value="solo",
            baseValue="solo",
            selectedSource="wiki_tags",
            originalRaw=None,
            originalNormalized=None,
            wikiValue="solo",
            matchedTags=("1girl",),
            conflict=False,
            issueCodes=(),
            warnings=(),
            appliedLowerBounds=(),
        )
        result = ClassifyResultV1(
            sampleId=item.sampleId,
            leaseId=item.leaseId,
            relativeImagePath=item.relativeImagePath,
            projection=ClassifyProjectionV1(
                quality=(),
                count="solo",
                character="hatsune_miku",
                series="vocaloid",
                artist="",
                appearance=(),
                tags=("1girl",),
                environment=(),
                nl="",
            ),
            countDecision=decision,
            inputTagCount=3,
            outputTagCount=3,
            droppedTagCount=0,
            source="danbooru",
        )
        return self._response(request, "result", {
            "schemaVersion": 1,
            "payloadType": "classify_process_result",
            "outcomes": [result.to_dict()],
        })


class DanbooruCoreWiringTests(unittest.TestCase):
    def test_preflight_freezes_current_resources_without_a_task_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            catalog, cl, classification = _fixture_catalog(root)
            config = _config(dataset, cl, classification, active=True)
            preparation = JobPreparationService(root / "state.db", resource_catalog=catalog)  # type: ignore[arg-type]
            try:
                summary = preparation.preflight(config.to_dict())
                self.assertEqual("fixture-replacement", summary.replaceIndex["resourceId"])
                self.assertEqual({"caption", "classify", "replace", "dropout"}, set(summary.resources))
                database = StateDatabase.open(root / "state.db")
                try:
                    job = database.get_job(summary.jobId)
                    frozen = json.loads(str(job["config_json"]))
                    self.assertEqual(10, job["config_schema_version"])
                    self.assertNotIn("profile", job.keys())
                    self.assertNotIn("profile", frozen)
                    self.assertEqual("fixture-replacement", frozen["replace"]["resourceId"])
                    self.assertEqual(
                        str(classification.metadata["wikiDataSourceId"]),
                        frozen["classify"]["wikiDataSourceId"],
                    )
                    self.assertEqual(
                        {"general": 0.61, "character": 0.52, "copyright": 0.55},
                        frozen["caption"]["categoryThresholds"],
                    )
                    self.assertEqual(
                        "danbooru", database.page_samples(summary.jobId, limit=1)[0]["source"]
                    )
                finally:
                    database.close()

                workspace = preparation.confirm_workspace(
                    summary.jobId, confirmed=True, confirmed_rebuild=False
                )
                database = StateDatabase.open(root / "state.db")
                try:
                    layout = OverlayLayout.open_existing(
                        str(workspace["overlayRoot"]), summary.jobId
                    )
                    view = WorkingAnnotationView(BaselineView(dataset), layout)
                    policy = PolicyRunner(
                        database, None, None, view,  # type: ignore[arg-type]
                        job_id=summary.jobId,
                        worker_instance_id="policy-fixture",
                        install_root=catalog.root,
                    )
                    export = ExportRunner(
                        database, None, None, view,  # type: ignore[arg-type]
                        job_id=summary.jobId,
                        worker_instance_id="export-fixture",
                    )
                    nl = NlRunner(
                        database, None, None, view, None,  # type: ignore[arg-type]
                        job_id=summary.jobId,
                        worker_instance_id="nl-fixture",
                    )
                    self.assertEqual(summary.configHash, policy._config()[0])
                    self.assertEqual(summary.configHash, export._config()[0])
                    self.assertEqual(summary.configHash, nl._config()[0])
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_missing_selected_tagger_reports_the_current_resource_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            catalog, cl, _ = _fixture_catalog(root)
            snapshot = catalog.scan()
            missing_tagger_catalog = _StaticCatalog(
                catalog.root,
                ResourceCatalogSnapshot(
                    defaults_schema_version=snapshot.defaults_schema_version,
                    defaults=snapshot.defaults,
                    packages=tuple(
                        package for package in snapshot.packages
                        if package.resource_id != cl.resource_id
                    ),
                    invalid=snapshot.invalid,
                ),
            )
            config = JobConfig(
                workMode="in_place",
                overwriteMode="incremental",
                sourceRoot=str(dataset),
                schemaVersion=10,
            )
            config.caption["resourceId"] = "caption-danbooru-cl-tagger-v2-00"
            config.classify["resourceId"] = "danbooru-classify-20260727-v1"
            config.replace.clear()
            config.replace.update({"enabled": False, "indexMode": "bundled"})
            config.ocr["enabled"] = False
            config.nl.update({"enabled": False, "apiEnabled": False})
            assert config.tokenBudget is not None
            config.tokenBudget["enabled"] = False
            service = JobPreparationService(
                root / "state.db",
                resource_catalog=missing_tagger_catalog,  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(
                JobPreflightError,
                "selected tagging-model is unavailable: caption-danbooru-cl-tagger-v2-00",
            ):
                service.preflight(config.to_dict())
            service.close()

    def test_classify_runner_uses_frozen_resource_identity_and_persists_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.txt").write_text(
                "hatsune_miku, vocaloid, 1girl", encoding="utf-8"
            )
            catalog, cl, classification = _fixture_catalog(root)
            config = _config(dataset, cl, classification, active=False)
            config.classify["enabled"] = True
            # This fixture validates the Classify worker against an existing
            # TXT/JSON pair; opt into JSON replacement explicitly under the
            # incremental preservation contract.
            config.classify["overwriteJson"] = True
            preparation = JobPreparationService(root / "state.db", resource_catalog=catalog)  # type: ignore[arg-type]
            try:
                summary = preparation.preflight(config.to_dict())
                preparation.confirm_workspace(
                    summary.jobId, confirmed=True, confirmed_rebuild=False
                )
                database = StateDatabase.open(root / "state.db")
                try:
                    scheduler = BoundedScheduler(database, lease_id_factory=lambda: "classify-lease")
                    scheduler.start_module(summary.jobId, "caption", enabled=False, profile="danbooru")
                    scheduler.start_module(summary.jobId, "classify", enabled=True, profile="danbooru")
                    job = database.get_job(summary.jobId)
                    frozen = json.loads(str(job["config_json"]))
                    layout = OverlayLayout.open_existing(str(job["overlay_root"]), summary.jobId)
                    view = WorkingAnnotationView(BaselineView(dataset), layout)
                    transport = _DanbooruClassifyTransport(classification)
                    report = ClassifyRunner(
                        database,
                        scheduler,
                        transport,
                        view,
                        ClassifyOverlayWriter.open_for_job(database, summary.jobId),
                        job_id=summary.jobId,
                        worker_instance_id="classify-fixture",
                        install_root=catalog.root,
                        resource_manifest_relative_path=frozen["classify"]["resourceManifestRelativePath"],
                        resource_fingerprint=frozen["classify"]["resourceFingerprint"],
                    ).run()
                    self.assertEqual(("completed", 1), (report.status, report.completed))
                    self.assertEqual((1, 1, "danbooru"), (
                        transport.hello_requests,
                        transport.process_requests,
                        transport.hello_profile,
                    ))
                    projection = json.loads(
                        layout.annotation_path("image", ".json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(("hatsune_miku", "vocaloid", "solo"), (
                        projection["character"], projection["series"], projection["count"],
                    ))
                    self.assertEqual(
                        "solo", database.get_count_evidence(summary.jobId, 1)["value"]
                    )
                finally:
                    database.close()
            finally:
                preparation.close()

    def test_pipeline_skips_replace_before_any_active_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            catalog, cl, classification = _fixture_catalog(root)
            config = _config(dataset, cl, classification, active=False)
            preparation = JobPreparationService(root / "state.db", resource_catalog=catalog)  # type: ignore[arg-type]
            try:
                summary = preparation.preflight(config.to_dict())
                preparation.confirm_workspace(
                    summary.jobId, confirmed=True, confirmed_rebuild=False
                )
                pipeline = PipelineService(
                    root / "state.db",
                    install_root=ROOT / ".runtime-build",
                    resource_catalog=catalog,  # type: ignore[arg-type]
                )
                active_modules: list[str] = []

                def stop_at_export(_database, _scheduler, _job_id, module_id, _config):
                    active_modules.append(module_id)
                    return "paused"

                pipeline._run_active_module = stop_at_export  # type: ignore[method-assign]
                pipeline._run(summary.jobId)
                database = StateDatabase.open(root / "state.db")
                try:
                    self.assertEqual(["export"], active_modules)
                    self.assertEqual(
                        "skipped", database.module_summary(summary.jobId, "replace")["status"]
                    )
                    self.assertEqual(
                        "not_requested",
                        database.get_count_observation(summary.jobId, 1)["status"],
                    )
                    frozen = json.loads(str(database.get_job(summary.jobId)["config_json"]))
                    self.assertEqual("fixture-replacement", frozen["replace"]["resourceId"])
                finally:
                    database.close()
                    pipeline.close()
            finally:
                preparation.close()

    def test_repair_copies_the_exact_v10_resources_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (3, 3), "white").save(dataset / "image.png")
            (dataset / "image.txt").write_text("1girl", encoding="utf-8")
            catalog, cl, classification = _fixture_catalog(root)
            config = _config(dataset, cl, classification, active=False)
            config.classify["enabled"] = True
            preparation = JobPreparationService(root / "state.db", resource_catalog=catalog)  # type: ignore[arg-type]
            repair = RepairPreparationService(root / "state.db")
            try:
                parent_id = preparation.preflight(config.to_dict()).jobId
                preparation.confirm_workspace(
                    parent_id, confirmed=True, confirmed_rebuild=False
                )
                database = StateDatabase.open(root / "state.db")
                try:
                    sample = database.page_samples(parent_id, limit=1)[0]
                    database.set_job_status(parent_id, "running", current_module_id="classify")
                    database.set_job_status(parent_id, "failed", current_module_id="classify")
                    database.upsert_issue(SampleIssue(
                        issueId="danbooru-repair",
                        jobId=parent_id,
                        sampleId=int(sample["sample_id"]),
                        relativeImagePath=str(sample["relative_image_path"]),
                        moduleId="classify",
                        code="classify_wiki_io_failed",
                        severity="error",
                        blocking=True,
                        retriable=True,
                        repairStartModule="classify",
                        message="retry after restoring the frozen resource",
                        attempt=1,
                    ))
                    parent = database.get_job(parent_id)
                finally:
                    database.close()
                self.assertTrue(preparation.release_lock_for_repair(parent_id))
                result = repair.prepare(parent_id)
                database = StateDatabase.open(root / "state.db")
                try:
                    child = database.get_job(result.repairJobId)
                    self.assertEqual(
                        (
                            10,
                            parent["config_json"],
                            parent["config_hash"],
                        ),
                        (
                            child["config_schema_version"],
                            child["config_json"],
                            child["config_hash"],
                        ),
                    )
                    self.assertNotIn("profile", child.keys())
                finally:
                    database.close()
            finally:
                repair.close()
                preparation.close()


if __name__ == "__main__":
    unittest.main()
