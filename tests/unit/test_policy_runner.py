from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from PIL import Image

from anima_core.classify_overlay import serialize_annotation_json
from anima_core.contracts import JobConfig
from anima_core.db import StateDatabase
from anima_core.job_preflight import JobPreparationService
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.path_safety import sha256_file
from anima_core.policy_runner import PolicyRunner, PolicyRunnerError
from anima_core.scheduler import BoundedScheduler
from anima_core.worker_protocol import ProtocolEnvelopeV1
from test_job_preflight import _write_test_resource_library


def _business() -> dict[str, object]:
    return {
        "quality": [], "count": "solo", "character": "amy_rose", "series": "", "artist": "",
        "appearance": ["white hair"], "tags": ["smile"], "environment": ["outdoors"], "nl": "A person smiles.",
    }


class _PolicyTransport:
    def __init__(self, layout: OverlayLayout, *, mutate_count: bool = False, quality_device: str | None = None) -> None:
        self.layout = layout
        self.mutate_count = mutate_count
        self.quality_device = quality_device
        self.calls = 0
        self.hello_payload: dict[str, object] | None = None

    def exchange(self, request: ProtocolEnvelopeV1) -> ProtocolEnvelopeV1:
        self.calls += 1
        if request.method == "hello":
            self.hello_payload = dict(request.payload)
            quality = request.payload["policy"]["quality"]
            assert isinstance(quality, dict)
            assert set(quality) == {"enabled", "dropoutProbability", "device", "batchSize", "resourceId"}
            quality_enabled = self.quality_device is not None
            payload = {"schemaVersion": 1, "payloadType": "policy_hello_result", "ready": True,
                       "qualityEnabled": quality_enabled, "device": self.quality_device,
                       "modelLoadCount": 1 if quality_enabled else 0,
                       "resourceFingerprint": request.payload["resourceFingerprint"] if quality_enabled else None}
            method = "hello"
        else:
            items = request.payload["items"]
            assert isinstance(items, list)
            outcomes = []
            for item in items:
                assert isinstance(item, dict)
                lease = str(item["leaseId"])
                result = _business()
                result["artist"] = "@Artist"
                if self.mutate_count:
                    result["count"] = "group"
                prepared, digest = self.layout.write_prepared("dropout", lease, ".json", serialize_annotation_json(result))
                outcomes.append({"schemaVersion": 1, "status": "prepared", "sampleId": item["sampleId"],
                                 "leaseId": lease, "relativeImagePath": item["relativeImagePath"],
                                 "preparedRelativePath": str(prepared.relative_to(self.layout.root)).replace("/", "\\"),
                                 "sha256": digest, "aestheticScore": None, "quality": [], "decision": {}})
            payload = {"schemaVersion": 1, "payloadType": "policy_batch_result", "outcomes": outcomes,
                       "modelLoadCount": 1 if self.quality_device is not None else 0}
            method = "result"
        return ProtocolEnvelopeV1(protocolVersion="1.0", kind="response", messageId=f"reply-{request.messageId}",
            runtimeId="policy", owner="policy", method=method, payload=payload, replyTo=request.messageId,
            jobId=request.jobId, configHash=request.configHash)


class PolicyRunnerTests(unittest.TestCase):
    def _prepared_runner(
        self,
        root: Path,
        *,
        mutate_count: bool = False,
        quality_device: str | None = None,
        full_copy: bool = False,
    ) -> tuple[StateDatabase, JobPreparationService, str, OverlayLayout, PolicyRunner]:
        dataset = root / "10_RootArtist"
        artist = dataset / "1_Artist"
        artist.mkdir(parents=True)
        Image.new("RGB", (3, 3), "white").save(artist / "image.png")
        (artist / "image.json").write_bytes(serialize_annotation_json(_business()))
        config = JobConfig(
            profile="e621",
            workMode="full_copy" if full_copy else "in_place",
            overwriteMode="incremental",
            sourceRoot=str(dataset),
            outputRoot=str(root / "output-with-different-name") if full_copy else None,
            recursive=True,
        )
        config.caption["enabled"] = config.classify["enabled"] = config.replace["enabled"] = config.nl["enabled"] = False
        config.countReview["enabled"] = False  # type: ignore[index]
        config.dropout["enabled"] = True
        config.tokenBudget["enabled"] = False  # type: ignore[index]
        config.dropout["quality"]["enabled"] = quality_device is not None
        # This fixture exercises Policy against an existing JSON annotation;
        # incremental preservation must therefore be explicitly overridden.
        config.classify["overwriteJson"] = True
        catalog, _ = _write_test_resource_library(root, include_ocr=False)
        config.caption["resourceId"] = "tagger-default"
        config.classify["resourceId"] = "classify-default"
        config.replace["resourceId"] = "replace-default"
        config.dropout["quality"]["resourceId"] = "dropout-default"
        preparation = JobPreparationService(root / "state.db", resource_catalog=catalog)
        job_id = preparation.preflight(config.to_dict()).jobId
        preparation.confirm_workspace(job_id, confirmed=True, confirmed_rebuild=False)
        database = StateDatabase.open(root / "state.db")
        scheduler = BoundedScheduler(database)
        for module in ("caption", "classify", "replace", "ocr", "nl"):
            scheduler.start_module(job_id, module, enabled=False, profile="e621")
        scheduler.start_module(job_id, "count_review", enabled=False, profile="e621")
        scheduler.start_module(job_id, "dropout", enabled=True, profile="e621")
        job = database.get_job(job_id)
        working_dataset = Path(str(job["dataset_root"]))
        layout = OverlayLayout.open_existing(str(job["overlay_root"]), job_id)
        runner = PolicyRunner(database, scheduler, _PolicyTransport(layout, mutate_count=mutate_count, quality_device=quality_device),
            WorkingAnnotationView(BaselineView(working_dataset), layout), job_id=job_id,
            worker_instance_id="policy-test", install_root=root,
            resource_manifest_relative_path=r"resource-library\dropout-models\dropout-default\resource.json" if quality_device is not None else None,
            resource_fingerprint="a" * 64 if quality_device is not None else None)
        return database, preparation, job_id, layout, runner

    def test_policy_commits_only_valid_prepared_overlay_and_preserves_protected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, preparation, job_id, layout, runner = self._prepared_runner(root)
            try:
                self.assertEqual("completed", runner.run())
                outputs = list((layout.root / "annotations").rglob("*.json"))
                self.assertEqual(1, len(outputs))
                output = outputs[0]
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual("@Artist", result["artist"])
                self.assertEqual(_business()["count"], result["count"])
                self.assertEqual(_business()["tags"], result["tags"])
                self.assertEqual(1, database.module_summary(job_id, "dropout")["completed"])
                transport = runner.transport
                assert isinstance(transport, _PolicyTransport)
                assert transport.hello_payload is not None
                self.assertEqual("10_RootArtist", transport.hello_payload["artistRootName"])
                events = database.event_page(job_id, 0, limit=10)
                self.assertEqual(["running", "running", "completed"], [event["status"] for event in events])
                self.assertEqual([0, 1, 1], [event["completed"] for event in events])
            finally:
                database.close()
                preparation.close()

    def test_full_copy_uses_source_root_name_for_flat_artist_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, _job_id, _layout, runner = self._prepared_runner(
                Path(temporary),
                full_copy=True,
            )
            try:
                self.assertEqual("completed", runner.run())
                transport = runner.transport
                assert isinstance(transport, _PolicyTransport)
                assert transport.hello_payload is not None
                self.assertEqual("10_RootArtist", transport.hello_payload["artistRootName"])
                self.assertNotEqual("output-with-different-name", transport.hello_payload["artistRootName"])
            finally:
                database.close()
                preparation.close()

    def test_policy_observes_a_persisted_pause_without_starting_worker_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, _layout, runner = self._prepared_runner(Path(temporary))
            try:
                database.set_module_summary(job_id, "dropout", status="paused")
                database.set_job_status(job_id, "paused", current_module_id="dropout", resume_status="running")
                self.assertEqual("paused", runner.run())
                self.assertEqual(0, runner.transport.calls)  # type: ignore[attr-defined]
                self.assertEqual("paused", database.event_page(job_id, 0, limit=10)[-1]["status"])
            finally:
                database.close()
                preparation.close()

    def test_policy_rejects_prepared_output_that_changes_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, layout, runner = self._prepared_runner(Path(temporary), mutate_count=True)
            try:
                with self.assertRaises(PolicyRunnerError):
                    runner.run()
                self.assertEqual("failed", database.get_job(job_id)["status"])
                self.assertEqual([], list((layout.root / "annotations").rglob("*.json")))
            finally:
                database.close()
                preparation.close()

    def test_policy_persists_quality_device_evidence_after_overlay_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database, preparation, job_id, _layout, runner = self._prepared_runner(Path(temporary), quality_device="cuda")
            try:
                self.assertEqual("completed", runner.run())
                database.clear_workspace_metadata(job_id)
                evidence = database.get_runtime_evidence(job_id)
                self.assertEqual("cuda", evidence["dropout"]["device"])
            finally:
                database.close()
                preparation.close()


if __name__ == "__main__":
    unittest.main()
