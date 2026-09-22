from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from fastapi import HTTPException

from anima_core.api import (
    _ConfirmBody,
    _CountReviewBatchBody,
    _CountReviewDecisionBody,
    build_control_app,
)
from anima_core.classify_overlay import serialize_annotation_json
from anima_core.classify_protocol import ClassifyCountDecisionV1
from anima_core.contracts import JobConfig
from anima_core.count_review_overlay import CountReviewOverlayWriter
from anima_core.count_review_protocol import CountEvidenceV1, CountObservationV1
from anima_core.count_review_runner import CountReviewRunner
from anima_core.count_review_service import CountReviewError, CountReviewService
from anima_core.db import StateDatabase
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.path_safety import windows_key
from anima_core.pipeline import PipelineService
from anima_core.scheduler import BoundedScheduler


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


class _ApiPipeline:
    def __init__(self) -> None:
        self.confirmations: list[tuple[str, bool]] = []

    def startup_recovery(self) -> dict[str, int]:
        return {}

    def confirm_count_review(self, job_id: str, *, confirmed: bool) -> bool:
        self.confirmations.append((job_id, confirmed))
        return True


def _projection() -> dict[str, object]:
    return {
        "quality": ["high_quality"],
        "count": "group",
        "character": "",
        "series": "",
        "artist": "artist_name",
        "appearance": ["blue_fur"],
        "tags": ["standing"],
        "environment": ["indoors"],
        "nl": "A complete existing caption.",
    }


def _evidence(value: str, warnings: tuple[str, ...] = ()) -> CountEvidenceV1:
    decision = ClassifyCountDecisionV1(
        value=value,
        baseValue=value,
        selectedSource="wiki_tags" if value else "none",
        originalRaw=None,
        originalNormalized=None,
        wikiValue=value or None,
        matchedTags=(value,) if value else (),
        conflict=False,
        issueCodes=warnings,
        warnings=(),
        appliedLowerBounds=(),
    )
    return CountEvidenceV1.from_decision(decision)


def _observation(status: str, count: str | None = None) -> CountObservationV1:
    if status == "not_requested":
        return CountObservationV1.not_requested("nl_disabled")
    return CountObservationV1.from_dict({
        "schemaVersion": 1,
        "status": status,
        "countValue": count,
        "layoutValue": "multi_view" if status == "observed" else None,
        "sameCharacterRepeated": True if status == "observed" else None,
        "warningCodes": ["count_observation_unknown"] if count == "unknown" else (["count_observation_invalid"] if status == "invalid" else []),
        "notRequestedReason": None,
    })


MATRIX = (
    (_evidence("solo"), _observation("observed", "solo")),
    (_evidence("solo"), _observation("observed", "duo")),
    (_evidence("solo", ("wiki_missing",)), _observation("observed", "solo")),
    (_evidence(""), _observation("observed", "trio")),
    (_evidence("duo"), _observation("not_requested")),
    (_evidence(""), _observation("invalid")),
)


class CountReviewFixture:
    def __init__(
        self,
        root: Path,
        inputs: tuple[tuple[CountEvidenceV1, CountObservationV1], ...] = MATRIX,
        *,
        schema_version: int = 10,
        profile: str = "e621",
    ) -> None:
        self.database_path = root / "state.db"
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        for sample_id in range(1, len(inputs) + 1):
            key = f"sample-{sample_id}"
            (self.dataset / f"{key}.png").write_bytes(b"immutable-image")
            (self.dataset / f"{key}.json").write_text(json.dumps(_projection()), encoding="utf-8")
        self.layout = OverlayLayout.create(self.dataset, "job-review")
        self.database = StateDatabase.open(self.database_path)
        config_kwargs = {
            "workMode": "in_place",
            "overwriteMode": "incremental",
            "sourceRoot": str(self.dataset),
            "schemaVersion": schema_version,
        }
        if schema_version != 10:
            config_kwargs["profile"] = profile
        self.config = JobConfig(**config_kwargs)  # type: ignore[arg-type]
        self.config.nl["systemPrompt"] = "Describe the visible image."
        if profile == "danbooru":
            self.config.replace.clear()
            self.config.replace.update({"enabled": False, "indexMode": "bundled"})
        self.database.insert_job({
            "job_id": "job-review",
            "config_schema_version": schema_version,
            "config_json": json.dumps(self.config.to_dict()),
            "config_hash": self.config.config_hash,
            "profile": profile,
            "work_mode": "in_place",
            "overwrite_mode": "incremental",
            "source_root": str(self.dataset),
            "output_root": None,
            "dataset_root": str(self.dataset),
            "dataset_root_key": windows_key(self.dataset),
            "manifest_schema_version": 1,
            "recursive": 0,
            "sample_count": len(inputs),
            "manifest_generated_at": "2026-07-26T00:00:00Z",
            "status": "ready",
            "current_module_id": None,
            "last_event_id": 0,
            "pinned": 0,
            "api_budget_extra": 0,
            "api_budget_revision": 0,
            "overlay_root": str(self.layout.root),
            "commit_journal_path": None,
            "resume_status": None,
            "created_at": "2026-07-26T00:00:00Z",
            "started_at": None,
            "cancel_requested_at": None,
            "finished_at": None,
        })
        self.database.insert_samples("job-review", [
            {
                "sample_id": sample_id,
                "relative_image_path": f"sample-{sample_id}.png",
                "annotation_key": f"sample-{sample_id}",
                "source": profile,
                "in_processing_scope": True,
                "image_format": "png",
                "image_frame_count": 1,
                "original_txt_state": "missing_or_blank",
                "original_json_state": "nonblank",
                "image_file_id": f"volume:{sample_id}",
                "image_size": len(b"immutable-image"),
                "image_mtime_ns": sample_id,
            }
            for sample_id in range(1, len(inputs) + 1)
        ])
        for module_id in ("caption", "classify", "replace", "ocr", "nl"):
            self.database.initialize_module_summary(
                "job-review", module_id, total=len(inputs), status="completed"
            )
        self.database.connection.execute(
            "UPDATE sample_state SET current_module_id='nl',status='completed' WHERE job_id='job-review'"
        )
        self.database.set_job_status("job-review", "running", current_module_id="nl")
        now = "2026-07-26T00:00:00Z"
        for sample_id, (evidence, observation) in enumerate(inputs, start=1):
            self.database.connection.execute(
                """INSERT INTO count_evidence(
                       job_id,sample_id,schema_version,value,decision_json,
                       review_warning_codes_json,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    "job-review",
                    sample_id,
                    1,
                    evidence.value,
                    evidence.decision_json,
                    evidence.review_warning_codes_json,
                    now,
                    now,
                ),
            )
            self.database.connection.execute(
                """INSERT INTO count_observations(
                       job_id,sample_id,schema_version,status,count_value,layout_value,
                       same_character_repeated,warning_codes_json,not_requested_reason,
                       created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "job-review",
                    sample_id,
                    1,
                    observation.status,
                    observation.countValue,
                    observation.layoutValue,
                    None if observation.sameCharacterRepeated is None else int(observation.sameCharacterRepeated),
                    observation.warning_codes_json,
                    observation.notRequestedReason,
                    now,
                    now,
                ),
            )
        lease_ids = iter(f"review-lease-{index}" for index in range(1, len(inputs) + 20))
        self.scheduler = BoundedScheduler(self.database, lease_id_factory=lease_ids.__next__)
        self.assert_started = self.scheduler.start_module(
            "job-review", "count_review", enabled=True, profile=profile
        )
        self.view = WorkingAnnotationView(BaselineView(self.dataset), self.layout)
        self.writer = CountReviewOverlayWriter(
            self.database, self.layout, self.view, "job-review"
        )

    def runner(self) -> CountReviewRunner:
        return CountReviewRunner(
            self.database,
            self.scheduler,
            self.writer,
            job_id="job-review",
            worker_instance_id="count-review-core",
        )

    def close(self) -> None:
        self.database.close()
        if self.layout.root.exists():
            self.layout.discard()


class CountReviewTests(unittest.TestCase):
    def test_pipeline_confirmation_waits_for_the_review_initializer_to_unregister(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(
                Path(temporary),
                ((_evidence("solo"), _observation("observed", "duo")),),
            )
            pipeline = PipelineService(fixture.database_path, install_root=ROOT / ".runtime-build")

            class SettlingThread:
                joined = False

                def join(self, timeout: float | None = None) -> None:
                    self.joined = True
                    with pipeline._lock:
                        pipeline._threads.pop("job-review", None)

            settling = SettlingThread()
            try:
                self.assertEqual("reviewing", fixture.runner().run())
                CountReviewService(fixture.database, "job-review").resolve(
                    1, expected_version=1, source="vlm"
                )
                pipeline._run = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
                pipeline._threads["job-review"] = settling  # type: ignore[assignment]

                self.assertTrue(pipeline.confirm_count_review("job-review", confirmed=True))
                pipeline.close()
                self.assertTrue(settling.joined)
                self.assertEqual(
                    ("running", "count_review"),
                    (
                        fixture.database.get_job("job-review")["status"],
                        fixture.database.get_job("job-review")["current_module_id"],
                    ),
                )
            finally:
                pipeline._threads.pop("job-review", None)
                pipeline.close()
                fixture.close()

    def test_nl_disabled_count_review_runs_from_classify_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(
                Path(temporary),
                ((_evidence("solo"), _observation("not_requested")),),
            )
            try:
                self.assertEqual("completed", fixture.runner().run())
                decision = fixture.database.get_count_review_decision("job-review", 1)
                self.assertEqual(("auto_resolved", "solo", "classify"), (
                    decision["status"], decision["final_count"], decision["selected_source"],
                ))
            finally:
                fixture.close()

    def test_danbooru_v4_pending_review_confirms_and_applies_only_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(
                Path(temporary),
                ((_evidence("solo"), _observation("observed", "duo")),),
                schema_version=10,
                profile="e621",
            )
            try:
                self.assertEqual("reviewing", fixture.runner().run())
                service = CountReviewService(fixture.database, "job-review")
                page = service.page(limit=10)
                self.assertEqual((1, 1), (page["targetCount"], page["pendingCount"]))
                service.resolve(1, expected_version=1, source="manual", count="trio")
                self.assertTrue(service.confirm(
                    confirmed=True, expected_config_hash=fixture.config.config_hash
                ))
                self.assertEqual("completed", fixture.runner().run())
                result = json.loads(
                    fixture.layout.annotation_path("sample-1", ".json").read_text(encoding="utf-8")
                )
                self.assertEqual("trio", result["count"])
                self.assertEqual(
                    {key: value for key, value in _projection().items() if key != "count"},
                    {key: value for key, value in result.items() if key != "count"},
                )
            finally:
                fixture.close()

    def test_initialize_update_confirm_and_apply_only_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(Path(temporary))
            try:
                evidence_before = [tuple(row) for row in fixture.database.page_count_evidence("job-review", limit=100)]
                observations_before = [tuple(row) for row in fixture.database.page_count_observations("job-review", limit=100)]
                self.assertEqual("reviewing", fixture.runner().run())
                rows = fixture.database.page_count_review_decisions("job-review", limit=100)
                self.assertEqual(
                    [
                        ("auto_resolved", "solo", "consensus", "[]"),
                        ("pending", None, None, '["count_observation_mismatch"]'),
                        ("pending", None, None, '["wiki_missing"]'),
                        ("auto_resolved", "trio", "vlm", "[]"),
                        ("auto_resolved", "duo", "classify", "[]"),
                        ("pending", None, None, '["count_observation_invalid"]'),
                    ],
                    [(row["status"], row["final_count"], row["selected_source"], row["review_reasons_json"]) for row in rows],
                )
                service = CountReviewService(fixture.database, "job-review")
                with self.assertRaisesRegex(ValueError, "pending"):
                    service.confirm(confirmed=True, expected_config_hash=fixture.config.config_hash)

                with self.assertRaisesRegex(ValueError, "version conflict"):
                    service.resolve_batch([
                        {"sampleId": 2, "expectedVersion": 1, "source": "classify", "count": None},
                        {"sampleId": 3, "expectedVersion": 99, "source": "vlm", "count": None},
                    ])
                self.assertEqual("pending", fixture.database.get_count_review_decision("job-review", 2)["status"])

                updated = service.resolve_batch([
                    {"sampleId": 2, "expectedVersion": 1, "source": "classify", "count": None},
                    {"sampleId": 3, "expectedVersion": 1, "source": "vlm", "count": None},
                ])
                self.assertEqual([("classify", "solo", 2), ("vlm", "solo", 2)], [(row["selected_source"], row["final_count"], row["version"]) for row in updated])
                row = service.resolve(6, expected_version=1, source="manual", count="group")
                self.assertEqual(("manual_resolved", "group", 2), (row["status"], row["final_count"], row["version"]))
                row = service.resolve(6, expected_version=2, source="manual", count="duo")
                self.assertEqual(("duo", 3), (row["final_count"], row["version"]))
                with self.assertRaisesRegex(ValueError, "version conflict"):
                    service.resolve(6, expected_version=2, source="manual", count="solo")

                versions = [(row["status"], row["final_count"], row["version"]) for row in fixture.database.page_count_review_decisions("job-review", limit=100)]
                self.assertEqual(0, service.initialize().inserted)
                self.assertEqual(versions, [(row["status"], row["final_count"], row["version"]) for row in fixture.database.page_count_review_decisions("job-review", limit=100)])

                self.assertTrue(service.confirm(confirmed=True, expected_config_hash=fixture.config.config_hash))
                self.assertFalse(service.confirm(confirmed=True, expected_config_hash=fixture.config.config_hash))
                self.assertEqual("completed", fixture.runner().run())
                self.assertEqual("completed", fixture.database.module_summary("job-review", "count_review")["status"])
                expected_counts = ("solo", "solo", "solo", "trio", "duo", "duo")
                baseline = _projection()
                for sample_id, expected_count in enumerate(expected_counts, start=1):
                    value = json.loads(fixture.layout.annotation_path(f"sample-{sample_id}", ".json").read_text(encoding="utf-8"))
                    self.assertEqual(expected_count, value["count"])
                    self.assertEqual(
                        {key: item for key, item in baseline.items() if key != "count"},
                        {key: item for key, item in value.items() if key != "count"},
                    )
                decisions = fixture.database.page_count_review_decisions("job-review", limit=100)
                self.assertTrue(all(row["applied_at"] is not None for row in decisions))
                self.assertEqual(evidence_before, [tuple(row) for row in fixture.database.page_count_evidence("job-review", limit=100)])
                self.assertEqual(observations_before, [tuple(row) for row in fixture.database.page_count_observations("job-review", limit=100)])
            finally:
                fixture.close()

    def test_missing_evidence_blocks_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(Path(temporary), MATRIX[:1])
            try:
                fixture.database.connection.execute(
                    "DELETE FROM count_evidence WHERE job_id='job-review' AND sample_id=1"
                )
                with self.assertRaisesRegex(CountReviewError, "evidence is missing"):
                    fixture.runner().run()
                self.assertEqual(0, fixture.database.count_current_review_decisions("job-review"))
            finally:
                fixture.close()

    def test_runner_returns_paused_at_the_batch_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(Path(temporary), MATRIX[:1])
            try:
                fixture.database.set_module_summary("job-review", "count_review", status="paused")
                fixture.database.set_job_status(
                    "job-review", "paused", current_module_id="count_review", resume_status="running"
                )
                self.assertEqual("paused", fixture.runner().run())
            finally:
                fixture.close()

    def test_prepared_recovery_commits_and_marks_decision_applied(self) -> None:
        for target_already_committed in (False, True):
            with self.subTest(target_already_committed=target_already_committed), tempfile.TemporaryDirectory() as temporary:
                fixture = CountReviewFixture(Path(temporary), MATRIX[:1])
                try:
                    service = CountReviewService(fixture.database, "job-review")
                    self.assertEqual(0, service.initialize().pending)
                    lease = fixture.scheduler.claim_batch(
                        "job-review", "count_review", "count-review-core", fixture.config.config_hash, limit=1
                    )[0]
                    if target_already_committed:
                        fixture.writer.write(lease, annotation_key="sample-1", final_count="solo")
                    else:
                        value = _projection()
                        value["count"] = "solo"
                        prepared, digest = fixture.layout.write_prepared(
                            "count_review", str(lease.leaseId), ".json", serialize_annotation_json(value)
                        )
                        relative = os.path.relpath(prepared, fixture.layout.root).replace("/", "\\")
                        fixture.database.stage_prepared_artifact(
                            "job-review", 1, lease_id=str(lease.leaseId), relative_path=relative, sha256=digest
                        )
                    fixture.database.mark_interrupted("job-review")
                    report = fixture.scheduler.recover(
                        "job-review",
                        confirmed=True,
                        expected_config_hash=fixture.config.config_hash,
                        manifest_schema_version=1,
                        protocol_version="1.0",
                        verify_source_fingerprints=lambda: True,
                        commit_prepared=fixture.writer.recover_prepared,
                    )
                    self.assertEqual((1, 0), (report.committedPrepared, report.repeatedPrepared))
                    decision = fixture.database.get_count_review_decision("job-review", 1)
                    self.assertIsNotNone(decision["applied_at"])
                    self.assertEqual("completed", fixture.database.get_sample_state("job-review", 1)["status"])
                    value = json.loads(fixture.layout.annotation_path("sample-1", ".json").read_text(encoding="utf-8"))
                    self.assertEqual("solo", value["count"])
                    self.assertEqual(
                        {key: item for key, item in _projection().items() if key != "count"},
                        {key: item for key, item in value.items() if key != "count"},
                    )
                finally:
                    fixture.close()

    def test_api_pages_filters_updates_and_maps_version_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(Path(temporary))
            pipeline = _ApiPipeline()
            try:
                self.assertEqual("reviewing", fixture.runner().run())
                app = build_control_app(
                    database_path=fixture.database_path,
                    pipeline_service=pipeline,  # type: ignore[arg-type]
                )
                listing = _endpoint(app, "/api/jobs/{job_id}/count-review", "GET")
                update = _endpoint(app, "/api/jobs/{job_id}/count-review/{sample_id}", "PUT")
                update_batch = _endpoint(app, "/api/jobs/{job_id}/count-review/batch", "POST")
                confirm = _endpoint(app, "/api/jobs/{job_id}/count-review/confirm", "POST")

                def page(**overrides):
                    arguments = {
                        "afterSampleId": 0,
                        "status": None,
                        "reason": None,
                        "classifyCount": None,
                        "vlmCount": None,
                        "mismatchOnly": False,
                        "limit": 200,
                    }
                    arguments.update(overrides)
                    return listing("job-review", **arguments)

                first = page(limit=2)
                self.assertEqual([1, 2], [item["sampleId"] for item in first["items"]])
                self.assertEqual((6, 3, 2), (first["targetCount"], first["pendingCount"], first["nextAfterSampleId"]))
                second = page(afterSampleId=2, limit=2)
                self.assertEqual([3, 4], [item["sampleId"] for item in second["items"]])
                self.assertEqual([2, 3, 6], [item["sampleId"] for item in page(status="pending")["items"]])
                self.assertEqual([2], [item["sampleId"] for item in page(reason="count_observation_mismatch")["items"]])
                self.assertEqual([4, 6], [item["sampleId"] for item in page(classifyCount="unavailable")["items"]])
                self.assertEqual([5, 6], [item["sampleId"] for item in page(vlmCount="unavailable")["items"]])
                self.assertEqual([2], [item["sampleId"] for item in page(mismatchOnly=True)["items"]])
                serialized = json.dumps(first)
                self.assertNotIn(str(fixture.dataset), serialized)
                self.assertNotIn("decision_json", serialized)

                response = update(
                    "job-review",
                    2,
                    _CountReviewDecisionBody(expectedVersion=1, source="classify", count=None),
                )
                self.assertEqual(("solo", 2), (response["decision"]["finalCount"], response["decision"]["version"]))
                with self.assertRaises(HTTPException) as conflict:
                    update(
                        "job-review",
                        2,
                        _CountReviewDecisionBody(expectedVersion=1, source="classify", count=None),
                    )
                self.assertEqual(409, conflict.exception.status_code)

                with self.assertRaises(HTTPException) as batch_conflict:
                    update_batch("job-review", _CountReviewBatchBody(updates=[
                        {"sampleId": 3, "expectedVersion": 1, "source": "vlm", "count": None},
                        {"sampleId": 6, "expectedVersion": 99, "source": "manual", "count": "group"},
                    ]))
                self.assertEqual(409, batch_conflict.exception.status_code)
                self.assertEqual("pending", fixture.database.get_count_review_decision("job-review", 3)["status"])

                batch = update_batch("job-review", _CountReviewBatchBody(updates=[
                    {"sampleId": 3, "expectedVersion": 1, "source": "vlm", "count": None},
                    {"sampleId": 6, "expectedVersion": 1, "source": "manual", "count": "group"},
                ]))
                self.assertEqual([3, 6], [item["sampleId"] for item in batch["items"]])
                self.assertEqual({"jobId": "job-review", "started": True}, confirm("job-review", _ConfirmBody(confirmed=True)))
                self.assertEqual([("job-review", True)], pipeline.confirmations)
            finally:
                fixture.close()

    def test_image_preview_uses_manifest_membership_and_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(Path(temporary), MATRIX[:1])
            try:
                self.assertEqual("completed", fixture.runner().run())
                app = build_control_app(
                    database_path=fixture.database_path,
                    pipeline_service=_ApiPipeline(),  # type: ignore[arg-type]
                )
                preview = _endpoint(
                    app,
                    "/api/jobs/{job_id}/count-review/{sample_id}/image",
                    "GET",
                )
                response = preview("job-review", 1)
                self.assertEqual("image/png", response.media_type)
                self.assertEqual("no-store", response.headers["cache-control"])
                self.assertNotIn(str(fixture.dataset), json.dumps(dict(response.headers)))
                with self.assertRaises(HTTPException) as missing_sample:
                    preview("job-review", 999)
                self.assertEqual(404, missing_sample.exception.status_code)

                original = "sample-1.png"
                for unsafe in ("..\\secret.png", str(fixture.dataset / "sample-1.png")):
                    with self.subTest(unsafe=unsafe):
                        fixture.database.connection.execute(
                            "UPDATE samples SET relative_image_path=? WHERE job_id='job-review' AND sample_id=1",
                            (unsafe,),
                        )
                        with self.assertRaises(HTTPException) as rejected:
                            preview("job-review", 1)
                        self.assertEqual(400, rejected.exception.status_code)
                        self.assertNotIn(str(fixture.dataset), str(rejected.exception.detail))
                fixture.database.connection.execute(
                    "UPDATE samples SET relative_image_path=? WHERE job_id='job-review' AND sample_id=1",
                    ("missing.png",),
                )
                with self.assertRaises(HTTPException) as missing_file:
                    preview("job-review", 1)
                self.assertEqual(404, missing_file.exception.status_code)
                fixture.database.connection.execute(
                    "UPDATE samples SET relative_image_path=? WHERE job_id='job-review' AND sample_id=1",
                    (original,),
                )
                with patch(
                    "anima_core.path_safety._is_reparse",
                    side_effect=lambda path: Path(path).name == original,
                ):
                    with self.assertRaises(HTTPException) as reparse:
                        preview("job-review", 1)
                self.assertEqual(400, reparse.exception.status_code)
            finally:
                fixture.close()

    def test_pipeline_confirmation_enforces_pending_gate_and_rolls_back_thread_start_failure(self) -> None:
        class CapturedThread:
            instances: list["CapturedThread"] = []

            def __init__(self, *, target, args, daemon, name) -> None:
                self.target = target
                self.args = args
                self.daemon = daemon
                self.name = name
                self.started = False
                self.instances.append(self)

            def start(self) -> None:
                self.started = True

            def join(self, timeout=None) -> None:
                return None

        class FailingThread(CapturedThread):
            def start(self) -> None:
                raise RuntimeError("thread start failed")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = CountReviewFixture(Path(temporary))
            pipeline = PipelineService(fixture.database_path, install_root=ROOT / ".runtime-build")
            try:
                self.assertEqual("reviewing", fixture.runner().run())
                with self.assertRaisesRegex(ValueError, "pending"):
                    pipeline.confirm_count_review("job-review", confirmed=True)
                service = CountReviewService(fixture.database, "job-review")
                service.resolve(2, expected_version=1, source="classify")
                service.resolve(3, expected_version=1, source="vlm")
                service.resolve(6, expected_version=1, source="manual", count="group")
                with self.assertRaisesRegex(CountReviewError, "explicit"):
                    pipeline.confirm_count_review("job-review", confirmed=False)

                with patch("anima_core.pipeline.threading.Thread", FailingThread):
                    with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                        pipeline.confirm_count_review("job-review", confirmed=True)
                self.assertEqual("reviewing", fixture.database.get_job("job-review")["status"])

                with patch("anima_core.pipeline.threading.Thread", CapturedThread):
                    self.assertTrue(pipeline.confirm_count_review("job-review", confirmed=True))
                    self.assertFalse(pipeline.confirm_count_review("job-review", confirmed=True))
                self.assertEqual(("job-review", True), CapturedThread.instances[-1].args)
                self.assertTrue(CapturedThread.instances[-1].started)
                self.assertEqual("running", fixture.database.get_job("job-review")["status"])
                pipeline._threads.clear()
            finally:
                pipeline.close()
                fixture.close()


if __name__ == "__main__":
    unittest.main()
