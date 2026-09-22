from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared" / "anima_caption_format"))
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_caption_format import serialize_flat_txt
from anima_caption_format.normalizer import CaptionDisplayPolicy
from anima_core.caption_overlay import CaptionOverlayWriter
from anima_core.caption_protocol import CaptionResultV1, CaptionTagV1, CaptionWorkItemV1
from anima_core.contracts import JobConfig, WorkLease
from anima_core.db import StateDatabase
from anima_core.overlay import BaselineView, OverlayLayout, WorkingAnnotationView
from anima_core.path_safety import windows_key
from anima_core.scheduler import BoundedScheduler


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("/", "\\"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _job(job_id: str, dataset: Path, overlay: OverlayLayout) -> tuple[dict[str, object], JobConfig]:
    config = JobConfig(
        profile="e621",
        workMode="in_place",
        overwriteMode="incremental",
        sourceRoot=str(dataset),
    )
    config.caption["overwriteTxt"] = True
    return {
        "job_id": job_id,
        "config_schema_version": config.schemaVersion,
        "config_json": json.dumps(config.to_dict()),
        "config_hash": config.config_hash,
        "profile": "e621",
        "work_mode": "in_place",
        "overwrite_mode": "incremental",
        "source_root": str(dataset),
        "output_root": None,
        "dataset_root": str(dataset),
        "dataset_root_key": windows_key(dataset),
        "manifest_schema_version": 1,
        "recursive": 0,
        "sample_count": 1,
        "manifest_generated_at": "2026-07-24T00:00:00Z",
        "status": "ready",
        "current_module_id": None,
        "last_event_id": 0,
        "pinned": 0,
        "api_budget_extra": 0,
        "api_budget_revision": 0,
        "overlay_root": str(overlay.root),
        "commit_journal_path": None,
        "resume_status": None,
        "created_at": "2026-07-24T00:00:00Z",
        "started_at": None,
        "cancel_requested_at": None,
        "finished_at": None,
    }, config


class CaptionOverlayFixture:
    def __init__(self, root: Path) -> None:
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "sample.png").write_bytes(b"immutable-image")
        (self.dataset / "sample.txt").write_bytes(b"baseline caption")
        (self.dataset / "sample.json").write_bytes(b'{"nl":"baseline"}')
        (self.dataset / "sample.safetensors").write_bytes(b"immutable-latent")
        self.layout = OverlayLayout.create(self.dataset, "job-caption-overlay")
        self.database = StateDatabase.open(root / "state.db")
        job, self.config = _job("job-caption-overlay", self.dataset, self.layout)
        self.database.insert_job(job)
        self.database.insert_samples(
            "job-caption-overlay",
            [
                {
                    "sample_id": 1,
                    "relative_image_path": "sample.png",
                    "annotation_key": "sample",
                    "source": "e621",
                    "in_processing_scope": True,
                    "image_format": "png",
                    "image_frame_count": 1,
                    "original_txt_state": "nonblank",
                    "original_json_state": "nonblank",
                    "image_file_id": "volume:1",
                    "image_size": len(b"immutable-image"),
                    "image_mtime_ns": 1_000_000,
                    "original_txt_sha256": hashlib.sha256(b"baseline caption").hexdigest(),
                    "original_json_sha256": hashlib.sha256(b'{"nl":"baseline"}').hexdigest(),
                }
            ],
        )
        self.scheduler = BoundedScheduler(self.database, lease_id_factory=lambda: "lease-caption-1")
        self.scheduler.start_module("job-caption-overlay", "caption", enabled=True, profile="e621")
        self.lease = self.scheduler.claim_batch(
            "job-caption-overlay",
            "caption",
            "caption-worker-1",
            self.config.config_hash,
            limit=1,
        )[0]
        row = self.database.get_leased_sample(
            "job-caption-overlay",
            "caption",
            1,
            lease_id=str(self.lease.leaseId),
            worker_instance_id=str(self.lease.workerInstanceId),
        )
        self.item = CaptionWorkItemV1.from_dict(
            {
                "schemaVersion": 1,
                "sampleId": 1,
                "leaseId": str(row["lease_id"]),
                "source": "e621",
                "relativeImagePath": str(row["relative_image_path"]),
                "annotationKey": str(row["annotation_key"]),
                "imageFormat": str(row["image_format"]),
                "imageFrameCount": int(row["image_frame_count"]),
                "imageFileId": str(row["image_file_id"]),
                "imageSize": int(row["image_size"]),
                "imageMtimeNs": int(row["image_mtime_ns"]),
            }
        )
        self.result = CaptionResultV1(
            sampleId=1,
            leaseId=self.item.leaseId,
            relativeImagePath=self.item.relativeImagePath,
            tags=(CaptionTagV1("test_tag", 0.8, "general"),),
            formattedTxt="test_tag",
            provider="CPUExecutionProvider",
        )
        self.writer = CaptionOverlayWriter(self.database, self.layout, "job-caption-overlay")

    def write_prepared(self) -> tuple[str, str]:
        prepared, digest = self.layout.write_prepared(
            "caption",
            self.item.leaseId,
            ".txt",
            self.result.formattedTxt.encode("utf-8"),
        )
        return os.path.relpath(prepared, self.layout.root).replace("/", "\\"), digest

    def recover(self):
        self.database.mark_interrupted("job-caption-overlay")
        return self.scheduler.recover(
            "job-caption-overlay",
            confirmed=True,
            expected_config_hash=self.config.config_hash,
            manifest_schema_version=1,
            protocol_version="1.0",
            verify_source_fingerprints=lambda: True,
            commit_prepared=self.writer.recover_prepared,
        )

    def close(self) -> None:
        self.database.close()
        if self.layout.root.exists():
            self.layout.discard()


class CaptionOverlayTests(unittest.TestCase):
    def test_commit_prepared_retries_transient_windows_file_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptionOverlayFixture(Path(temporary))
            try:
                relative, digest = fixture.write_prepared()
                sharing_error = PermissionError(32, "sharing violation")
                sharing_error.winerror = 32
                with patch(
                    "anima_core.overlay.os.replace",
                    side_effect=[sharing_error, None],
                ) as replace:
                    fixture.layout.commit_prepared(relative, digest, "sample", ".txt")
                self.assertEqual(2, replace.call_count)
            finally:
                fixture.close()

    def test_commit_ocr_prepared_retries_transient_windows_file_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptionOverlayFixture(Path(temporary))
            try:
                prepared, digest = fixture.layout.write_ocr_prepared("lease-ocr-1", b"{}")
                relative = os.path.relpath(prepared, fixture.layout.root).replace("/", "\\")
                sharing_error = PermissionError(32, "sharing violation")
                sharing_error.winerror = 32
                with patch(
                    "anima_core.overlay.os.replace",
                    side_effect=[sharing_error, None],
                ) as replace:
                    fixture.layout.commit_ocr_prepared(relative, digest, "sample.png")
                self.assertEqual(2, replace.call_count)
            finally:
                fixture.close()

    def test_writer_serializes_caption_tags_with_the_shared_v10_serializer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptionOverlayFixture(Path(temporary))
            try:
                fixture.result = CaptionResultV1(
                    sampleId=1,
                    leaseId=fixture.item.leaseId,
                    relativeImagePath=fixture.item.relativeImagePath,
                    tags=(CaptionTagV1("blue_eyes", 0.9, "general"), CaptionTagV1("long_hair", 0.8, "general")),
                    formattedTxt="legacy worker preview",
                    provider="CPUExecutionProvider",
                )
                fixture.writer(fixture.item, fixture.result)

                policy = CaptionDisplayPolicy.from_mapping(fixture.config.to_dict()["captionFormat"])
                expected = serialize_flat_txt({
                    "quality": [], "count": "", "character": "", "series": "", "artist": "",
                    "appearance": [], "tags": ["blue_eyes", "long_hair"], "environment": [], "nl": "",
                }, policy)
                self.assertEqual(expected, fixture.layout.annotation_path("sample", ".txt").read_bytes())
            finally:
                fixture.close()

    def test_writer_stages_then_commits_overlay_without_mutating_the_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptionOverlayFixture(Path(temporary))
            before = _tree_hashes(fixture.dataset)
            try:
                fixture.writer(fixture.item, fixture.result)
                prepared_state = fixture.database.get_sample_state("job-caption-overlay", 1)
                self.assertEqual("prepared", prepared_state["status"])
                self.assertFalse(
                    fixture.layout.resolve_prepared(prepared_state["prepared_artifact_relative_path"]).exists()
                )
                self.assertEqual(b"test tag.", fixture.layout.annotation_path("sample", ".txt").read_bytes())
                fixture.scheduler.complete(fixture.lease, txt_provenance="module1_written")
                completed = fixture.database.get_sample_state("job-caption-overlay", 1)
                self.assertEqual(("completed", "module1_written"), (completed["status"], completed["txt_provenance"]))
                self.assertEqual(before, _tree_hashes(fixture.dataset))
                self.assertEqual(b"baseline caption", BaselineView(fixture.dataset).read("sample", ".txt"))
                self.assertEqual(
                    b"test tag.",
                    WorkingAnnotationView(BaselineView(fixture.dataset), fixture.layout).read("sample", ".txt"),
                )
                annotation_files = [path.name for path in fixture.layout.root.joinpath("annotations").rglob("*") if path.is_file()]
                self.assertEqual(["sample.txt"], annotation_files)
            finally:
                fixture.close()

    def test_recovery_handles_all_caption_prepared_crash_windows(self) -> None:
        for window in ("orphan_before_state", "prepared_before_move", "target_before_complete", "invalid_prepared"):
            with self.subTest(window=window), tempfile.TemporaryDirectory() as temporary:
                fixture = CaptionOverlayFixture(Path(temporary))
                before = _tree_hashes(fixture.dataset)
                try:
                    relative, digest = fixture.write_prepared()
                    if window != "orphan_before_state":
                        fixture.database.stage_prepared_artifact(
                            "job-caption-overlay",
                            1,
                            lease_id=fixture.item.leaseId,
                            relative_path=relative,
                            sha256=digest,
                        )
                    if window == "target_before_complete":
                        fixture.layout.commit_prepared(relative, digest, "sample", ".txt")
                    if window == "invalid_prepared":
                        fixture.layout.resolve_prepared(relative).write_bytes(b"tampered")
                    report = fixture.recover()
                    state = fixture.database.get_sample_state("job-caption-overlay", 1)
                    if window == "orphan_before_state":
                        self.assertEqual((1, 0, 0), (report.returnedLeases, report.committedPrepared, report.repeatedPrepared))
                        self.assertEqual("pending", state["status"])
                        self.assertFalse(fixture.layout.annotation_path("sample", ".txt").exists())
                    elif window == "invalid_prepared":
                        self.assertEqual((0, 0, 1), (report.returnedLeases, report.committedPrepared, report.repeatedPrepared))
                        self.assertEqual("pending", state["status"])
                        self.assertFalse(fixture.layout.annotation_path("sample", ".txt").exists())
                    else:
                        self.assertEqual((0, 1, 0), (report.returnedLeases, report.committedPrepared, report.repeatedPrepared))
                        self.assertEqual(("completed", "module1_written"), (state["status"], state["txt_provenance"]))
                        self.assertEqual(b"test_tag", fixture.layout.annotation_path("sample", ".txt").read_bytes())
                        summary = fixture.database.module_summary("job-caption-overlay", "caption")
                        self.assertEqual(1, summary["completed"])
                    self.assertEqual(before, _tree_hashes(fixture.dataset))
                finally:
                    fixture.close()


if __name__ == "__main__":
    unittest.main()
