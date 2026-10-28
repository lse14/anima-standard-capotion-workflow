from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core/src"))
sys.path.insert(0, str(ROOT / "tests/unit"))
from test_nl_runner import NlFixture, FakeNlTransport, _projection
from anima_core.overlay import OverlayLayout
from anima_core.moss_sidecar import parse_moss_json


def _moss_output():
    return {
        "caption": {
            "long": "A cat sits beside a sunlit window.",
            "medium": "A cat sits beside a window.",
            "short": "A cat by a window.",
        },
        "watermark": [],
        "borders": ["none"],
    }


class NlHistoryAcceptanceTests(unittest.TestCase):
    def test_reused_original_survives_job_deletion_and_checks_image_and_text(self):
        with tempfile.TemporaryDirectory() as temp:
            f = NlFixture(Path(temp))
            try:
                f.config.nl.update({"apiEnabled": False, "reuseOriginalNl": True})
                f.refresh_config()
                self.assertEqual("completed", f.runner(FakeNlTransport()).run())
                history = f.database.verified_nl_history_for_sample("job-nl", 1)
                self.assertEqual("old caption", history["text"])
                self.assertEqual(hashlib.sha256(b"immutable-image").hexdigest(), history["image_sha256"])
                (f.dataset / "sample.png").write_bytes(b"different-image")
                self.assertIsNone(f.database.verified_nl_history_for_sample("job-nl", 1))
                (f.dataset / "sample.png").write_bytes(b"immutable-image")
                f.database.connection.execute("UPDATE nl_history SET text='tampered'")
                self.assertIsNone(f.database.verified_nl_history_for_sample("job-nl", 1))
                f.database.connection.execute("DELETE FROM jobs WHERE job_id='job-nl'")
                self.assertEqual(1, f.database.connection.execute("SELECT COUNT(*) FROM nl_history").fetchone()[0])
            finally:
                f.close()

    def test_history_restores_blank_nl_without_api_and_protects_current_manual_text(self):
        for current in ("", "New manual text"):
            with self.subTest(current=current), tempfile.TemporaryDirectory() as temp:
                f = NlFixture(Path(temp), baseline_nl=current)
                try:
                    f.database.archive_nl_for_sample("job-nl", 1, source="dropout", text="Archived caption")
                    f.config.nl.update({"apiEnabled": False, "reuseOriginalNl": True})
                    f.refresh_config()
                    transport = FakeNlTransport()
                    self.assertEqual("completed", f.runner(transport).run())
                    value = json.loads(f.writer.view.read("sample", ".json"))
                    self.assertEqual(current or "Archived caption", value["nl"])
                    self.assertEqual(0, transport.process)
                finally:
                    f.close()

    def test_prepared_recovery_is_idempotent_and_never_archives_unverified_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            f = NlFixture(Path(temp))
            try:
                lease = f.scheduler.claim_batch("job-nl", "nl", "test", f.config.config_hash)[0]
                with patch.object(OverlayLayout, "commit_prepared", side_effect=OSError("crash")):
                    with self.assertRaises(OSError):
                        f.writer.write_value(sample_id=1, lease_id=lease.leaseId, annotation_key="sample", projection=_projection(), nl="Recovered caption")
                row = f.database.get_sample_state("job-nl", 1)
                path = f.layout.resolve_prepared(row["prepared_artifact_relative_path"])
                valid = path.read_bytes()
                path.write_text(json.dumps(_projection("UNVERIFIED")), encoding="utf-8")
                self.assertFalse(f.writer.recover_prepared("job-nl", 1, row["prepared_artifact_relative_path"], row["prepared_artifact_sha256"]))
                self.assertEqual(0, f.database.connection.execute("SELECT COUNT(*) FROM nl_history WHERE text='UNVERIFIED'").fetchone()[0])
                path.write_bytes(valid)
                before = f.database.connection.execute("SELECT COUNT(*) FROM nl_history").fetchone()[0]
                self.assertTrue(f.writer.recover_prepared("job-nl", 1, row["prepared_artifact_relative_path"], row["prepared_artifact_sha256"]))
                self.assertTrue(f.writer.recover_prepared("job-nl", 1, row["prepared_artifact_relative_path"], row["prepared_artifact_sha256"]))
                self.assertEqual(before, f.database.connection.execute("SELECT COUNT(*) FROM nl_history").fetchone()[0])
            finally:
                f.close()

    def test_prepared_moss_recovery_only_commits_the_moss_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            f = NlFixture(Path(temp))
            try:
                baseline_path = f.dataset / "sample.json"
                baseline_bytes = baseline_path.read_bytes()
                f.config.nl.update({
                    "promptVersion": "nl-default-prompt-moss-v1",
                    "captionPreset": "moss-general",
                    "apiEnabled": True,
                    "reuseOriginalNl": False,
                    "useImage": True,
                    "useFullJson": False,
                })
                f.refresh_config()
                lease = f.scheduler.claim_batch("job-nl", "nl", "test", f.config.config_hash)[0]
                f.database.mark_nl_request_started("job-nl", 1, lease_id=str(lease.leaseId))
                with patch.object(OverlayLayout, "commit_moss_prepared", side_effect=OSError("crash")):
                    with self.assertRaises(OSError):
                        f.writer.write_moss_value(
                            sample_id=1,
                            lease_id=lease.leaseId,
                            relative_image_path="sample.png",
                            moss=_moss_output(),
                        )

                row = f.database.get_sample_state("job-nl", 1)
                self.assertEqual("prepared", row["status"])
                self.assertEqual(0, f.database.connection.execute(
                    "SELECT COUNT(*) FROM nl_history WHERE job_id='job-nl'"
                ).fetchone()[0])
                self.assertTrue(f.writer.recover_prepared(
                    "job-nl", 1, row["prepared_artifact_relative_path"], row["prepared_artifact_sha256"]
                ))
                self.assertTrue(f.writer.recover_prepared(
                    "job-nl", 1, row["prepared_artifact_relative_path"], row["prepared_artifact_sha256"]
                ))

                self.assertEqual(baseline_bytes, baseline_path.read_bytes())
                self.assertEqual(_moss_output(), parse_moss_json(
                    f.layout.moss_sidecar_path("sample.png").read_bytes()
                ))
                self.assertFalse(f.layout.annotation_path("sample", ".json").exists())
                self.assertEqual(0, f.database.connection.execute(
                    "SELECT COUNT(*) FROM nl_history WHERE job_id='job-nl'"
                ).fetchone()[0])
            finally:
                f.close()


if __name__ == "__main__":
    unittest.main()
