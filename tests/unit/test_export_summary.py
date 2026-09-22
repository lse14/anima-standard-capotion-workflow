from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_core.commit_journal import CommitJournal
from anima_core.export_summary import build_export_summary, commit_status


class ExportSummaryTests(unittest.TestCase):
    def _journal(
        self,
        state: str,
        *,
        schema_version: int = 1,
        phase: str | None = None,
    ) -> CommitJournal:
        root = Path(tempfile.gettempdir()) / "anima-export-summary"
        return CommitJournal(
            "job", state, root / "dataset", root / ".dataset.stage",
            root / ".dataset.rollback", root / ".dataset.anima-backups" / "job.zip",
            schema_version=schema_version, phase=phase,
            updated_at="2026-09-16T00:00:00Z" if schema_version == 2 else None,
        )  # type: ignore[arg-type]

    def test_every_persisted_journal_state_has_one_frozen_ui_status(self) -> None:
        expected = {
            "prepared": "staged", "backup_verified": "backup_verified", "rollback_created": "committing",
            "committed": "committed", "rolled_back": "rolled_back", "rollback_required": "rollback_required",
        }
        for state, status in expected.items():
            self.assertEqual(status, commit_status(job_status="committing", module_status="completed", journal=self._journal(state)))

    def test_v2_journal_phase_is_the_stable_in_progress_ui_status(self) -> None:
        active = {
            ("declared", "preparing"),
            ("declared", "creating_stage"),
            ("staging", "writing_annotations"),
            ("staging", "copying_unlabeled"),
            ("staging", "copying_ocr"),
            ("staging", "applying_provenance"),
            ("staged", "verifying_baseline"),
            ("staged", "writing_backup"),
            ("backup_verified", "switching_directories"),
            ("rollback_created", "switching_directories"),
        }
        for state, phase in active:
            with self.subTest(state=state, phase=phase):
                self.assertEqual(phase, commit_status(
                    job_status="committing",
                    module_status="completed",
                    journal=self._journal(state, schema_version=2, phase=phase),
                ))
        self.assertEqual("committed", commit_status(
            job_status="committing", module_status="completed",
            journal=self._journal("committed", schema_version=2, phase="cleanup"),
        ))

    def test_summary_never_claims_export_before_directory_commit(self) -> None:
        module = {"status": "completed", "completed": 12, "failed": 3, "skipped": 4, "issue_count": 3}
        staging = build_export_summary(job_id="job", format_value="both", job_status="exporting", module_summary=module, journal=None)
        self.assertEqual(("staging", 19, 12, 3, 0, 4, 3, 0, None), (staging.commitStatus, staging.scanned, staging.valid, staging.invalid, staging.exported, staging.skipped, staging.issueCount, staging.unlabeled, staging.unlabeledFolder))
        committed = build_export_summary(job_id="job", format_value="both", job_status="succeeded", module_summary=module, journal=self._journal("committed"))
        self.assertEqual(("committed", 12), (committed.commitStatus, committed.exported))

    def test_partial_success_is_committed_and_unlabeled_count_is_set_not_summed(self) -> None:
        module = {"status": "completed", "completed": 12, "failed": 0, "skipped": 2, "issue_count": 0}
        for status in ("succeeded", "succeeded_with_issues"):
            with self.subTest(status=status):
                summary = build_export_summary(
                    job_id="job", format_value="both", job_status=status,
                    module_summary=module, journal=None,
                    diagnostics=[
                        {"code": "unlabeled_images", "count": 186},
                        {"code": "unlabeled_images", "count": 186},
                    ],
                )
                self.assertEqual("committed", summary.commitStatus)
                self.assertEqual(186, summary.unlabeled)

    def test_committed_summary_reports_task_scoped_unlabeled_folder(self) -> None:
        summary = build_export_summary(
            job_id="12345678-1234-1234-1234-123456789abc",
            format_value="json",
            job_status="succeeded",
            module_summary={"status": "completed", "completed": 8, "failed": 1, "skipped": 2, "issue_count": 1},
            journal=None,
            diagnostics=[{"code": "unlabeled_images", "count": 3}],
        )
        self.assertEqual(3, summary.unlabeled)
        self.assertEqual("_unlabeled\\123456781234123412341234", summary.unlabeledFolder)

    def test_safe_conversions_are_reported_by_type_and_by_sample(self) -> None:
        # F32: ROADMAP.md:988 — aggregated counts only, never a per-sample log.
        module = {"status": "completed", "completed": 12, "failed": 0, "skipped": 0, "issue_count": 0}
        diagnostics = [
            {"code": "samples_converted", "count": 7},
            {"code": "array_duplicate_removed", "count": 9},
            {"code": "array_string_split", "count": 4},
            {"code": "some_unrelated_diagnostic", "count": 3},
        ]
        summary = build_export_summary(
            job_id="job", format_value="both", job_status="exporting",
            module_summary=module, journal=None, diagnostics=diagnostics,
        ).to_dict()
        self.assertEqual(7, summary["convertedSamples"])
        # Frozen vocabulary order, unrelated diagnostics excluded.
        self.assertEqual(["array_string_split", "array_duplicate_removed"], list(summary["conversions"]))
        self.assertEqual({"array_string_split": 4, "array_duplicate_removed": 9}, summary["conversions"])
        empty = build_export_summary(job_id="job", format_value="json", job_status="exporting", module_summary=module, journal=None).to_dict()
        self.assertEqual((0, {}), (empty["convertedSamples"], empty["conversions"]))

