"""Bounded module-6 result summary for the local control plane."""
from __future__ import annotations

from dataclasses import dataclass, field

from .commit_journal import CommitJournal


# Frozen module-6 safe structural conversion vocabulary (ROADMAP.md:988).
CONVERSION_CODES = (
    "missing_field_defaulted",
    "array_empty_to_list",
    "array_string_split",
    "array_duplicate_removed",
    "string_empty_to_empty",
    "single_string_array_unwrapped",
    "character_duplicate_removed",
    "character_normalized",
)
CONVERTED_SAMPLES_CODE = "samples_converted"
UNLABELED_IMAGES_CODE = "unlabeled_images"


def unlabeled_output_folder(job_id: str) -> str:
    token = job_id.replace("-", "")[:24]
    if not token:
        raise ValueError("job id cannot produce an unlabeled output folder")
    return f"_unlabeled\\{token}"


@dataclass(frozen=True)
class ExportSummary:
    format: str
    commitStatus: str
    scanned: int
    valid: int
    invalid: int
    exported: int
    skipped: int
    issueCount: int
    issuesPageEndpoint: str
    convertedSamples: int = 0
    conversions: dict[str, int] = field(default_factory=dict)
    unlabeled: int = 0
    unlabeledFolder: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "commitStatus": self.commitStatus,
            "scanned": self.scanned,
            "valid": self.valid,
            "invalid": self.invalid,
            "exported": self.exported,
            "skipped": self.skipped,
            "issueCount": self.issueCount,
            "issuesPageEndpoint": self.issuesPageEndpoint,
            "convertedSamples": self.convertedSamples,
            "conversions": dict(self.conversions),
            "unlabeled": self.unlabeled,
            "unlabeledFolder": self.unlabeledFolder,
        }


def _conversion_counts(diagnostics: object) -> tuple[int, dict[str, int]]:
    """Project persisted export diagnostics onto the frozen conversion vocabulary."""
    counts: dict[str, int] = {}
    for row in diagnostics or ():
        counts[str(row["code"])] = int(row["count"])
    return counts.get(CONVERTED_SAMPLES_CODE, 0), {code: counts[code] for code in CONVERSION_CODES if code in counts}


def commit_status(*, job_status: object, module_status: object, journal: CommitJournal | None) -> str:
    """Project persisted states onto the frozen, UI-facing export vocabulary."""
    if journal is not None:
        if journal.schema_version == 2:
            if journal.state == "committed":
                return "committed"
            if journal.state in {"rollback_required", "rolled_back"}:
                return str(journal.state)
            if journal.phase is not None:
                return str(journal.phase)
        return {
            "prepared": "staged",
            "backup_verified": "backup_verified",
            "rollback_created": "committing",
            "committed": "committed",
            "rolled_back": "rolled_back",
            "rollback_required": "rollback_required",
        }[journal.state]
    if job_status in {"succeeded", "succeeded_with_issues"}:
        return "committed"
    if job_status == "failed":
        return "failed"
    if module_status == "completed_with_issues":
        return "validation_failed"
    if job_status == "exporting" and module_status == "completed":
        return "staging"
    return "not_started"


def build_export_summary(
    *,
    job_id: str,
    format_value: str,
    job_status: object,
    module_summary: object | None,
    journal: CommitJournal | None,
    diagnostics: object | None = None,
) -> ExportSummary:
    if format_value not in {"json", "flat_txt", "both"}:
        raise ValueError("export summary format is invalid")
    values = module_summary or {}
    completed = int(values["completed"]) if values else 0
    failed = int(values["failed"]) if values else 0
    skipped = int(values["skipped"]) if values else 0
    issues = int(values["issue_count"]) if values else 0
    status = commit_status(
        job_status=job_status,
        module_status=values["status"] if values else None,
        journal=journal,
    )
    diagnostic_rows = list(diagnostics or ())
    converted_samples, conversions = _conversion_counts(diagnostic_rows)
    diagnostic_counts = {str(row["code"]): int(row["count"]) for row in diagnostic_rows}
    unlabeled = diagnostic_counts.get(UNLABELED_IMAGES_CODE, 0)
    return ExportSummary(
        format=format_value,
        commitStatus=status,
        scanned=completed + failed + skipped,
        valid=completed,
        invalid=failed,
        exported=completed if status == "committed" else 0,
        skipped=skipped,
        issueCount=issues,
        issuesPageEndpoint=f"/api/jobs/{job_id}?issueAfterSampleId=0",
        convertedSamples=converted_samples,
        conversions=conversions,
        unlabeled=unlabeled,
        unlabeledFolder=unlabeled_output_folder(job_id) if unlabeled else None,
    )
