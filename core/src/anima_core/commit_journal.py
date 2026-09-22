"""Strict module-6 commit journal schema and crash-recovery classification."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .overlay import OverlayLayout, OverlayError
from .dataset_workspace import DatasetWorkspace
from .path_safety import canonicalize, windows_paths_equal

JournalState = Literal[
    "prepared", "declared", "staging", "staged", "backup_verified",
    "rollback_created", "committed", "rollback_required", "rolled_back",
]
CommitPhase = Literal[
    "preparing", "creating_stage", "writing_annotations", "copying_unlabeled",
    "copying_ocr", "applying_provenance", "verifying_baseline",
    "writing_backup", "switching_directories", "cleanup",
]
LEGACY_STATES = frozenset({
    "prepared", "backup_verified", "rollback_created", "committed",
    "rollback_required", "rolled_back",
})
STATES = frozenset({
    "declared", "staging", "staged", "backup_verified", "rollback_created",
    "committed", "rollback_required", "rolled_back",
})
PHASES = frozenset({
    "preparing", "creating_stage", "writing_annotations", "copying_unlabeled",
    "copying_ocr", "applying_provenance", "verifying_baseline",
    "writing_backup", "switching_directories", "cleanup",
})
STATE_PHASES = {
    "declared": frozenset({"preparing", "creating_stage"}),
    "staging": frozenset({
        "writing_annotations", "copying_unlabeled", "copying_ocr",
        "applying_provenance",
    }),
    "staged": frozenset({"verifying_baseline", "writing_backup"}),
    "backup_verified": frozenset({"switching_directories"}),
    "rollback_created": frozenset({"switching_directories"}),
    "committed": frozenset({"cleanup"}),
    "rollback_required": frozenset({"switching_directories"}),
    "rolled_back": frozenset({"cleanup"}),
}

class CommitJournalError(RuntimeError): pass

@dataclass(frozen=True)
class CommitJournal:
    job_id: str
    state: JournalState
    dataset_root: Path
    staging_root: Path
    rollback_root: Path
    backup_zip: Path
    schema_version: int = 1
    phase: CommitPhase | None = None
    updated_at: str | None = None

    @classmethod
    def v2(
        cls,
        job_id: str,
        state: JournalState,
        dataset_root: Path,
        staging_root: Path,
        rollback_root: Path,
        backup_zip: Path,
        *,
        phase: CommitPhase,
    ) -> "CommitJournal":
        updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        journal = cls(
            job_id, state, dataset_root, staging_root, rollback_root, backup_zip,
            schema_version=2, phase=phase, updated_at=updated_at,
        )
        journal._validate_version_fields()
        return journal

    @classmethod
    def from_value(cls, value: object, *, job_id: str) -> "CommitJournal":
        if not isinstance(value, dict):
            raise CommitJournalError("commit journal fields are invalid")
        schema_version = value.get("schemaVersion")
        shared_fields = {
            "schemaVersion", "jobId", "state", "datasetRoot", "stagingRoot",
            "rollbackRoot", "backupZip",
        }
        if type(schema_version) is not int or schema_version not in {1, 2}:
            raise CommitJournalError("commit journal identity is invalid")
        expected_fields = shared_fields if schema_version == 1 else shared_fields | {"phase", "updatedAt"}
        if set(value) != expected_fields:
            raise CommitJournalError("commit journal fields are invalid")
        allowed_states = LEGACY_STATES if schema_version == 1 else STATES
        if value["jobId"] != job_id or value["state"] not in allowed_states:
            raise CommitJournalError("commit journal identity is invalid")
        try:
            dataset=canonicalize(value["datasetRoot"],must_exist=False,directory=True).value; staging=canonicalize(value["stagingRoot"],must_exist=False,directory=True).value; rollback=canonicalize(value["rollbackRoot"],must_exist=False,directory=True).value; backup=canonicalize(value["backupZip"],must_exist=False,directory=False).value
        except Exception as exc: raise CommitJournalError("commit journal paths are invalid") from exc
        expected_backup_directory = dataset.parent / f".{dataset.name}.anima-backups"
        workspace_ok = False
        try:
            workspace = DatasetWorkspace.for_dataset(dataset, must_exist=False)
            workspace_ok = (
                windows_paths_equal(staging, workspace.stage_root(job_id))
                and windows_paths_equal(rollback, workspace.rollback_root(job_id))
                and windows_paths_equal(backup, workspace.backup_zip(job_id))
            )
            restore_ok = (
                staging.parent == workspace.job_root(job_id)
                and rollback.parent == workspace.job_root(job_id)
                and staging.name == "restore-stage"
                and rollback.name == "restore-rollback"
                and (
                    backup.parent == workspace.backup_root(job_id)
                    or backup.parent == expected_backup_directory
                )
            )
        except Exception:
            workspace_ok = False
            restore_ok = False
        legacy_ok = (
            staging.parent == dataset.parent and rollback.parent == dataset.parent
            and backup.parent == expected_backup_directory
        )
        if (
            (not workspace_ok and not restore_ok and not legacy_ok)
            or windows_paths_equal(dataset, staging) or windows_paths_equal(dataset, rollback)
        ):
            raise CommitJournalError("commit journal paths are not sibling-safe")
        journal = cls(
            job_id, value["state"], dataset, staging, rollback, backup,
            schema_version=schema_version,
            phase=value.get("phase"),
            updated_at=value.get("updatedAt"),
        )
        journal._validate_version_fields()
        return journal

    def _validate_version_fields(self) -> None:
        if self.schema_version == 1:
            if self.state not in LEGACY_STATES or self.phase is not None or self.updated_at is not None:
                raise CommitJournalError("commit journal v1 fields are invalid")
            return
        if self.schema_version != 2 or self.state not in STATES:
            raise CommitJournalError("commit journal identity is invalid")
        if self.phase not in PHASES or self.phase not in STATE_PHASES[self.state]:
            raise CommitJournalError("commit journal phase is invalid")
        if not isinstance(self.updated_at, str) or not self.updated_at.endswith("Z"):
            raise CommitJournalError("commit journal timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(self.updated_at[:-1] + "+00:00")
        except ValueError as exc:
            raise CommitJournalError("commit journal timestamp is invalid") from exc
        canonical = parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        if canonical != self.updated_at:
            raise CommitJournalError("commit journal timestamp is invalid")

    def to_value(self) -> dict[str,object]:
        self._validate_version_fields()
        value: dict[str, object] = {
            "schemaVersion": self.schema_version,
            "jobId": self.job_id,
            "state": self.state,
            "datasetRoot": str(self.dataset_root),
            "stagingRoot": str(self.staging_root),
            "rollbackRoot": str(self.rollback_root),
            "backupZip": str(self.backup_zip),
        }
        if self.schema_version == 2:
            value["phase"] = self.phase
            value["updatedAt"] = self.updated_at
        return value

def load(layout: OverlayLayout) -> CommitJournal | None:
    path=layout.commit_journal_path()
    if not path.exists(): return None
    try:
        import json; value=json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc: raise CommitJournalError("commit journal is unreadable") from exc
    return CommitJournal.from_value(value,job_id=layout.job_id)

def write(layout: OverlayLayout,journal: CommitJournal) -> Path:
    if journal.job_id!=layout.job_id or not windows_paths_equal(journal.dataset_root,layout.dataset_root): raise CommitJournalError("commit journal does not match overlay")
    try: return layout.write_journal(journal.to_value())
    except OverlayError as exc: raise CommitJournalError("commit journal cannot be persisted") from exc

def recovery_action(journal: CommitJournal) -> Literal["finish_commit","restore_rollback","verify_committed"]:
    if journal.state in {"prepared","backup_verified","rollback_required"}: return "restore_rollback"
    if journal.state=="rollback_created": return "finish_commit"
    return "verify_committed"
