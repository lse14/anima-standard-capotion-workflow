"""Directory-level restoration of a module-6 ZIP64 original-annotation backup."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .annotation_backup import AnnotationBackupError, restore_to_staging
from .dataset_workspace import DatasetWorkspace
from .commit_journal import CommitJournal, CommitJournalError, write as write_journal
from .directory_commit import DirectoryCommitError, restore_rollback
from .export_staging import create_hardlink_staging, reveal_staging
from .overlay import OverlayLayout


class AnnotationRestoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnnotationRestoreResult:
    restored: int
    backupZip: Path


class AnnotationRestoreCoordinator:
    """Explicit user restoration; it never modifies a formal dataset in place."""

    def __init__(self, layout: OverlayLayout) -> None:
        self.layout = layout

    def restore(self, backup_zip: str | Path) -> AnnotationRestoreResult:
        dataset = self.layout.dataset_root
        backup = Path(backup_zip)
        if not dataset.is_dir() or not backup.is_file():
            raise AnnotationRestoreError("dataset or annotation backup is unavailable")
        token = self.layout.job_id.replace("-", "")[:24]
        if DatasetWorkspace.is_new_overlay(self.layout.root, dataset, self.layout.job_id):
            workspace = DatasetWorkspace.for_dataset(dataset)
            staging = workspace.job_root(self.layout.job_id) / "restore-stage"
            rollback = workspace.job_root(self.layout.job_id) / "restore-rollback"
        else:
            parent = dataset.parent
            staging = parent / f".{dataset.name}.anima-restore-stage-{token}"
            rollback = parent / f".{dataset.name}.anima-restore-rollback-{token}"
        if staging.exists() or rollback.exists():
            raise AnnotationRestoreError("annotation restore workspace already exists")
        try:
            staging = create_hardlink_staging(dataset, staging)
            restored = restore_to_staging(backup, staging)
            journal = CommitJournal(self.layout.job_id, "prepared", dataset, staging, rollback, backup)
            write_journal(self.layout, journal)
            write_journal(self.layout, CommitJournal(self.layout.job_id, "backup_verified", dataset, staging, rollback, backup))
            os.replace(dataset, rollback)
            write_journal(self.layout, CommitJournal(self.layout.job_id, "rollback_created", dataset, staging, rollback, backup))
            reveal_staging(staging)
            os.replace(staging, dataset)
            write_journal(self.layout, CommitJournal(self.layout.job_id, "committed", dataset, staging, rollback, backup))
            if not dataset.is_dir():
                raise AnnotationRestoreError("restored dataset is unavailable")
            shutil.rmtree(rollback)
            return AnnotationRestoreResult(restored, backup)
        except Exception as exc:
            try:
                if not dataset.exists() and rollback.exists():
                    restore_rollback(dataset, rollback)
            except (DirectoryCommitError, OSError):
                pass
            # The staging tree is a full hard-link mirror of the dataset, so leaving
            # it behind wastes a directory entry per sample and makes every retry
            # fail on `annotation restore workspace already exists`. It is only kept
            # when the dataset itself still has to be rebuilt from the rollback.
            if dataset.is_dir() and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise AnnotationRestoreError("annotation restore failed") from exc
