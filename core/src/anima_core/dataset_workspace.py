"""Paths for new task workspaces kept beside a dataset on the same volume."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .path_safety import canonicalize, ensure_within


class DatasetWorkspaceError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetWorkspace:
    dataset_root: Path
    root: Path
    dataset_key: str

    @classmethod
    def for_dataset(cls, dataset_root: str | Path, *, must_exist: bool = True) -> "DatasetWorkspace":
        # Recovery can inspect a journal while the formal dataset is still
        # temporarily renamed to rollback.  Path identity and volume checks
        # remain strict even when the directory itself is absent.
        dataset = canonicalize(dataset_root, must_exist=must_exist, directory=True).value
        # The key is stable for the canonical dataset path and cannot collide
        # merely because two datasets share the same final directory name.
        dataset_key = hashlib.sha256(str(dataset).casefold().encode("utf-8")).hexdigest()[:32]
        root = dataset.parent / ".anima-dataset-tool" / dataset_key
        try:
            parent_device = os.stat(dataset.parent).st_dev
            if root.exists() and os.stat(root).st_dev != parent_device:
                raise DatasetWorkspaceError("dataset workspace is on another volume")
        except OSError as exc:
            raise DatasetWorkspaceError("dataset workspace volume cannot be verified") from exc
        ensure_within(dataset.parent, root)
        return cls(dataset, root, dataset_key)

    def job_root(self, job_id: str, *, create: bool = False) -> Path:
        safe = "".join(character for character in job_id if character.isalnum() or character in "-_")
        if not safe or safe != job_id:
            raise DatasetWorkspaceError("job id contains unsafe characters")
        target = ensure_within(self.root, self.root / "jobs" / safe)
        if create:
            target.mkdir(parents=True, exist_ok=False)
        return target

    def overlay_root(self, job_id: str) -> Path:
        return self.job_root(job_id) / "overlay"

    def stage_root(self, job_id: str) -> Path:
        return self.job_root(job_id) / "stage"

    def rollback_root(self, job_id: str) -> Path:
        return self.job_root(job_id) / "rollback"

    def journal_root(self, job_id: str) -> Path:
        return self.job_root(job_id) / "journal"

    def backup_root(self, job_id: str) -> Path:
        return self.job_root(job_id) / "backup"

    def backup_zip(self, job_id: str) -> Path:
        return self.backup_root(job_id) / f"{job_id}.zip"

    @classmethod
    def is_new_overlay(cls, root: str | Path, dataset_root: str | Path, job_id: str) -> bool:
        try:
            workspace = cls.for_dataset(dataset_root, must_exist=False)
            return workspace.overlay_root(job_id).resolve() == Path(root).resolve()
        except (OSError, ValueError):
            return False
