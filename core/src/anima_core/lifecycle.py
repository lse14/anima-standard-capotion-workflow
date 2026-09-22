from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .db import StateDatabase
from .overlay import OverlayLayout
from .state_machine import can_discard, require_discard_confirmation


class JobLifecycleError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscardResult:
    jobId: str
    overlayDeleted: bool


@dataclass(frozen=True)
class HistoryDeletionResult:
    jobId: str
    deletedControlRecord: bool


class JobLifecycle:
    """Explicit user-facing lifecycle operations outside worker business logic."""

    def __init__(self, database: StateDatabase) -> None:
        self.database = database

    def ensure_delete_allowed(self, job_id: str) -> None:
        if self.database.has_repair_children(job_id):
            raise JobLifecycleError("请先删除修复子任务")

    def ensure_history_delete_allowed(self, job_id: str, *, confirmed: bool) -> None:
        if not confirmed:
            raise JobLifecycleError("completed history deletion requires explicit confirmation")
        job = self.database.get_job(job_id)
        if job["status"] not in {"succeeded", "succeeded_with_issues"}:
            raise JobLifecycleError("only a completed task history can be deleted")
        self.ensure_delete_allowed(job_id)
        overlay_value = job["overlay_root"]
        if overlay_value:
            root = Path(str(overlay_value))
            if root.exists():
                try:
                    journal_state = OverlayLayout.open_existing(root, job_id).journal_state()
                except Exception as exc:
                    raise JobLifecycleError("overlay cannot be safely inspected for history deletion") from exc
                if journal_state not in (None, "resolved", "committed", "rolled_back"):
                    raise JobLifecycleError("completed history cannot be deleted before journal resolution")
            elif job["commit_journal_path"]:
                raise JobLifecycleError("completed history has a journal pointer without a recoverable overlay")
        elif job["commit_journal_path"]:
            raise JobLifecycleError("completed history has a journal pointer without a recoverable overlay")

    def delete_history(self, job_id: str, *, confirmed: bool) -> HistoryDeletionResult:
        """Remove only a completed task's control-plane record; business files stay untouched."""
        self.ensure_history_delete_allowed(job_id, confirmed=confirmed)
        with self.database.transaction(immediate=True):
            self.ensure_history_delete_allowed(job_id, confirmed=confirmed)
            if self.database.connection.execute(
                "SELECT 1 FROM dataset_claims WHERE job_id=?", (job_id,)
            ).fetchone() is not None:
                raise JobLifecycleError("completed history cannot be deleted before dataset lock release")
            self.database.delete_job_control_record(job_id)
        return HistoryDeletionResult(job_id, True)

    def discard(self, job_id: str, *, confirmed: bool) -> DiscardResult:
        require_discard_confirmation(confirmed)
        job = self.database.get_job(job_id)
        overlay_deleted = False
        overlay_value = job["overlay_root"]
        journal_state: str | None = None
        layout: OverlayLayout | None = None
        if overlay_value:
            root = Path(str(overlay_value))
            if root.exists():
                try:
                    layout = OverlayLayout.open_existing(root, job_id)
                    journal_state = layout.journal_state()
                except Exception as exc:
                    raise JobLifecycleError("overlay cannot be safely inspected for discard") from exc
        elif job["commit_journal_path"]:
            raise JobLifecycleError("job has a journal pointer without a recoverable overlay")
        with self.database.transaction(immediate=True):
            self.ensure_delete_allowed(job_id)
            current = self.database.get_job(job_id)
            if not can_discard(str(current["status"]), journal_state=journal_state):
                raise JobLifecycleError("job cannot be discarded before recovery and journal resolution")
            if layout is not None:
                layout.discard()
                overlay_deleted = True
            self.database.set_job_status(job_id, "discarded")
            # A discarded job can never run again, so it must stop claiming its dataset.
            self.database.release_dataset_claim(job_id)
        return DiscardResult(job_id, overlay_deleted)
