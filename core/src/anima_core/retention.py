from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .db import StateDatabase
from .overlay import OverlayLayout


DEFAULT_SUCCESS_RETENTION = 20


@dataclass(frozen=True)
class RetentionCandidate:
    jobId: str
    overlayRoot: str | None


@dataclass(frozen=True)
class RetentionResult:
    deletedJobIds: tuple[str, ...]
    deletedOverlays: int
    deletedLogs: int


class RetentionManager:
    """Removes only old successful control-plane records and task workspaces.

    It deliberately has no backup-directory parameter. ZIP64 annotation backups
    are outside this API and can only be removed by their dedicated future UI
    operation.
    """

    def __init__(self, database: StateDatabase, *, logs_root: str | Path | None = None, keep_successes: int = DEFAULT_SUCCESS_RETENTION) -> None:
        if keep_successes < 0:
            raise ValueError("keep_successes cannot be negative")
        self.database = database
        self.logs_root = Path(logs_root) if logs_root is not None else None
        self.keep_successes = keep_successes

    def candidates(self) -> list[RetentionCandidate]:
        rows = list(self.database.connection.execute(
            """SELECT job_id,overlay_root,commit_journal_path FROM jobs
               WHERE status IN ('succeeded','succeeded_with_issues') AND pinned=0
                 AND NOT EXISTS(
                     SELECT 1 FROM repair_jobs AS child WHERE child.parent_job_id=jobs.job_id
                 )
               ORDER BY COALESCE(finished_at,created_at) DESC, job_id DESC"""
        ))
        # A succeeded job with a journal pointer remains protected until its
        # overlay says the journal reached a final, resolved state.
        result: list[RetentionCandidate] = []
        for row in rows[self.keep_successes:]:
            overlay_root = row["overlay_root"]
            if row["commit_journal_path"] and not overlay_root:
                continue
            if overlay_root and not self._journal_resolved(str(row["job_id"]), Path(str(overlay_root))):
                continue
            result.append(RetentionCandidate(str(row["job_id"]), str(overlay_root) if overlay_root else None))
        return result

    def _journal_resolved(self, job_id: str, root: Path) -> bool:
        if not root.exists():
            # No overlay cannot be used as evidence that an old journal is
            # resolved. Keep the control record for explicit recovery review.
            return False
        try:
            layout = OverlayLayout.open_existing(root, job_id)
            return not layout.has_unresolved_journal()
        except Exception:
            return False

    def cleanup(self) -> RetentionResult:
        deleted: list[str] = []
        overlays = 0
        logs = 0
        for candidate in self.candidates():
            # The control record is the only piece that can fail on a constraint,
            # so it is removed first: a failure here must not leave the overlay
            # deleted while the record survives and blocks every later startup.
            try:
                self.database.delete_job_control_record(candidate.jobId)
            except (KeyError, ValueError):
                continue
            except Exception:
                # A foreign-key or locking failure must never take the whole
                # backend down at startup; the record is simply retried later.
                continue
            deleted.append(candidate.jobId)
            if candidate.overlayRoot:
                root = Path(candidate.overlayRoot)
                if root.exists():
                    try:
                        OverlayLayout.open_existing(root, candidate.jobId).discard()
                        overlays += 1
                    except Exception:
                        # The record is already gone, so a leftover directory is
                        # orphaned scratch data and must not fail retention.
                        pass
            log = self._log_path(candidate.jobId)
            if log is not None and log.exists():
                try:
                    log.unlink()
                    logs += 1
                except OSError:
                    pass
        return RetentionResult(tuple(deleted), overlays, logs)

    def _log_path(self, job_id: str) -> Path | None:
        if self.logs_root is None:
            return None
        root = self.logs_root.resolve()
        candidate = (root / f"{job_id}.log").resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("retention log path escapes logs root") from exc
        return candidate
