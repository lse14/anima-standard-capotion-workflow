from __future__ import annotations

from dataclasses import dataclass


class InvalidTransition(ValueError):
    """Raised when a persisted state transition is not allowed."""


JOB_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"preflighting", "discarded"}),
    # A repair job is derived, not user-confirmed, so it prepares its workspace
    # straight out of preflight.
    "preflighting": frozenset({"draft", "ready", "preparing_workspace", "failed", "interrupted"}),
    # start_module accepts a ready job and makes it the running module owner;
    # confirm_workspace can also reject a stale configuration while still ready.
    "ready": frozenset({"preparing_workspace", "running", "cancelling", "failed", "discarded"}),
    "preparing_workspace": frozenset({"running", "cancelling", "failed", "interrupted"}),
    "running": frozenset({"paused", "reviewing", "exporting", "cancelling", "failed", "interrupted"}),
    "paused": frozenset({"running", "exporting", "cancelling", "interrupted", "failed"}),
    # Startup journal recovery can find an interrupted commit already committed.
    "interrupted": frozenset({"preparing_workspace", "running", "reviewing", "exporting", "cancelling", "succeeded", "succeeded_with_issues", "failed", "discarded"}),
    "reviewing": frozenset({"running", "exporting", "cancelling", "failed", "discarded", "interrupted"}),
    "exporting": frozenset({"paused", "committing", "reviewing", "cancelling", "failed", "interrupted"}),
    "committing": frozenset({"succeeded", "succeeded_with_issues", "cancelled_recoverable", "failed", "interrupted"}),
    "cancelling": frozenset({"cancelled_recoverable", "succeeded", "failed", "interrupted"}),
    "cancelled_recoverable": frozenset({"preparing_workspace", "running", "reviewing", "exporting", "interrupted", "discarded"}),
    "succeeded": frozenset(),
    "succeeded_with_issues": frozenset(),
    "failed": frozenset({"preparing_workspace", "running", "reviewing", "discarded"}),
    "discarded": frozenset(),
}

MODULE_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running", "paused", "skipped", "skipped_not_available", "failed"}),
    "running": frozenset({"paused", "completed", "completed_with_issues", "failed", "skipped"}),
    "paused": frozenset({"running", "failed", "skipped"}),
    "completed": frozenset(),
    "completed_with_issues": frozenset({"running"}),
    "failed": frozenset({"running"}),
    "skipped": frozenset(),
    "skipped_not_available": frozenset(),
}

SAMPLE_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"leased", "skipped"}),
    "leased": frozenset({"prepared", "request_started", "failed", "pending", "skipped"}),
    "prepared": frozenset({"completed", "failed", "pending"}),
    "request_started": frozenset({"response_staged", "failed"}),
    "response_staged": frozenset({"completed", "failed", "pending"}),
    "completed": frozenset(),
    "failed": frozenset({"pending"}),
    "skipped": frozenset(),
}


def transition(current: str, target: str, table: dict[str, frozenset[str]]) -> str:
    if target not in table.get(current, frozenset()):
        raise InvalidTransition(f"invalid transition {current!r} -> {target!r}")
    return target


def transition_job(current: str, target: str) -> str:
    return transition(current, target, JOB_TRANSITIONS)


def transition_module(current: str, target: str, *, module_id: str | None = None) -> str:
    """Validate a module transition, retaining the legacy unavailable status for dropout jobs."""
    if module_id != "dropout" and target == "skipped_not_available":
        raise InvalidTransition("only dropout may be skipped_not_available")
    return transition(current, target, MODULE_TRANSITIONS)


def transition_sample(current: str, target: str) -> str:
    return transition(current, target, SAMPLE_TRANSITIONS)


def can_discard(job_status: str, *, journal_state: str | None) -> bool:
    if journal_state not in (None, "resolved", "committed", "rolled_back"):
        return False
    return job_status in {"draft", "ready", "interrupted", "reviewing", "cancelled_recoverable", "failed"}


@dataclass(frozen=True)
class RecoveryDecision:
    nextStatus: str
    requiresUserConfirmation: bool
    reason: str


def decide_startup_recovery(status: str, journal_state: str | None) -> RecoveryDecision:
    if journal_state in {"prepared", "declared", "staging", "staged", "backup_verified"}:
        return RecoveryDecision("interrupted", True, "pre-swap Export requires confirmed recommit")
    if journal_state in {"rollback_created", "rollback_required"}:
        return RecoveryDecision("committing", False, "finish the journaled directory switch")
    if journal_state not in (None, "resolved", "committed", "rolled_back"):
        return RecoveryDecision("committing", False, "commit journal must be resolved before any new work")
    # Kept aligned with db.NON_INTERRUPTIBLE_JOB_STATUSES: whatever startup
    # recovery refuses to freeze must also be reported as needing no recovery.
    if status in {"succeeded", "succeeded_with_issues", "discarded", "draft", "ready", "failed", "cancelled_recoverable"}:
        return RecoveryDecision(status, False, "terminal or not-started state")
    if status == "cancelling":
        return RecoveryDecision("cancelling", False, "finish cancellation drain")
    if status == "interrupted":
        return RecoveryDecision("interrupted", True, "manual recovery is required")
    return RecoveryDecision("interrupted", True, "backend restart interrupted an active task")


def request_cancellation(status: str) -> str:
    """Persist the cancellation barrier before stopping workers or API retries."""
    if status in {"running", "paused", "preparing_workspace", "exporting", "reviewing"}:
        return "cancelling"
    if status == "cancelling":
        return status
    raise InvalidTransition(f"cannot cancel job in state {status!r}")


def finish_cancellation(status: str, *, journal_state: str | None, commit_succeeded: bool = False) -> str:
    if status != "cancelling":
        raise InvalidTransition(f"cancellation drain requires cancelling, got {status!r}")
    if journal_state not in (None, "resolved", "committed", "rolled_back"):
        return "committing"
    return "succeeded" if commit_succeeded else "cancelled_recoverable"


def require_discard_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise InvalidTransition("discard requires explicit second confirmation")
