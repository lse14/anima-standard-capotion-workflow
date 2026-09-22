"""Connection-level operations shared by state database domain mixins."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .contracts import SampleIssue, pipeline_module_ids, utc_now
from .db_schema import (
    DEFAULT_PAGE_SIZE,
    FINISHED_JOB_STATUSES,
    MAX_COUNT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    STARTED_JOB_STATUSES,
)
from .state_machine import transition_job


def get_job_row(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(f"job does not exist: {job_id}")
    return row


def validate_page_limit(value: int | None) -> int:
    if value is None:
        return DEFAULT_PAGE_SIZE
    if not 1 <= value <= MAX_PAGE_SIZE:
        raise ValueError(f"page size must be between 1 and {MAX_PAGE_SIZE}")
    return value


def validate_count_page_limit(value: int | None) -> int:
    if value is None:
        return DEFAULT_PAGE_SIZE
    if not 1 <= value <= MAX_COUNT_PAGE_SIZE:
        raise ValueError(f"count page size must be between 1 and {MAX_COUNT_PAGE_SIZE}")
    return value


def repair_target_clause(
    connection: sqlite3.Connection,
    job_id: str,
    module_id: str,
    *,
    sample_alias: str = "s",
) -> tuple[str, list[Any]]:
    """Restrict pipeline modules to their versioned repair start point."""
    row = connection.execute(
        "SELECT config_schema_version FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"job does not exist: {job_id}")
    try:
        module_order = pipeline_module_ids(int(row["config_schema_version"]))
    except ValueError as exc:
        raise ValueError("repair job configuration schema is invalid") from exc
    if module_id not in module_order:
        raise ValueError(f"unknown module for repair selection: {module_id}")
    parent = connection.execute(
        "SELECT parent_job_id FROM repair_jobs WHERE repair_job_id=?", (job_id,)
    ).fetchone()
    if parent is None:
        return "", []
    allowed_starts = module_order[: module_order.index(module_id) + 1]
    placeholders = ",".join("?" for _ in allowed_starts)
    return (
        " AND EXISTS (SELECT 1 FROM repair_targets AS rt "
        f"WHERE rt.repair_job_id={sample_alias}.job_id AND rt.sample_id={sample_alias}.sample_id "
        f"AND rt.repair_start_module IN ({placeholders}))",
        list(allowed_starts),
    )


def increment_module_counts(
    connection: sqlite3.Connection,
    job_id: str,
    module_id: str,
    *,
    completed: int = 0,
    failed: int = 0,
    skipped: int = 0,
    issues: int = 0,
) -> None:
    result = connection.execute(
        """UPDATE module_summary SET completed=completed+?,failed=failed+?,skipped=skipped+?,issue_count=issue_count+?
           WHERE job_id=? AND module_id=?""",
        (completed, failed, skipped, issues, job_id, module_id),
    )
    if result.rowcount != 1:
        raise KeyError(f"module summary does not exist: {job_id}/{module_id}")


def upsert_issue(connection: sqlite3.Connection, issue: SampleIssue) -> None:
    connection.execute(
        """INSERT INTO issues(issue_id,job_id,sample_id,relative_image_path,module_id,code,severity,blocking,retriable,
           repair_start_module,message,field_errors_json,attempt,updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(job_id,sample_id,module_id,code) DO UPDATE SET
             issue_id=excluded.issue_id,relative_image_path=excluded.relative_image_path,severity=excluded.severity,
             blocking=excluded.blocking,retriable=excluded.retriable,repair_start_module=excluded.repair_start_module,
             message=excluded.message,field_errors_json=excluded.field_errors_json,attempt=excluded.attempt,
             resolved_at=NULL,updated_at=excluded.updated_at""",
        (
            issue.issueId, issue.jobId, issue.sampleId, issue.relativeImagePath, issue.moduleId, issue.code,
            issue.severity, int(issue.blocking), int(issue.retriable), issue.repairStartModule, issue.message,
            json.dumps(list(issue.fieldErrors), ensure_ascii=False, separators=(",", ":")), issue.attempt, utc_now(),
        ),
    )


def set_job_status(
    connection: sqlite3.Connection,
    job_id: str,
    status: str,
    *,
    current_module_id: str | None = None,
    resume_status: str | None = None,
) -> None:
    """Persist one validated job transition and its lifecycle timestamps."""
    row = get_job_row(connection, job_id)
    current = str(row["status"])
    if status != current:
        transition_job(current, status)
    connection.execute(
        """UPDATE jobs SET status=?,current_module_id=COALESCE(?,current_module_id),resume_status=?,
           started_at=COALESCE(started_at,?),cancel_requested_at=COALESCE(cancel_requested_at,?),finished_at=?
           WHERE job_id=?""",
        (
            status, current_module_id, resume_status,
            utc_now() if status in STARTED_JOB_STATUSES else None,
            utc_now() if status == "cancelling" else None,
            utc_now() if status in FINISHED_JOB_STATUSES else (
                row["finished_at"] if current == "cancelled_recoverable" and status == "interrupted" else None
            ),
            job_id,
        ),
    )


def insert_count_observation_consistent(
    connection: sqlite3.Connection,
    job_id: str,
    sample_id: int,
    observation: object,
    *,
    now: str,
) -> None:
    from .count_review_protocol import CountObservationV1

    parsed = CountObservationV1.from_dict(
        observation.to_dict() if isinstance(observation, CountObservationV1) else observation
    )
    repeated = None if parsed.sameCharacterRepeated is None else int(parsed.sameCharacterRepeated)
    values = (
        job_id,
        sample_id,
        1,
        parsed.status,
        parsed.countValue,
        parsed.layoutValue,
        repeated,
        parsed.warning_codes_json,
        parsed.notRequestedReason,
        now,
        now,
    )
    connection.execute(
        """INSERT OR IGNORE INTO count_observations(
               job_id,sample_id,schema_version,status,count_value,layout_value,
               same_character_repeated,warning_codes_json,not_requested_reason,created_at,updated_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        values,
    )
    persisted = connection.execute(
        """SELECT schema_version,status,count_value,layout_value,same_character_repeated,
                  warning_codes_json,not_requested_reason
           FROM count_observations WHERE job_id=? AND sample_id=?""",
        (job_id, sample_id),
    ).fetchone()
    expected = (
        1,
        parsed.status,
        parsed.countValue,
        parsed.layoutValue,
        repeated,
        parsed.warning_codes_json,
        parsed.notRequestedReason,
    )
    if persisted is None or tuple(persisted) != expected:
        raise ValueError("count observation retry does not match persisted observation")
