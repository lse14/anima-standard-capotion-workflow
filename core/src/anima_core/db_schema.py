"""Schema and migration ownership for the state database."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .contracts import utc_now
from .path_safety import canonicalize, windows_compare, windows_path_is_within


SCHEMA_VERSION = 5
MAX_EVENT_RING = 10_000
DEFAULT_PAGE_SIZE = 200
MAX_PAGE_SIZE = 1_000
MAX_COUNT_PAGE_SIZE = 500


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    config_schema_version INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    work_mode TEXT NOT NULL CHECK(work_mode IN ('in_place', 'full_copy')),
    overwrite_mode TEXT NOT NULL CHECK(overwrite_mode IN ('incremental', 'rebuild')),
    source_root TEXT NOT NULL,
    output_root TEXT,
    dataset_root TEXT NOT NULL,
    dataset_root_key TEXT NOT NULL,
    manifest_schema_version INTEGER NOT NULL,
    recursive INTEGER NOT NULL CHECK(recursive IN (0, 1)),
    sample_count INTEGER NOT NULL DEFAULT 0,
    manifest_generated_at TEXT,
    status TEXT NOT NULL,
    current_module_id TEXT,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0 CHECK(pinned IN (0, 1)),
    api_budget_extra INTEGER NOT NULL DEFAULT 0,
    api_budget_revision INTEGER NOT NULL DEFAULT 0,
    overlay_root TEXT,
    commit_journal_path TEXT,
    runtime_evidence_json TEXT,
    resume_status TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    cancel_requested_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS preflight_progress (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('queued', 'preflighting', 'completed', 'failed', 'cancelled')),
    checked_files INTEGER NOT NULL DEFAULT 0 CHECK(checked_files >= 0),
    total_files INTEGER NOT NULL DEFAULT 0 CHECK(total_files >= 0),
    error TEXT,
    summary_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_claims (
    dataset_root TEXT COLLATE WIN_ORDINAL_NOCASE PRIMARY KEY,
    dataset_root_key TEXT NOT NULL,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id) ON DELETE CASCADE,
    lock_path TEXT NOT NULL,
    acquired_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    relative_image_path TEXT NOT NULL,
    annotation_key TEXT COLLATE WIN_ORDINAL_NOCASE NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('e621', 'danbooru')),
    in_processing_scope INTEGER NOT NULL CHECK(in_processing_scope IN (0, 1)),
    image_format TEXT NOT NULL CHECK(image_format IN ('jpeg', 'png', 'webp', 'bmp')),
    image_frame_count INTEGER NOT NULL,
    original_txt_state TEXT NOT NULL CHECK(original_txt_state IN ('missing_or_blank', 'nonblank')),
    original_json_state TEXT NOT NULL CHECK(original_json_state IN ('missing_or_blank', 'nonblank')),
    image_file_id TEXT,
    image_size INTEGER,
    image_mtime_ns INTEGER,
    original_txt_sha256 TEXT,
    original_json_sha256 TEXT,
    incremental_preserve INTEGER NOT NULL DEFAULT 0 CHECK(incremental_preserve IN (0, 1)),
    incremental_preserve_reason TEXT,
    PRIMARY KEY(job_id, sample_id),
    UNIQUE(job_id, annotation_key)
);

CREATE TABLE IF NOT EXISTS sample_state (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    current_module_id TEXT,
    txt_provenance TEXT NOT NULL CHECK(txt_provenance IN ('missing', 'original_preserved', 'module1_written')),
    status TEXT NOT NULL CHECK(status IN ('pending', 'leased', 'prepared', 'request_started', 'response_staged', 'completed', 'failed', 'skipped')),
    attempt INTEGER NOT NULL DEFAULT 0,
    lease_id TEXT,
    worker_instance_id TEXT,
    lease_expires_at TEXT,
    prepared_artifact_relative_path TEXT,
    prepared_artifact_sha256 TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id, sample_id),
    FOREIGN KEY(job_id, sample_id) REFERENCES samples(job_id, sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS issues (
    issue_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER,
    relative_image_path TEXT,
    module_id TEXT NOT NULL,
    code TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('info', 'warning', 'error')),
    blocking INTEGER NOT NULL CHECK(blocking IN (0, 1)),
    retriable INTEGER NOT NULL CHECK(retriable IN (0, 1)),
    repair_start_module TEXT,
    message TEXT NOT NULL,
    field_errors_json TEXT NOT NULL DEFAULT '[]',
    attempt INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(job_id, sample_id, module_id, code)
);

CREATE TABLE IF NOT EXISTS module_summary (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    module_id TEXT NOT NULL,
    status TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    issue_count INTEGER NOT NULL DEFAULT 0,
    worker_restart_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    PRIMARY KEY(job_id, module_id)
);

CREATE TABLE IF NOT EXISTS module_diagnostics (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    module_id TEXT NOT NULL,
    code TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('info', 'warning', 'error')),
    count INTEGER NOT NULL CHECK(count >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id, module_id, code)
);

CREATE TABLE IF NOT EXISTS event_ring (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL,
    module_id TEXT NOT NULL,
    status TEXT NOT NULL,
    completed INTEGER NOT NULL,
    total INTEGER NOT NULL,
    sample_id INTEGER,
    issue_code TEXT,
    attempt INTEGER NOT NULL,
    config_hash TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    message TEXT,
    PRIMARY KEY(job_id, event_id)
);

CREATE TABLE IF NOT EXISTS staged_nl (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    lease_id TEXT NOT NULL,
    nl TEXT NOT NULL CHECK(length(CAST(nl AS BLOB)) <= 16384),
    sha256 TEXT NOT NULL,
    staged_at TEXT NOT NULL,
    PRIMARY KEY(job_id, sample_id)
);

CREATE TABLE IF NOT EXISTS count_evidence (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    value TEXT NOT NULL CHECK(value IN ('', 'solo', 'duo', 'trio', 'group')),
    decision_json TEXT NOT NULL CHECK(length(CAST(decision_json AS BLOB)) <= 262144),
    review_warning_codes_json TEXT NOT NULL CHECK(length(CAST(review_warning_codes_json AS BLOB)) <= 4096),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id, sample_id),
    FOREIGN KEY(job_id, sample_id) REFERENCES samples(job_id, sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS count_observations (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    status TEXT NOT NULL CHECK(status IN ('observed', 'not_requested', 'invalid')),
    count_value TEXT CHECK(count_value IS NULL OR count_value IN ('solo', 'duo', 'trio', 'group', 'unknown')),
    layout_value TEXT CHECK(layout_value IS NULL OR layout_value IN ('single_scene', 'multi_view', 'character_sheet', 'multi_panel', 'unknown')),
    same_character_repeated INTEGER CHECK(same_character_repeated IS NULL OR same_character_repeated IN (0, 1)),
    warning_codes_json TEXT NOT NULL CHECK(length(CAST(warning_codes_json AS BLOB)) <= 4096),
    not_requested_reason TEXT CHECK(not_requested_reason IS NULL OR length(CAST(not_requested_reason AS BLOB)) BETWEEN 1 AND 128),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id, sample_id),
    FOREIGN KEY(job_id, sample_id) REFERENCES samples(job_id, sample_id) ON DELETE CASCADE,
    CHECK(
        (status = 'observed' AND count_value IS NOT NULL AND layout_value IS NOT NULL
          AND same_character_repeated IS NOT NULL AND not_requested_reason IS NULL)
        OR (status = 'invalid' AND not_requested_reason IS NULL)
        OR (status = 'not_requested' AND count_value IS NULL AND layout_value IS NULL
          AND same_character_repeated IS NULL AND not_requested_reason IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS count_review_decisions (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    status TEXT NOT NULL CHECK(status IN ('pending', 'auto_resolved', 'manual_resolved')),
    final_count TEXT CHECK(final_count IS NULL OR final_count IN ('solo', 'duo', 'trio', 'group')),
    selected_source TEXT CHECK(selected_source IS NULL OR selected_source IN ('consensus', 'classify', 'vlm', 'manual')),
    review_reasons_json TEXT NOT NULL CHECK(length(CAST(review_reasons_json AS BLOB)) <= 4096),
    version INTEGER NOT NULL CHECK(version >= 1),
    resolved_at TEXT,
    applied_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id, sample_id),
    FOREIGN KEY(job_id, sample_id) REFERENCES samples(job_id, sample_id) ON DELETE CASCADE,
    CHECK(
        (status = 'pending' AND final_count IS NULL AND selected_source IS NULL AND resolved_at IS NULL AND applied_at IS NULL)
        OR (status = 'auto_resolved' AND final_count IS NOT NULL
          AND selected_source IN ('consensus', 'classify', 'vlm') AND resolved_at IS NOT NULL)
        OR (status = 'manual_resolved' AND final_count IS NOT NULL
          AND selected_source IN ('classify', 'vlm', 'manual') AND resolved_at IS NOT NULL)
    )
);

-- Export retains only verified overlay artifact identities. Business JSON
-- and TXT bytes remain in the task overlay and never enter the control plane.
CREATE TABLE IF NOT EXISTS export_artifacts (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('json', 'txt')),
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY(job_id, sample_id, kind),
    FOREIGN KEY(job_id, sample_id) REFERENCES samples(job_id, sample_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS nl_outcome_window (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    succeeded INTEGER NOT NULL CHECK(succeeded IN (0, 1))
);

-- Durable NL business archive.  It intentionally has no foreign key to jobs:
-- deleting task control history must not destroy the only verified caption.
CREATE TABLE IF NOT EXISTS nl_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_root_key TEXT NOT NULL,
    dataset_root TEXT NOT NULL,
    sample_identity TEXT NOT NULL,
    relative_image_path TEXT NOT NULL,
    annotation_key TEXT NOT NULL,
    image_file_id TEXT,
    image_sha256 TEXT,
    job_id TEXT,
    sample_id INTEGER,
    source TEXT NOT NULL CHECK(source IN ('api','original_json','nl_txt','manual','dropout','unknown')),
    text TEXT NOT NULL CHECK(length(CAST(text AS BLOB)) <= 16384),
    text_sha256 TEXT NOT NULL,
    source_sha256 TEXT,
    version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nl_history_identity
    ON nl_history(dataset_root_key, sample_identity, created_at DESC, history_id DESC);
CREATE INDEX IF NOT EXISTS idx_nl_history_text
    ON nl_history(dataset_root_key, text_sha256);

CREATE TABLE IF NOT EXISTS nl_format_outcome_window (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    format_failed INTEGER NOT NULL CHECK(format_failed IN (0, 1))
);

-- Repair jobs use a fresh overlay and current sample state while retaining a
-- stable link to the original task's retriable issue history.
CREATE TABLE IF NOT EXISTS repair_jobs (
    repair_job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    parent_job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repair_targets (
    repair_job_id TEXT NOT NULL REFERENCES repair_jobs(repair_job_id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL,
    repair_start_module TEXT NOT NULL,
    PRIMARY KEY(repair_job_id, sample_id)
);

CREATE TABLE IF NOT EXISTS repair_target_issues (
    repair_job_id TEXT NOT NULL,
    sample_id INTEGER NOT NULL,
    parent_issue_id TEXT NOT NULL REFERENCES issues(issue_id) ON DELETE RESTRICT,
    PRIMARY KEY(repair_job_id, sample_id, parent_issue_id),
    FOREIGN KEY(repair_job_id, sample_id)
      REFERENCES repair_targets(repair_job_id, sample_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_samples_page ON samples(job_id, sample_id);
CREATE INDEX IF NOT EXISTS idx_issues_sample_page ON issues(job_id, sample_id, issue_id);
CREATE INDEX IF NOT EXISTS idx_state_lease ON sample_state(job_id, current_module_id, status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_count_observations_status_page ON count_observations(job_id, status, sample_id);
CREATE INDEX IF NOT EXISTS idx_count_review_status_page ON count_review_decisions(job_id, status, sample_id);
CREATE INDEX IF NOT EXISTS idx_count_review_count_page ON count_review_decisions(job_id, final_count, sample_id);
CREATE INDEX IF NOT EXISTS idx_export_artifacts_page ON export_artifacts(job_id, sample_id, kind);
CREATE INDEX IF NOT EXISTS idx_repair_targets_page ON repair_targets(repair_job_id, sample_id);
"""


TERMINAL_JOB_STATUSES = ("succeeded", "succeeded_with_issues", "failed", "discarded", "cancelled_recoverable")
# Startup recovery freezes everything else; cancellation drains on its own and a
# job that is already interrupted must keep its original resume_status.
NON_INTERRUPTIBLE_JOB_STATUSES = (
    "draft", "ready", "interrupted", "cancelling", *TERMINAL_JOB_STATUSES,
)
STARTED_JOB_STATUSES = frozenset({"preparing_workspace", "running", "reviewing", "exporting", "committing"})
FINISHED_JOB_STATUSES = frozenset({"succeeded", "succeeded_with_issues", "failed", "discarded"})


def _ordinal_nocase(left: str, right: str) -> int:
    return windows_compare(str(left), str(right))


def _schema_checksum() -> str:
    return hashlib.sha256(SCHEMA_SQL.encode("utf-8")).hexdigest()


def _expected_columns() -> dict[str, list[sqlite3.Row]]:
    """Read the target column layout from SCHEMA_SQL itself, not a second copy."""
    probe = sqlite3.connect(":memory:")
    try:
        probe.row_factory = sqlite3.Row
        probe.create_collation("WIN_ORDINAL_NOCASE", _ordinal_nocase)
        probe.executescript(SCHEMA_SQL)
        names = [str(row[0]) for row in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {name: list(probe.execute(f"PRAGMA table_info({name})")) for name in names}
    finally:
        probe.close()


def _add_missing_columns(connection: sqlite3.Connection) -> int:
    """ALTER already existing tables so an older database gains new columns."""
    added = 0
    for table, columns in _expected_columns().items():
        present = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if not present:
            continue
        for column in columns:
            name = str(column["name"])
            if name in present:
                continue
            if column["notnull"] and column["dflt_value"] is None:
                raise RuntimeError(f"cannot migrate {table}.{name} without a backfill default")
            not_null = " NOT NULL" if column["notnull"] else ""
            default = "" if column["dflt_value"] is None else f" DEFAULT {column['dflt_value']}"
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column['type']}{not_null}{default}")
            added += 1
    return added


def _missing_columns(connection: sqlite3.Connection) -> list[str]:
    missing: list[str] = []
    for table, columns in _expected_columns().items():
        present = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if not present:
            missing.append(table)
            continue
        missing.extend(f"{table}.{row['name']}" for row in columns if str(row["name"]) not in present)
    return missing


def migrate(connection: sqlite3.Connection) -> int:
    """Bring one control-plane database to SCHEMA_VERSION and return it.

    The version is inspected before anything is written, missing tables and
    missing columns are both repaired, and user_version only advances after the
    result verifies against SCHEMA_SQL.
    """
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"unsupported database schema version: {current}")
    if current in {1, 2, 3}:
        raise RuntimeError(
            f"legacy state database schema {current} is incompatible; reinitialize the state database"
        )
    if current == 0:
        existing_tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if existing_tables:
            raise RuntimeError("unversioned state database is incompatible; reinitialize the state database")
    checksum = _schema_checksum()
    recorded = connection.execute(
        "SELECT checksum FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)
    ).fetchone() if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone() else None
    if current == SCHEMA_VERSION and recorded is not None and str(recorded[0]) == checksum:
        return current
    connection.executescript(SCHEMA_SQL)
    _add_missing_columns(connection)
    missing = _missing_columns(connection)
    if missing:
        raise RuntimeError("schema migration is incomplete: " + ", ".join(sorted(missing)))
    connection.execute(
        """INSERT INTO schema_migrations(version,checksum,applied_at) VALUES (?,?,?)
           ON CONFLICT(version) DO UPDATE SET checksum=excluded.checksum,applied_at=excluded.applied_at""",
        (SCHEMA_VERSION, checksum, utc_now()),
    )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return SCHEMA_VERSION


def default_state_database_path(*, local_app_data: Path | str | None = None) -> Path:
    base = Path(local_app_data) if local_app_data is not None else Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    return base / "AnimaDatasetTool" / "state.db"


def assert_database_outside_datasets(database_path: Path | str, dataset_roots: Iterable[Path | str]) -> None:
    """Reject a control-plane database inside any source/output/overlay tree."""
    target = canonicalize(Path(database_path).expanduser().absolute()).value
    for root_value in dataset_roots:
        root = canonicalize(Path(root_value).expanduser().absolute()).value
        if windows_path_is_within(root, target):
            raise ValueError(f"state database must not be stored in dataset/workspace path: {root}")
