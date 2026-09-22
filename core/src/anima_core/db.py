from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .contracts import utc_now
from .db_count_review import CountReviewDatabaseMixin
from .db_jobs import JobDatabaseMixin
from .db_nl_export import NlExportDatabaseMixin
from .db_scheduler import SchedulerDatabaseMixin
from .db_schema import (
    DEFAULT_PAGE_SIZE,
    FINISHED_JOB_STATUSES,
    MAX_COUNT_PAGE_SIZE,
    MAX_EVENT_RING,
    MAX_PAGE_SIZE,
    NON_INTERRUPTIBLE_JOB_STATUSES,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    STARTED_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    _add_missing_columns,
    _expected_columns,
    _missing_columns,
    _ordinal_nocase,
    _schema_checksum,
    assert_database_outside_datasets,
    default_state_database_path,
    migrate,
)


class StateDatabase(
    JobDatabaseMixin,
    SchedulerDatabaseMixin,
    CountReviewDatabaseMixin,
    NlExportDatabaseMixin,
):
    """Small control-plane repository; business payloads never enter this class."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self.connection = connection

    @classmethod
    def open(cls, path: Path | str) -> "StateDatabase":
        target = Path(path).expanduser().resolve()
        if target.suffix.lower() not in {".db", ".sqlite3"}:
            raise ValueError("control-plane database must be a SQLite .db file")
        target.parent.mkdir(parents=True, exist_ok=True)
        # A live dataset lock can outlive the request thread that created it;
        # FastAPI may dispatch repair/release calls to another worker thread.
        connection = sqlite3.connect(
            target, timeout=30, isolation_level=None, check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.create_collation("WIN_ORDINAL_NOCASE", _ordinal_nocase)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            migrate(connection)
        except Exception:
            connection.close()
            raise
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA journal_size_limit=67108864")
        connection.execute("PRAGMA wal_autocheckpoint=1000")
        return cls(target, connection)

    @classmethod
    def open_default(cls, *, local_app_data: Path | str | None = None) -> "StateDatabase":
        return cls.open(default_state_database_path(local_app_data=local_app_data))

    def close(self) -> None:
        self.connection.close()

    @contextlib.contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def record_migration(self, checksum: str, *, version: int = SCHEMA_VERSION) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, checksum, applied_at) VALUES (?, ?, ?)",
            (version, checksum, utc_now()),
        )

    def checkpoint(self, *, truncate: bool = False) -> tuple[Any, ...]:
        mode = "TRUNCATE" if truncate else "PASSIVE"
        return tuple(self.connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())

    def count(self, table: str, job_id: str) -> int:
        if table not in {"samples", "sample_state", "issues", "event_ring"}:
            raise ValueError("unsupported count table")
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table} WHERE job_id=?", (job_id,)).fetchone()[0])
