"""Scheduling ownership for the state database."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any, Callable

from .contracts import (
    COUNT_REVIEW_SCHEMA_VERSIONS,
    SampleIssue,
    SampleRunState,
    utc_now,
)
from .db_primitives import (
    increment_module_counts,
    repair_target_clause,
    upsert_issue,
    validate_page_limit,
)
from .db_schema import MAX_PAGE_SIZE, NON_INTERRUPTIBLE_JOB_STATUSES
from .state_machine import transition_job, transition_module


# Public list endpoints remain capped by MAX_PAGE_SIZE (1000). OCR's v10
# worker protocol permits a distinct internal lease batch of 1024 items.
MAX_LEASE_BOUND = 1024


def _complete_leased_sample_with_issue(
    database: Any,
    job_id: str,
    module_id: str,
    sample_id: int,
    *,
    lease_id: str,
    issue: SampleIssue,
    allowed_statuses: tuple[str, ...] = ("leased", "prepared"),
) -> None:
    """Atomically settle a nonblocking completed result and its sample issue."""
    if (
        issue.jobId != job_id
        or issue.sampleId != sample_id
        or issue.moduleId != module_id
    ):
        raise ValueError("issue identity does not match the completed lease")
    if not allowed_statuses:
        raise ValueError("issue completion allowed statuses cannot be empty")
    placeholders = ",".join("?" for _ in allowed_statuses)
    with database.transaction(immediate=True):
        previous = database.connection.execute(
            """SELECT resolved_at FROM issues
               WHERE job_id=? AND sample_id=? AND module_id=? AND code=?""",
            (job_id, sample_id, module_id, issue.code),
        ).fetchone()
        result = database.connection.execute(
            """UPDATE sample_state SET status='completed',lease_id=NULL,worker_instance_id=NULL,
               lease_expires_at=NULL,prepared_artifact_relative_path=NULL,
               prepared_artifact_sha256=NULL,updated_at=?
               WHERE job_id=? AND sample_id=? AND current_module_id=?
                 AND status IN (""" + placeholders + ") AND lease_id=?",
            (utc_now(), job_id, sample_id, module_id, *allowed_statuses, lease_id),
        )
        if result.rowcount != 1:
            raise ValueError("issue completion does not belong to an active lease")
        upsert_issue(database.connection, issue)
        issue_delta = int(previous is None or previous["resolved_at"] is not None)
        increment_module_counts(database.connection, job_id, module_id, completed=1, issues=issue_delta)


class SchedulerDatabaseMixin:
    """Lease, heartbeat, sample state, prepared artifact, and recovery operations."""

    def get_sample_state(self, job_id: str, sample_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM sample_state WHERE job_id=? AND sample_id=?", (job_id, sample_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"sample state does not exist: {job_id}/{sample_id}")
        return row

    def get_leased_sample(
        self,
        job_id: str,
        module_id: str,
        sample_id: int,
        *,
        lease_id: str,
        worker_instance_id: str,
    ) -> sqlite3.Row:
        """Return one manifest row only when it still belongs to the exact lease."""
        row = self.connection.execute(
            """SELECT s.*,st.current_module_id,st.txt_provenance,st.status,st.attempt,st.lease_id,
                      st.worker_instance_id,st.lease_expires_at
               FROM samples AS s JOIN sample_state AS st
                 ON st.job_id=s.job_id AND st.sample_id=s.sample_id
               WHERE s.job_id=? AND s.sample_id=? AND st.current_module_id=?
                 AND st.status='leased' AND st.lease_id=? AND st.worker_instance_id=?""",
            (job_id, sample_id, module_id, lease_id, worker_instance_id),
        ).fetchone()
        if row is None:
            raise ValueError("sample does not belong to the active worker lease")
        return row

    def get_sample_with_state(self, job_id: str, sample_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            """SELECT s.*,st.current_module_id,st.txt_provenance,st.status,st.attempt,
                      st.lease_id,st.worker_instance_id,st.lease_expires_at,
                      st.prepared_artifact_relative_path,st.prepared_artifact_sha256
               FROM samples AS s JOIN sample_state AS st
                 ON st.job_id=s.job_id AND st.sample_id=s.sample_id
               WHERE s.job_id=? AND s.sample_id=?""",
            (job_id, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"sample does not exist: {job_id}/{sample_id}")
        return row

    def claim_leases(
        self,
        job_id: str,
        module_id: str,
        worker_instance_id: str,
        config_hash: str,
        *,
        limit: int,
        max_in_flight: int,
        single_worker: bool = False,
        lease_bound: int = MAX_PAGE_SIZE,
        lease_id_factory: Callable[[], str],
        expires_at: str,
    ) -> list[sqlite3.Row]:
        if not 1 <= lease_bound <= MAX_LEASE_BOUND:
            raise ValueError("lease bound out of bounds")
        if not 1 <= limit <= lease_bound:
            raise ValueError("lease limit out of bounds")
        if not 1 <= max_in_flight <= lease_bound:
            raise ValueError("in-flight lease limit out of bounds")
        if not worker_instance_id or len(worker_instance_id) > 128:
            raise ValueError("worker instance id must be a non-empty bounded string")
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id,config_hash FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            if job["config_hash"] != config_hash:
                raise ValueError("config hash does not match immutable job configuration")
            expected_job_status = "exporting" if module_id == "export" else "running"
            if job["status"] != expected_job_status or job["current_module_id"] != module_id:
                return []
            active_workers = {
                str(row[0])
                for row in self.connection.execute(
                    """SELECT DISTINCT worker_instance_id FROM sample_state
                       WHERE job_id=? AND current_module_id=?
                         AND status IN ('leased','prepared','request_started','response_staged')
                         AND worker_instance_id IS NOT NULL""",
                    (job_id, module_id),
                )
            }
            if single_worker and active_workers and active_workers != {worker_instance_id}:
                raise ValueError(f"{module_id} allows only one active worker")
            in_flight = int(self.connection.execute(
                """SELECT COUNT(*) FROM sample_state WHERE job_id=? AND current_module_id=?
                   AND status IN ('leased','prepared','request_started','response_staged')""",
                (job_id, module_id),
            ).fetchone()[0])
            capacity = max_in_flight - in_flight
            if capacity <= 0:
                return []
            limit = min(limit, capacity)
            repair_clause, repair_parameters = repair_target_clause(self.connection, job_id, module_id)
            rows = list(self.connection.execute(
                """SELECT s.* FROM samples AS s JOIN sample_state AS st
                   ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                   WHERE s.job_id=? AND s.in_processing_scope=1
                      AND (st.current_module_id=? OR (?='caption' AND st.current_module_id IS NULL))
                      AND st.status='pending'""" + repair_clause + """
                   ORDER BY s.sample_id LIMIT ?""",
                (job_id, module_id, module_id, *repair_parameters, limit),
            ))
            if not rows:
                return []
            lease_ids: list[tuple[int, str]] = []
            for row in rows:
                lease_id = lease_id_factory()
                if not lease_id or lease_id != str(lease_id) or len(lease_id) > 128:
                    raise ValueError("lease id must be a non-empty bounded string")
                self.connection.execute(
                    """UPDATE sample_state SET current_module_id=?,status='leased',attempt=attempt+1,
                       lease_id=?,worker_instance_id=?,lease_expires_at=?,updated_at=?
                       WHERE job_id=? AND sample_id=? AND status='pending'""",
                     (module_id, lease_id, worker_instance_id, expires_at, utc_now(), job_id, row["sample_id"]),
                )
                lease_ids.append((int(row["sample_id"]), lease_id))
            placeholders = ",".join("?" for _ in lease_ids)
            updated = list(self.connection.execute(
                f"""SELECT s.*, st.current_module_id, st.status, st.attempt, st.lease_id,
                           st.worker_instance_id, st.lease_expires_at
                    FROM samples AS s JOIN sample_state AS st
                      ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                    WHERE s.job_id=? AND st.sample_id IN ({placeholders})
                    ORDER BY s.sample_id""",
                (job_id, *(sample_id for sample_id, _ in lease_ids)),
            ))
            return updated

    def reset_next_module_page(
        self,
        job_id: str,
        module_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = 1_000,
    ) -> list[int]:
        """Move one bounded page from the completed prior module into a new module."""
        limit = validate_page_limit(limit)
        with self.transaction(immediate=True):
            repair_clause, repair_parameters = repair_target_clause(self.connection, job_id, module_id)
            query = """SELECT s.sample_id FROM samples AS s JOIN sample_state AS st
                       ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                       WHERE s.job_id=? AND s.in_processing_scope=1
                         AND ((st.current_module_id IS NULL AND st.status='pending')
                           OR (st.current_module_id IS NOT NULL AND st.current_module_id<>?
                               AND st.status IN ('completed','skipped')))""" + repair_clause
            params: list[Any] = [job_id, module_id, *repair_parameters]
            if after_sample_id is not None:
                query += " AND s.sample_id>?"
                params.append(after_sample_id)
            query += " ORDER BY s.sample_id LIMIT ?"
            params.append(limit)
            sample_ids = [int(row[0]) for row in self.connection.execute(query, params)]
            if not sample_ids:
                return []
            placeholders = ",".join("?" for _ in sample_ids)
            self.connection.execute(
                f"""UPDATE sample_state SET current_module_id=?,status='pending',lease_id=NULL,worker_instance_id=NULL,
                    lease_expires_at=NULL,prepared_artifact_relative_path=NULL,prepared_artifact_sha256=NULL,updated_at=?
                    WHERE job_id=? AND sample_id IN ({placeholders})""",
                (module_id, utc_now(), job_id, *sample_ids),
            )
            return sample_ids

    def skip_failed_samples_for_export_page(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = 1_000,
    ) -> list[int]:
        """Carry samples that failed before Export into Export as explicit skips.

        `reset_next_module_page` only advances completed/skipped samples, so a
        failed one would otherwise stall in its old module: invisible to the
        export queue, missing from the artifact index and fatal at commit time.
        """
        limit = validate_page_limit(limit)
        with self.transaction(immediate=True):
            query = """SELECT s.sample_id FROM samples AS s JOIN sample_state AS st
                       ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                       WHERE s.job_id=? AND s.in_processing_scope=1 AND st.status='failed'
                         AND st.current_module_id IS NOT NULL AND st.current_module_id<>'export'"""
            params: list[Any] = [job_id]
            if after_sample_id is not None:
                query += " AND s.sample_id>?"
                params.append(after_sample_id)
            query += " ORDER BY s.sample_id LIMIT ?"
            params.append(limit)
            sample_ids = [int(row[0]) for row in self.connection.execute(query, params)]
            if not sample_ids:
                return []
            placeholders = ",".join("?" for _ in sample_ids)
            self.connection.execute(
                f"""UPDATE sample_state SET current_module_id='export',status='skipped',lease_id=NULL,
                    worker_instance_id=NULL,lease_expires_at=NULL,prepared_artifact_relative_path=NULL,
                    prepared_artifact_sha256=NULL,updated_at=?
                    WHERE job_id=? AND sample_id IN ({placeholders})""",
                (utc_now(), job_id, *sample_ids),
            )
            return sample_ids

    def skip_disabled_caption_page(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = 1_000,
    ) -> list[int]:
        """Apply the disabled-caption provenance matrix to one keyset page."""
        limit = validate_page_limit(limit)
        with self.transaction(immediate=True):
            repair_clause, repair_parameters = repair_target_clause(self.connection, job_id, "caption")
            query = """SELECT s.sample_id FROM samples AS s JOIN sample_state AS st
                       ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                       WHERE s.job_id=? AND s.in_processing_scope=1
                         AND st.current_module_id IS NULL AND st.status='pending'""" + repair_clause
            params: list[Any] = [job_id, *repair_parameters]
            if after_sample_id is not None:
                query += " AND s.sample_id>?"
                params.append(after_sample_id)
            query += " ORDER BY s.sample_id LIMIT ?"
            params.append(limit)
            sample_ids = [int(row[0]) for row in self.connection.execute(query, params)]
            if not sample_ids:
                return []
            placeholders = ",".join("?" for _ in sample_ids)
            self.connection.execute(
                f"""UPDATE sample_state
                    SET current_module_id='caption',status='skipped',
                        txt_provenance=CASE
                          WHEN sample_id IN (
                            SELECT sample_id FROM samples
                            WHERE job_id=? AND original_txt_state='nonblank'
                          ) THEN 'original_preserved' ELSE 'missing' END,
                        lease_id=NULL,worker_instance_id=NULL,lease_expires_at=NULL,updated_at=?
                    WHERE job_id=? AND sample_id IN ({placeholders})""",
                (job_id, utc_now(), job_id, *sample_ids),
            )
            return sample_ids

    def skip_incremental_preserved_page(
        self,
        job_id: str,
        module_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = 1_000,
    ) -> list[int]:
        """Settle samples frozen as existing-annotation preserves for a module.

        Export is included: it commits the untouched baseline bytes without
        claiming a worker lease. All business-mutating modules are skipped so
        a preserved annotation cannot consume API/token budget or be rewritten.
        """
        if module_id not in {"caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"}:
            raise ValueError("invalid preservation module")
        limit = validate_page_limit(limit)
        with self.transaction(immediate=True):
            query = """SELECT s.sample_id FROM samples AS s JOIN sample_state AS st
                       ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                       WHERE s.job_id=? AND s.in_processing_scope=1
                         AND s.incremental_preserve=1 AND st.status='pending'
                         AND (st.current_module_id=? OR (?='caption' AND st.current_module_id IS NULL))"""
            params: list[Any] = [job_id, module_id, module_id]
            if after_sample_id is not None:
                query += " AND s.sample_id>?"
                params.append(after_sample_id)
            query += " ORDER BY s.sample_id LIMIT ?"
            params.append(limit)
            sample_ids = [int(row[0]) for row in self.connection.execute(query, params)]
            if not sample_ids:
                return []
            placeholders = ",".join("?" for _ in sample_ids)
            self.connection.execute(
                f"""UPDATE sample_state SET current_module_id=?,status='skipped',
                    txt_provenance=CASE WHEN txt_provenance='missing' AND EXISTS(
                        SELECT 1 FROM samples WHERE job_id=? AND sample_id IN ({placeholders})
                        AND original_txt_state='nonblank') THEN 'original_preserved' ELSE txt_provenance END,
                    lease_id=NULL,worker_instance_id=NULL,lease_expires_at=NULL,updated_at=?
                    WHERE job_id=? AND sample_id IN ({placeholders}) AND status='pending'""",
                (module_id, job_id, *sample_ids, utc_now(), job_id, *sample_ids),
            )
            increment_module_counts(self.connection, job_id, module_id, skipped=len(sample_ids))
            return sample_ids

    def heartbeat(
        self,
        job_id: str,
        worker_instance_id: str,
        expires_at: str,
        *,
        lease_ids: Iterable[str] | None = None,
    ) -> int:
        params: list[Any] = [expires_at, utc_now(), job_id, worker_instance_id]
        predicate = ""
        if lease_ids is not None:
            bounded = list(lease_ids)
            # OCR may lease up to 1024 samples per batch, which exceeds the generic
            # page bound, so the heartbeat tolerates that module's own ceiling.
            if len(bounded) > MAX_LEASE_BOUND:
                raise ValueError("heartbeat lease list is unbounded")
            if not bounded:
                return 0
            predicate = " AND lease_id IN (" + ",".join("?" for _ in bounded) + ")"
            params.extend(bounded)
        result = self.connection.execute(
            "UPDATE sample_state SET lease_expires_at=?,updated_at=? WHERE job_id=? AND worker_instance_id=? AND status IN ('leased','prepared','request_started','response_staged')" + predicate,
            params,
        )
        return result.rowcount

    def return_expired_leases(self, job_id: str, now: str) -> int:
        result = self.connection.execute(
            """UPDATE sample_state SET status='pending',lease_id=NULL,worker_instance_id=NULL,
               lease_expires_at=NULL,updated_at=?
               WHERE job_id=? AND status='leased' AND lease_expires_at IS NOT NULL AND lease_expires_at<?""",
            (utc_now(), job_id, now),
        )
        return result.rowcount

    def return_module_leases(self, job_id: str, module_id: str) -> int:
        """Return a dead worker's plain leases; staged and API-uncertain work stays."""
        result = self.connection.execute(
            """UPDATE sample_state SET status='pending',lease_id=NULL,worker_instance_id=NULL,
               lease_expires_at=NULL,prepared_artifact_relative_path=NULL,
               prepared_artifact_sha256=NULL,updated_at=?
               WHERE job_id=? AND current_module_id=? AND status='leased'""",
            (utc_now(), job_id, module_id),
        )
        return result.rowcount

    def fail_unsettled_module_samples_page(
        self,
        job_id: str,
        module_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = 1_000,
    ) -> list[int]:
        """Settle one page after a module exhausts its worker restart budget."""
        limit = validate_page_limit(limit)
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            expected_job_status = "exporting" if module_id == "export" else "running"
            if job["status"] != expected_job_status or job["current_module_id"] != module_id:
                raise ValueError("only an active module can settle an unavailable worker")
            repair_clause, repair_parameters = repair_target_clause(
                self.connection, job_id, module_id,
            )
            query = """SELECT s.sample_id,s.relative_image_path,st.attempt
                       FROM samples AS s JOIN sample_state AS st
                         ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                       WHERE s.job_id=? AND s.in_processing_scope=1
                         AND st.status IN ('pending','leased','prepared','request_started','response_staged')
                         AND (st.current_module_id=? OR (?='caption' AND st.current_module_id IS NULL))""" + repair_clause
            parameters: list[Any] = [job_id, module_id, module_id, *repair_parameters]
            if after_sample_id is not None:
                query += " AND s.sample_id>?"
                parameters.append(after_sample_id)
            query += " ORDER BY s.sample_id LIMIT ?"
            parameters.append(limit)
            rows = list(self.connection.execute(query, parameters))
            if not rows:
                return []
            sample_ids = [int(row["sample_id"]) for row in rows]
            placeholders = ",".join("?" for _ in sample_ids)
            sample_status = "skipped" if module_id == "export" else "failed"
            self.connection.execute(
                f"""UPDATE sample_state SET current_module_id=?,status=?,lease_id=NULL,
                    worker_instance_id=NULL,lease_expires_at=NULL,
                    prepared_artifact_relative_path=NULL,prepared_artifact_sha256=NULL,updated_at=?
                    WHERE job_id=? AND sample_id IN ({placeholders})""",
                (module_id, sample_status, utc_now(), job_id, *sample_ids),
            )
            issue_delta = 0
            for row in rows:
                sample_id = int(row["sample_id"])
                code = "module_worker_unavailable"
                previous = self.connection.execute(
                    """SELECT resolved_at FROM issues
                       WHERE job_id=? AND sample_id=? AND module_id=? AND code=?""",
                    (job_id, sample_id, module_id, code),
                ).fetchone()
                upsert_issue(self.connection, SampleIssue(
                    issueId=hashlib.sha256(
                        f"{job_id}\0{sample_id}\0{module_id}\0{code}".encode("utf-8")
                    ).hexdigest(),
                    jobId=job_id,
                    sampleId=sample_id,
                    relativeImagePath=str(row["relative_image_path"]),
                    moduleId=module_id,  # type: ignore[arg-type]
                    code=code,
                    severity="error",
                    blocking=True,
                    retriable=False,
                    message="Module worker remained unavailable after repeated abnormal exits; the sample was skipped.",
                    attempt=int(row["attempt"]),
                ))
                issue_delta += int(previous is None or previous["resolved_at"] is not None)
            increment_module_counts(
                self.connection, job_id, module_id, failed=len(rows), issues=issue_delta,
            )
            return sample_ids

    def count_in_flight(self, job_id: str) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM sample_state WHERE job_id=? AND status IN ('leased','prepared','request_started','response_staged')",
            (job_id,),
        ).fetchone()[0])

    def count_module_unsettled(self, job_id: str, module_id: str) -> int:
        repair_clause, repair_parameters = repair_target_clause(self.connection, job_id, module_id)
        return int(self.connection.execute(
            """SELECT COUNT(*) FROM samples AS s JOIN sample_state AS st
                 ON st.job_id=s.job_id AND st.sample_id=s.sample_id
               WHERE s.job_id=? AND s.in_processing_scope=1
                 AND st.status IN ('pending','leased','prepared','request_started','response_staged')
                 AND (st.current_module_id=? OR (?='caption' AND st.current_module_id IS NULL))""" + repair_clause,
            (job_id, module_id, module_id, *repair_parameters),
        ).fetchone()[0])

    def stage_prepared_artifact(
        self,
        job_id: str,
        sample_id: int,
        *,
        lease_id: str,
        relative_path: str,
        sha256: str,
        allowed_statuses: tuple[str, ...] = ("leased",),
    ) -> None:
        if not allowed_statuses:
            raise ValueError("prepared artifact allowed statuses cannot be empty")
        placeholders = ",".join("?" for _ in allowed_statuses)
        result = self.connection.execute(
            f"""UPDATE sample_state SET status='prepared',prepared_artifact_relative_path=?,prepared_artifact_sha256=?,updated_at=?
               WHERE job_id=? AND sample_id=? AND status IN ({placeholders}) AND lease_id=?""",
            (relative_path, sha256, utc_now(), job_id, sample_id, *allowed_statuses, lease_id),
        )
        if result.rowcount != 1:
            raise ValueError("prepared artifact does not belong to an active lease")

    def stage_classify_prepared_artifact(
        self,
        job_id: str,
        sample_id: int,
        *,
        lease_id: str,
        relative_path: str,
        sha256: str,
        count_decision: Mapping[str, object],
    ) -> None:
        """Atomically stage Classify output and its validated v3 count evidence."""
        from .classify_protocol import ClassifyCountDecisionV1
        from .count_review_protocol import CountEvidenceV1

        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        ):
            raise ValueError("classify prepared artifact identity is invalid")
        evidence = CountEvidenceV1.from_decision(ClassifyCountDecisionV1.from_dict(count_decision))
        now = utc_now()
        with self.transaction(immediate=True):
            row = self.connection.execute(
                """SELECT st.status,st.current_module_id,st.lease_id,
                          st.prepared_artifact_relative_path,st.prepared_artifact_sha256,
                          j.config_schema_version
                   FROM sample_state AS st JOIN jobs AS j ON j.job_id=st.job_id
                   WHERE st.job_id=? AND st.sample_id=?""",
                (job_id, sample_id),
            ).fetchone()
            if (
                row is None
                or row["current_module_id"] != "classify"
                or row["lease_id"] != lease_id
                or row["status"] not in {"leased", "prepared"}
            ):
                raise ValueError("classify artifact does not belong to an active lease")
            if row["status"] == "prepared" and (
                row["prepared_artifact_relative_path"] != relative_path
                or row["prepared_artifact_sha256"] != sha256
            ):
                raise ValueError("classify prepared retry does not match persisted artifact")
            if row["status"] == "leased":
                result = self.connection.execute(
                    """UPDATE sample_state
                       SET status='prepared',prepared_artifact_relative_path=?,prepared_artifact_sha256=?,updated_at=?
                       WHERE job_id=? AND sample_id=? AND current_module_id='classify'
                         AND status='leased' AND lease_id=?""",
                    (relative_path, sha256, now, job_id, sample_id, lease_id),
                )
                if result.rowcount != 1:
                    raise ValueError("classify artifact does not belong to an active lease")

            config_schema_version = int(row["config_schema_version"])
            if config_schema_version == 2:
                return
            if config_schema_version not in COUNT_REVIEW_SCHEMA_VERSIONS:
                raise ValueError("classify task configuration schema is invalid")
            values = (
                job_id,
                sample_id,
                1,
                evidence.value,
                evidence.decision_json,
                evidence.review_warning_codes_json,
                now,
                now,
            )
            self.connection.execute(
                """INSERT OR IGNORE INTO count_evidence(
                       job_id,sample_id,schema_version,value,decision_json,
                       review_warning_codes_json,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?)""",
                values,
            )
            persisted = self.connection.execute(
                """SELECT schema_version,value,decision_json,review_warning_codes_json
                   FROM count_evidence WHERE job_id=? AND sample_id=?""",
                (job_id, sample_id),
            ).fetchone()
            expected = (1, evidence.value, evidence.decision_json, evidence.review_warning_codes_json)
            if persisted is None or tuple(persisted) != expected:
                raise ValueError("classify count evidence retry does not match persisted evidence")

    def complete_leased_sample(
        self,
        job_id: str,
        sample_id: int,
        *,
        lease_id: str,
        allowed_statuses: tuple[str, ...] = ("leased", "prepared", "response_staged"),
        txt_provenance: str | None = None,
    ) -> None:
        if not allowed_statuses:
            raise ValueError("allowed statuses cannot be empty")
        if txt_provenance is not None and txt_provenance not in {
            "missing",
            "original_preserved",
            "module1_written",
        }:
            raise ValueError("invalid TXT provenance")
        placeholders = ",".join("?" for _ in allowed_statuses)
        result = self.connection.execute(
            f"""UPDATE sample_state SET status='completed',txt_provenance=COALESCE(?,txt_provenance),
                lease_id=NULL,worker_instance_id=NULL,
                lease_expires_at=NULL,updated_at=? WHERE job_id=? AND sample_id=? AND lease_id=?
                AND status IN ({placeholders})""",
            (txt_provenance, utc_now(), job_id, sample_id, lease_id, *allowed_statuses),
        )
        if result.rowcount != 1:
            raise ValueError("completion does not belong to an active lease")

    def complete_leased_sample_and_count(
        self,
        job_id: str,
        module_id: str,
        sample_id: int,
        *,
        lease_id: str,
        allowed_statuses: tuple[str, ...] = ("leased", "prepared", "response_staged"),
        txt_provenance: str | None = None,
    ) -> None:
        """Settle a successful lease and its aggregate count in one short transaction."""
        with self.transaction(immediate=True):
            if module_id == "count_review":
                decision = self.connection.execute(
                    """SELECT d.status,d.final_count,d.applied_at,st.status AS sample_status
                         FROM count_review_decisions AS d
                         JOIN sample_state AS st ON st.job_id=d.job_id AND st.sample_id=d.sample_id
                        WHERE d.job_id=? AND d.sample_id=? AND st.current_module_id='count_review'
                          AND st.lease_id=?""",
                    (job_id, sample_id, lease_id),
                ).fetchone()
                if (
                    decision is None
                    or decision["status"] not in {"auto_resolved", "manual_resolved"}
                    or decision["final_count"] is None
                    or decision["applied_at"] is not None
                    or decision["sample_status"] != "prepared"
                ):
                    raise ValueError("count review completion has no prepared resolved decision")
            self.complete_leased_sample(
                job_id,
                sample_id,
                lease_id=lease_id,
                allowed_statuses=allowed_statuses,
                txt_provenance=txt_provenance,
            )
            if module_id == "count_review":
                result = self.connection.execute(
                    """UPDATE count_review_decisions SET applied_at=?,updated_at=?
                       WHERE job_id=? AND sample_id=? AND applied_at IS NULL""",
                    (utc_now(), utc_now(), job_id, sample_id),
                )
                if result.rowcount != 1:
                    raise ValueError("count review decision could not be marked applied")
            increment_module_counts(self.connection, job_id, module_id, completed=1)

    def skip_leased_caption_sample(
        self,
        job_id: str,
        sample_id: int,
        *,
        lease_id: str,
    ) -> None:
        """Settle an incremental non-overwrite decision without invoking a worker."""
        with self.transaction(immediate=True):
            result = self.connection.execute(
                """UPDATE sample_state
                   SET status='skipped',txt_provenance='original_preserved',lease_id=NULL,
                       worker_instance_id=NULL,lease_expires_at=NULL,updated_at=?
                   WHERE job_id=? AND sample_id=? AND current_module_id='caption'
                     AND status='leased' AND lease_id=?""",
                (utc_now(), job_id, sample_id, lease_id),
            )
            if result.rowcount != 1:
                raise ValueError("caption skip does not belong to an active lease")
            increment_module_counts(self.connection, job_id, "caption", skipped=1)

    def fail_leased_sample_with_issue(
        self,
        job_id: str,
        module_id: str,
        sample_id: int,
        *,
        lease_id: str,
        issue: SampleIssue,
        allowed_statuses: tuple[str, ...] = ("leased",),
    ) -> None:
        """Persist one failed lease, its stable issue, and aggregate counts atomically."""
        if (
            issue.jobId != job_id
            or issue.sampleId != sample_id
            or issue.moduleId != module_id
        ):
            raise ValueError("issue identity does not match the failed lease")
        if not allowed_statuses:
            raise ValueError("failure allowed statuses cannot be empty")
        placeholders = ",".join("?" for _ in allowed_statuses)
        with self.transaction(immediate=True):
            previous = self.connection.execute(
                """SELECT resolved_at FROM issues
                   WHERE job_id=? AND sample_id=? AND module_id=? AND code=?""",
                (job_id, sample_id, module_id, issue.code),
            ).fetchone()
            result = self.connection.execute(
                """UPDATE sample_state SET status='failed',lease_id=NULL,worker_instance_id=NULL,
                   lease_expires_at=NULL,prepared_artifact_relative_path=NULL,
                   prepared_artifact_sha256=NULL,updated_at=?
                   WHERE job_id=? AND sample_id=? AND current_module_id=?
                     AND status IN (""" + placeholders + ") AND lease_id=?",
                (utc_now(), job_id, sample_id, module_id, *allowed_statuses, lease_id),
            )
            if result.rowcount != 1:
                raise ValueError("failure does not belong to an active lease")
            upsert_issue(self.connection, issue)
            issue_delta = int(previous is None or previous["resolved_at"] is not None)
            increment_module_counts(
                self.connection, job_id, module_id, failed=1, issues=issue_delta,
            )

    def skip_leased_sample_with_issue(
        self,
        job_id: str,
        module_id: str,
        sample_id: int,
        *,
        lease_id: str,
        issue: SampleIssue,
    ) -> None:
        """Retain an Export validation failure while omitting its new artifacts."""
        if (
            module_id != "export"
            or issue.jobId != job_id
            or issue.sampleId != sample_id
            or issue.moduleId != module_id
        ):
            raise ValueError("issue identity does not match the skipped Export lease")
        with self.transaction(immediate=True):
            previous = self.connection.execute(
                """SELECT resolved_at FROM issues
                   WHERE job_id=? AND sample_id=? AND module_id=? AND code=?""",
                (job_id, sample_id, module_id, issue.code),
            ).fetchone()
            result = self.connection.execute(
                """UPDATE sample_state SET status='skipped',lease_id=NULL,worker_instance_id=NULL,
                   lease_expires_at=NULL,prepared_artifact_relative_path=NULL,
                   prepared_artifact_sha256=NULL,updated_at=?
                   WHERE job_id=? AND sample_id=? AND current_module_id='export'
                     AND status='leased' AND lease_id=?""",
                (utc_now(), job_id, sample_id, lease_id),
            )
            if result.rowcount != 1:
                raise ValueError("Export skip does not belong to an active lease")
            upsert_issue(self.connection, issue)
            issue_delta = int(previous is None or previous["resolved_at"] is not None)
            # `failed` reports invalid Export inputs; the sample state is
            # `skipped` so commit does not expect an artifact for it.
            increment_module_counts(
                self.connection, job_id, module_id, failed=1, issues=issue_delta,
            )

    def return_lease_to_pending(self, job_id: str, sample_id: int, *, lease_id: str) -> None:
        result = self.connection.execute(
            """UPDATE sample_state SET status='pending',lease_id=NULL,worker_instance_id=NULL,lease_expires_at=NULL,
               prepared_artifact_relative_path=NULL,prepared_artifact_sha256=NULL,updated_at=?
               WHERE job_id=? AND sample_id=? AND lease_id=? AND status IN ('leased','prepared')""",
            (utc_now(), job_id, sample_id, lease_id),
        )
        if result.rowcount != 1:
            raise ValueError("lease cannot be returned")

    def recovery_state_page(self, job_id: str, *, after_sample_id: int | None = None, limit: int = 500) -> list[sqlite3.Row]:
        limit = validate_page_limit(limit)
        query = """SELECT s.*,st.current_module_id,st.status,st.attempt,st.lease_id,st.worker_instance_id,
                          st.lease_expires_at,st.prepared_artifact_relative_path,st.prepared_artifact_sha256
                   FROM samples AS s JOIN sample_state AS st ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                   WHERE s.job_id=?"""
        params: list[Any] = [job_id]
        if after_sample_id is not None:
            query += " AND s.sample_id>?"
            params.append(after_sample_id)
        query += " ORDER BY s.sample_id LIMIT ?"
        params.append(limit)
        return list(self.connection.execute(query, params))

    def mark_interrupted(self, job_id: str, *, allow_failed: bool = False) -> None:
        """Freeze one active job for manual recovery, preserving its resume target.

        `cancelling` keeps draining and an already interrupted job must not
        overwrite `resume_status` with 'interrupted' on a second startup pass;
        both are excluded here to match `decide_startup_recovery`.
        """
        non_interruptible = tuple(
            status for status in NON_INTERRUPTIBLE_JOB_STATUSES
            if not allow_failed or status != "failed"
        )
        placeholders = ",".join("?" for _ in non_interruptible)
        self.connection.execute(
            f"""UPDATE jobs SET resume_status=CASE
                    WHEN status='paused' THEN COALESCE(resume_status,'running')
                    ELSE status
                END,status='interrupted'
                WHERE job_id=? AND status NOT IN ({placeholders})""",
            (job_id, *non_interruptible),
        )

    def pause_active_module(self, job_id: str, module_id: str, *, active_status: str) -> None:
        """Atomically pause the exact active job and module pair."""
        if active_status not in {"running", "exporting"}:
            raise ValueError("active module status is invalid")
        transition_job(active_status, "paused")
        transition_module("running", "paused", module_id=module_id)
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            if job["status"] != active_status or job["current_module_id"] != module_id:
                raise ValueError("active module state changed before pause")
            paused_at = utc_now()
            summary = self.connection.execute(
                """UPDATE module_summary
                      SET status='paused',started_at=COALESCE(started_at,?)
                    WHERE job_id=? AND module_id=? AND status='running'""",
                (paused_at, job_id, module_id),
            )
            if summary.rowcount != 1:
                raise ValueError("active module state changed before pause")
            result = self.connection.execute(
                """UPDATE jobs SET status='paused',resume_status=?
                   WHERE job_id=? AND status=? AND current_module_id=?""",
                (active_status, job_id, active_status, module_id),
            )
            if result.rowcount != 1:
                raise ValueError("active module state changed before pause")

    def pause_future_module(
        self, job_id: str, module_id: str, *, total: int, current_module_id: str, active_status: str,
    ) -> None:
        """Create a no-work paused summary for the selected future module."""
        if total < 0:
            raise ValueError("module total must not be negative")
        transition_module("pending", "paused", module_id=module_id)
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            if job["status"] != active_status or job["current_module_id"] != current_module_id:
                raise ValueError("task state changed before future pause")
            if self.connection.execute(
                "SELECT 1 FROM module_summary WHERE job_id=? AND module_id=?", (job_id, module_id)
            ).fetchone() is not None:
                raise ValueError("future module has already been initialized")
            if self.connection.execute(
                "SELECT 1 FROM sample_state WHERE job_id=? AND current_module_id=? LIMIT 1",
                (job_id, module_id),
            ).fetchone() is not None:
                raise ValueError("future module already has work")
            self.connection.execute(
                "INSERT INTO module_summary(job_id,module_id,status,total) VALUES (?,?,?,?)",
                (job_id, module_id, "paused", total),
            )

    def cancel_future_pause(
        self, job_id: str, module_id: str, *, current_module_id: str, active_status: str,
    ) -> None:
        """Remove only an unstarted future pause summary."""
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            if (
                job["status"] != active_status
                or job["current_module_id"] != current_module_id
                or current_module_id == module_id
            ):
                raise ValueError("task state changed before future pause cancellation")
            result = self.connection.execute(
                """DELETE FROM module_summary
                   WHERE job_id=? AND module_id=? AND status='paused'
                     AND completed=0 AND failed=0 AND skipped=0 AND issue_count=0
                     AND worker_restart_count=0 AND started_at IS NULL AND finished_at IS NULL
                     AND NOT EXISTS (
                         SELECT 1 FROM sample_state
                         WHERE job_id=? AND current_module_id=?
                     )""",
                (job_id, module_id, job_id, module_id),
            )
            if result.rowcount != 1:
                raise ValueError("future pause is not a zero-work summary")

    def start_prepaused_module(self, job_id: str, module_id: str, *, resume_status: str) -> None:
        """Make a selected future pause the current task pause without queue work."""
        if resume_status not in {"running", "exporting"}:
            raise ValueError("prepaused module resume status is invalid")
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            transition_job(str(job["status"]), "paused")
            summary = self.connection.execute(
                """SELECT 1 FROM module_summary
                   WHERE job_id=? AND module_id=? AND status='paused'
                     AND completed=0 AND failed=0 AND skipped=0 AND issue_count=0
                     AND worker_restart_count=0 AND started_at IS NULL AND finished_at IS NULL""",
                (job_id, module_id),
            ).fetchone()
            if summary is None or job["current_module_id"] == module_id:
                raise ValueError("prepaused module state changed before start")
            if self.connection.execute(
                "SELECT 1 FROM sample_state WHERE job_id=? AND current_module_id=? LIMIT 1",
                (job_id, module_id),
            ).fetchone() is not None:
                raise ValueError("prepaused module already has work")
            result = self.connection.execute(
                """UPDATE jobs SET status='paused',resume_status=?,current_module_id=?
                   WHERE job_id=? AND status=? AND current_module_id=?""",
                (resume_status, module_id, job_id, job["status"], job["current_module_id"]),
            )
            if result.rowcount != 1:
                raise ValueError("prepaused module state changed before start")

    def resume_prepaused_module(self, job_id: str, module_id: str, *, target_status: str, total: int) -> None:
        """Release an arrived zero-work pause before its first worker starts."""
        if target_status not in {"running", "exporting"} or total < 0:
            raise ValueError("prepaused module resume state is invalid")
        transition_job("paused", target_status)
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id,resume_status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            summary = self.connection.execute(
                """SELECT total FROM module_summary
                   WHERE job_id=? AND module_id=? AND status='paused'
                     AND completed=0 AND failed=0 AND skipped=0 AND issue_count=0
                     AND worker_restart_count=0 AND started_at IS NULL AND finished_at IS NULL""",
                (job_id, module_id),
            ).fetchone()
            if (
                job["status"] != "paused"
                or job["current_module_id"] != module_id
                or job["resume_status"] != target_status
                or summary is None
                or int(summary["total"]) != total
            ):
                raise ValueError("prepaused module state changed before resume")
            result = self.connection.execute(
                """UPDATE module_summary SET status='pending'
                   WHERE job_id=? AND module_id=? AND status='paused'
                     AND total=? AND completed=0 AND failed=0 AND skipped=0 AND issue_count=0
                     AND worker_restart_count=0 AND started_at IS NULL AND finished_at IS NULL""",
                (job_id, module_id, total),
            )
            if result.rowcount != 1:
                raise ValueError("prepaused module state changed before resume")
            self.connection.execute(
                "UPDATE jobs SET status=?,resume_status=NULL WHERE job_id=?", (target_status, job_id)
            )

    def restore_prepaused_module(self, job_id: str, module_id: str, *, active_status: str, total: int) -> None:
        """Restore a zero-work pause when its replacement pipeline thread cannot start."""
        if active_status not in {"running", "exporting"} or total < 0:
            raise ValueError("prepaused module restore state is invalid")
        transition_job(active_status, "paused")
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            if job["status"] != active_status or job["current_module_id"] != module_id:
                raise ValueError("prepaused module state changed before restore")
            summary = self.connection.execute(
                """UPDATE module_summary SET status='paused'
                   WHERE job_id=? AND module_id=? AND status='pending'
                     AND total=? AND completed=0 AND failed=0 AND skipped=0 AND issue_count=0
                     AND worker_restart_count=0 AND started_at IS NULL AND finished_at IS NULL""",
                (job_id, module_id, total),
            )
            if summary.rowcount != 1:
                raise ValueError("prepaused module state changed before restore")
            self.connection.execute(
                "UPDATE jobs SET status='paused',resume_status=? WHERE job_id=?", (active_status, job_id)
            )

    def resume_paused_module(self, job_id: str, module_id: str, *, target_status: str) -> None:
        """Atomically resume the exact paused job and module pair."""
        if target_status not in {"running", "exporting"}:
            raise ValueError("resume target status is invalid")
        transition_job("paused", target_status)
        transition_module("paused", "running", module_id=module_id)
        with self.transaction(immediate=True):
            job = self.connection.execute(
                "SELECT status,current_module_id,resume_status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"job does not exist: {job_id}")
            if (
                job["status"] != "paused"
                or job["current_module_id"] != module_id
                or job["resume_status"] != target_status
            ):
                raise ValueError("paused module state changed before resume")
            summary = self.connection.execute(
                "UPDATE module_summary SET status='running' WHERE job_id=? AND module_id=? AND status='paused'",
                (job_id, module_id),
            )
            if summary.rowcount != 1:
                raise ValueError("paused module state changed before resume")
            result = self.connection.execute(
                """UPDATE jobs SET status=?,resume_status=NULL
                   WHERE job_id=? AND status='paused' AND current_module_id=? AND resume_status=?""",
                (target_status, job_id, module_id, target_status),
            )
            if result.rowcount != 1:
                raise ValueError("paused module state changed before resume")

    def begin_cancellation(self, job_id: str) -> None:
        result = self.connection.execute(
            """UPDATE jobs SET resume_status=CASE
                    WHEN status='cancelling' THEN resume_status
                    WHEN status='paused' THEN COALESCE(resume_status,'running')
                    ELSE status
                END,
                status='cancelling',cancel_requested_at=COALESCE(cancel_requested_at,?)
               WHERE job_id=? AND status IN ('preparing_workspace','running','paused','reviewing','exporting','cancelling')""",
            (utc_now(), job_id),
        )
        if result.rowcount != 1:
            raise ValueError("job is not in a cancellable state")

    def abandon_cancellation_work(self, job_id: str) -> int:
        """Drop the in-flight sample rows a dead process left behind during cancellation.

        A restart has no worker left to finish or release those rows, and
        `settle_cancellation` only accepts an empty in-flight set, so the job would
        otherwise stay `cancelling` forever while it still held its dataset claim.
        """
        result = self.connection.execute(
            """UPDATE sample_state SET status='pending',lease_id=NULL,worker_instance_id=NULL,
               lease_expires_at=NULL,prepared_artifact_relative_path=NULL,
               prepared_artifact_sha256=NULL,updated_at=?
               WHERE job_id=? AND status IN ('leased','prepared','request_started','response_staged')""",
            (utc_now(), job_id),
        )
        return result.rowcount

    def settle_cancellation(self, job_id: str, *, succeeded: bool = False) -> None:
        if self.count_in_flight(job_id):
            raise ValueError("cannot settle cancellation while work remains in flight")
        status = "succeeded" if succeeded else "cancelled_recoverable"
        result = self.connection.execute(
            "UPDATE jobs SET status=?,finished_at=? WHERE job_id=? AND status='cancelling'",
            (status, utc_now(), job_id),
        )
        if result.rowcount != 1:
            raise ValueError("job is not cancelling")

    def set_sample_state(self, job_id: str, sample_id: int, state: SampleRunState) -> None:
        result = self.connection.execute(
            """UPDATE sample_state SET current_module_id=?,txt_provenance=?,status=?,attempt=?,lease_id=?,
               worker_instance_id=?,lease_expires_at=?,prepared_artifact_relative_path=?,prepared_artifact_sha256=?,updated_at=?
               WHERE job_id=? AND sample_id=?""",
            (
                state.currentModuleId, state.txtProvenance, state.status, state.attempt, state.leaseId,
                state.workerInstanceId, state.leaseExpiresAt, state.preparedArtifactRelativePath,
                state.preparedArtifactSha256, utc_now(), job_id, sample_id,
            ),
        )
        if result.rowcount != 1:
            raise KeyError(f"sample does not exist: {job_id}/{sample_id}")
