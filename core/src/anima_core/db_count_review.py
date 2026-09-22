"""Count review ownership for the state database."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping

from .contracts import COUNT_REVIEW_SCHEMA_VERSIONS, pipeline_module_ids, utc_now
from .db_primitives import (
    get_job_row,
    insert_count_observation_consistent,
    repair_target_clause,
    set_job_status as set_job_status_primitive,
    validate_count_page_limit,
)
from .db_schema import DEFAULT_PAGE_SIZE, MAX_COUNT_PAGE_SIZE


class CountReviewDatabaseMixin:
    """Count evidence, observation, review decision, and batch confirmation operations."""

    def inherit_repair_count_evidence(self, repair_job_id: str, parent_job_id: str) -> int:
        """Copy validated Classify evidence needed by downstream repair targets."""
        from .count_review_protocol import CountEvidenceV1

        repair_job = get_job_row(self.connection, repair_job_id)
        parent_job = get_job_row(self.connection, parent_job_id)
        repair_version = int(repair_job["config_schema_version"])
        if repair_version != int(parent_job["config_schema_version"]):
            raise ValueError("repair task configuration version does not match its parent")
        if repair_version == 2:
            return 0
        if repair_version not in COUNT_REVIEW_SCHEMA_VERSIONS:
            raise ValueError("repair task configuration schema is invalid")

        inherited = 0
        cursor: int | None = None
        while True:
            cursor_clause = "" if cursor is None else " AND rt.sample_id>?"
            parameters: list[object] = [parent_job_id, repair_job_id]
            if cursor is not None:
                parameters.append(cursor)
            parameters.append(MAX_COUNT_PAGE_SIZE)
            rows = list(self.connection.execute(
                """SELECT rt.sample_id,e.schema_version,e.value,e.decision_json,
                          e.review_warning_codes_json
                     FROM repair_targets AS rt
                     LEFT JOIN count_evidence AS e
                       ON e.job_id=? AND e.sample_id=rt.sample_id
                    WHERE rt.repair_job_id=? AND rt.repair_start_module IN ('ocr','replace','nl')"""
                + cursor_clause + " ORDER BY rt.sample_id LIMIT ?",
                parameters,
            ))
            if not rows:
                return inherited
            parsed: list[tuple[int, CountEvidenceV1]] = []
            for row in rows:
                try:
                    decision = json.loads(str(row["decision_json"]))
                    warning_codes = json.loads(str(row["review_warning_codes_json"]))
                    evidence = CountEvidenceV1.from_dict({
                        "schemaVersion": row["schema_version"],
                        "value": row["value"],
                        "decision": decision,
                        "reviewWarningCodes": warning_codes,
                    })
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("parent Classify count evidence is missing or invalid") from exc
                parsed.append((int(row["sample_id"]), evidence))
            with self.transaction(immediate=True):
                now = utc_now()
                self.connection.executemany(
                    """INSERT INTO count_evidence(
                           job_id,sample_id,schema_version,value,decision_json,
                           review_warning_codes_json,created_at,updated_at
                       ) VALUES (?,?,?,?,?,?,?,?)""",
                    [
                        (
                            repair_job_id,
                            sample_id,
                            evidence.schemaVersion,
                            evidence.value,
                            evidence.decision_json,
                            evidence.review_warning_codes_json,
                            now,
                            now,
                        )
                        for sample_id, evidence in parsed
                    ],
                )
            inherited += len(parsed)
            cursor = int(rows[-1]["sample_id"])

    def get_count_evidence(self, job_id: str, sample_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM count_evidence WHERE job_id=? AND sample_id=?",
            (job_id, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"count evidence does not exist: {job_id}/{sample_id}")
        return row

    def page_count_evidence(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> list[sqlite3.Row]:
        limit = validate_count_page_limit(limit)
        if after_sample_id is None:
            return list(self.connection.execute(
                "SELECT * FROM count_evidence WHERE job_id=? ORDER BY sample_id LIMIT ?",
                (job_id, limit),
            ))
        return list(self.connection.execute(
            """SELECT * FROM count_evidence WHERE job_id=? AND sample_id>?
               ORDER BY sample_id LIMIT ?""",
            (job_id, after_sample_id, limit),
        ))

    def get_count_observation(self, job_id: str, sample_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM count_observations WHERE job_id=? AND sample_id=?",
            (job_id, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"count observation does not exist: {job_id}/{sample_id}")
        return row

    def _insert_count_observation_consistent(
        self,
        job_id: str,
        sample_id: int,
        observation: object,
        *,
        now: str,
    ) -> None:
        insert_count_observation_consistent(
            self.connection, job_id, sample_id, observation, now=now,
        )

    def record_count_observation_not_requested(
        self,
        job_id: str,
        sample_id: int,
        *,
        lease_id: str,
        reason: str,
    ) -> None:
        from .count_review_protocol import CountObservationV1

        observation = CountObservationV1.not_requested(reason)
        with self.transaction(immediate=True):
            row = self.connection.execute(
                """SELECT j.config_schema_version,st.current_module_id,st.status,st.lease_id
                   FROM jobs AS j JOIN sample_state AS st ON st.job_id=j.job_id
                   WHERE j.job_id=? AND st.sample_id=?""",
                (job_id, sample_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"sample state does not exist: {job_id}/{sample_id}")
            if int(row["config_schema_version"]) == 2:
                return
            if (
                int(row["config_schema_version"]) not in COUNT_REVIEW_SCHEMA_VERSIONS
                or row["current_module_id"] != "nl"
                or row["status"] != "leased"
                or row["lease_id"] != lease_id
            ):
                raise ValueError("not-requested observation does not belong to an active NL lease")
            self._insert_count_observation_consistent(job_id, sample_id, observation, now=utc_now())

    def record_nl_disabled_observations_page(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = MAX_COUNT_PAGE_SIZE,
    ) -> list[int]:
        from .count_review_protocol import CountObservationV1

        limit = validate_count_page_limit(limit)
        job = get_job_row(self.connection, job_id)
        if int(job["config_schema_version"]) == 2:
            return []
        if int(job["config_schema_version"]) not in COUNT_REVIEW_SCHEMA_VERSIONS:
            raise ValueError("NL task configuration schema is invalid")
        clause, parameters = repair_target_clause(self.connection, job_id, "nl")
        cursor = "" if after_sample_id is None else " AND s.sample_id>?"
        query_parameters: list[object] = [job_id]
        if after_sample_id is not None:
            query_parameters.append(after_sample_id)
        query_parameters.extend(parameters)
        query_parameters.append(limit)
        rows = list(self.connection.execute(
            """SELECT s.sample_id FROM samples AS s
               WHERE s.job_id=? AND s.in_processing_scope=1"""
            + cursor + clause + " ORDER BY s.sample_id LIMIT ?",
            query_parameters,
        ))
        if not rows:
            return []
        observation = CountObservationV1.not_requested("nl_disabled")
        with self.transaction(immediate=True):
            now = utc_now()
            for row in rows:
                self._insert_count_observation_consistent(
                    job_id, int(row["sample_id"]), observation, now=now
                )
        return [int(row["sample_id"]) for row in rows]

    def page_count_observations(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> list[sqlite3.Row]:
        limit = validate_count_page_limit(limit)
        predicate = "" if after_sample_id is None else " AND sample_id>?"
        parameters: list[object] = [job_id]
        if after_sample_id is not None:
            parameters.append(after_sample_id)
        parameters.append(limit)
        return list(self.connection.execute(
            "SELECT * FROM count_observations WHERE job_id=?" + predicate + " ORDER BY sample_id LIMIT ?",
            parameters,
        ))

    def get_count_review_decision(self, job_id: str, sample_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM count_review_decisions WHERE job_id=? AND sample_id=?",
            (job_id, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"count review decision does not exist: {job_id}/{sample_id}")
        return row

    def page_count_review_decisions(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        status: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> list[sqlite3.Row]:
        limit = validate_count_page_limit(limit)
        if status is not None and status not in {"pending", "auto_resolved", "manual_resolved"}:
            raise ValueError("count review status filter is invalid")
        predicates: list[str] = ["job_id=?"]
        parameters: list[object] = [job_id]
        if after_sample_id is not None:
            predicates.append("sample_id>?")
            parameters.append(after_sample_id)
        if status is not None:
            predicates.append("status=?")
            parameters.append(status)
        parameters.append(limit)
        return list(self.connection.execute(
            "SELECT * FROM count_review_decisions WHERE " + " AND ".join(predicates)
            + " ORDER BY sample_id LIMIT ?",
            parameters,
        ))

    def page_count_review_items(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        status: str | None = None,
        reason: str | None = None,
        classify_count: str | None = None,
        vlm_count: str | None = None,
        mismatch_only: bool = False,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> list[sqlite3.Row]:
        """Keyset-page bounded Count Review summaries without loading decision payloads."""
        from .count_review_protocol import COUNT_REVIEW_REASON_CODES, FINAL_COUNT_VALUES

        limit = validate_count_page_limit(limit)
        if after_sample_id is not None and (type(after_sample_id) is not int or after_sample_id < 0):
            raise ValueError("count review cursor is invalid")
        if status is not None and status not in {"pending", "auto_resolved", "manual_resolved"}:
            raise ValueError("count review status filter is invalid")
        if reason is not None and reason not in COUNT_REVIEW_REASON_CODES:
            raise ValueError("count review reason filter is invalid")
        classify_values = {*FINAL_COUNT_VALUES, "unavailable"}
        vlm_values = {*FINAL_COUNT_VALUES, "unknown", "unavailable"}
        if classify_count is not None and classify_count not in classify_values:
            raise ValueError("count review Classify filter is invalid")
        if vlm_count is not None and vlm_count not in vlm_values:
            raise ValueError("count review VLM filter is invalid")
        if type(mismatch_only) is not bool:
            raise ValueError("count review mismatch filter is invalid")

        predicates = ["d.job_id=?", "st.current_module_id='count_review'"]
        parameters: list[object] = [job_id]
        if after_sample_id is not None:
            predicates.append("d.sample_id>?")
            parameters.append(after_sample_id)
        if status is not None:
            predicates.append("d.status=?")
            parameters.append(status)
        if reason is not None:
            predicates.append("instr(d.review_reasons_json,?)>0")
            parameters.append(f'"{reason}"')
        if classify_count == "unavailable":
            predicates.append("(e.value IS NULL OR e.value='')")
        elif classify_count is not None:
            predicates.append("e.value=?")
            parameters.append(classify_count)
        if vlm_count == "unavailable":
            placeholders = ",".join("?" for _ in (*FINAL_COUNT_VALUES, "unknown"))
            predicates.append(
                f"(o.status IS NULL OR o.status<>'observed' OR o.count_value IS NULL OR o.count_value NOT IN ({placeholders}))"
            )
            parameters.extend((*FINAL_COUNT_VALUES, "unknown"))
        elif vlm_count is not None:
            predicates.extend(("o.status='observed'", "o.count_value=?"))
            parameters.append(vlm_count)
        if mismatch_only:
            placeholders = ",".join("?" for _ in FINAL_COUNT_VALUES)
            predicates.append(
                f"e.value IN ({placeholders}) AND o.status='observed' "
                f"AND o.count_value IN ({placeholders}) AND e.value<>o.count_value"
            )
            parameters.extend((*FINAL_COUNT_VALUES, *FINAL_COUNT_VALUES))
        parameters.append(limit)
        return list(self.connection.execute(
            """SELECT d.sample_id,s.relative_image_path,
                      e.value AS evidence_value,e.review_warning_codes_json,
                      o.schema_version AS observation_schema_version,o.status AS observation_status,
                      o.count_value AS observation_count_value,o.layout_value AS observation_layout_value,
                      o.same_character_repeated,o.warning_codes_json AS observation_warning_codes_json,
                      o.not_requested_reason,
                      d.status AS decision_status,d.final_count,d.selected_source,
                      d.review_reasons_json,d.version,d.resolved_at,d.applied_at
                 FROM count_review_decisions AS d
                 JOIN samples AS s ON s.job_id=d.job_id AND s.sample_id=d.sample_id
                 JOIN sample_state AS st ON st.job_id=d.job_id AND st.sample_id=d.sample_id
                 LEFT JOIN count_evidence AS e ON e.job_id=d.job_id AND e.sample_id=d.sample_id
                 LEFT JOIN count_observations AS o ON o.job_id=d.job_id AND o.sample_id=d.sample_id
                WHERE """ + " AND ".join(predicates) + " ORDER BY d.sample_id LIMIT ?",
            parameters,
        ))

    def page_count_review_inputs(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = MAX_COUNT_PAGE_SIZE,
    ) -> list[sqlite3.Row]:
        """Page immutable Classify/NL inputs for current Count Review targets."""
        limit = validate_count_page_limit(limit)
        cursor = "" if after_sample_id is None else " AND s.sample_id>?"
        parameters: list[object] = [job_id]
        if after_sample_id is not None:
            parameters.append(after_sample_id)
        parameters.append(limit)
        return list(self.connection.execute(
            """SELECT s.sample_id,s.annotation_key,s.relative_image_path,
                      e.schema_version AS evidence_schema_version,e.value AS evidence_value,
                      e.decision_json,e.review_warning_codes_json,
                      o.schema_version AS observation_schema_version,o.status AS observation_status,
                      o.count_value AS observation_count_value,o.layout_value AS observation_layout_value,
                      o.same_character_repeated,o.warning_codes_json AS observation_warning_codes_json,
                      o.not_requested_reason
                 FROM samples AS s
                 JOIN sample_state AS st ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                 LEFT JOIN count_evidence AS e ON e.job_id=s.job_id AND e.sample_id=s.sample_id
                 LEFT JOIN count_observations AS o ON o.job_id=s.job_id AND o.sample_id=s.sample_id
                WHERE s.job_id=? AND s.in_processing_scope=1 AND st.current_module_id='count_review'"""
            + cursor + " ORDER BY s.sample_id LIMIT ?",
            parameters,
        ))

    def insert_initial_count_review_decisions(
        self,
        job_id: str,
        decisions: Iterable[tuple[int, object]],
    ) -> int:
        """Insert one bounded initialization page without replacing saved decisions."""
        from .count_review_protocol import InitialCountReviewDecisionV1

        values = list(decisions)
        if not values or len(values) > MAX_COUNT_PAGE_SIZE:
            raise ValueError("count review initialization page is empty or too large")
        sample_ids = [sample_id for sample_id, _ in values]
        if any(type(sample_id) is not int or sample_id < 1 for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("count review initialization sample ids are invalid")
        parsed = [
            (
                sample_id,
                InitialCountReviewDecisionV1.from_dict(
                    decision.to_dict() if isinstance(decision, InitialCountReviewDecisionV1) else decision
                ),
            )
            for sample_id, decision in values
        ]
        with self.transaction(immediate=True):
            job = get_job_row(self.connection, job_id)
            if int(job["config_schema_version"]) not in COUNT_REVIEW_SCHEMA_VERSIONS:
                raise ValueError("count review requires a v3/v4 task")
            placeholders = ",".join("?" for _ in sample_ids)
            targets = {
                int(row[0])
                for row in self.connection.execute(
                    f"""SELECT sample_id FROM sample_state WHERE job_id=?
                          AND current_module_id='count_review' AND sample_id IN ({placeholders})""",
                    (job_id, *sample_ids),
                )
            }
            if targets != set(sample_ids):
                raise ValueError("count review initialization includes a non-target sample")
            now = utc_now()
            before = self.connection.total_changes
            self.connection.executemany(
                """INSERT OR IGNORE INTO count_review_decisions(
                       job_id,sample_id,schema_version,status,final_count,selected_source,
                       review_reasons_json,version,resolved_at,applied_at,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        job_id,
                        sample_id,
                        1,
                        decision.status,
                        decision.finalCount,
                        decision.selectedSource,
                        decision.review_reasons_json,
                        1,
                        now if decision.status == "auto_resolved" else None,
                        None,
                        now,
                        now,
                    )
                    for sample_id, decision in parsed
                ],
            )
            return self.connection.total_changes - before

    def count_current_review_targets(self, job_id: str) -> int:
        return int(self.connection.execute(
            """SELECT COUNT(*) FROM sample_state
               WHERE job_id=? AND current_module_id='count_review'""",
            (job_id,),
        ).fetchone()[0])

    def count_current_review_decisions(self, job_id: str, *, status: str | None = None) -> int:
        if status is not None and status not in {"pending", "auto_resolved", "manual_resolved"}:
            raise ValueError("count review status is invalid")
        predicate = "" if status is None else " AND d.status=?"
        parameters: tuple[object, ...] = (job_id,) if status is None else (job_id, status)
        return int(self.connection.execute(
            """SELECT COUNT(*) FROM count_review_decisions AS d
               JOIN sample_state AS st ON st.job_id=d.job_id AND st.sample_id=d.sample_id
               WHERE d.job_id=? AND st.current_module_id='count_review'""" + predicate,
            parameters,
        ).fetchone()[0])

    def _resolve_count_review_decision_in_transaction(
        self,
        job_id: str,
        *,
        sample_id: int,
        expected_version: int,
        source: str,
        count: str | None,
    ) -> None:
        from .count_review_protocol import FINAL_COUNT_VALUES

        if type(sample_id) is not int or sample_id < 1 or type(expected_version) is not int or expected_version < 1:
            raise ValueError("count review update identity is invalid")
        if not isinstance(source, str) or source not in {"classify", "vlm", "manual"}:
            raise ValueError("count review source is invalid")
        if source == "manual":
            if not isinstance(count, str) or count not in FINAL_COUNT_VALUES:
                raise ValueError("manual count review value is invalid")
            final_count = count
        elif count is not None:
            raise ValueError("source-based count review updates must not supply a count")
        else:
            final_count = None
        job = get_job_row(self.connection, job_id)
        if int(job["config_schema_version"]) not in COUNT_REVIEW_SCHEMA_VERSIONS or job["status"] != "reviewing" or job["current_module_id"] != "count_review":
            raise ValueError("count review decisions can be changed only while reviewing")
        row = self.connection.execute(
            """SELECT d.status,d.version,d.applied_at,e.value AS classify_count,
                      o.status AS observation_status,o.count_value AS observation_count
                 FROM count_review_decisions AS d
                 JOIN sample_state AS st ON st.job_id=d.job_id AND st.sample_id=d.sample_id
                 LEFT JOIN count_evidence AS e ON e.job_id=d.job_id AND e.sample_id=d.sample_id
                 LEFT JOIN count_observations AS o ON o.job_id=d.job_id AND o.sample_id=d.sample_id
                WHERE d.job_id=? AND d.sample_id=? AND st.current_module_id='count_review'""",
            (job_id, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"count review decision does not exist: {job_id}/{sample_id}")
        if int(row["version"]) != expected_version:
            raise ValueError("count review decision version conflict")
        if row["status"] not in {"pending", "manual_resolved"} or row["applied_at"] is not None:
            raise ValueError("count review decision is not manually editable")
        if source == "classify":
            if row["classify_count"] not in FINAL_COUNT_VALUES:
                raise ValueError("classify count is unavailable for this review")
            final_count = str(row["classify_count"])
        elif source == "vlm":
            if row["observation_status"] != "observed" or row["observation_count"] not in FINAL_COUNT_VALUES:
                raise ValueError("VLM count is unavailable for this review")
            final_count = str(row["observation_count"])
        assert final_count is not None
        now = utc_now()
        result = self.connection.execute(
            """UPDATE count_review_decisions
                  SET status='manual_resolved',final_count=?,selected_source=?,version=version+1,
                      resolved_at=?,updated_at=?
                WHERE job_id=? AND sample_id=? AND version=? AND applied_at IS NULL""",
            (final_count, source, now, now, job_id, sample_id, expected_version),
        )
        if result.rowcount != 1:
            raise ValueError("count review decision version conflict")

    def update_count_review_decisions(
        self,
        job_id: str,
        updates: Iterable[Mapping[str, object]],
    ) -> list[sqlite3.Row]:
        items = list(updates)
        if not 1 <= len(items) <= MAX_COUNT_PAGE_SIZE:
            raise ValueError("count review update batch is empty or too large")
        expected_fields = {"sampleId", "expectedVersion", "source", "count"}
        if any(not isinstance(item, Mapping) or set(item) != expected_fields for item in items):
            raise ValueError("count review update fields are invalid")
        sample_ids = [item["sampleId"] for item in items]
        if any(type(sample_id) is not int or sample_id < 1 for sample_id in sample_ids):
            raise ValueError("count review update sample ids are invalid")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("count review update sample ids are duplicated")
        with self.transaction(immediate=True):
            for item in items:
                self._resolve_count_review_decision_in_transaction(
                    job_id,
                    sample_id=item["sampleId"],  # type: ignore[arg-type]
                    expected_version=item["expectedVersion"],  # type: ignore[arg-type]
                    source=item["source"],  # type: ignore[arg-type]
                    count=item["count"],  # type: ignore[arg-type]
                )
            return [self.get_count_review_decision(job_id, int(sample_id)) for sample_id in sample_ids]

    def update_count_review_decision(
        self,
        job_id: str,
        sample_id: int,
        *,
        expected_version: int,
        source: str,
        count: str | None = None,
    ) -> sqlite3.Row:
        return self.update_count_review_decisions(job_id, [{
            "sampleId": sample_id,
            "expectedVersion": expected_version,
            "source": source,
            "count": count,
        }])[0]

    def confirm_count_review(
        self,
        job_id: str,
        *,
        expected_config_hash: str,
    ) -> bool:
        """Open the application phase after a database-level pending gate."""
        with self.transaction(immediate=True):
            job = get_job_row(self.connection, job_id)
            if int(job["config_schema_version"]) not in COUNT_REVIEW_SCHEMA_VERSIONS or job["config_hash"] != expected_config_hash:
                raise ValueError("count review configuration does not match")
            if job["current_module_id"] == "count_review" and job["status"] == "running":
                return False
            if job["current_module_id"] != "count_review" or job["status"] != "reviewing":
                try:
                    order = pipeline_module_ids(int(job["config_schema_version"]))
                    if str(job["current_module_id"]) in order and order.index(str(job["current_module_id"])) > order.index("count_review"):
                        return False
                except ValueError:
                    pass
                raise ValueError("task is not awaiting count review confirmation")
            targets = self.count_current_review_targets(job_id)
            decisions = self.count_current_review_decisions(job_id)
            pending = self.count_current_review_decisions(job_id, status="pending")
            if targets != decisions:
                raise ValueError("count review decisions are incomplete")
            if pending:
                raise ValueError("count review still has pending decisions")
            set_job_status_primitive(
                self.connection, job_id, "running", current_module_id="count_review",
            )
            return True
