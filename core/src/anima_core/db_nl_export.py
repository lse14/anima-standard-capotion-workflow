"""NL and export ownership for the state database."""

from __future__ import annotations

import re
import hashlib
import sqlite3
import json
from collections.abc import Mapping
from typing import Any

from .contracts import COUNT_REVIEW_SCHEMA_VERSIONS, utc_now
from .db_primitives import (
    get_job_row,
    insert_count_observation_consistent,
    repair_target_clause,
    validate_page_limit,
)
from .db_schema import MAX_PAGE_SIZE
from .path_safety import sha256_file, ensure_within


class NlExportDatabaseMixin:
    """NL request and response staging plus export artifact operations."""

    def record_nl_history(
        self,
        *,
        dataset_root_key: str,
        dataset_root: str,
        sample_identity: str,
        relative_image_path: str,
        annotation_key: str,
        source: str,
        text: str,
        image_file_id: str | None = None,
        image_sha256: str | None = None,
        source_sha256: str | None = None,
        job_id: str | None = None,
        sample_id: int | None = None,
        version: int = 1,
    ) -> int:
        """Append one durable NL revision after identity and size validation.

        This table deliberately survives task-history deletion.  Callers must
        provide the current image identity; reuse never falls back to filename
        matching alone.
        """
        allowed = {"api", "original_json", "nl_txt", "manual", "dropout", "unknown"}
        if source not in allowed or not isinstance(text, str) or not text.strip():
            raise ValueError("NL history source/text is invalid")
        if len(text.encode("utf-8")) > 16_384:
            raise ValueError("NL history text exceeds its limit")
        if not dataset_root_key or not dataset_root or not sample_identity or not relative_image_path or not annotation_key:
            raise ValueError("NL history identity is incomplete")
        if not isinstance(version, int) or version < 1:
            raise ValueError("NL history version is invalid")
        for name, value in (("image_sha256", image_sha256), ("source_sha256", source_sha256)):
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError(f"NL history {name} is invalid")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.transaction(immediate=True):
            existing = self.connection.execute(
                """SELECT history_id FROM nl_history
                   WHERE dataset_root_key=? AND sample_identity=? AND source=?
                     AND text_sha256=? AND image_sha256 IS ? AND version=?
                   ORDER BY history_id LIMIT 1""",
                (dataset_root_key, sample_identity, source, digest, image_sha256, version),
            ).fetchone()
            if existing is not None:
                return int(existing["history_id"])
            cursor = self.connection.execute(
                """INSERT INTO nl_history(
                       dataset_root_key,dataset_root,sample_identity,relative_image_path,
                       annotation_key,image_file_id,image_sha256,job_id,sample_id,source,
                       text,text_sha256,source_sha256,version,created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    dataset_root_key, dataset_root, sample_identity, relative_image_path,
                    annotation_key, image_file_id, image_sha256, job_id, sample_id, source,
                    text, digest, source_sha256, version, utc_now(),
                ),
            )
        return int(cursor.lastrowid)

    def archive_nl_for_sample(
        self,
        job_id: str,
        sample_id: int,
        *,
        source: str,
        text: str,
        image_sha256: str | None = None,
    ) -> int:
        """Archive text using immutable sample and image identity from SQLite."""
        row = self.connection.execute(
            """SELECT j.dataset_root,j.dataset_root_key,s.relative_image_path,s.annotation_key,
                      s.image_file_id,s.image_size,s.image_mtime_ns
               FROM samples AS s JOIN jobs AS j ON j.job_id=s.job_id
               WHERE s.job_id=? AND s.sample_id=?""", (job_id, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"sample does not exist: {job_id}/{sample_id}")
        image_path = ensure_within(
            str(row["dataset_root"]),
            str(row["dataset_root"]) + "/" + str(row["relative_image_path"]).replace("\\", "/"),
        )
        if not image_path.is_file():
            raise ValueError("NL history image is unavailable")
        actual_image_sha256 = sha256_file(image_path)
        if image_sha256 is not None and image_sha256 != actual_image_sha256:
            raise ValueError("NL history image identity does not match")
        return self.record_nl_history(
            dataset_root_key=str(row["dataset_root_key"]), dataset_root=str(row["dataset_root"]),
            sample_identity=str(row["relative_image_path"]), relative_image_path=str(row["relative_image_path"]),
            annotation_key=str(row["annotation_key"]), image_file_id=row["image_file_id"],
            image_sha256=actual_image_sha256, source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), job_id=job_id, sample_id=sample_id,
            source=source, text=text,
        )

    def find_nl_history(
        self,
        *,
        dataset_root_key: str,
        sample_identity: str,
        image_file_id: str | None = None,
        image_sha256: str | None = None,
        source: str | None = None,
    ) -> sqlite3.Row | None:
        """Return newest history only when the persisted image identity matches."""
        params: list[object] = [dataset_root_key, sample_identity]
        query = """SELECT * FROM nl_history WHERE dataset_root_key=? AND sample_identity=?"""
        if image_file_id is None and image_sha256 is None:
            raise ValueError("NL history lookup requires image identity")
        if image_file_id is not None:
            query += " AND image_file_id=?"
            params.append(image_file_id)
        if image_sha256 is not None:
            query += " AND image_sha256=?"
            params.append(image_sha256)
        if source is not None:
            query += " AND source=?"
            params.append(source)
        query += " ORDER BY created_at DESC,history_id DESC LIMIT 1"
        row = self.connection.execute(query, params).fetchone()
        if row is None:
            return None
        text = str(row["text"])
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != str(row["text_sha256"]):
            return None
        return row

    def verified_nl_history_for_sample(self, job_id: str, sample_id: int, *, exclude_nonempty: bool = True) -> sqlite3.Row | None:
        """Find reusable history only when the current image bytes match its SHA-256."""
        sample = self.connection.execute(
            """SELECT j.dataset_root,j.dataset_root_key,s.relative_image_path,s.image_file_id
               FROM samples s JOIN jobs j ON j.job_id=s.job_id
               WHERE s.job_id=? AND s.sample_id=?""", (job_id, sample_id),
        ).fetchone()
        if sample is None:
            raise KeyError(f"sample does not exist: {job_id}/{sample_id}")
        image = ensure_within(str(sample["dataset_root"]), str(sample["dataset_root"]) + "/" + str(sample["relative_image_path"]).replace("\\", "/"))
        if not image.is_file():
            return None
        digest = sha256_file(image)
        rows = self.list_nl_history(dataset_root_key=str(sample["dataset_root_key"]), sample_identity=str(sample["relative_image_path"]))
        for row in rows:
            if row["image_sha256"] != digest:
                continue
            if exclude_nonempty and str(row["source"]) == "manual":
                continue
            if hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest() != str(row["text_sha256"]):
                continue
            return row
        return None

    def list_nl_history(self, *, dataset_root_key: str, sample_identity: str | None = None) -> list[sqlite3.Row]:
        if sample_identity is None:
            return list(self.connection.execute(
                "SELECT * FROM nl_history WHERE dataset_root_key=? ORDER BY created_at DESC,history_id DESC",
                (dataset_root_key,),
            ))
        return list(self.connection.execute(
            "SELECT * FROM nl_history WHERE dataset_root_key=? AND sample_identity=? ORDER BY created_at DESC,history_id DESC",
            (dataset_root_key, sample_identity),
        ))

    def record_nl_outcome(self, job_id: str, *, succeeded: bool) -> tuple[int, int]:
        with self.transaction(immediate=True):
            self.connection.execute(
                "INSERT INTO nl_outcome_window(job_id,succeeded) VALUES (?,?)", (job_id, int(succeeded))
            )
            self.connection.execute(
                "DELETE FROM nl_outcome_window WHERE job_id=? AND outcome_id NOT IN (SELECT outcome_id FROM nl_outcome_window WHERE job_id=? ORDER BY outcome_id DESC LIMIT 100)",
                (job_id, job_id),
            )
            row = self.connection.execute(
                "SELECT COUNT(*) AS processed, COALESCE(SUM(1-succeeded),0) AS failed FROM nl_outcome_window WHERE job_id=?",
                (job_id,),
            ).fetchone()
        return int(row["processed"]), int(row["failed"])

    def record_nl_format_outcome(self, job_id: str, *, format_failed: bool) -> tuple[int, int]:
        """Record the independent 20-sample window used by the format breaker."""
        with self.transaction(immediate=True):
            self.connection.execute(
                "INSERT INTO nl_format_outcome_window(job_id,format_failed) VALUES (?,?)",
                (job_id, int(format_failed)),
            )
            self.connection.execute(
                """DELETE FROM nl_format_outcome_window
                   WHERE job_id=? AND outcome_id NOT IN (
                       SELECT outcome_id FROM nl_format_outcome_window
                       WHERE job_id=? ORDER BY outcome_id DESC LIMIT 20
                   )""",
                (job_id, job_id),
            )
            row = self.connection.execute(
                """SELECT COUNT(*) AS processed,
                          COALESCE(SUM(format_failed),0) AS failed
                   FROM nl_format_outcome_window WHERE job_id=?""",
                (job_id,),
            ).fetchone()
        return int(row["processed"]), int(row["failed"])

    def get_nl_api_runtime_override(self, job_id: str) -> dict[str, object] | None:
        """Return a pending connection-only API revision, if one is queued."""
        evidence = self.get_runtime_evidence(job_id)
        value = evidence.get("_nlApiOverride")
        return dict(value) if isinstance(value, dict) else None

    def get_nl_api_task_revision(self, job_id: str) -> int:
        """Return the last queued/applied task-local API revision counter."""
        evidence = self.get_runtime_evidence(job_id)
        pending = evidence.get("_nlApiOverride")
        if isinstance(pending, dict) and type(pending.get("taskRevision")) is int:
            return int(pending["taskRevision"])
        applied = evidence.get("_nlApiTaskRevision")
        return int(applied) if type(applied) is int and applied >= 0 else 0

    def set_nl_api_runtime_override(self, job_id: str, value: Mapping[str, object]) -> None:
        if not isinstance(value, Mapping):
            raise ValueError("NL API runtime override is invalid")
        current = self.get_runtime_evidence(job_id)
        current["_nlApiOverride"] = dict(value)
        serialized = json.dumps(current, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > 262_144:
            raise ValueError("runtime evidence is too large")
        result = self.connection.execute(
            "UPDATE jobs SET runtime_evidence_json=? WHERE job_id=?", (serialized, job_id),
        )
        if result.rowcount != 1:
            raise KeyError(f"job does not exist: {job_id}")

    def apply_nl_api_runtime_override(self, job_id: str, value: Mapping[str, object]) -> None:
        """Promote a hello-confirmed revision and remove its pending marker."""
        if not isinstance(value, Mapping):
            raise ValueError("NL API runtime is invalid")
        current = self.get_runtime_evidence(job_id)
        pending = current.get("_nlApiOverride")
        if isinstance(pending, dict) and pending.get("runtime") == dict(value):
            current["_nlApiRuntime"] = dict(value)
            task_revision = pending.get("taskRevision")
            if type(task_revision) is int and task_revision >= 0:
                current["_nlApiTaskRevision"] = task_revision
            current.pop("_nlApiOverride", None)
            serialized = json.dumps(current, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len(serialized.encode("utf-8")) > 262_144:
                raise ValueError("runtime evidence is too large")
            result = self.connection.execute(
                "UPDATE jobs SET runtime_evidence_json=? WHERE job_id=?", (serialized, job_id),
            )
            if result.rowcount != 1:
                raise KeyError(f"job does not exist: {job_id}")

    def add_api_budget_extra(self, job_id: str, amount: int) -> int:
        if type(amount) is not int or amount < 1:
            raise ValueError("API budget extra must be a positive integer")
        with self.transaction(immediate=True):
            self.connection.execute(
                "UPDATE jobs SET api_budget_extra=api_budget_extra+?,api_budget_revision=api_budget_revision+1 WHERE job_id=?",
                (amount, job_id),
            )
            row = get_job_row(self.connection, job_id)
        return int(row["api_budget_revision"])

    def stage_export_artifacts(
        self,
        job_id: str,
        sample_id: int,
        *,
        lease_id: str,
        artifacts: Mapping[str, tuple[str, str]],
    ) -> None:
        """Persist only verified Export artifact identities before completion."""
        if not artifacts or set(artifacts) - {"json", "txt"}:
            raise ValueError("export artifacts are invalid")
        for relative_path, digest in artifacts.values():
            if not isinstance(relative_path, str) or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("export artifact identity is invalid")
        with self.transaction(immediate=True):
            row = self.connection.execute(
                """SELECT 1 FROM sample_state WHERE job_id=? AND sample_id=?
                   AND current_module_id='export' AND status='leased' AND lease_id=?""",
                (job_id, sample_id, lease_id),
            ).fetchone()
            if row is None:
                raise ValueError("export artifacts do not belong to an active lease")
            self.connection.execute(
                "DELETE FROM export_artifacts WHERE job_id=? AND sample_id=?",
                (job_id, sample_id),
            )
            self.connection.executemany(
                """INSERT INTO export_artifacts(job_id,sample_id,kind,relative_path,sha256)
                   VALUES (?,?,?,?,?)""",
                [(job_id, sample_id, kind, relative_path, digest) for kind, (relative_path, digest) in artifacts.items()],
            )

    def page_export_artifacts(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        """Keyset-page manifest rows with their module-6 artifact identities."""
        limit = validate_page_limit(limit)
        predicate = "" if after_sample_id is None else " AND s.sample_id>?"
        parameters: list[Any] = [job_id]
        if after_sample_id is not None:
            parameters.append(after_sample_id)
        parameters.append(limit)
        return list(self.connection.execute(
            """SELECT s.sample_id,s.annotation_key,s.in_processing_scope,e.kind,e.relative_path,e.sha256
               FROM samples AS s JOIN export_artifacts AS e
                 ON e.job_id=s.job_id AND e.sample_id=s.sample_id
               WHERE s.job_id=?""" + predicate + " ORDER BY s.sample_id,e.kind LIMIT ?",
            parameters,
        ))

    def page_export_artifact_groups(
        self,
        job_id: str,
        *,
        after_sample_id: int | None = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        """Page whole sample groups so a JSON/TXT pair can never split pages.

        Samples explicitly skipped by Export own no artifact and must keep
        their existing annotation, so they never enter the commit projection.
        """
        limit = validate_page_limit(limit)
        repair_clause, repair_parameters = repair_target_clause(
            self.connection, job_id, "export",
        )
        predicate = "" if after_sample_id is None else " AND s.sample_id>?"
        parameters: list[Any] = [job_id]
        if after_sample_id is not None:
            parameters.append(after_sample_id)
        parameters.extend(repair_parameters)
        parameters.append(limit)
        return list(self.connection.execute(
            """WITH page AS (
                   SELECT s.sample_id AS sample_id,s.annotation_key AS annotation_key
                   FROM samples AS s JOIN sample_state AS st
                     ON st.job_id=s.job_id AND st.sample_id=s.sample_id
                   WHERE s.job_id=? AND s.in_processing_scope=1
                     AND NOT (st.current_module_id='export' AND st.status='skipped')"""
            + predicate + repair_clause + " ORDER BY s.sample_id LIMIT ?"
            + """ )
               SELECT page.sample_id,page.annotation_key,e.kind,e.relative_path,e.sha256
               FROM page LEFT JOIN export_artifacts AS e
                 ON e.job_id=? AND e.sample_id=page.sample_id
               ORDER BY page.sample_id,e.kind""",
            (*parameters, job_id),
        ))

    def mark_nl_request_started(self, job_id: str, sample_id: int, *, lease_id: str) -> None:
        result = self.connection.execute(
            """UPDATE sample_state SET status='request_started',updated_at=?
               WHERE job_id=? AND sample_id=? AND current_module_id='nl' AND status='leased' AND lease_id=?""",
            (utc_now(), job_id, sample_id, lease_id),
        )
        if result.rowcount != 1:
            raise ValueError("NL request does not belong to an active lease")

    def return_unsubmitted_nl_request(self, job_id: str, sample_id: int, *, lease_id: str) -> None:
        result = self.connection.execute(
            """UPDATE sample_state SET status='pending',lease_id=NULL,worker_instance_id=NULL,lease_expires_at=NULL,updated_at=?
               WHERE job_id=? AND sample_id=? AND current_module_id='nl' AND status='request_started' AND lease_id=?""",
            (utc_now(), job_id, sample_id, lease_id),
        )
        if result.rowcount != 1:
            raise ValueError("unsubmitted NL request does not belong to an active lease")

    def confirm_nl_unknown_requests(self, job_id: str, *, limit: int = MAX_PAGE_SIZE) -> int:
        """Return only crash-uncertain API requests after explicit user confirmation."""
        limit = validate_page_limit(limit)
        with self.transaction(immediate=True):
            rows = list(self.connection.execute(
                """SELECT sample_id FROM sample_state WHERE job_id=? AND current_module_id='nl'
                   AND status='request_started' ORDER BY sample_id LIMIT ?""",
                (job_id, limit),
            ))
            if not rows:
                return 0
            ids = [int(row["sample_id"]) for row in rows]
            placeholders = ",".join("?" for _ in ids)
            self.connection.execute(
                f"""UPDATE sample_state SET status='pending',lease_id=NULL,worker_instance_id=NULL,
                    lease_expires_at=NULL,updated_at=? WHERE job_id=? AND sample_id IN ({placeholders})
                    AND current_module_id='nl' AND status='request_started'""",
                (utc_now(), job_id, *ids),
            )
        return len(ids)

    def stage_nl_response(
        self,
        job_id: str,
        sample_id: int,
        *,
        lease_id: str,
        nl: str,
        sha256: str,
        observation: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(nl, str) or len(nl.encode("utf-8")) > 16_384 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("staged NL is invalid")
        job = get_job_row(self.connection, job_id)
        config_schema_version = int(job["config_schema_version"])
        if config_schema_version in COUNT_REVIEW_SCHEMA_VERSIONS:
            from .count_review_protocol import CountObservationV1

            parsed_observation = CountObservationV1.from_dict(observation)
            if parsed_observation.status not in {"observed", "invalid"}:
                raise ValueError("requested NL response must contain an observed or invalid observation")
        elif config_schema_version == 2:
            if observation is not None:
                raise ValueError("legacy NL response must not contain a count observation")
            parsed_observation = None
        else:
            raise ValueError("NL task configuration schema is invalid")
        with self.transaction(immediate=True):
            now = utc_now()
            result = self.connection.execute(
                """UPDATE sample_state SET status='response_staged',updated_at=?
                   WHERE job_id=? AND sample_id=? AND current_module_id='nl' AND status='request_started' AND lease_id=?""",
                (now, job_id, sample_id, lease_id),
            )
            if result.rowcount != 1:
                raise ValueError("NL response does not belong to a started request")
            if parsed_observation is not None:
                insert_count_observation_consistent(
                    self.connection, job_id, sample_id, parsed_observation, now=now
                )
            self.connection.execute(
                """INSERT INTO staged_nl(job_id,sample_id,lease_id,nl,sha256,staged_at) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(job_id,sample_id) DO UPDATE SET lease_id=excluded.lease_id,nl=excluded.nl,sha256=excluded.sha256,staged_at=excluded.staged_at""",
                (job_id, sample_id, lease_id, nl, sha256, now),
            )

    def staged_nl(self, job_id: str, sample_id: int, *, lease_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM staged_nl WHERE job_id=? AND sample_id=? AND lease_id=?", (job_id, sample_id, lease_id),
        ).fetchone()
        if row is None:
            raise KeyError("staged NL does not exist for the current lease")
        return row

    def delete_staged_nl(self, job_id: str, sample_id: int, *, lease_id: str) -> None:
        result = self.connection.execute(
            "DELETE FROM staged_nl WHERE job_id=? AND sample_id=? AND lease_id=?", (job_id, sample_id, lease_id),
        )
        if result.rowcount != 1:
            raise ValueError("staged NL does not belong to the current lease")
