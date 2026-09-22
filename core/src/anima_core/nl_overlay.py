from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from .classify_overlay import ClassifyJsonError, parse_annotation_json, serialize_annotation_json
from .db import StateDatabase
from .overlay import OverlayError, OverlayLayout, WorkingAnnotationView
from .path_safety import safe_relative_path, sha256_file, windows_paths_equal


SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class NlOverlayWriter:
    database: StateDatabase
    layout: OverlayLayout
    view: WorkingAnnotationView
    job_id: str

    def __post_init__(self) -> None:
        job = self.database.get_job(self.job_id)
        if self.layout.job_id != self.job_id or not windows_paths_equal(self.layout.dataset_root, str(job["dataset_root"])):
            raise OverlayError("NL overlay does not match the immutable job")
        if not isinstance(job["overlay_root"], str) or not windows_paths_equal(self.layout.root, str(job["overlay_root"])):
            raise OverlayError("NL overlay root does not match the persisted job")

    def commit_staged(self, job_id: str, sample_id: int, lease_id: str) -> bool:
        if job_id != self.job_id:
            raise OverlayError("NL staged response belongs to another job")
        row = self.database.get_sample_with_state(job_id, sample_id)
        if row["current_module_id"] != "nl" or row["status"] != "response_staged" or row["lease_id"] != lease_id:
            raise OverlayError("NL staged response does not belong to the active lease")
        staged = self.database.staged_nl(job_id, sample_id, lease_id=lease_id)
        if hashlib.sha256(str(staged["nl"]).encode("utf-8")).hexdigest() != staged["sha256"]:
            raise OverlayError("staged NL digest does not match")
        try:
            projection = parse_annotation_json(self.view.read(str(row["annotation_key"]), ".json"))
        except ClassifyJsonError as exc:
            raise OverlayError("working JSON is invalid while committing staged NL") from exc
        if projection is None:
            raise OverlayError("working JSON is missing while committing staged NL")
        self.write_value(sample_id=sample_id, lease_id=lease_id, annotation_key=str(row["annotation_key"]), projection=projection, nl=str(staged["nl"]), allowed_statuses=("response_staged",), history_source="api")
        self.database.delete_staged_nl(job_id, sample_id, lease_id=lease_id)
        return True

    def write_value(
        self, *, sample_id: int, lease_id: str, annotation_key: str, projection: dict[str, object], nl: str,
        allowed_statuses: tuple[str, ...] = ("leased",), history_source: str = "api",
    ) -> None:
        if not isinstance(nl, str) or len(nl.encode("utf-8")) > 16_384:
            raise OverlayError("NL output exceeds its limit")
        previous = projection.get("nl")
        if isinstance(previous, str) and previous.strip() and previous != nl:
            source = "original_json" if self.database.get_sample_with_state(self.job_id, sample_id)["original_json_state"] == "nonblank" else "nl_txt" if self.database.get_sample_with_state(self.job_id, sample_id)["original_txt_state"] == "nonblank" else "unknown"
            self.database.archive_nl_for_sample(self.job_id, sample_id, source=source, text=previous)
        result = dict(projection)
        result["nl"] = nl
        data = serialize_annotation_json(result)
        prepared, digest = self.layout.write_prepared("nl", lease_id, ".json", data)
        relative = os.path.relpath(prepared, self.layout.root).replace("/", "\\")
        self.database.stage_prepared_artifact(
            self.job_id, sample_id, lease_id=lease_id, relative_path=relative, sha256=digest,
            allowed_statuses=allowed_statuses,
        )
        # Persist the result before the filesystem replacement.  Recovery can
        # safely repeat this idempotent archive if a process stops between
        # staging and the replacement.
        if nl.strip():
            self.database.archive_nl_for_sample(self.job_id, sample_id, source=history_source, text=nl)
        self.layout.commit_prepared(relative, digest, annotation_key, ".json")

    def recover_prepared(self, job_id: str, sample_id: int, prepared_relative_path: str, expected_sha256: str) -> bool:
        if job_id != self.job_id or not SHA256.fullmatch(expected_sha256):
            raise OverlayError("NL recovery identity is invalid")
        row = self.database.get_sample_with_state(job_id, sample_id)
        if row["current_module_id"] != "nl" or row["status"] != "prepared":
            raise OverlayError("NL recovery state is not prepared")
        if not isinstance(row["prepared_artifact_relative_path"], str) or safe_relative_path(str(row["prepared_artifact_relative_path"])) != safe_relative_path(prepared_relative_path) or row["prepared_artifact_sha256"] != expected_sha256:
            raise OverlayError("NL recovery metadata does not match SQLite")
        target = self.layout.annotation_path(str(row["annotation_key"]), ".json")
        prepared = self.layout.resolve_prepared(prepared_relative_path)
        if target.is_file() and sha256_file(target) == expected_sha256:
            if prepared.is_file() and sha256_file(prepared) == expected_sha256:
                prepared.unlink()
            return True
        if not prepared.is_file() or sha256_file(prepared) != expected_sha256:
            return False
        try:
            recovered = parse_annotation_json(prepared.read_bytes())
        except (OSError, ClassifyJsonError):
            recovered = None
        if recovered is not None and isinstance(recovered.get("nl"), str) and recovered["nl"].strip():
            source = "manual" if str(row["lease_id"] or "").startswith("manual-nl-") else "api"
            self.database.archive_nl_for_sample(job_id, sample_id, source=source, text=str(recovered["nl"]))
        return sha256_file(self.layout.commit_prepared(prepared_relative_path, expected_sha256, str(row["annotation_key"]), ".json")) == expected_sha256
