from __future__ import annotations

import hashlib
import uuid

from .classify_overlay import ClassifyJsonError, parse_annotation_json
from .contracts import ProgressEvent, SampleRunState, utc_now
from .db import StateDatabase
from .nl_overlay import NlOverlayWriter
from .overlay import BaselineView, OverlayLayout, WorkingAnnotationView


class NlManualWriteService:
    """Seed a confirmed NL edit into a prepared NL repair child."""

    def __init__(self, database: StateDatabase, job_id: str) -> None:
        self.database = database
        self.job_id = job_id

    @staticmethod
    def validate_text(nl: str) -> bytes:
        try:
            encoded = nl.encode("utf-8")
        except (AttributeError, UnicodeEncodeError) as exc:
            raise ValueError("manual NL text must be valid UTF-8") from exc
        if not nl.strip():
            raise ValueError("manual NL text must not be blank")
        if len(encoded) > 16_384:
            raise ValueError("manual NL text exceeds 16 KiB")
        return encoded

    def _parent_issue(self, *, sample_id: int, issue_id: str | None) -> object:
        parent_job_id = self.database.repair_parent_job_id(self.job_id)
        if parent_job_id is None:
            raise ValueError("manual NL write requires a repair child")
        if issue_id is not None:
            row = self.database.connection.execute(
                "SELECT * FROM issues WHERE job_id=? AND issue_id=? AND resolved_at IS NULL",
                (parent_job_id, issue_id),
            ).fetchone()
        else:
            row = self.database.connection.execute(
                """SELECT * FROM issues WHERE job_id=? AND sample_id=? AND module_id='nl'
                   AND resolved_at IS NULL ORDER BY issue_id""",
                (parent_job_id, sample_id),
            ).fetchone()
        if row is None or row["module_id"] != "nl" or int(row["sample_id"]) != sample_id:
            raise ValueError("manual NL write issue identity is invalid")
        return row

    def seed(self, *, sample_id: int, issue_id: str | None, nl: str, confirmed: bool) -> dict[str, object]:
        if not confirmed:
            raise ValueError("manual NL write requires explicit confirmation")
        encoded = self.validate_text(nl)
        job = self.database.get_job(self.job_id)
        if job["status"] != "preparing_workspace" or job["current_module_id"] != "workspace":
            raise ValueError("manual NL write requires a prepared repair child")
        issue = self._parent_issue(sample_id=sample_id, issue_id=issue_id)
        target = self.database.connection.execute(
            "SELECT repair_start_module FROM repair_targets WHERE repair_job_id=? AND sample_id=?",
            (self.job_id, sample_id),
        ).fetchone()
        if target is None or target["repair_start_module"] != "nl":
            raise ValueError("manual NL write sample is not an NL repair target")
        row = self.database.get_sample_with_state(self.job_id, sample_id)
        if row["status"] != "pending":
            raise ValueError("manual NL write sample is not pending")
        overlay_root = job["overlay_root"]
        if not isinstance(overlay_root, str) or not overlay_root:
            raise ValueError("manual NL write requires an annotation overlay")
        layout = OverlayLayout.open_existing(overlay_root, self.job_id)
        view = WorkingAnnotationView(BaselineView(layout.dataset_root), layout)
        try:
            projection = parse_annotation_json(view.read(str(row["annotation_key"]), ".json"))
        except ClassifyJsonError as exc:
            raise ValueError("manual NL write requires valid working JSON") from exc
        if projection is None:
            raise ValueError("manual NL write requires existing working JSON")
        lease_id = f"manual-nl-{uuid.uuid4().hex}"
        before = SampleRunState(
            sampleId=sample_id,
            txtProvenance=row["txt_provenance"],
            currentModuleId=row["current_module_id"],
            status=row["status"],
            attempt=int(row["attempt"]),
            leaseId=row["lease_id"],
            workerInstanceId=row["worker_instance_id"],
            leaseExpiresAt=row["lease_expires_at"],
            preparedArtifactRelativePath=row["prepared_artifact_relative_path"],
            preparedArtifactSha256=row["prepared_artifact_sha256"],
        )
        self.database.set_sample_state(
            self.job_id,
            sample_id,
            SampleRunState(
                sampleId=sample_id,
                txtProvenance=row["txt_provenance"],
                currentModuleId="nl",
                status="leased",
                attempt=int(row["attempt"]) + 1,
                leaseId=lease_id,
                workerInstanceId="manual-review",
                leaseExpiresAt=utc_now(),
            ),
        )
        try:
            NlOverlayWriter(self.database, layout, view, self.job_id).write_value(
                sample_id=sample_id,
                lease_id=lease_id,
                annotation_key=str(row["annotation_key"]),
                projection=projection,
                nl=nl,
                history_source="manual",
            )
            self.database.complete_leased_sample(
                self.job_id, sample_id, lease_id=lease_id, allowed_statuses=("prepared",),
            )
            self.database.append_event(ProgressEvent(
                self.job_id,
                int(job["last_event_id"]) + 1,
                "nl",
                "completed",
                1,
                int(job["sample_count"]),
                str(job["config_hash"]),
                int(row["attempt"]) + 1,
                sampleId=sample_id,
                issueCode=str(issue["code"]),
                message=f"manual NL write sha256={hashlib.sha256(encoded).hexdigest()}",
            ))
        except Exception:
            self.database.set_sample_state(self.job_id, sample_id, before)
            raise
        return {"jobId": self.job_id, "sampleId": sample_id, "issueId": str(issue["issue_id"]), "written": True}
