"""Creation of isolated, bounded issue-repair tasks."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from .contracts import JobConfig, pipeline_module_ids, utc_now
from .custom_classification_resource import (
    CustomClassificationResourceError,
    freeze_custom_classification_resource,
    inspect_custom_classification_resource,
)
from .db import MAX_PAGE_SIZE, StateDatabase
from .job_preflight import config_from_dict
from .locks import DatasetLock, release_completed_dataset_lock
from .overlay import BaselineView, OverlayError, OverlayLayout, WorkingAnnotationView
from .path_safety import PathSafetyError, ensure_within, windows_key


class RepairPreparationError(ValueError):
    pass


@dataclass(frozen=True)
class RepairPreparationResult:
    repairJobId: str
    parentJobId: str
    targetCount: int
    datasetRoot: str
    overlayRoot: str


class RepairPreparationService:
    """Creates a new task state without copying the complete dataset again."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._locks: dict[str, DatasetLock] = {}

    @staticmethod
    def _job_row(repair_job_id: str, parent: object, dataset: Path) -> dict[str, object]:
        return {
            "job_id": repair_job_id,
            "config_schema_version": parent["config_schema_version"], "config_json": parent["config_json"],
            "config_hash": parent["config_hash"],
            "work_mode": parent["work_mode"], "overwrite_mode": parent["overwrite_mode"],
            "source_root": str(dataset), "output_root": parent["output_root"], "dataset_root": str(dataset),
            "dataset_root_key": windows_key(dataset), "manifest_schema_version": 1, "recursive": parent["recursive"],
            "sample_count": 0, "manifest_generated_at": None, "status": "preflighting", "current_module_id": None,
            "last_event_id": 0, "pinned": 0, "api_budget_extra": 0, "api_budget_revision": 0,
            "overlay_root": None, "commit_journal_path": None, "resume_status": None,
            "created_at": utc_now(), "started_at": None, "cancel_requested_at": None,
            "finished_at": None,
        }

    @staticmethod
    def _copy_parent_manifest(database: StateDatabase, parent_job_id: str, repair_job_id: str) -> int:
        """Reuse the parent's frozen manifest instead of decoding the dataset again.

        A repair task runs against the same locked dataset, so re-running the
        full image preflight would cost two opens per image for every sample.
        """
        cursor: int | None = None
        count = 0
        while True:
            page = database.page_samples(parent_job_id, after_sample_id=cursor, limit=500)
            if not page:
                break
            cursor = int(page[-1]["sample_id"])
            count += database.insert_samples(repair_job_id, [dict(row) for row in page], batch_size=500)
        database.connection.execute(
            "UPDATE jobs SET sample_count=?,manifest_generated_at=? WHERE job_id=?",
            (count, utc_now(), repair_job_id),
        )
        return count

    @staticmethod
    def _inherit_custom_classification_resource(
        parent: object,
        config: JobConfig,
        child_layout: OverlayLayout,
    ) -> None:
        if config.classify.get("indexMode") != "custom":
            return
        parent_overlay = parent["overlay_root"]
        manifest_relative = config.classify.get("resourceManifestRelativePath")
        fingerprint = config.classify.get("resourceFingerprint")
        if not all(isinstance(value, str) and value for value in (
            parent_overlay, manifest_relative, fingerprint,
        )):
            raise RepairPreparationError("parent frozen custom classification reference is invalid")
        try:
            parent_layout = OverlayLayout.open_existing(str(parent_overlay), str(parent["job_id"]))
            inspected = inspect_custom_classification_resource(
                parent_layout.resource_path(str(manifest_relative)),
            )
            if inspected.package.fingerprint != fingerprint:
                raise CustomClassificationResourceError(
                    "parent frozen classification resource fingerprint changed"
                )
            inherited_relative, inherited = freeze_custom_classification_resource(
                child_layout, inspected,
            )
            if inherited_relative != manifest_relative or inherited.fingerprint != fingerprint:
                raise CustomClassificationResourceError(
                    "repair classification resource identity changed"
                )
        except (CustomClassificationResourceError, OSError, ValueError) as exc:
            raise RepairPreparationError(
                "parent frozen custom classification resource is invalid"
            ) from exc

    @staticmethod
    def _inherit_target_annotations(
        database: StateDatabase,
        parent: object,
        parent_job_id: str,
        repair_job_id: str,
        dataset: Path,
        child_layout: OverlayLayout,
    ) -> None:
        parent_overlay = parent["overlay_root"]
        if not isinstance(parent_overlay, str) or not parent_overlay:
            raise RepairPreparationError("parent annotation overlay is unavailable")
        try:
            parent_layout = OverlayLayout.open_existing(parent_overlay, parent_job_id)
            parent_view = WorkingAnnotationView(BaselineView(dataset), parent_layout)
            rows = database.connection.execute(
                """SELECT s.sample_id,s.annotation_key
                   FROM repair_targets AS target
                   JOIN samples AS s ON s.job_id=target.repair_job_id AND s.sample_id=target.sample_id
                  WHERE target.repair_job_id=? ORDER BY s.sample_id""",
                (repair_job_id,),
            )
            for row in rows:
                raw = parent_view.read(str(row["annotation_key"]), ".json")
                if raw is not None:
                    child_layout.write_annotation(str(row["annotation_key"]), ".json", raw)
        except (OSError, OverlayError, ValueError) as exc:
            raise RepairPreparationError("parent target annotations are unavailable") from exc

    def prepare(self, parent_job_id: str) -> RepairPreparationResult:
        database = StateDatabase.open(self.database_path)
        layout: OverlayLayout | None = None
        acquired: DatasetLock | None = None
        retain_database = False
        repair_job_id = uuid.uuid4().hex
        try:
            parent = database.get_job(parent_job_id)
            if parent["status"] not in {"reviewing", "failed"}:
                raise RepairPreparationError("only a reviewed or failed task can create a repair task")
            try:
                config = config_from_dict(json.loads(str(parent["config_json"])))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RepairPreparationError("parent frozen JobConfig is invalid") from exc
            if (
                config.config_hash != parent["config_hash"]
                or config.schemaVersion != int(parent["config_schema_version"])
            ):
                raise RepairPreparationError("parent frozen JobConfig identity does not match")
            dataset = Path(str(parent["dataset_root"]))
            database.insert_job(self._job_row(repair_job_id, parent, dataset))
            acquired = DatasetLock.acquire(database, dataset, repair_job_id)
            try:
                self._copy_parent_manifest(database, parent_job_id, repair_job_id)
            except (OSError, ValueError) as exc:
                raise RepairPreparationError("current formal dataset cannot produce a repair manifest") from exc
            database.create_repair_link(repair_job_id, parent_job_id)
            cursor: int | None = None
            while True:
                page = database.stage_repair_target_page(repair_job_id, parent_job_id, after_sample_id=cursor, limit=500)
                if not page:
                    break
                cursor = database.repair_parent_cursor(repair_job_id)
            target_count = database.count_repair_targets(repair_job_id)
            if target_count == 0:
                raise RepairPreparationError("task has no eligible retriable sample issues")
            if isinstance(config.countReview, dict) and config.countReview.get("enabled") is True:
                try:
                    database.inherit_repair_count_evidence(repair_job_id, parent_job_id)
                except ValueError as exc:
                    raise RepairPreparationError(str(exc)) from exc
            layout = OverlayLayout.create(dataset, repair_job_id)
            self._inherit_custom_classification_resource(parent, config, layout)
            self._inherit_target_annotations(database, parent, parent_job_id, repair_job_id, dataset, layout)
            database.set_workspace_metadata(repair_job_id, dataset_root=str(dataset), dataset_root_key=windows_key(dataset), overlay_root=str(layout.root))
            database.set_job_status(repair_job_id, "preparing_workspace", current_module_id="workspace")
            self._locks[repair_job_id] = acquired
            acquired = None
            retain_database = True
            return RepairPreparationResult(repair_job_id, parent_job_id, target_count, str(dataset), str(layout.root))
        except Exception:
            if layout is not None and layout.root.exists():
                layout.discard()
            try:
                database.delete_job_control_record(repair_job_id)
            except KeyError:
                pass
            if acquired is not None:
                acquired.release(recovery_complete=True)
            raise
        finally:
            if not retain_database:
                database.close()

    def prepare_manual_nl(
        self,
        parent_job_id: str,
        *,
        sample_id: int | None = None,
        issue_id: str | None = None,
        confirmed: bool,
        for_manual_write: bool = False,
    ) -> RepairPreparationResult:
        """Create a one-sample NL repair only after an explicit user choice.

        This path deliberately does not widen the normal repair candidate query:
        callers must identify one unresolved, non-retriable NL issue.
        """
        if not confirmed:
            raise RepairPreparationError("manual NL retry requires explicit confirmation")
        if (sample_id is None) == (issue_id is None):
            raise RepairPreparationError("manual NL retry requires exactly one sampleId or issueId")
        database = StateDatabase.open(self.database_path)
        layout: OverlayLayout | None = None
        acquired: DatasetLock | None = None
        retain_database = False
        repair_job_id = uuid.uuid4().hex
        try:
            parent = database.get_job(parent_job_id)
            if parent["status"] not in {"reviewing", "failed", "succeeded"}:
                raise RepairPreparationError("manual NL retry requires a reviewed NL task")
            if issue_id is not None:
                issue = database.connection.execute(
                    "SELECT * FROM issues WHERE job_id=? AND issue_id=? AND resolved_at IS NULL",
                    (parent_job_id, issue_id),
                ).fetchone()
            else:
                issue = database.connection.execute(
                    """SELECT * FROM issues WHERE job_id=? AND sample_id=? AND module_id='nl'
                       AND resolved_at IS NULL AND retriable=0 ORDER BY issue_id""",
                    (parent_job_id, sample_id),
                ).fetchone()
            if issue is None or issue["module_id"] != "nl" or bool(issue["retriable"]):
                raise RepairPreparationError("manual NL retry requires one unresolved non-retriable NL issue")
            selected_sample_id = int(issue["sample_id"])
            if sample_id is not None and selected_sample_id != sample_id:
                raise RepairPreparationError("sampleId does not match issueId")
            try:
                config = config_from_dict(json.loads(str(parent["config_json"])))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RepairPreparationError("parent frozen JobConfig is invalid") from exc
            if (
                config.config_hash != parent["config_hash"]
                or config.schemaVersion != int(parent["config_schema_version"])
            ):
                raise RepairPreparationError("parent frozen JobConfig identity does not match")
            try:
                order = pipeline_module_ids(config.schemaVersion)
                current_index = order.index(str(parent["current_module_id"]))
                nl_index = order.index("nl")
            except ValueError as exc:
                raise RepairPreparationError("manual NL retry requires a review stage at or after NL") from exc
            if current_index < nl_index:
                raise RepairPreparationError("manual NL retry requires a review stage at or after NL")
            sample = database.get_sample_with_state(parent_job_id, selected_sample_id)
            dataset = Path(str(parent["dataset_root"]))
            if not for_manual_write:
                nl = config.nl
                if nl.get("enabled") is not True or nl.get("apiEnabled") is not True or nl.get("useImage") is not True:
                    raise RepairPreparationError("manual NL retry requires image-enabled API NL configuration")
                policy = nl.get("apiPolicy")
                if not isinstance(policy, dict) or type(policy.get("maxHttpAttempts")) is not int:
                    raise RepairPreparationError("frozen NL API budget is invalid")
                used = database.module_diagnostic_count(parent_job_id, "nl", "nl_http_attempts")
                if used >= int(policy["maxHttpAttempts"]) + int(parent["api_budget_extra"]):
                    raise RepairPreparationError("NL API budget is exhausted")
                try:
                    image = ensure_within(dataset, dataset / str(sample["relative_image_path"]))
                    with image.open("rb") as stream:
                        if not stream.read(1):
                            raise OSError("image is empty")
                except (OSError, PathSafetyError) as exc:
                    raise RepairPreparationError("manual NL retry image is missing or unreadable") from exc
            database.insert_job(self._job_row(repair_job_id, parent, dataset))
            acquired = DatasetLock.acquire(database, dataset, repair_job_id)
            try:
                self._copy_parent_manifest(database, parent_job_id, repair_job_id)
                database.create_repair_link(repair_job_id, parent_job_id)
                child_sample = database.connection.execute(
                    "SELECT sample_id FROM samples WHERE job_id=? AND relative_image_path=?",
                    (repair_job_id, sample["relative_image_path"]),
                ).fetchone()
                if child_sample is None:
                    raise RepairPreparationError("manual NL retry sample is not present in the child manifest")
                child_sample_id = int(child_sample["sample_id"])
                database.connection.execute(
                    "INSERT INTO repair_targets(repair_job_id,sample_id,repair_start_module) VALUES (?,?,?)",
                    (repair_job_id, child_sample_id, "nl"),
                )
                database.connection.execute(
                    "INSERT INTO repair_target_issues(repair_job_id,sample_id,parent_issue_id) VALUES (?,?,?)",
                    (repair_job_id, child_sample_id, str(issue["issue_id"])),
                )
                if isinstance(config.countReview, dict) and config.countReview.get("enabled") is True:
                    database.inherit_repair_count_evidence(repair_job_id, parent_job_id)
                layout = OverlayLayout.create(dataset, repair_job_id)
                self._inherit_custom_classification_resource(parent, config, layout)
                parent_overlay = parent["overlay_root"]
                if isinstance(parent_overlay, str) and parent_overlay:
                    parent_layout = OverlayLayout.open_existing(parent_overlay, parent_job_id)
                    raw_annotation = WorkingAnnotationView(
                        BaselineView(dataset), parent_layout,
                    ).read(str(sample["annotation_key"]), ".json")
                    if raw_annotation is not None:
                        layout.write_annotation(str(sample["annotation_key"]), ".json", raw_annotation)
                database.set_workspace_metadata(
                    repair_job_id, dataset_root=str(dataset), dataset_root_key=windows_key(dataset), overlay_root=str(layout.root),
                )
                database.set_job_status(repair_job_id, "preparing_workspace", current_module_id="workspace")
                self._locks[repair_job_id] = acquired
                acquired = None
                retain_database = True
                return RepairPreparationResult(repair_job_id, parent_job_id, 1, str(dataset), str(layout.root))
            except Exception:
                raise
        except Exception:
            if layout is not None and layout.root.exists():
                layout.discard()
            try:
                database.delete_job_control_record(repair_job_id)
            except KeyError:
                pass
            if acquired is not None:
                acquired.release(recovery_complete=True)
            raise
        finally:
            if not retain_database:
                database.close()

    def prepare_manual_nl_batch(
        self,
        parent_job_id: str,
        *,
        issue_ids: tuple[str, ...] | list[str],
        confirmed: bool,
    ) -> RepairPreparationResult:
        """Create one NL repair task for a bounded selection of unresolved NL issues."""
        requested_issue_ids = tuple(issue_ids)
        if not confirmed:
            raise RepairPreparationError("manual NL retry requires explicit confirmation")
        if (
            not requested_issue_ids
            or len(requested_issue_ids) > MAX_PAGE_SIZE
            or len(set(requested_issue_ids)) != len(requested_issue_ids)
            or any(not isinstance(issue_id, str) or not 1 <= len(issue_id) <= 128 for issue_id in requested_issue_ids)
        ):
            raise RepairPreparationError("manual NL retry requires a unique current-page issue selection")
        database = StateDatabase.open(self.database_path)
        layout: OverlayLayout | None = None
        acquired: DatasetLock | None = None
        retain_database = False
        repair_job_id = uuid.uuid4().hex
        try:
            parent = database.get_job(parent_job_id)
            if parent["status"] not in {"reviewing", "failed", "succeeded"}:
                raise RepairPreparationError("manual NL retry requires a reviewed NL task")
            placeholders = ",".join("?" for _ in requested_issue_ids)
            found = list(database.connection.execute(
                f"""SELECT * FROM issues WHERE job_id=? AND resolved_at IS NULL AND issue_id IN ({placeholders})
                    ORDER BY sample_id,issue_id""",
                (parent_job_id, *requested_issue_ids),
            ))
            issues_by_id = {str(issue["issue_id"]): issue for issue in found}
            if len(issues_by_id) != len(requested_issue_ids):
                raise RepairPreparationError("manual NL retry requires unresolved current-page NL issues")
            issues = [issues_by_id[issue_id] for issue_id in requested_issue_ids]
            if any(issue["module_id"] != "nl" for issue in issues):
                raise RepairPreparationError("manual NL retry requires unresolved NL issues")
            selected_sample_ids = [int(issue["sample_id"]) for issue in issues]
            if len(set(selected_sample_ids)) != len(selected_sample_ids):
                raise RepairPreparationError("manual NL retry requires at most one issue per sample")
            try:
                config = config_from_dict(json.loads(str(parent["config_json"])))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RepairPreparationError("parent frozen JobConfig is invalid") from exc
            if (
                config.config_hash != parent["config_hash"]
                or config.schemaVersion != int(parent["config_schema_version"])
            ):
                raise RepairPreparationError("parent frozen JobConfig identity does not match")
            try:
                order = pipeline_module_ids(config.schemaVersion)
                current_index = order.index(str(parent["current_module_id"]))
                nl_index = order.index("nl")
            except ValueError as exc:
                raise RepairPreparationError("manual NL retry requires a review stage at or after NL") from exc
            if current_index < nl_index:
                raise RepairPreparationError("manual NL retry requires a review stage at or after NL")
            nl = config.nl
            if nl.get("enabled") is not True or nl.get("apiEnabled") is not True or nl.get("useImage") is not True:
                raise RepairPreparationError("manual NL retry requires image-enabled API NL configuration")
            policy = nl.get("apiPolicy")
            if not isinstance(policy, dict) or type(policy.get("maxHttpAttempts")) is not int:
                raise RepairPreparationError("frozen NL API budget is invalid")
            used = database.module_diagnostic_count(parent_job_id, "nl", "nl_http_attempts")
            if used >= int(policy["maxHttpAttempts"]) + int(parent["api_budget_extra"]):
                raise RepairPreparationError("NL API budget is exhausted")
            dataset = Path(str(parent["dataset_root"]))
            samples = [database.get_sample_with_state(parent_job_id, sample_id) for sample_id in selected_sample_ids]
            for sample in samples:
                try:
                    image = ensure_within(dataset, dataset / str(sample["relative_image_path"]))
                    with image.open("rb") as stream:
                        if not stream.read(1):
                            raise OSError("image is empty")
                except (OSError, PathSafetyError) as exc:
                    raise RepairPreparationError("manual NL retry image is missing or unreadable") from exc
            database.insert_job(self._job_row(repair_job_id, parent, dataset))
            acquired = DatasetLock.acquire(database, dataset, repair_job_id)
            self._copy_parent_manifest(database, parent_job_id, repair_job_id)
            database.create_repair_link(repair_job_id, parent_job_id)
            relative_paths = [str(sample["relative_image_path"]) for sample in samples]
            child_placeholders = ",".join("?" for _ in relative_paths)
            child_samples = {
                str(row["relative_image_path"]): int(row["sample_id"])
                for row in database.connection.execute(
                    f"SELECT sample_id,relative_image_path FROM samples WHERE job_id=? AND relative_image_path IN ({child_placeholders})",
                    (repair_job_id, *relative_paths),
                )
            }
            targets: list[tuple[int, str]] = []
            for issue, sample in zip(issues, samples, strict=True):
                child_sample_id = child_samples.get(str(sample["relative_image_path"]))
                if child_sample_id is None:
                    raise RepairPreparationError("manual NL retry sample is not present in the child manifest")
                targets.append((child_sample_id, str(issue["issue_id"])))
            database.connection.executemany(
                "INSERT INTO repair_targets(repair_job_id,sample_id,repair_start_module) VALUES (?,?,?)",
                [(repair_job_id, sample_id, "nl") for sample_id, _ in targets],
            )
            database.connection.executemany(
                "INSERT INTO repair_target_issues(repair_job_id,sample_id,parent_issue_id) VALUES (?,?,?)",
                [(repair_job_id, sample_id, issue_id) for sample_id, issue_id in targets],
            )
            if isinstance(config.countReview, dict) and config.countReview.get("enabled") is True:
                database.inherit_repair_count_evidence(repair_job_id, parent_job_id)
            layout = OverlayLayout.create(dataset, repair_job_id)
            self._inherit_custom_classification_resource(parent, config, layout)
            parent_overlay = parent["overlay_root"]
            if isinstance(parent_overlay, str) and parent_overlay:
                parent_layout = OverlayLayout.open_existing(parent_overlay, parent_job_id)
                parent_view = WorkingAnnotationView(BaselineView(dataset), parent_layout)
                for sample in samples:
                    raw_annotation = parent_view.read(str(sample["annotation_key"]), ".json")
                    if raw_annotation is not None:
                        layout.write_annotation(str(sample["annotation_key"]), ".json", raw_annotation)
            database.set_workspace_metadata(
                repair_job_id, dataset_root=str(dataset), dataset_root_key=windows_key(dataset), overlay_root=str(layout.root),
            )
            database.set_job_status(repair_job_id, "preparing_workspace", current_module_id="workspace")
            self._locks[repair_job_id] = acquired
            acquired = None
            retain_database = True
            return RepairPreparationResult(repair_job_id, parent_job_id, len(targets), str(dataset), str(layout.root))
        except Exception:
            if layout is not None and layout.root.exists():
                layout.discard()
            try:
                database.delete_job_control_record(repair_job_id)
            except KeyError:
                pass
            if acquired is not None:
                acquired.release(recovery_complete=True)
            raise
        finally:
            if not retain_database:
                database.close()

    def close(self) -> None:
        for lock in tuple(self._locks.values()):
            try:
                lock.release(recovery_complete=True)
            finally:
                lock.database.close()
        self._locks.clear()

    def release_lock_for_history_deletion(self, job_id: str) -> bool:
        return release_completed_dataset_lock(self._locks, self.database_path, job_id)

    def release_lock_for_repair(self, job_id: str) -> bool:
        """Release a repair task lock while its parent is being repaired."""
        lock = self._locks.get(job_id)
        if lock is None:
            return False
        job = lock.database.get_job(job_id)
        if job["status"] not in {"reviewing", "failed", "succeeded"}:
            raise RepairPreparationError(
                "only a reviewed, failed, or succeeded task can release its lock for repair"
            )
        overlay_root = job["overlay_root"]
        if isinstance(overlay_root, str) and overlay_root:
            try:
                state = OverlayLayout.open_existing(overlay_root, job_id).journal_state()
            except Exception as exc:
                raise RepairPreparationError("task commit journal cannot be safely inspected") from exc
            if state not in {None, "committed", "rolled_back"}:
                raise RepairPreparationError("task lock cannot be released before commit journal recovery")
        lock.release(recovery_complete=True)
        lock.database.close()
        del self._locks[job_id]
        return True

    def restore_lock_after_repair_failure(self, job_id: str) -> None:
        """Restore a repair-owned lock after a child repair could not be prepared."""
        if job_id in self._locks:
            return
        database = StateDatabase.open(self.database_path)
        try:
            job = database.get_job(job_id)
            dataset = Path(str(job["dataset_root"]))
            self._locks[job_id] = DatasetLock.acquire(database, dataset, job_id)
        except Exception:
            database.close()
            raise
