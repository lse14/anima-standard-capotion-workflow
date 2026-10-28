"""Confirmed, fail-closed recovery for interrupted pre-swap Export commits."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from .annotation_backup import AnnotationBackupError, verify_backup
from .commit_journal import CommitJournal, load as load_commit_journal
from .contracts import sha256_json
from .db import StateDatabase
from .dataset_workspace import DatasetWorkspace
from .export_commit import ExportCommitCoordinator, ExportCommitError
from .overlay import OverlayLayout
from .path_safety import canonicalize, sha256_file, windows_paths_equal
from .pipeline_recovery import classify_export_recovery


class ExportRecommitError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RecommitInventory:
    path: str
    fileCount: int
    directoryCount: int
    apparentBytes: int
    errorCount: int
    skippedCount: int
    truncated: bool
    exitCode: int
    exists: bool
    type: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "type": self.type,
            "fileCount": self.fileCount,
            "directoryCount": self.directoryCount,
            "apparentBytes": self.apparentBytes,
            "errorCount": self.errorCount,
            "skippedCount": self.skippedCount,
            "truncated": self.truncated,
            "exitCode": self.exitCode,
        }


def _is_reparse_stat(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(value.st_mode) or bool(attributes & reparse_flag)


def _scan_inventory(root: Path, *, max_entries: int) -> RecommitInventory:
    path = str(root)
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError:
        return RecommitInventory(path, 0, 0, 0, 0, 0, False, 0, False, "missing")
    except OSError:
        return RecommitInventory(path, 0, 0, 0, 1, 1, False, 1, False, "unreadable")
    if _is_reparse_stat(root_stat):
        return RecommitInventory(path, 0, 0, 0, 1, 1, False, 1, True, "reparse_point")
    if not stat.S_ISDIR(root_stat.st_mode):
        size = int(root_stat.st_size) if stat.S_ISREG(root_stat.st_mode) else 0
        return RecommitInventory(path, int(stat.S_ISREG(root_stat.st_mode)), 0, size, 1, 1, False, 1, True, "file")

    files = 0
    directories = 0
    apparent_bytes = 0
    errors = 0
    skipped = 0
    truncated = False
    seen = 0
    stack = [root]
    while stack:
        current = stack.pop()
        if seen >= max_entries:
            truncated = True
            break
        seen += 1
        directories += 1
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda item: item.name.casefold())
        except OSError:
            errors += 1
            skipped += 1
            continue
        for entry in reversed(entries):
            if seen >= max_entries:
                truncated = True
                break
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError:
                errors += 1
                skipped += 1
                seen += 1
                continue
            seen += 1
            if _is_reparse_stat(entry_stat):
                errors += 1
                skipped += 1
            elif stat.S_ISDIR(entry_stat.st_mode):
                stack.append(Path(entry.path))
                seen -= 1
            elif stat.S_ISREG(entry_stat.st_mode):
                files += 1
                apparent_bytes += int(entry_stat.st_size)
            else:
                skipped += 1
        if truncated:
            break
    exit_code = 0 if errors == 0 and skipped == 0 and not truncated else 1
    return RecommitInventory(
        path, files, directories, apparent_bytes, errors, skipped,
        truncated, exit_code, True, "directory",
    )


@dataclass(frozen=True)
class _RecommitContext:
    job_id: str
    dataset: Path
    layout: OverlayLayout
    journal: CommitJournal
    format_value: str
    ocr_enabled: bool


class ExportRecommitService:
    def __init__(self, database_path: str | Path, *, max_inventory_entries: int = 2_000_000) -> None:
        if type(max_inventory_entries) is not int or max_inventory_entries < 1:
            raise ValueError("max_inventory_entries must be a positive integer")
        self.database_path = Path(database_path)
        self.max_inventory_entries = max_inventory_entries
        self._active_lock = threading.Lock()
        self._active_jobs: set[str] = set()

    @staticmethod
    def _expected_paths(job_id: str, dataset: Path) -> tuple[Path, Path, Path]:
        token = job_id.replace("-", "")[:24]
        if not token:
            raise ExportRecommitError("invalid_job_id", "job id cannot identify an Export stage")
        parent = dataset.parent
        return (
            parent / f".{dataset.name}.anima-stage-{token}",
            parent / f".{dataset.name}.anima-rollback-{token}",
            parent / f".{dataset.name}.anima-backups" / f"{job_id}.zip",
        )

    @staticmethod
    def _workspace_paths(job_id: str, dataset: Path) -> tuple[Path, Path, Path]:
        workspace = DatasetWorkspace.for_dataset(dataset, must_exist=False)
        return workspace.stage_root(job_id), workspace.rollback_root(job_id), workspace.backup_zip(job_id)

    def _context(self, database: StateDatabase, job_id: str) -> _RecommitContext:
        job = database.get_job(job_id)
        if (
            job["status"] != "interrupted"
            or job["current_module_id"] != "export"
            or database.module_summary(job_id, "export")["status"] != "completed"
        ):
            raise ExportRecommitError("not_recommittable", "job is not an interrupted completed Export")
        overlay_root = job["overlay_root"]
        if not isinstance(overlay_root, str) or not overlay_root:
            raise ExportRecommitError("overlay_unavailable", "job has no Export recovery overlay")
        try:
            layout = OverlayLayout.open_existing(overlay_root, job_id)
            journal = load_commit_journal(layout)
        except Exception as exc:
            raise ExportRecommitError("journal_invalid", "Export recovery journal is invalid") from exc
        if journal is None or classify_export_recovery(journal) != "manual_recommit":
            raise ExportRecommitError("not_recommittable", "Export journal does not require manual recommit")
        # During a directory swap the formal dataset can be temporarily absent;
        # the authenticated journal still provides the recovery identity.
        dataset = canonicalize(str(job["dataset_root"]), must_exist=False, directory=True).value
        legacy_paths = self._expected_paths(job_id, dataset)
        workspace_paths = self._workspace_paths(job_id, dataset)
        allowed_paths = (legacy_paths, workspace_paths)
        if not windows_paths_equal(journal.dataset_root, dataset) or not any(
            all(windows_paths_equal(actual, expected) for actual, expected in zip(
                (journal.staging_root, journal.rollback_root, journal.backup_zip), paths
            )) for paths in allowed_paths
        ):
            raise ExportRecommitError("invalid_stage_path", "journal stage path does not match the task")
        try:
            config = json.loads(str(job["config_json"]))
            format_value = config["export"]["format"]
            ocr = config.get("ocr", {})
            ocr_enabled = bool(ocr.get("enabled", False)) if isinstance(ocr, dict) else False
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ExportRecommitError("config_invalid", "frozen Export configuration is invalid") from exc
        if format_value not in {"json", "flat_txt", "both"}:
            raise ExportRecommitError("config_invalid", "frozen Export format is invalid")
        return _RecommitContext(job_id, dataset, layout, journal, str(format_value), ocr_enabled)

    @staticmethod
    def _overlay_state(
        database: StateDatabase, context: _RecommitContext,
    ) -> dict[str, object]:
        coordinator = ExportCommitCoordinator(database, context.layout, job_id=context.job_id)
        records: list[dict[str, object]] = []
        try:
            for annotation_key, artifacts in coordinator._artifacts(context.format_value):
                for kind, (relative_path, expected_digest) in sorted(artifacts.items()):
                    source = context.layout.resolve_prepared(relative_path)
                    actual_digest = sha256_file(source) if source.is_file() else None
                    if actual_digest != expected_digest:
                        raise ExportCommitError("verified Export artifact changed")
                    records.append({
                        "annotationKey": annotation_key,
                        "kind": kind,
                        "relativePath": relative_path,
                        "sha256": actual_digest,
                    })
            if context.ocr_enabled:
                for relative_image_path, raw in coordinator._overlay_ocr_sidecars():
                    records.append({
                        "relativeImagePath": relative_image_path,
                        "kind": "ocr",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    })
        except Exception:
            return {
                "path": str(context.layout.root),
                "artifactsValid": False,
                "artifactCount": len(records),
                "artifactFingerprint": None,
            }
        return {
            "path": str(context.layout.root),
            "artifactsValid": True,
            "artifactCount": len(records),
            "artifactFingerprint": sha256_json(records),
        }

    @staticmethod
    def _backup_state(path: Path) -> dict[str, object]:
        partial = path.with_suffix(path.suffix + ".partial")
        target_exists = path.exists() or path.is_symlink()
        partial_exists = partial.exists() or partial.is_symlink()
        cleanup_safe = True
        valid = False
        if target_exists:
            try:
                target_stat = os.lstat(path)
                cleanup_safe = stat.S_ISREG(target_stat.st_mode) and not _is_reparse_stat(target_stat)
                if cleanup_safe:
                    verify_backup(path)
                    valid = True
            except (OSError, AnnotationBackupError):
                valid = False
        if partial_exists:
            try:
                partial_stat = os.lstat(partial)
                cleanup_safe = cleanup_safe and stat.S_ISREG(partial_stat.st_mode) and not _is_reparse_stat(partial_stat)
            except OSError:
                cleanup_safe = False
        if valid and partial_exists:
            state = "valid_with_partial"
        elif valid:
            state = "valid"
        elif target_exists:
            state = "invalid"
        elif partial_exists:
            state = "partial"
        else:
            state = "missing"
        return {
            "path": str(path),
            "state": state,
            "partialPath": str(partial),
            "partialExists": partial_exists,
            "cleanupSafe": cleanup_safe,
        }

    def _build_preview(
        self, database: StateDatabase, job_id: str, *, persist_child_baselines: bool = False,
    ) -> dict[str, object]:
        context = self._context(database, job_id)
        # A successful nested repair may have committed a subset of this task's
        # targets first. Advance only the parent baselines covered by those
        # persisted child links before validating the outer Export overlay.
        # The same fingerprints are computed for a read-only preview; persisting
        # them is the recommit's job, so a GET never writes the database.
        for child in database.repair_children(job_id):
            if str(child["status"]) not in {"succeeded", "succeeded_with_issues"}:
                continue
            child_job_id = str(child["job_id"])
            fingerprints = ExportCommitCoordinator(
                database, context.layout, job_id=child_job_id,
            )._repair_parent_baselines(context.dataset)
            if persist_child_baselines:
                database.update_repair_parent_baselines(child_job_id, fingerprints)
        inventory = _scan_inventory(
            context.journal.staging_root,
            max_entries=self.max_inventory_entries,
        )
        coordinator = ExportCommitCoordinator(database, context.layout, job_id=job_id)
        try:
            coordinator._verify_baseline(context.dataset)
            baseline_matches = True
        except Exception:
            baseline_matches = False
        overlay = self._overlay_state(database, context)
        backup = self._backup_state(context.journal.backup_zip)
        rollback_exists = context.journal.rollback_root.exists() or context.journal.rollback_root.is_symlink()
        stage_safe = (
            inventory.type in {"directory", "missing"}
            and inventory.errorCount == 0
            and inventory.skippedCount == 0
            and not inventory.truncated
            and inventory.exitCode == 0
        )
        safe = (
            stage_safe
            and baseline_matches
            and bool(overlay["artifactsValid"])
            and not rollback_exists
            and bool(backup["cleanupSafe"])
        )
        preview: dict[str, object] = {
            "jobId": job_id,
            "recoveryKind": "export_recommit",
            "journalSchemaVersion": context.journal.schema_version,
            "journalState": context.journal.state,
            "journalPhase": context.journal.phase,
            "journalUpdatedAt": context.journal.updated_at,
            "stage": inventory.to_dict(),
            "dataset": {"path": str(context.dataset), "baselineMatches": baseline_matches},
            "overlay": overlay,
            "rollback": {"path": str(context.journal.rollback_root), "exists": rollback_exists},
            "backup": backup,
            "safeToRecommit": safe,
        }
        preview["inventoryFingerprint"] = sha256_json(preview)
        return preview

    def preview(self, job_id: str) -> dict[str, object]:
        database = StateDatabase.open(self.database_path)
        try:
            return self._build_preview(database, job_id)
        finally:
            database.close()

    @staticmethod
    def _remove_confirmed_file(path: Path) -> None:
        value = os.lstat(path)
        if not stat.S_ISREG(value.st_mode) or _is_reparse_stat(value):
            raise ExportRecommitError("cleanup_unsafe", "backup cleanup target is not a regular file")
        path.unlink()

    def recommit(
        self,
        job_id: str,
        *,
        confirmed: bool,
        inventory_fingerprint: str,
    ) -> dict[str, object]:
        if not confirmed:
            raise ExportRecommitError("confirmation_required", "explicit recommit confirmation is required")
        if not isinstance(inventory_fingerprint, str) or len(inventory_fingerprint) != 64:
            raise ExportRecommitError("fingerprint_invalid", "inventory fingerprint is invalid")
        with self._active_lock:
            if job_id in self._active_jobs:
                raise ExportRecommitError("recommit_active", "Export recommit is already running")
            self._active_jobs.add(job_id)
        try:
            current = self.preview(job_id)
            if current["inventoryFingerprint"] != inventory_fingerprint:
                raise ExportRecommitError("fingerprint_conflict", "inventory fingerprint changed before recommit")
            if not current["safeToRecommit"]:
                raise ExportRecommitError("inventory_unsafe", "recommit inventory is incomplete or unsafe")

            database = StateDatabase.open(self.database_path)
            try:
                context = self._context(database, job_id)
                final_check = self._build_preview(database, job_id, persist_child_baselines=True)
                if final_check["inventoryFingerprint"] != inventory_fingerprint:
                    raise ExportRecommitError("fingerprint_conflict", "inventory fingerprint changed before cleanup")
                stage = context.journal.staging_root
                if stage.exists():
                    stage_stat = os.lstat(stage)
                    if not stat.S_ISDIR(stage_stat.st_mode) or _is_reparse_stat(stage_stat):
                        raise ExportRecommitError("stage_cleanup_unsafe", "confirmed Export stage is no longer a safe directory")
                    shutil.rmtree(stage)
                    if stage.exists():
                        raise ExportRecommitError("stage_cleanup_failed", "confirmed Export stage could not be removed")

                backup_state = final_check["backup"]
                assert isinstance(backup_state, dict)
                state = str(backup_state["state"])
                partial = Path(str(backup_state["partialPath"]))
                if state == "invalid":
                    self._remove_confirmed_file(context.journal.backup_zip)
                if bool(backup_state["partialExists"]):
                    self._remove_confirmed_file(partial)
                reuse_backup = state in {"valid", "valid_with_partial"}

                database.set_job_status(
                    job_id, "exporting", current_module_id="export", resume_status=None,
                )
                result = ExportCommitCoordinator(
                    database, context.layout, job_id=job_id,
                ).commit(reuse_verified_backup=reuse_backup)
                status = str(database.get_job(job_id)["status"])
                return {
                    "jobId": job_id,
                    "status": status,
                    "exported": result.exported,
                    "unlabeled": result.unlabeled,
                    "unlabeledFolder": result.unlabeledFolder,
                    "backupZip": str(result.backupZip),
                }
            finally:
                database.close()
        finally:
            with self._active_lock:
                self._active_jobs.discard(job_id)
