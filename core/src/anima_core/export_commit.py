"""Core-owned directory commit for Export prepared artifacts."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .annotation_backup import verify_backup, write_backup
from .dataset_workspace import DatasetWorkspace
from .commit_journal import (
    CommitJournal,
    CommitJournalError,
    CommitPhase,
    JournalState,
    load as load_journal,
    write as write_journal,
)
from .db import StateDatabase
from .db_primitives import repair_target_clause
from .directory_commit import DirectoryCommitError, restore_rollback
from .export_staging import StagingError, create_hardlink_staging, replace_business_annotation, replace_moss_sidecar, replace_ocr_sidecar, reveal_staging
from .export_summary import UNLABELED_IMAGES_CODE, unlabeled_output_folder
from .moss_sidecar import MossSidecarError, parse_moss_json
from .overlay import MOSS_ANNOTATIONS_DIRECTORY, OverlayLayout
from .ocr_sidecar import OcrSidecarError, parse_ocr_sidecar
from .path_safety import PathSafetyError, assert_no_reparse_tree, canonicalize, ensure_within, long_path, read_annotation_state, safe_relative_path, sha256_file, windows_paths_equal
from .replace_overlay import ReplaceOverlayWriter
from .replace_provenance import ReplaceProvenanceChange, apply_provenance_changes, provenance_database_path


STAGING_ENTRY_BYTES = 4096
"""Conservative per-sample directory-entry cost of a hard-linked staging tree."""


class ExportCommitError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExportCommitResult:
    exported: int
    backupZip: Path
    unlabeled: int
    unlabeledFolder: str


def _advance_journal(
    layout: OverlayLayout,
    *,
    job_id: str,
    state: JournalState,
    phase: CommitPhase,
    dataset: Path,
    staging: Path,
    rollback: Path,
    backup: Path,
) -> CommitJournal:
    journal = CommitJournal.v2(
        job_id, state, dataset, staging, rollback, backup, phase=phase,
    )
    write_journal(layout, journal)
    return journal


class ExportCommitCoordinator:
    """Turns verified overlay artifacts into one recoverable directory switch."""

    def __init__(self, database: StateDatabase, layout: OverlayLayout, *, job_id: str) -> None:
        self.database = database
        self.layout = layout
        self.job_id = job_id

    def _configuration(self) -> tuple[str, Path, bool, bool]:
        job = self.database.get_job(self.job_id)
        try:
            config = json.loads(str(job["config_json"]))
            format_value = config["export"]["format"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ExportCommitError("frozen export configuration is invalid") from exc
        if format_value not in {"json", "flat_txt", "both"}:
            raise ExportCommitError("frozen export format is invalid")
        ocr_config = config.get("ocr")
        if ocr_config is None:
            ocr_enabled = False
        elif isinstance(ocr_config, dict) and isinstance(ocr_config.get("enabled"), bool):
            ocr_enabled = ocr_config["enabled"]
        else:
            raise ExportCommitError("frozen OCR configuration is invalid")
        nl_config = config.get("nl")
        moss_enabled = (
            isinstance(nl_config, dict)
            and nl_config.get("enabled") is True
            and nl_config.get("promptVersion") == "nl-default-prompt-moss-v1"
        )
        if job["status"] != "exporting" or job["current_module_id"] != "export":
            raise ExportCommitError("export commit is not active")
        return str(format_value), Path(str(job["dataset_root"])), ocr_enabled, moss_enabled

    @staticmethod
    def _expected_kinds(format_value: str) -> set[str]:
        return {"json"} if format_value == "json" else ({"txt"} if format_value == "flat_txt" else {"json", "txt"})

    @staticmethod
    def _replace_directory_with_retry(source: Path, destination: Path) -> None:
        """Retry short-lived Windows directory locks from sync/indexing tools."""
        source, destination = long_path(source), long_path(destination)
        for attempt in range(20):
            try:
                os.replace(source, destination)
                return
            except PermissionError as exc:
                if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32} or attempt == 19:
                    raise
                time.sleep(0.5)

    def _artifacts(self, format_value: str):
        expected = self._expected_kinds(format_value)
        cursor: int | None = None
        while True:
            page = self.database.page_export_artifact_groups(self.job_id, after_sample_id=cursor, limit=500)
            if not page:
                return
            grouped: dict[int, tuple[str, dict[str, tuple[str, str]]]] = {}
            for row in page:
                sample_id = int(row["sample_id"])
                grouped.setdefault(sample_id, (str(row["annotation_key"]), {}))
                kind = row["kind"]
                if kind is not None:
                    if kind not in expected or not isinstance(row["relative_path"], str) or not isinstance(row["sha256"], str):
                        raise ExportCommitError("export artifact index is invalid")
                    grouped[sample_id][1][str(kind)] = (str(row["relative_path"]), str(row["sha256"]))
            for sample_id, (key, artifacts) in grouped.items():
                if set(artifacts) != expected:
                    raise ExportCommitError(f"sample {sample_id} has incomplete verified export artifacts")
                yield key, artifacts
            cursor = int(page[-1]["sample_id"])

    def _copy_to_staging(self, staging: Path, format_value: str) -> int:
        exported = 0
        for annotation_key, artifacts in self._artifacts(format_value):
            for kind, (relative, digest) in artifacts.items():
                source = self.layout.resolve_prepared(relative)
                if not source.is_file() or sha256_file(source) != digest:
                    raise ExportCommitError("verified export artifact is missing or changed")
                replace_business_annotation(staging, annotation_key, ".json" if kind == "json" else ".txt", source.read_bytes())
            if format_value == "json":
                replace_business_annotation(staging, annotation_key, ".txt", None)
            elif format_value == "flat_txt":
                replace_business_annotation(staging, annotation_key, ".json", None)
            exported += 1
        return exported

    def _scoped_samples(self):
        """Yield only samples eligible for this task's Export projection."""
        clause, clause_parameters = repair_target_clause(
            self.database.connection, self.job_id, "export",
        )
        cursor: int | None = None
        while True:
            cursor_clause = "" if cursor is None else " AND s.sample_id>?"
            parameters: list[object] = [self.job_id]
            if cursor is not None:
                parameters.append(cursor)
            parameters.extend(clause_parameters)
            parameters.append(500)
            rows = list(self.database.connection.execute(
                "SELECT s.* FROM samples AS s WHERE s.job_id=? AND s.in_processing_scope=1"
                + cursor_clause + clause + " ORDER BY s.sample_id LIMIT ?",
                parameters,
            ))
            if not rows:
                return
            yield from rows
            cursor = int(rows[-1]["sample_id"])

    @staticmethod
    def _has_annotation(staging: Path, annotation_key: str, *, relative_image_path: str | None = None, moss_enabled: bool = False) -> bool:
        for suffix in (".txt", ".json"):
            target = ExportCommitCoordinator._baseline_annotation(staging, annotation_key, suffix)
            try:
                if read_annotation_state(target) == "nonblank":
                    return True
            except PathSafetyError:
                # An existing malformed annotation is still user data.  It is
                # preserved and must not make the image look unannotated.
                return True
        if moss_enabled and relative_image_path is not None:
            try:
                relative = safe_relative_path(f"{MOSS_ANNOTATIONS_DIRECTORY}\\{relative_image_path}.moss.json")
                sidecar = ensure_within(staging, staging / Path(relative.replace("\\", os.sep)))
                if sidecar.is_file():
                    return True
            except PathSafetyError:
                return True
        return False

    def _copy_unlabeled_images_to_staging(self, staging: Path, *, moss_enabled: bool = False) -> tuple[int, str]:
        relative_root = safe_relative_path(unlabeled_output_folder(self.job_id))
        output_root = ensure_within(staging, staging / Path(relative_root.replace("\\", os.sep)))
        if output_root.exists():
            raise ExportCommitError("task-scoped unlabeled output already exists")
        copied = 0
        for row in self._scoped_samples():
            if self._has_annotation(
                staging,
                str(row["annotation_key"]),
                relative_image_path=str(row["relative_image_path"]),
                moss_enabled=moss_enabled,
            ):
                continue
            relative_image_path = safe_relative_path(str(row["relative_image_path"]))
            source = ensure_within(staging, staging / Path(relative_image_path.replace("\\", os.sep)))
            if not source.is_file():
                raise ExportCommitError("unlabeled source image is missing")
            target = ensure_within(output_root, output_root / Path(relative_image_path.replace("\\", os.sep)))
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(long_path(source), long_path(target))
            except OSError as link_error:
                if target.exists() or target.is_symlink():
                    raise ExportCommitError("unlabeled image target already exists") from link_error
                try:
                    shutil.copy2(source, target)
                except OSError as exc:
                    raise ExportCommitError("unlabeled image could not be staged") from exc
            copied += 1
        return copied, relative_root

    def _overlay_ocr_sidecars(self):
        """Yield only scoped, strict task-overlay sidecars in keyset pages."""
        try:
            assert_no_reparse_tree(self.layout.root / "ocr_annotations")
        except PathSafetyError as exc:
            raise ExportCommitError("OCR overlay contains a reparse point") from exc
        for row in self._scoped_samples():
            relative_image_path = str(row["relative_image_path"])
            try:
                source = self.layout.ocr_sidecar_path(relative_image_path)
                if not source.exists():
                    if source.is_symlink():
                        raise PathSafetyError("OCR overlay sidecar is a symbolic link")
                    continue
                source = canonicalize(source, must_exist=True, directory=False).value
                raw = source.read_bytes()
                parse_ocr_sidecar(raw, expected_relative_image_path=relative_image_path)
            except (OSError, OcrSidecarError, PathSafetyError) as exc:
                raise ExportCommitError("OCR overlay sidecar is invalid") from exc
            yield relative_image_path, raw

    def _copy_ocr_sidecars_to_staging(self, staging: Path) -> int:
        copied = 0
        for relative_image_path, raw in self._overlay_ocr_sidecars():
            try:
                replace_ocr_sidecar(staging, relative_image_path, raw)
            except StagingError as exc:
                raise ExportCommitError("OCR sidecar staging write failed") from exc
            copied += 1
        return copied

    def _overlay_moss_sidecars(self):
        """Yield only path-safe, scoped and schema-valid MOSS sidecars."""
        try:
            assert_no_reparse_tree(self.layout.root / MOSS_ANNOTATIONS_DIRECTORY)
        except PathSafetyError as exc:
            raise ExportCommitError("MOSS overlay contains a reparse point") from exc
        for row in self._scoped_samples():
            relative_image_path = str(row["relative_image_path"])
            try:
                source = self.layout.moss_sidecar_path(relative_image_path)
                if not source.exists():
                    if source.is_symlink():
                        raise PathSafetyError("MOSS overlay sidecar is a symbolic link")
                    continue
                source = canonicalize(source, must_exist=True, directory=False).value
                raw = source.read_bytes()
                parse_moss_json(raw)
            except (OSError, MossSidecarError, PathSafetyError) as exc:
                raise ExportCommitError("MOSS overlay sidecar is invalid") from exc
            yield relative_image_path, raw

    def _copy_moss_sidecars_to_staging(self, staging: Path) -> int:
        copied = 0
        for relative_image_path, raw in self._overlay_moss_sidecars():
            try:
                replace_moss_sidecar(staging, relative_image_path, raw)
            except StagingError as exc:
                raise ExportCommitError("MOSS sidecar staging write failed") from exc
            copied += 1
        return copied

    def _provenance_changes(self, staging: Path, format_value: str):
        writer = ReplaceOverlayWriter(self.database, self.layout, self.job_id)
        for annotation_key, _ in self._artifacts(format_value):
            if format_value == "flat_txt":
                yield ReplaceProvenanceChange.delete(annotation_key)
                continue
            fingerprint = writer.provenance(annotation_key)
            if fingerprint is None:
                continue
            json_path = self._baseline_annotation(staging, annotation_key, ".json")
            if not json_path.is_file():
                raise ExportCommitError("replace provenance has no final JSON artifact")
            yield ReplaceProvenanceChange.upsert(annotation_key, fingerprint, sha256_file(json_path))

    @staticmethod
    def _baseline_annotation(dataset: Path, annotation_key: str, suffix: str) -> Path:
        relative = safe_relative_path(annotation_key + suffix)
        return ensure_within(dataset, dataset / Path(relative.replace("\\", os.sep)))

    @staticmethod
    def _probe_hardlink_support(parent: Path) -> None:
        """ROADMAP.md:1047 — create, verify and delete a probe link before touching the dataset."""
        descriptor, probe = tempfile.mkstemp(prefix=".anima-linkprobe-", suffix=".tmp", dir=parent)
        source, link = Path(probe), Path(probe + ".link")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"anima-hardlink-probe")
            os.link(source, link)
            if link.read_bytes() != source.read_bytes() or link.stat().st_ino != source.stat().st_ino:
                raise ExportCommitError("hard link probe produced an inconsistent file")
        except OSError as exc:
            raise ExportCommitError("dataset parent directory does not support hard links") from exc
        finally:
            for path in (link, source):
                if path.exists():
                    path.unlink()

    def _verify_capacity(self, dataset: Path, parent: Path, *, ocr_enabled: bool, moss_enabled: bool) -> None:
        """ROADMAP.md:1051 — hard links still cost directory entries, new annotations and a backup."""
        required = 0
        # The hard-link staging tree mirrors the complete dataset, while
        # annotation replacements and unlabeled links are repair-target scoped.
        cursor: int | None = None
        while True:
            page = self.database.page_samples(self.job_id, after_sample_id=cursor, limit=500)
            if not page:
                break
            required += len(page) * STAGING_ENTRY_BYTES * 2
            cursor = int(page[-1]["sample_id"])
        for row in self._scoped_samples():
            for suffix in (".txt", ".json"):
                baseline = self._baseline_annotation(dataset, str(row["annotation_key"]), suffix)
                if baseline.is_file():
                    required += baseline.stat().st_size * 2
        if ocr_enabled:
            for _, raw in self._overlay_ocr_sidecars():
                required += STAGING_ENTRY_BYTES + len(raw)
        if moss_enabled:
            for _, raw in self._overlay_moss_sidecars():
                required += STAGING_ENTRY_BYTES + len(raw)
        provenance = provenance_database_path(dataset)
        required += provenance.stat().st_size if provenance.is_file() else 64 * 1024
        try:
            free = shutil.disk_usage(parent).free
        except OSError as exc:
            raise ExportCommitError("dataset volume free space cannot be measured") from exc
        if free < required:
            raise ExportCommitError(f"export commit needs {required} free bytes but only {free} are available")

    def _verify_baseline(self, dataset: Path) -> None:
        """ROADMAP.md:1058 — re-check the baseline annotations before any journal or directory switch."""
        for row in self._scoped_samples():
            for suffix, recorded in ((".txt", row["original_txt_sha256"]), (".json", row["original_json_sha256"])):
                baseline = self._baseline_annotation(dataset, str(row["annotation_key"]), suffix)
                try:
                    digest = sha256_file(baseline) if read_annotation_state(baseline) == "nonblank" else None
                except (OSError, PathSafetyError) as exc:
                    raise ExportCommitError("baseline annotation cannot be re-checked") from exc
                if digest != recorded:
                    raise ExportCommitError("baseline annotations changed since preflight; run preflight again")

    def _repair_parent_baselines(self, dataset: Path) -> dict[int, tuple[str | None, str | None]]:
        """Record the post-commit baseline for direct parent repair targets."""
        rows = self.database.connection.execute(
            """SELECT DISTINCT parent_issue.sample_id,parent_sample.annotation_key
                 FROM repair_target_issues AS link
                 JOIN issues AS parent_issue ON parent_issue.issue_id=link.parent_issue_id
                 JOIN samples AS parent_sample ON parent_sample.job_id=parent_issue.job_id
                    AND parent_sample.sample_id=parent_issue.sample_id
                WHERE link.repair_job_id=?""",
            (self.job_id,),
        )
        fingerprints: dict[int, tuple[str | None, str | None]] = {}
        for row in rows:
            values: list[str | None] = []
            for suffix in (".txt", ".json"):
                path = self._baseline_annotation(dataset, str(row["annotation_key"]), suffix)
                values.append(
                    sha256_file(path) if read_annotation_state(path) == "nonblank" else None
                )
            fingerprints[int(row["sample_id"])] = (values[0], values[1])
        return fingerprints

    def _backup(self, dataset: Path, backup_zip: Path) -> Path:
        return write_backup(
            dataset,
            backup_zip,
            lambda cursor: self.database.page_samples(self.job_id, after_sample_id=cursor, limit=500),
        )

    def commit(self, *, reuse_verified_backup: bool = False) -> ExportCommitResult:
        format_value, dataset, ocr_enabled, moss_enabled = self._configuration()
        summary = self.database.module_summary(self.job_id, "export")
        if summary["status"] != "completed":
            raise ExportCommitError("Export must settle every sample before staging")
        token = self.job_id.replace("-", "")[:24]
        # Keep the journal's authenticated workspace paths for every recommit
        # attempt.  A damaged backup is rebuilt in the same workspace; it must
        # not silently switch to the new default layout.
        existing_journal = load_journal(self.layout)
        if existing_journal is not None and windows_paths_equal(existing_journal.dataset_root, dataset):
            # Recommit must reuse the journal's exact, authenticated paths so
            # legacy sibling layouts remain recoverable after new workspaces
            # became the default.
            staging = existing_journal.staging_root
            rollback = existing_journal.rollback_root
            backup = existing_journal.backup_zip
            backup_dir = backup.parent
            workspace_parent = staging.parent
        elif DatasetWorkspace.is_new_overlay(self.layout.root, dataset, self.job_id):
            workspace = DatasetWorkspace.for_dataset(dataset)
            staging = workspace.stage_root(self.job_id)
            rollback = workspace.rollback_root(self.job_id)
            backup_dir = workspace.backup_root(self.job_id)
            backup = workspace.backup_zip(self.job_id)
            workspace.journal_root(self.job_id).mkdir(parents=True, exist_ok=True)
            workspace_parent = workspace.root
        else:
            parent = dataset.parent
            staging = parent / f".{dataset.name}.anima-stage-{token}"
            rollback = parent / f".{dataset.name}.anima-rollback-{token}"
            backup_dir = parent / f".{dataset.name}.anima-backups"
            backup = backup_dir / f"{self.job_id}.zip"
            workspace_parent = parent
        if staging.exists() or rollback.exists() or (backup.exists() and not reuse_verified_backup):
            raise ExportCommitError("export commit workspace already exists")
        if reuse_verified_backup:
            verify_backup(backup)
        self._probe_hardlink_support(workspace_parent)
        self._verify_capacity(dataset, workspace_parent, ocr_enabled=ocr_enabled, moss_enabled=moss_enabled)
        rollback_created = False
        staging_promoted = False
        try:
            _advance_journal(
                self.layout, job_id=self.job_id, state="declared", phase="preparing",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            self.database.set_job_status(self.job_id, "committing", current_module_id="export")
            _advance_journal(
                self.layout, job_id=self.job_id, state="declared", phase="creating_stage",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            staging = create_hardlink_staging(dataset, staging)
            _advance_journal(
                self.layout, job_id=self.job_id, state="staging", phase="writing_annotations",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            exported = self._copy_to_staging(staging, format_value)
            if moss_enabled:
                _advance_journal(
                    self.layout, job_id=self.job_id, state="staging", phase="copying_moss",
                    dataset=dataset, staging=staging, rollback=rollback, backup=backup,
                )
                self._copy_moss_sidecars_to_staging(staging)
            _advance_journal(
                self.layout, job_id=self.job_id, state="staging", phase="copying_unlabeled",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            unlabeled, unlabeled_folder = self._copy_unlabeled_images_to_staging(staging, moss_enabled=moss_enabled)
            self.database.set_module_diagnostic_count(
                self.job_id, "export", UNLABELED_IMAGES_CODE, severity="warning", count=unlabeled,
            )
            if ocr_enabled:
                _advance_journal(
                    self.layout, job_id=self.job_id, state="staging", phase="copying_ocr",
                    dataset=dataset, staging=staging, rollback=rollback, backup=backup,
                )
                self._copy_ocr_sidecars_to_staging(staging)
            _advance_journal(
                self.layout, job_id=self.job_id, state="staging", phase="applying_provenance",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            apply_provenance_changes(staging, self._provenance_changes(staging, format_value))
            _advance_journal(
                self.layout, job_id=self.job_id, state="staged", phase="verifying_baseline",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            self._verify_baseline(dataset)
            backup_dir.mkdir(exist_ok=True)
            _advance_journal(
                self.layout, job_id=self.job_id, state="staged", phase="writing_backup",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            if reuse_verified_backup:
                verify_backup(backup)
            else:
                self._backup(dataset, backup)
            _advance_journal(
                self.layout, job_id=self.job_id, state="backup_verified", phase="switching_directories",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            self._replace_directory_with_retry(dataset, rollback)
            rollback_created = True
            _advance_journal(
                self.layout, job_id=self.job_id, state="rollback_created", phase="switching_directories",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            reveal_staging(staging)
            self._replace_directory_with_retry(staging, dataset)
            staging_promoted = True
            _advance_journal(
                self.layout, job_id=self.job_id, state="committed", phase="cleanup",
                dataset=dataset, staging=staging, rollback=rollback, backup=backup,
            )
            if not dataset.is_dir():
                raise ExportCommitError("committed dataset is unavailable")
            # Everything below runs after the irreversible directory switch, so a
            # failure here must never turn a committed dataset into a failed job.
            # Cleanup is therefore best effort and the terminal status is written
            # first, before any bookkeeping that could raise.
            self.database.set_job_status(
                self.job_id, "succeeded_with_issues" if unlabeled > 0 else "succeeded", current_module_id="export",
            )
            shutil.rmtree(rollback, ignore_errors=True)
            try:
                self.database.update_repair_parent_baselines(
                    self.job_id,
                    self._repair_parent_baselines(dataset),
                )
                self.database.resolve_repaired_parent_issues(self.job_id)
            except Exception:
                # Parent bookkeeping is retried by the next repair commit; the
                # exported dataset itself is already correct on disk.
                pass
            try:
                # A committed journal is resolved, so the task-local overlay is no
                # longer needed for recovery.
                self.layout.discard()
                self.database.clear_workspace_metadata(self.job_id)
            except Exception:
                pass
            return ExportCommitResult(exported, backup, unlabeled, unlabeled_folder)
        except Exception as exc:
            restored = False
            journal_failure: Exception | None = None
            state: str | None = None
            current_journal: CommitJournal | None = None
            if self.layout.commit_journal_path().exists():
                try:
                    current_journal = load_journal(self.layout)
                except CommitJournalError as journal_error:
                    journal_failure = journal_error
            if (
                current_journal is not None
                and current_journal.schema_version == 2
                and current_journal.state in {"declared", "staging", "staged", "backup_verified"}
                and not rollback_created
                and dataset.is_dir()
            ):
                raise ExportCommitError("export directory commit stopped before directory switching") from exc
            try:
                if not dataset.exists() and rollback.exists():
                    restored = restore_rollback(dataset, rollback)
            except (DirectoryCommitError, CommitJournalError, OSError):
                pass
            if current_journal is not None:
                try:
                    if staging_promoted:
                        state = "committed"
                    else:
                        restored = restored or (
                            rollback_created and dataset.is_dir()
                            and not rollback.exists() and staging.is_dir()
                        )
                        state = "rolled_back" if restored else "rollback_required"
                    phase: CommitPhase = "cleanup" if state in {"committed", "rolled_back"} else "switching_directories"
                    _advance_journal(
                        self.layout, job_id=self.job_id, state=state, phase=phase,
                        dataset=dataset, staging=staging, rollback=rollback, backup=backup,
                    )
                except (CommitJournalError, OSError) as journal_error:
                    journal_failure = journal_error
            # A failed commit must not leave a full hard-link mirror of the dataset behind;
            # only a rollback that still needs recovery keeps its staging tree for diagnosis.
            if journal_failure is None and state != "rollback_required" and dataset.is_dir() and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if self.database.get_job(self.job_id)["status"] in {"exporting", "committing"}:
                self.database.set_job_status(self.job_id, "failed", current_module_id="export")
            if journal_failure is not None:
                raise ExportCommitError("export rollback journal cannot be persisted") from journal_failure
            raise ExportCommitError("export directory commit failed") from exc

    @staticmethod
    def recover(layout: OverlayLayout) -> str | None:
        """Resolve a persisted module-6 directory switch before new work begins."""
        journal = load_journal(layout)
        if journal is None:
            return None
        if journal.state in {"committed", "rolled_back"}:
            # A resolved journal must not keep a stale hard-link mirror of the dataset alive.
            if journal.dataset_root.is_dir() and journal.staging_root.exists():
                shutil.rmtree(journal.staging_root, ignore_errors=True)
            if journal.state == "committed" and journal.dataset_root.is_dir() and journal.rollback_root.exists():
                shutil.rmtree(journal.rollback_root)
            return str(journal.state)
        dataset, staging, rollback = journal.dataset_root, journal.staging_root, journal.rollback_root
        try:
            if journal.state in {"prepared", "backup_verified", "rollback_required"}:
                if not dataset.exists() and rollback.exists():
                    restore_rollback(dataset, rollback)
                if staging.exists():
                    shutil.rmtree(staging)
                _advance_journal(
                    layout, job_id=journal.job_id, state="rolled_back", phase="cleanup",
                    dataset=dataset, staging=staging, rollback=rollback, backup=journal.backup_zip,
                )
                return "rolled_back"
            if journal.state == "rollback_created":
                # A failed second rename can already have restored the old
                # directory while the journal update itself was interrupted.
                if dataset.is_dir() and not rollback.exists() and staging.is_dir():
                    shutil.rmtree(staging)
                    _advance_journal(
                        layout, job_id=journal.job_id, state="rolled_back", phase="cleanup",
                        dataset=dataset, staging=staging, rollback=rollback, backup=journal.backup_zip,
                    )
                    return "rolled_back"
                if not dataset.exists() and staging.is_dir():
                    reveal_staging(staging)
                    ExportCommitCoordinator._replace_directory_with_retry(staging, dataset)
                if not dataset.is_dir():
                    raise ExportCommitError("commit recovery has no consistent dataset directory")
                if rollback.exists():
                    shutil.rmtree(rollback)
                _advance_journal(
                    layout, job_id=journal.job_id, state="committed", phase="cleanup",
                    dataset=dataset, staging=staging, rollback=rollback, backup=journal.backup_zip,
                )
                return "committed"
        except (OSError, DirectoryCommitError, CommitJournalError) as exc:
            raise ExportCommitError("export commit recovery failed") from exc
        raise ExportCommitError("unknown commit journal state")
