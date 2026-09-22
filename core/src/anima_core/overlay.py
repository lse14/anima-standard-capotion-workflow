from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .contracts import canonical_json, utc_now
from .dataset_workspace import DatasetWorkspace
from .path_safety import (
    IMAGE_EXTENSIONS,
    PathSafetyError,
    atomic_write_bytes,
    canonicalize,
    ensure_within,
    replace_with_sharing_retry,
    safe_relative_path,
    sha256_file,
)


OVERLAY_MARKER = "overlay-manifest.json"
RESOLVED_JOURNAL_STATES = frozenset({"resolved", "committed", "rolled_back"})
OCR_ANNOTATIONS_DIRECTORY = "ocr_annotations"


class OverlayError(RuntimeError):
    pass


def _set_hidden(path: Path) -> None:
    if os.name != "nt":
        return
    import ctypes

    get_attributes = ctypes.windll.kernel32.GetFileAttributesW
    set_attributes = ctypes.windll.kernel32.SetFileAttributesW
    attributes = get_attributes(str(path))
    if attributes == 0xFFFFFFFF or not set_attributes(str(path), attributes | 0x2):
        raise OverlayError(f"unable to set hidden attribute: {path}")


@dataclass(frozen=True)
class OverlayLayout:
    job_id: str
    dataset_root: Path
    root: Path

    @classmethod
    def create(cls, dataset_root: str | Path, job_id: str) -> "OverlayLayout":
        dataset = canonicalize(dataset_root, must_exist=True, directory=True).value
        safe_job = "".join(character for character in job_id if character.isalnum() or character in "-_")
        if not safe_job or safe_job != job_id:
            raise OverlayError("jobId contains unsafe characters")
        workspace = DatasetWorkspace.for_dataset(dataset)
        job_root = workspace.job_root(job_id, create=False)
        workspace.root.mkdir(parents=True, exist_ok=True)
        (workspace.root / "jobs").mkdir(parents=True, exist_ok=True)
        if job_root.exists():
            raise OverlayError(f"task workspace already exists: {job_root}")
        job_root.mkdir()
        overlay = workspace.overlay_root(job_id)
        if overlay.exists():
            raise OverlayError(f"overlay already exists: {overlay}")
        overlay.mkdir()
        try:
            (overlay / "annotations").mkdir()
            (overlay / OCR_ANNOTATIONS_DIRECTORY).mkdir()
            (overlay / "prepared").mkdir()
            (overlay / "commit").mkdir()
            (overlay / "resources").mkdir()
            marker = {
                "schemaVersion": 1,
                "jobId": job_id,
                "datasetRoot": str(dataset),
                "layout": "dataset_workspace_v1",
                "createdAt": utc_now(),
            }
            atomic_write_bytes(overlay, OVERLAY_MARKER, (canonical_json(marker) + "\n").encode("utf-8"))
            _set_hidden(overlay)
        except Exception:
            shutil.rmtree(job_root, ignore_errors=True)
            raise
        return cls(job_id, dataset, overlay)

    @classmethod
    def open_existing(cls, root: str | Path, expected_job_id: str, *, allow_missing_dataset: bool = False) -> "OverlayLayout":
        overlay = canonicalize(root, must_exist=True, directory=True).value
        marker_path = overlay / OVERLAY_MARKER
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OverlayError("overlay marker is missing or invalid") from exc
        if marker.get("schemaVersion") != 1 or marker.get("jobId") != expected_job_id:
            raise OverlayError("overlay marker does not match job")
        dataset = canonicalize(marker.get("datasetRoot", ""), must_exist=not allow_missing_dataset, directory=True).value
        ensure_within(dataset.parent, overlay)
        return cls(expected_job_id, dataset, overlay)

    def annotation_path(self, annotation_key: str, suffix: str) -> Path:
        if suffix not in {".txt", ".json"}:
            raise OverlayError("unsupported annotation suffix")
        key = safe_relative_path(annotation_key)
        if key.split("\\", 1)[0].lower() == OCR_ANNOTATIONS_DIRECTORY:
            raise OverlayError("OCR sidecars must not use the business annotation tree")
        relative = safe_relative_path(key + suffix)
        return ensure_within(self.root / "annotations", self.root / "annotations" / Path(relative.replace("\\", os.sep)))

    def ocr_sidecar_path(self, relative_image_path: str) -> Path:
        try:
            relative = safe_relative_path(relative_image_path)
            image_path = Path(relative.replace("\\", os.sep))
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise OverlayError("OCR sidecar path requires a supported image extension")
            sidecar_relative = safe_relative_path(f"{OCR_ANNOTATIONS_DIRECTORY}\\{relative}.ocr.json")
            return ensure_within(
                self.root / OCR_ANNOTATIONS_DIRECTORY,
                self.root / Path(sidecar_relative.replace("\\", os.sep)),
            )
        except PathSafetyError as exc:
            raise OverlayError("OCR sidecar path is unsafe") from exc

    def prepared_path(self, module_id: str, lease_id: str, suffix: str) -> Path:
        if module_id not in {"caption", "classify", "replace", "ocr", "nl", "count_review", "dropout", "token_budget", "export"}:
            raise OverlayError("unsupported module")
        safe_lease = "".join(character for character in lease_id if character.isalnum() or character in "-_")
        if not safe_lease or safe_lease != lease_id or suffix not in {".txt", ".json"}:
            raise OverlayError("unsafe prepared artifact name")
        relative = safe_relative_path(f"{module_id}\\{safe_lease}{suffix}")
        return ensure_within(self.root / "prepared", self.root / "prepared" / Path(relative.replace("\\", os.sep)))

    def write_annotation(self, annotation_key: str, suffix: str, data: bytes) -> Path:
        self.annotation_path(annotation_key, suffix)
        relative = "annotations\\" + safe_relative_path(annotation_key + suffix)
        return atomic_write_bytes(self.root, relative, data)

    def write_ocr_sidecar(self, relative_image_path: str, data: bytes) -> Path:
        self.ocr_sidecar_path(relative_image_path)
        relative = safe_relative_path(f"{OCR_ANNOTATIONS_DIRECTORY}\\{relative_image_path}.ocr.json")
        return atomic_write_bytes(self.root, relative, data)

    def write_resource(self, relative_path: str, data: bytes) -> Path:
        relative = safe_relative_path(relative_path)
        destination = ensure_within(self.root / "resources", self.root / "resources" / Path(relative.replace("\\", os.sep)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        return atomic_write_bytes(self.root, "resources\\" + relative, data)

    def resource_path(self, relative_path: str) -> Path:
        relative = safe_relative_path(relative_path)
        return ensure_within(self.root / "resources", self.root / "resources" / Path(relative.replace("\\", os.sep)))

    def write_prepared(self, module_id: str, lease_id: str, suffix: str, data: bytes) -> tuple[Path, str]:
        path = self.prepared_path(module_id, lease_id, suffix)
        relative = os.path.relpath(path, self.root)
        written = atomic_write_bytes(self.root, relative, data)
        return written, sha256_file(written)

    def resolve_prepared(self, prepared_relative_path: str) -> Path:
        safe_prepared = safe_relative_path(prepared_relative_path)
        return ensure_within(
            self.root / "prepared",
            self.root / Path(safe_prepared.replace("\\", os.sep)),
        )

    def commit_prepared(self, prepared_relative_path: str, expected_sha256: str, annotation_key: str, suffix: str) -> Path:
        prepared = self.resolve_prepared(prepared_relative_path)
        if not prepared.is_file() or sha256_file(prepared) != expected_sha256:
            raise OverlayError("prepared artifact is missing or has an invalid digest")
        destination = self.annotation_path(annotation_key, suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        replace_with_sharing_retry(prepared, destination)
        return destination

    def write_ocr_prepared(self, lease_id: str, data: bytes) -> tuple[Path, str]:
        return self.write_prepared("ocr", lease_id, ".json", data)

    def commit_ocr_prepared(
        self,
        prepared_relative_path: str,
        expected_sha256: str,
        relative_image_path: str,
    ) -> Path:
        prepared = self.resolve_prepared(prepared_relative_path)
        try:
            ensure_within(self.root / "prepared" / "ocr", prepared)
        except PathSafetyError as exc:
            raise OverlayError("OCR prepared artifact is outside the OCR staging directory") from exc
        if not prepared.is_file() or sha256_file(prepared) != expected_sha256:
            raise OverlayError("OCR prepared artifact is missing or has an invalid digest")
        destination = self.ocr_sidecar_path(relative_image_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        replace_with_sharing_retry(prepared, destination)
        return destination

    def commit_journal_path(self) -> Path:
        return self.root / "commit" / "journal.json"

    def journal_state(self) -> str | None:
        journal = self.commit_journal_path()
        if not journal.exists():
            return None
        try:
            value = json.loads(journal.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OverlayError("commit journal is unreadable") from exc
        state = value.get("state")
        if not isinstance(state, str):
            raise OverlayError("commit journal has no state")
        return state

    def write_journal(self, value: dict[str, object]) -> Path:
        state = value.get("state")
        if not isinstance(state, str) or not state:
            raise OverlayError("commit journal requires a state")
        payload = dict(value)
        payload.setdefault("schemaVersion", 1)
        payload.setdefault("jobId", self.job_id)
        if payload["schemaVersion"] not in {1, 2} or payload["jobId"] != self.job_id:
            raise OverlayError("commit journal does not match overlay")
        return atomic_write_bytes(self.root, "commit\\journal.json", (canonical_json(payload) + "\n").encode("utf-8"))

    def has_unresolved_journal(self) -> bool:
        state = self.journal_state()
        return state is not None and state not in RESOLVED_JOURNAL_STATES

    def discard(self) -> None:
        verified = self.open_existing(self.root, self.job_id)
        if verified.has_unresolved_journal():
            raise OverlayError("commit journal must be resolved before discard")
        job_root = verified.root.parent
        shutil.rmtree(verified.root)
        # Reclaim only this resolved task directory.  Sibling jobs and
        # dataset-level backups remain untouched.
        try:
            if job_root.is_dir() and not any(job_root.iterdir()):
                job_root.rmdir()
        except OSError:
            pass


@dataclass(frozen=True)
class BaselineView:
    dataset_root: Path

    def annotation_path(self, annotation_key: str, suffix: str) -> Path:
        relative = safe_relative_path(annotation_key + suffix)
        return ensure_within(self.dataset_root, self.dataset_root / Path(relative.replace("\\", os.sep)))

    def read(self, annotation_key: str, suffix: str) -> bytes | None:
        path = self.annotation_path(annotation_key, suffix)
        return path.read_bytes() if path.is_file() else None


@dataclass(frozen=True)
class WorkingAnnotationView:
    baseline: BaselineView
    overlay: OverlayLayout
    allow_baseline_fallback: bool = True

    def read(self, annotation_key: str, suffix: str) -> bytes | None:
        overlay_path = self.overlay.annotation_path(annotation_key, suffix)
        if overlay_path.is_file():
            return overlay_path.read_bytes()
        return self.baseline.read(annotation_key, suffix) if self.allow_baseline_fallback else None


@dataclass(frozen=True)
class OriginalNlView:
    baseline: BaselineView

    def read_json(self, annotation_key: str) -> dict[str, object] | None:
        raw = self.baseline.read(annotation_key, ".json")
        if raw is None or not raw.decode("utf-8-sig").strip():
            return None
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise OverlayError("baseline JSON must be an object")
        return value
