from __future__ import annotations

import os
import sqlite3
import warnings
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Callable, Iterator, Any, Literal

from .contracts import SampleIssue, SampleManifest, SampleRecord, utc_now
from .path_safety import (
    PathSafetyError,
    annotation_key,
    assert_no_reparse_tree,
    file_fingerprint,
    image_format,
    read_annotation_state,
    sha256_file,
    windows_compare,
)


class ManifestError(ValueError):
    def __init__(self, message: str, *, code: str = "manifest_invalid") -> None:
        super().__init__(message)
        self.code = code


class ManifestCancelled(ManifestError):
    """Raised when a caller cancels a manifest walk between filesystem entries."""

    def __init__(self, message: str = "manifest scan cancelled") -> None:
        super().__init__(message, code="manifest_cancelled")


@dataclass(frozen=True)
class ManifestScan:
    manifest: SampleManifest
    records: Iterator[SampleRecord]


def annotation_paths(image_path: Path) -> tuple[Path, Path]:
    """Sibling annotation paths using exactly ``annotation_key`` semantics.

    ``with_suffix("")`` followed by ``with_suffix(".txt")`` strips a second
    time for names such as ``d.va_overwatch.png``; only the final extension
    may be replaced.
    """
    return image_path.with_name(image_path.stem + ".txt"), image_path.with_name(image_path.stem + ".json")


def _image_info(path: Path, expected_format: str) -> tuple[int, str]:
    try:
        from PIL import Image, ImageFile
        from PIL import UnidentifiedImageError
    except ImportError as exc:
        raise ManifestError("Pillow is required for image preflight") from exc
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getattr(Image, "DecompressionBombWarning", Warning))
            with Image.open(path) as image:
                actual = str(image.format or "").lower()
                expected = "jpeg" if expected_format == "jpeg" else expected_format
                if actual not in ({"jpg", "jpeg"} if expected == "jpeg" else {expected}):
                    raise ManifestError(f"image extension/format mismatch: {path}", code="image_format_mismatch")
                frames = 1
                try:
                    while True:
                        image.seek(frames)
                        frames += 1
                except EOFError:
                    pass
                for frame_index in range(frames):
                    image.seek(frame_index)
                    image.load()
            # Pillow requires verify() to be the first operation after open.
            # The separate handle keeps both complete decode and structural
            # verification without retaining image pixels between samples.
            with Image.open(path) as verification:
                verification.verify()
    except ManifestError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Warning) as exc:
        raise ManifestError(f"image cannot be decoded: {path}", code="image_decode_failed") from exc
    if frames > 1:
        raise ManifestError(f"multi-frame image is not supported: {path}", code="image_multi_frame")
    return frames, expected


class ManifestBuilder:
    """Iterative discovery; it never builds a Python list of all paths."""

    def __init__(
        self, source_root: str | Path, *, recursive: bool, profile: str = "e621",
        invalid_image_action: Literal["block", "skip"] = "block",
    ) -> None:
        if invalid_image_action not in {"block", "skip"}:
            raise ValueError("invalid image action must be block or skip")
        self.source_root = Path(source_root)
        self.recursive = recursive
        self.profile = profile
        self.invalid_image_action = invalid_image_action
        # Bounded counters for the preflight backup/space estimate.
        self.annotation_bytes = 0
        self.annotation_files = 0
        self.image_issue_count = 0

    def _iter_paths(self, *, cancel_check: Callable[[], bool] | None = None) -> Iterator[tuple[Path, bool]]:
        root = self.source_root
        stack: list[tuple[Path, bool]] = [(root, True)]
        while stack:
            if cancel_check is not None and cancel_check():
                raise ManifestCancelled()
            current, is_scope = stack.pop()
            directories: list[tuple[Path, bool]] = []
            with os.scandir(current) as entries:
                # Sorting one directory at a time makes sample IDs stable while
                # keeping memory bounded by the largest single directory.
                for entry in sorted(
                    entries,
                    key=cmp_to_key(lambda left, right: windows_compare(left.name, right.name)),
                ):
                    if cancel_check is not None and cancel_check():
                        raise ManifestCancelled()
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in {".mikazuki-cache", ".anima-idg"} or (
                            current == root and entry.name == "_unlabeled"
                        ):
                            continue
                        directories.append((path, is_scope and self.recursive))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        image_format(path)
                    except PathSafetyError:
                        continue
                    yield path, is_scope
            # Push descending so the stack pops subdirectories in ascending
            # order, matching the ascending file order inside each directory.
            stack.extend(reversed(directories))

    def _record(self, sample_id: int, path: Path, in_scope: bool) -> tuple[SampleRecord, ManifestError | None]:
        relative = os.path.relpath(path, self.source_root).replace("/", "\\")
        fmt = image_format(path)
        defect: ManifestError | None = None
        try:
            frames, actual_format = _image_info(path, fmt)
        except ManifestError as exc:
            # ROADMAP.md:50/173: one unusable image is a per-sample blocking
            # issue; the scan keeps going instead of failing the whole task.
            frames, actual_format, defect = 0, fmt, exc
        processing_scope = in_scope and not (defect is not None and self.invalid_image_action == "skip")
        txt, json_path = annotation_paths(path)
        try:
            txt_state = read_annotation_state(txt) if txt.is_file() else "missing_or_blank"
            json_state = read_annotation_state(json_path) if json_path.is_file() else "missing_or_blank"
        except PathSafetyError as exc:
            raise ManifestError(str(exc)) from exc
        if processing_scope:
            for annotation in (txt, json_path):
                if annotation.is_file():
                    self.annotation_bytes += annotation.stat().st_size
                    self.annotation_files += 1
        return SampleRecord(
            sampleId=sample_id,
            relativeImagePath=relative,
            annotationKey=annotation_key(relative),
            source=self.profile,  # type: ignore[arg-type]
            inProcessingScope=processing_scope,
            imageFormat=actual_format,  # type: ignore[arg-type]
            imageFrameCount=frames,
            originalTxtState=txt_state,  # type: ignore[arg-type]
            originalJsonState=json_state,  # type: ignore[arg-type]
        ), defect

    def iter_records(self) -> Iterator[SampleRecord]:
        for record, _ in self.iter_scan_records():
            yield record

    def iter_scan_records(self) -> Iterator[tuple[SampleRecord, ManifestError | None]]:
        """Yield every discovered sample with its per-sample image defect, if any."""
        for record, defect, _in_scope in self._iter_scan_records_with_scope():
            yield record, defect

    def _iter_scan_records_with_scope(
        self,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[SampleRecord, ManifestError | None, bool]]:
        """Yield scan records together with the original scope classification."""
        if self.profile not in {"e621", "danbooru"}:
            raise ManifestError(f"unsupported profile:{self.profile}")
        assert_no_reparse_tree(self.source_root)
        self.annotation_bytes = 0
        self.annotation_files = 0
        collision_db = sqlite3.connect(":memory:")
        collision_db.create_collation("WIN_ORDINAL_NOCASE", windows_compare)
        collision_db.execute("CREATE TABLE keys(value TEXT COLLATE WIN_ORDINAL_NOCASE PRIMARY KEY)")
        try:
            sample_id = 0
            for path, in_scope in self._iter_paths(cancel_check=cancel_check):
                sample_id += 1
                record, defect = self._record(sample_id, path, in_scope)
                try:
                    collision_db.execute("INSERT INTO keys(value) VALUES (?)", (record.annotationKey,))
                except sqlite3.IntegrityError as exc:
                    raise ManifestError(f"annotationKey collision: {record.annotationKey}") from exc
                yield record, defect, in_scope
        finally:
            collision_db.close()

    def count(self) -> int:
        return sum(1 for _ in self.iter_records())

    def count_candidate_images(self, *, cancel_check: Callable[[], bool] | None = None) -> int:
        """Count in-scope supported image files without decoding image data."""
        assert_no_reparse_tree(self.source_root)
        return sum(1 for _path, in_scope in self._iter_paths(cancel_check=cancel_check) if in_scope)

    def create_manifest(self, job_id: str) -> SampleManifest:
        return SampleManifest(jobId=job_id, recursive=self.recursive, sampleCount=self.count(), generatedAt=utc_now())

    def scan_into(
        self,
        database: Any,
        job_id: str,
        *,
        batch_size: int = 500,
        total_files: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> SampleManifest:
        """Decode and persist a manifest in one bounded pass.

        Callers that need both the immutable header and rows should use this
        method instead of calling ``count()`` followed by ``iter_records()``;
        that avoids decoding every image twice.
        """
        if not 1 <= batch_size <= 1_000:
            raise ValueError("manifest batch_size must be between 1 and 1000")
        if total_files is None:
            total_files = self.count_candidate_images()
        if type(total_files) is not int or total_files < 0:
            raise ValueError("manifest total_files must be a non-negative integer")
        count = 0
        checked = 0
        self.image_issue_count = 0
        batch: list[dict[str, object]] = []
        completed = False
        try:
            for record, defect, in_scope in self._iter_scan_records_with_scope(cancel_check=cancel_check):
                count += 1
                batch.append(record_storage_values(self, record))
                if defect is not None:
                    self.image_issue_count += 1
                    skipped = self.invalid_image_action == "skip"
                    database.upsert_issue(SampleIssue(
                        issueId=f"{job_id}-image-{record.sampleId}", jobId=job_id, sampleId=record.sampleId,
                        relativeImagePath=record.relativeImagePath, moduleId="workspace",  # type: ignore[arg-type]
                        code=defect.code, severity="warning" if skipped else "error", blocking=not skipped, retriable=False,
                        message=str(defect), attempt=1,
                    ))
                if len(batch) >= batch_size:
                    database.insert_samples(job_id, batch, batch_size=batch_size)
                    batch.clear()
                if in_scope:
                    checked += 1
                    if checked > total_files:
                        raise ManifestError(
                            f"supported image set changed during scan: checked {checked} of {total_files}"
                        )
                    if on_progress is not None:
                        on_progress(checked, total_files)
            if batch:
                database.insert_samples(job_id, batch, batch_size=batch_size)
            if checked != total_files:
                raise ManifestError(
                    f"supported image set changed during scan: checked {checked} of {total_files}"
                )
            generated = utc_now()
            database.connection.execute(
                "UPDATE jobs SET sample_count=?,manifest_generated_at=? WHERE job_id=?",
                (count, generated, job_id),
            )
            completed = True
            return SampleManifest(jobId=job_id, recursive=self.recursive, sampleCount=count, generatedAt=generated)
        finally:
            if not completed:
                database.clear_manifest_rows(job_id)


def record_storage_values(builder: ManifestBuilder, record: SampleRecord) -> dict[str, object]:
    image = builder.source_root / record.relativeImagePath
    txt, json_path = annotation_paths(image)
    fingerprint = file_fingerprint(image)
    return {
        "sample_id": record.sampleId,
        "relative_image_path": record.relativeImagePath,
        "annotation_key": record.annotationKey,
        "source": record.source,
        "in_processing_scope": record.inProcessingScope,
        "image_format": record.imageFormat,
        "image_frame_count": record.imageFrameCount,
        "original_txt_state": record.originalTxtState,
        "original_json_state": record.originalJsonState,
        "image_file_id": fingerprint["file_id"],
        "image_size": fingerprint["size"],
        "image_mtime_ns": fingerprint["mtime_ns"],
        "original_txt_sha256": sha256_file(txt) if txt.is_file() and record.originalTxtState == "nonblank" else None,
        "original_json_sha256": sha256_file(json_path) if json_path.is_file() and record.originalJsonState == "nonblank" else None,
    }
