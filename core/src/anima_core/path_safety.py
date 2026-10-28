from __future__ import annotations

import ctypes
import hashlib
import os
import sqlite3
import stat
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp", ".bmp": "bmp"}
RESERVED_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class PathSafetyError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalPath:
    value: Path
    key: str


def windows_key(path: str | Path) -> str:
    text = os.path.normpath(os.path.abspath(os.fspath(path))).replace("/", "\\")
    return _ordinal_fold(text)


def _ordinal_fold(value: str) -> str:
    """Compatibility key for persisted metadata and non-security lookups.

    Security and collision decisions use ``windows_compare`` directly because
    a persisted lowercase form is not equivalent to Win32 ordinal comparison
    for every Unicode string.
    """
    return unicodedata.normalize("NFC", value).lower()


def windows_compare(left: str, right: str) -> int:
    if os.name == "nt":
        compare = ctypes.windll.kernel32.CompareStringOrdinal
        compare.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int, ctypes.c_bool]
        compare.restype = ctypes.c_int
        result = compare(left, -1, right, -1, True)
        if result:
            return result - 2
        raise PathSafetyError(f"CompareStringOrdinal failed with Win32 error {ctypes.get_last_error()}")
    left_key = _ordinal_fold(left)
    right_key = _ordinal_fold(right)
    return (left_key > right_key) - (left_key < right_key)


_LONG_PATH_PREFIX = "\\\\?\\"


def long_path(path: str | Path) -> str | Path:
    """Return an extended-length path for filesystem calls on Windows.

    Windows still enforces MAX_PATH (260) on most APIs unless the path carries
    the ``\\\\?\\`` prefix. A dataset near that limit plus the workspace's
    `stage`/`rollback`/`backup` segments easily exceeds it, so every path that
    creates, renames or links gets this prefix. Short paths are returned unchanged
    so normal behaviour, error messages and comparisons stay identical.
    """
    raw = os.fspath(path)
    if os.name != "nt" or not raw or raw.startswith(_LONG_PATH_PREFIX):
        return path
    absolute = os.path.abspath(os.path.normpath(raw))
    if len(absolute) < 240:
        return path
    if absolute.startswith("\\\\"):
        return _LONG_PATH_PREFIX + "UNC" + absolute[1:]
    return _LONG_PATH_PREFIX + absolute


def _is_reparse(path: Path) -> bool:
    try:
        information = os.lstat(path)
    except OSError as exc:
        raise PathSafetyError(f"cannot inspect path without following links: {path}") from exc
    attributes = getattr(information, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)) or stat.S_ISLNK(information.st_mode)


def _check_component(component: str) -> None:
    if not component or component in {".", ".."}:
        raise PathSafetyError("empty or traversal path component")
    if ":" in component:
        raise PathSafetyError(f"alternate data stream is not allowed: {component}")
    if component.endswith((".", " ")):
        raise PathSafetyError(f"ambiguous trailing path component: {component}")
    base = component.split(".", 1)[0].upper()
    if base in RESERVED_DEVICE_NAMES:
        raise PathSafetyError(f"reserved Windows device name: {component}")


def _existing_components(path: Path) -> Iterable[Path]:
    anchor = Path(path.anchor)
    current = anchor
    for component in path.parts[1:] if path.anchor else path.parts:
        current = current / component
        try:
            os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise PathSafetyError(f"cannot inspect path component: {current}") from exc
        yield current


def _absolute_path_parts(path: str | Path) -> tuple[str, ...]:
    normalized = Path(os.path.abspath(os.path.normpath(os.fspath(path))))
    return tuple(normalized.parts)


def windows_paths_equal(left: str | Path, right: str | Path) -> bool:
    left_parts = _absolute_path_parts(left)
    right_parts = _absolute_path_parts(right)
    return len(left_parts) == len(right_parts) and all(
        windows_compare(left_part, right_part) == 0
        for left_part, right_part in zip(left_parts, right_parts)
    )


def windows_path_is_within(root: str | Path, child: str | Path, *, allow_equal: bool = True) -> bool:
    root_parts = _absolute_path_parts(root)
    child_parts = _absolute_path_parts(child)
    if len(child_parts) < len(root_parts) or (not allow_equal and len(child_parts) == len(root_parts)):
        return False
    return all(
        windows_compare(root_part, child_part) == 0
        for root_part, child_part in zip(root_parts, child_parts)
    )


def canonicalize(path: str | Path, *, must_exist: bool = False, directory: bool | None = None) -> CanonicalPath:
    raw = os.fspath(path)
    if "\x00" in raw or not raw:
        raise PathSafetyError("path is empty or contains NUL")
    if not os.path.isabs(raw):
        raise PathSafetyError("path must be absolute")
    candidate = Path(os.path.abspath(os.path.normpath(raw)))
    if not candidate.is_absolute():
        raise PathSafetyError("path must be absolute")
    for component in candidate.parts:
        if component not in (candidate.anchor, ""):
            _check_component(component)
    for existing in _existing_components(candidate):
        if _is_reparse(existing):
            raise PathSafetyError(f"reparse point is not allowed: {existing}")
    if must_exist and not candidate.exists():
        raise PathSafetyError(f"path does not exist: {candidate}")
    if directory is True and candidate.exists() and not candidate.is_dir():
        raise PathSafetyError(f"path is not a directory: {candidate}")
    if directory is False and candidate.exists() and not candidate.is_file():
        raise PathSafetyError(f"path is not a file: {candidate}")
    return CanonicalPath(candidate, windows_key(candidate))


def assert_no_reparse_tree(root: str | Path) -> None:
    canonical = canonicalize(root, must_exist=True, directory=True).value
    stack = [canonical]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                child = Path(entry.path)
                if _is_reparse(child):
                    raise PathSafetyError(f"reparse point found: {child}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(child)


def ensure_within(root: str | Path, child: str | Path) -> Path:
    root_path = canonicalize(root).value
    child_path = canonicalize(child).value
    if not windows_path_is_within(root_path, child_path):
        raise PathSafetyError(f"path escapes root: {child_path}")
    return child_path


def safe_relative_path(value: str) -> str:
    if not value or "\x00" in value:
        raise PathSafetyError("relative path is empty or contains NUL")
    normalized = value.replace("/", "\\")
    path = Path(normalized)
    if normalized.startswith("\\") or path.is_absolute() or Path(normalized).drive or (len(normalized) >= 2 and normalized[1] == ":"):
        raise PathSafetyError("relative path must not have a root or drive")
    parts = normalized.split("\\")
    for part in parts:
        _check_component(part)
    return "\\".join(parts)


def validate_source_output(source: str | Path, output: str | Path | None, work_mode: str) -> tuple[CanonicalPath, CanonicalPath | None]:
    source_path = canonicalize(source, must_exist=True, directory=True)
    if work_mode not in {"in_place", "full_copy"}:
        raise PathSafetyError("invalid work mode")
    if work_mode == "in_place":
        if output is not None:
            raise PathSafetyError("in_place must not have outputRoot")
        return source_path, None
    if output is None:
        raise PathSafetyError("full_copy requires outputRoot")
    output_path = canonicalize(output, directory=True)
    if windows_paths_equal(output_path.value, source_path.value):
        raise PathSafetyError("source and output must differ")
    if output_path.value.exists() and any(output_path.value.iterdir()):
        raise PathSafetyError("full_copy output must be absent or empty")
    if windows_path_is_within(source_path.value, output_path.value, allow_equal=False) or windows_path_is_within(
        output_path.value, source_path.value, allow_equal=False
    ):
        raise PathSafetyError("source and output must not be parent/child")
    parent = output_path.value.parent
    canonicalize(parent, must_exist=True, directory=True)
    return source_path, output_path


def image_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    try:
        return IMAGE_EXTENSIONS[suffix]
    except KeyError as exc:
        raise PathSafetyError(f"unsupported image extension: {suffix}") from exc


def annotation_key(relative_image_path: str | Path) -> str:
    safe = safe_relative_path(os.fspath(relative_image_path))
    path = Path(safe)
    return str(path.with_suffix("")).replace("/", "\\")


def detect_annotation_collisions(relative_image_paths: Iterable[str]) -> None:
    collision_db = sqlite3.connect(":memory:")
    collision_db.create_collation("WIN_ORDINAL_NOCASE", windows_compare)
    collision_db.execute(
        "CREATE TABLE keys(value TEXT COLLATE WIN_ORDINAL_NOCASE PRIMARY KEY, relative_path TEXT NOT NULL)"
    )
    try:
        for relative in relative_image_paths:
            key = annotation_key(relative)
            try:
                collision_db.execute("INSERT INTO keys(value,relative_path) VALUES (?,?)", (key, relative))
            except sqlite3.IntegrityError as exc:
                previous = collision_db.execute(
                    "SELECT relative_path FROM keys WHERE value=?", (key,)
                ).fetchone()
                raise PathSafetyError(
                    f"annotationKey collision: {previous[0] if previous else key} and {relative}"
                ) from exc
    finally:
        collision_db.close()


def read_annotation_state(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return "missing_or_blank"
    if not target.is_file():
        raise PathSafetyError(f"annotation path is not a file: {path}")
    data = target.read_bytes()
    if not data:
        return "missing_or_blank"
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PathSafetyError(f"annotation is not strict UTF-8: {path}") from exc
    return "missing_or_blank" if not text.strip() else "nonblank"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, int | str]:
    target = Path(path)
    info = target.stat()
    return {
        "file_id": f"{getattr(info, 'st_dev', 0)}:{getattr(info, 'st_ino', 0)}",
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def replace_with_sharing_retry(source: str | Path, destination: str | Path) -> None:
    source, destination = long_path(source), long_path(destination)
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) != 32 or attempt == 4:
                raise
            time.sleep(0.05)


def atomic_write_bytes(root: str | Path, relative_path: str, data: bytes) -> Path:
    root_path = canonicalize(root, must_exist=True, directory=True).value
    safe = safe_relative_path(relative_path)
    destination = ensure_within(root_path, root_path / Path(safe.replace("\\", os.sep)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_reparse(destination.parent):
        raise PathSafetyError(f"destination parent is a reparse point: {destination.parent}")
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=long_path(destination.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_sharing_retry(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
