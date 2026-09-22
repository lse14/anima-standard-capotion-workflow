"""ZIP64 backup writer and streaming restore staging for business annotations."""
from __future__ import annotations
import hashlib,json,os,tempfile,zipfile
from collections.abc import Callable,Sequence
from pathlib import Path
from .path_safety import safe_relative_path,sha256_file
from .export_staging import replace_business_annotation

class AnnotationBackupError(RuntimeError):
    pass


def _verify_archive_against_manifest(archive: zipfile.ZipFile) -> None:
    """Verify every recorded entry item by item before the backup is renamed into place."""
    with archive.open("manifest.jsonl") as manifest:
        for raw in manifest:
            try:
                entry = json.loads(raw.decode("utf-8"))
                if not entry["exists"]:
                    continue
                info = archive.getinfo(entry["path"])
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise AnnotationBackupError("backup verification failed") from exc
            if info.file_size != entry["size"] or hashlib.sha256(archive.read(info)).hexdigest() != entry["sha256"]:
                raise AnnotationBackupError("backup verification failed")


def verify_backup(backup_zip: str | Path) -> Path:
    """Run the same archive and manifest verification used before publication."""
    backup = Path(backup_zip)
    if not backup.is_file():
        raise AnnotationBackupError("backup is unavailable")
    try:
        with zipfile.ZipFile(backup) as archive:
            if archive.testzip() is not None or "manifest.jsonl" not in archive.namelist():
                raise AnnotationBackupError("backup verification failed")
            _verify_archive_against_manifest(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AnnotationBackupError("backup verification failed") from exc
    return backup


def write_backup(dataset_root: str | Path, backup_zip: str | Path, page: Callable[[int | None], Sequence[object]]) -> Path:
    root, target = Path(dataset_root), Path(backup_zip)
    partial = target.with_suffix(target.suffix + ".partial")
    if target.exists() or partial.exists():
        raise AnnotationBackupError("backup destination already exists")
    cursor: int | None = None
    manifest_path: Path | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".manifest.", suffix=".jsonl", dir=target.parent)
        manifest_path = Path(temporary)
        with os.fdopen(fd, "wb") as manifest, zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            while True:
                rows = page(cursor)
                if not rows:
                    break
                cursor = int(rows[-1]["sample_id"])
                for row in rows:
                    if not row["in_processing_scope"]:
                        continue
                    key = safe_relative_path(str(row["annotation_key"]))
                    for suffix in (".txt", ".json"):
                        # ZipInfo always stores "/" separators, so manifest paths use them too.
                        relative = (key + suffix).replace("\\", "/")
                        source = root / Path(relative.replace("/", os.sep))
                        entry: dict[str, object] = {"path": relative, "exists": source.is_file()}
                        if source.is_file():
                            entry.update({"size": source.stat().st_size, "mtimeNs": source.stat().st_mtime_ns, "sha256": sha256_file(source)})
                            archive.write(source, relative)
                        manifest.write((json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
            manifest.flush()
            os.fsync(manifest.fileno())
            archive.write(manifest_path, "manifest.jsonl")
        verify_backup(partial)
        os.replace(partial, target)
        return target
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    finally:
        if manifest_path and manifest_path.exists():
            manifest_path.unlink()


def restore_to_staging(backup_zip: str | Path, staging_root: str | Path) -> int:
    """Restore exactly the ZIP manifest's business files into an existing staging tree."""
    backup, staging = Path(backup_zip), Path(staging_root)
    if not backup.is_file() or not staging.is_dir():
        raise AnnotationBackupError("restore source or staging directory is unavailable")
    try:
        with zipfile.ZipFile(backup) as archive:
            if archive.testzip() is not None or "manifest.jsonl" not in archive.namelist():
                raise AnnotationBackupError("backup verification failed")
            restored, last_key, last_suffix_rank = 0, "", -1
            with archive.open("manifest.jsonl") as manifest:
                for raw in manifest:
                    try:
                        entry = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise AnnotationBackupError("backup manifest is invalid") from exc
                    if not isinstance(entry, dict) or set(entry) not in ({"path", "exists"}, {"path", "exists", "size", "mtimeNs", "sha256"}):
                        raise AnnotationBackupError("backup manifest fields are invalid")
                    relative = safe_relative_path(entry.get("path", ""))
                    if not relative.endswith((".txt", ".json")):
                        raise AnnotationBackupError("backup manifest path is invalid")
                    suffix = Path(relative).suffix.lower()
                    key, rank = relative[:-len(suffix)], 0 if suffix == ".txt" else 1
                    if key.casefold() < last_key.casefold() or (key.casefold() == last_key.casefold() and rank <= last_suffix_rank):
                        raise AnnotationBackupError("backup manifest ordering is invalid")
                    if key.casefold() != last_key.casefold():
                        last_key, last_suffix_rank = key, -1
                    last_suffix_rank = rank
                    if type(entry.get("exists")) is not bool:
                        raise AnnotationBackupError("backup manifest existence flag is invalid")
                    if not entry["exists"]:
                        replace_business_annotation(staging, key, suffix, None)
                        restored += 1
                        continue
                    if not isinstance(entry.get("size"), int) or entry["size"] < 0 or not isinstance(entry.get("mtimeNs"), int) or entry["mtimeNs"] < 0 or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
                        raise AnnotationBackupError("backup manifest metadata is invalid")
                    try:
                        info = archive.getinfo(relative.replace("\\", "/"))
                    except KeyError as exc:
                        raise AnnotationBackupError("backup archive entry is missing") from exc
                    if info.file_size != entry["size"]:
                        raise AnnotationBackupError("backup archive size mismatch")
                    data = archive.read(info)
                    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                        raise AnnotationBackupError("backup archive digest mismatch")
                    target = replace_business_annotation(staging, key, suffix, data)
                    assert target is not None
                    os.utime(target, ns=(entry["mtimeNs"], entry["mtimeNs"]))
                    restored += 1
            return restored
    except (OSError, zipfile.BadZipFile) as exc:
        raise AnnotationBackupError("backup cannot be restored") from exc
