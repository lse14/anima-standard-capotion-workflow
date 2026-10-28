"""Same-volume hard-link staging builder; it never commits a directory."""
from __future__ import annotations
import os, shutil
from pathlib import Path
from .moss_sidecar import MossSidecarError, parse_moss_json
from .ocr_sidecar import OcrSidecarError, ocr_sidecar_relative_path, parse_ocr_sidecar
from .overlay import MOSS_ANNOTATIONS_DIRECTORY
from .path_safety import PathSafetyError, assert_no_reparse_tree, atomic_write_bytes, canonicalize, ensure_within, image_format, long_path, safe_relative_path

class StagingError(RuntimeError): pass

FILE_ATTRIBUTE_HIDDEN=0x2

def _set_hidden(path:Path,hidden:bool)->None:
    if os.name!='nt': return
    import ctypes
    kernel32=ctypes.windll.kernel32
    attributes=kernel32.GetFileAttributesW(str(path))
    if attributes==0xFFFFFFFF: raise StagingError(f"unable to read file attributes: {path}")
    updated=attributes|FILE_ATTRIBUTE_HIDDEN if hidden else attributes&~FILE_ATTRIBUTE_HIDDEN
    if updated!=attributes and not kernel32.SetFileAttributesW(str(path),updated): raise StagingError(f"unable to update hidden attribute: {path}")

def reveal_staging(staging_root:str|Path)->None:
    """Drop the hidden attribute before staging is renamed into the formal dataset path."""
    _set_hidden(Path(staging_root),False)

def create_hardlink_staging(dataset_root:str|Path,staging_root:str|Path)->Path:
    dataset=canonicalize(dataset_root,must_exist=True,directory=True).value; staging=canonicalize(staging_root,directory=True).value
    if staging.exists(): raise StagingError("staging must be a new workspace path")
    try:
        if dataset.stat().st_dev != staging.parent.stat().st_dev:
            raise StagingError("staging must remain on the dataset volume")
    except OSError as exc:
        raise StagingError("staging volume cannot be verified") from exc
    assert_no_reparse_tree(dataset);staging.mkdir()
    stack=[(dataset,staging)]
    try:
        _set_hidden(staging,True)
        while stack:
            source,target=stack.pop()
            with os.scandir(source) as entries:
                for entry in entries:
                    source_path=Path(entry.path);target_path=ensure_within(staging, target/entry.name)
                    if entry.is_dir(follow_symlinks=False):target_path.mkdir();stack.append((source_path,target_path))
                    elif entry.is_file(follow_symlinks=False):os.link(long_path(source_path),long_path(target_path))
                    else:raise StagingError(f"unsupported filesystem entry: {source_path}")
    except Exception:
        shutil.rmtree(staging,ignore_errors=True);raise
    return staging

def replace_business_annotation(staging_root:str|Path,annotation_key:str,suffix:str,data:bytes|None)->Path|None:
    """Create or remove only a processing-scope business annotation in staging."""
    if suffix not in {'.txt','.json'}: raise StagingError('unsupported annotation suffix')
    root=canonicalize(staging_root,must_exist=True,directory=True).value;relative=safe_relative_path(annotation_key+suffix)
    target=ensure_within(root,root/Path(relative.replace('\\',os.sep)))
    if data is None:
        if target.exists(): target.unlink()
        return None
    return atomic_write_bytes(root,relative,data)


def replace_ocr_sidecar(staging_root: str | Path, relative_image_path: str, data: bytes) -> Path:
    """Atomically replace one validated OCR sidecar in the staging tree only."""
    try:
        image_format(relative_image_path)
        parse_ocr_sidecar(data, expected_relative_image_path=relative_image_path)
        relative = ocr_sidecar_relative_path(relative_image_path)
        root = canonicalize(staging_root, must_exist=True, directory=True).value
        target = ensure_within(root, root / Path(relative.replace("\\", os.sep)))
        if target.is_symlink():
            raise StagingError("OCR staging sidecar path is a symbolic link")
        if target.exists():
            canonicalize(target, must_exist=True, directory=False)
        return atomic_write_bytes(root, relative, data)
    except (OcrSidecarError, PathSafetyError) as exc:
        raise StagingError("OCR sidecar staging input is invalid") from exc


def replace_moss_sidecar(staging_root: str | Path, relative_image_path: str, data: bytes) -> Path:
    """Atomically replace one MOSS JSON sidecar in the export staging tree."""
    try:
        image_format(relative_image_path)
        relative = safe_relative_path(f"{MOSS_ANNOTATIONS_DIRECTORY}\\{relative_image_path}.moss.json")
        root = canonicalize(staging_root, must_exist=True, directory=True).value
        target = ensure_within(root, root / Path(relative.replace("\\", os.sep)))
        if target.is_symlink():
            raise StagingError("MOSS staging sidecar path is a symbolic link")
        if target.exists():
            canonicalize(target, must_exist=True, directory=False)
        if not isinstance(data, bytes) or not data:
            raise StagingError("MOSS staging sidecar is empty")
        parse_moss_json(data)
        return atomic_write_bytes(root, relative, data)
    except (MossSidecarError, PathSafetyError) as exc:
        raise StagingError("MOSS sidecar staging path is invalid") from exc
