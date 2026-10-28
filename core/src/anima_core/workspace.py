from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .path_safety import assert_no_reparse_tree, canonicalize, ensure_within, validate_source_output


@dataclass(frozen=True)
class CopyProgress:
    files: int
    bytes: int
    currentRelativePath: str


@dataclass(frozen=True)
class CopyResult:
    datasetRoot: Path
    files: int
    bytes: int
    copied: bool


def prepare_dataset(
    source_root: str | Path,
    output_root: str | Path | None,
    work_mode: str,
    job_id: str,
    report: Callable[[CopyProgress], None] | None = None,
) -> CopyResult:
    source, output = validate_source_output(source_root, output_root, work_mode)
    assert_no_reparse_tree(source.value)
    if work_mode == "in_place":
        return CopyResult(source.value, 0, 0, False)
    assert output is not None
    safe_job = "".join(character for character in job_id if character.isalnum() or character in "-_")
    if not safe_job or safe_job != job_id:
        raise ValueError("unsafe jobId")
    staging = output.value.parent / f".{output.value.name}.anima-copy-{safe_job}"
    ensure_within(output.value.parent, staging)
    discarded_path = output.value.with_name(output.value.name + ".anima-empty")
    # A crash during the publish step can leave a staging tree or emptied
    # placeholder behind. Both are discardable scratch: publishing only ever
    # requires an absent or empty output directory, so the copy restarts cleanly
    # instead of failing forever on `copy staging already exists`.
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    if discarded_path.exists():
        shutil.rmtree(discarded_path, ignore_errors=True)
    staging.mkdir()
    files = 0
    total_bytes = 0
    stack: list[tuple[Path, Path, os.ScandirIterator[str]]] = []
    try:
        stack.append((source.value, staging, os.scandir(source.value)))
        while stack:
            source_dir, target_dir, entries = stack[-1]
            try:
                entry = next(entries)
            except StopIteration:
                entries.close()
                stack.pop()
                shutil.copystat(source_dir, target_dir, follow_symlinks=False)
                continue
            source_path = Path(entry.path)
            target_path = target_dir / entry.name
            if entry.is_dir(follow_symlinks=False):
                target_path.mkdir()
                stack.append((source_path, target_path, os.scandir(source_path)))
                continue
            if not entry.is_file(follow_symlinks=False):
                raise OSError(f"unsupported filesystem entry: {source_path}")
            shutil.copy2(source_path, target_path)
            size = source_path.stat().st_size
            files += 1
            total_bytes += size
            if report:
                report(CopyProgress(files, total_bytes, os.path.relpath(source_path, source.value)))
        if output.value.exists():
            if any(output.value.iterdir()):
                raise OSError("output changed and is no longer empty")
            # Move the emptied placeholder aside instead of deleting it, so the
            # publish step below is a single atomic rename. A crash between an
            # rmdir and the replace would otherwise leave no output directory and
            # a stranded staging tree that blocks every later attempt.
            if discarded_path.exists():
                shutil.rmtree(discarded_path, ignore_errors=True)
            os.replace(output.value, discarded_path)
        os.replace(staging, output.value)
        if discarded_path.exists():
            shutil.rmtree(discarded_path, ignore_errors=True)
    except Exception:
        for _, _, entries in reversed(stack):
            try:
                entries.close()
            except OSError:
                pass
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return CopyResult(output.value, files, total_bytes, True)
