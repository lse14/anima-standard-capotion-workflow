from __future__ import annotations

import ctypes
import hashlib
import json
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

from .contracts import utc_now
from .db import StateDatabase
from .path_safety import canonicalize


class DatasetLockError(RuntimeError):
    pass


class DatasetClaimConflict(DatasetLockError):
    def __init__(self, claiming_job_id: str) -> None:
        self.claiming_job_id = claiming_job_id
        super().__init__(f"dataset is claimed by task {claiming_job_id}")


GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_ALWAYS = 4
FILE_ATTRIBUTE_HIDDEN = 0x2
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def _lock_path(dataset_root: Path) -> Path:
    information = os.stat(dataset_root, follow_symlinks=False)
    if not information.st_ino:
        raise DatasetLockError(f"cannot obtain a stable Windows file identity for: {dataset_root}")
    identity = f"{information.st_dev}:{information.st_ino}".encode("ascii")
    digest = hashlib.sha256(identity).hexdigest()[:24]
    return dataset_root.parent / f".anima-dataset-{digest}.lock"


def _open_windows_lock(path: Path) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(str(path), GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_ALWAYS, FILE_ATTRIBUTE_HIDDEN, None)
    if handle in (None, INVALID_HANDLE_VALUE):
        error = ctypes.get_last_error()
        raise DatasetLockError(f"dataset lock is already held or cannot be created: {path} (winerror={error})")
    return int(handle)


def _write_windows_lock(handle: int, payload: bytes) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_pointer = kernel32.SetFilePointerEx
    set_pointer.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_uint32]
    set_pointer.restype = ctypes.c_int
    set_end = kernel32.SetEndOfFile
    set_end.argtypes = [ctypes.c_void_p]
    set_end.restype = ctypes.c_int
    write_file = kernel32.WriteFile
    write_file.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    write_file.restype = ctypes.c_int
    flush = kernel32.FlushFileBuffers
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    if not set_pointer(ctypes.c_void_p(handle), 0, None, 0) or not set_end(ctypes.c_void_p(handle)):
        raise DatasetLockError("unable to reset dataset lock metadata")
    buffer = ctypes.create_string_buffer(payload)
    written = ctypes.c_uint32()
    if not write_file(ctypes.c_void_p(handle), buffer, len(payload), ctypes.byref(written), None) or written.value != len(payload):
        raise DatasetLockError("unable to write dataset lock metadata")
    if not flush(ctypes.c_void_p(handle)):
        raise DatasetLockError("unable to flush dataset lock metadata")


def _close_windows_handle(handle: int) -> None:
    if os.name == "nt":
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(ctypes.c_void_p(handle))


def _remove_dataset_lock_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise DatasetLockError(f"cannot remove dataset lock file: {path}") from exc


@dataclass
class DatasetLock:
    database: StateDatabase
    dataset_root: Path
    job_id: str
    lock_path: Path
    handle: int
    released: bool = False
    handle_closed: bool = False

    @classmethod
    def acquire(cls, database: StateDatabase, dataset_root: str | Path, job_id: str) -> "DatasetLock":
        dataset = canonicalize(dataset_root, must_exist=True, directory=True).value
        path = _lock_path(dataset)
        if os.name != "nt":
            raise DatasetLockError("first release requires Windows lock semantics")
        handle = _open_windows_lock(path)
        payload = (
            json.dumps(
                {"schemaVersion": 1, "jobId": job_id, "datasetRoot": str(dataset), "acquiredAt": utc_now()},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        try:
            _write_windows_lock(handle, payload)
            with database.transaction(immediate=True):
                existing = database.connection.execute(
                    "SELECT job_id FROM dataset_claims WHERE dataset_root=?", (str(dataset),)
                ).fetchone()
                if existing is not None and existing["job_id"] != job_id:
                    raise DatasetClaimConflict(str(existing["job_id"]))
                if existing is None:
                    database.connection.execute(
                        """INSERT INTO dataset_claims(
                               dataset_root,dataset_root_key,job_id,lock_path,acquired_at
                           ) VALUES (?,?,?,?,?)""",
                        (str(dataset), str(dataset), job_id, str(path), utc_now()),
                    )
                else:
                    database.connection.execute(
                        """UPDATE dataset_claims SET dataset_root=?,dataset_root_key=?,lock_path=?,acquired_at=?
                           WHERE dataset_root=? AND job_id=?""",
                        (str(dataset), str(dataset), str(path), utc_now(), str(dataset), job_id),
                    )
        except Exception:
            _close_windows_handle(handle)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        return cls(database, dataset, job_id, path, handle)

    def release(self, *, recovery_complete: bool) -> None:
        if self.released:
            return
        if not recovery_complete:
            raise DatasetLockError("dataset lock cannot be released before recovery is complete")
        if not self.handle_closed:
            _close_windows_handle(self.handle)
            self.handle_closed = True
        _remove_dataset_lock_file(self.lock_path)
        with self.database.transaction(immediate=True):
            self.database.connection.execute(
                "DELETE FROM dataset_claims WHERE dataset_root=? AND job_id=?",
                (str(self.dataset_root), self.job_id),
            )
        self.released = True

    def __enter__(self) -> "DatasetLock":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.release(recovery_complete=True)
            return
        # Preserve the database claim after an exception.  The next startup
        # must inspect the job/journal before it can release this dataset.
        if not self.handle_closed:
            _close_windows_handle(self.handle)
            self.handle_closed = True
        self.released = True


def release_completed_dataset_lock(
    locks: MutableMapping[str, DatasetLock],
    database_path: str | Path,
    job_id: str,
) -> bool:
    """Release a successful task's process and durable dataset lock before history deletion."""
    lock = locks.get(job_id)
    if lock is not None:
        if lock.database.get_job(job_id)["status"] not in {"succeeded", "succeeded_with_issues"}:
            raise DatasetLockError("only a completed task can release its lock for history deletion")
        lock.release(recovery_complete=True)
        lock.database.close()
        del locks[job_id]
        return True

    database = StateDatabase.open(database_path)
    try:
        if database.get_job(job_id)["status"] not in {"succeeded", "succeeded_with_issues"}:
            raise DatasetLockError("only a completed task can release its lock for history deletion")
        claim = database.connection.execute(
            "SELECT lock_path FROM dataset_claims WHERE job_id=?", (job_id,)
        ).fetchone()
        if claim is None:
            return False
        lock_path = claim["lock_path"]
        if isinstance(lock_path, str) and lock_path:
            _remove_dataset_lock_file(Path(lock_path))
        database.release_dataset_claim(job_id)
    finally:
        database.close()
    return True
