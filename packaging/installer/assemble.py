"""Deterministic component planning and staged assembly for source bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
import filecmp
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Callable, Literal, Mapping
import zipfile

from manifest import Component, ComponentVariant, InstallManifest, ManifestError
from paths import ProjectLayout, PathSafetyError, assert_within_root, safe_extract_zip, safe_relative


class AssemblyError(RuntimeError):
    """A selected component cannot be assembled safely from its frozen inputs."""


MANDATORY_E621_COMPONENTS = frozenset(
    {
        "core",
        "caption-e621",
        "classify-e621",
        "replace-e621",
        "nl",
        "export",
        "token-budget",
        "e621-indexes",
        "e621-replacement-indexes",
        "e621-tagger",
        "qwen3-tokenizer",
    }
)

_OPTIONAL_COMPONENT_IDS = frozenset({"ocr-cpu", "ocr-gpu", "ocr-models", "policy", "quality-stack"})


def validate_mandatory_e621_components(manifest: InstallManifest) -> None:
    present = {component.component_id for component in manifest.components}
    missing = sorted(MANDATORY_E621_COMPONENTS - present)
    if missing:
        raise ManifestError("mandatory E621 components are missing: " + ", ".join(missing))
    invalid_optional = sorted(
        component.component_id
        for component in manifest.components
        if component.component_id in _OPTIONAL_COMPONENT_IDS and component.required
    )
    if invalid_optional:
        raise ManifestError(
            "optional component must set required=false: " + ", ".join(invalid_optional)
        )


_VARIANT = Literal["cpu", "cuda", "shared"]
_LOCK_NAMES: dict[str, dict[str, str]] = {
    "core": {"cpu": "core", "shared": "core"},
    "caption-e621": {"cpu": "caption-e621-cpu", "cuda": "caption-e621-cuda"},
    "classify-e621": {"cpu": "classify-e621", "shared": "classify-e621"},
    "replace-e621": {"cpu": "replace-e621", "shared": "replace-e621"},
    "nl": {"cpu": "nl", "shared": "nl"},
    "policy": {"cpu": "policy-cpu", "cuda": "policy-cuda"},
    "export": {"cpu": "export", "shared": "export"},
    "token-budget": {"cpu": "token-budget", "shared": "token-budget"},
    "ocr-cpu": {"cpu": "ocr-paddle", "shared": "ocr-paddle"},
    "ocr-gpu": {"cuda": "ocr-paddle-gpu"},
}


@dataclass(frozen=True)
class PlannedComponent:
    component: Component
    variant: ComponentVariant
    runtime_id: str | None
    lock_name: str | None


@dataclass(frozen=True)
class InstallationPlan:
    accelerator: Literal["cpu", "nvidia"]
    components: tuple[PlannedComponent, ...]

    @property
    def runtime_ids(self) -> frozenset[str]:
        return frozenset(item.runtime_id for item in self.components if item.runtime_id is not None)

    @property
    def lock_names(self) -> frozenset[str]:
        return frozenset(item.lock_name for item in self.components if item.lock_name is not None)


def _artifact_identity(artifact: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "id": artifact.artifact_id,
        "delivery": artifact.delivery,
        "sizeBytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "relativePath": artifact.relative_path,
    }
    if artifact.delivery == "source-tree":
        identity["sourceRelativePath"] = artifact.source_relative_path
    elif artifact.delivery == "candidate-release":
        identity["candidatePath"] = artifact.candidate_path
    else:
        identity["url"] = artifact.url
        identity["allowedHosts"] = list(artifact.allowed_hosts)
        if artifact.repository is not None:
            identity["repository"] = artifact.repository
            identity["revision"] = artifact.revision
    return identity


def component_fingerprint(item: PlannedComponent) -> str:
    """Return the immutable identity of one selected component variant."""
    value = {
        "componentId": item.component.component_id,
        "kind": item.component.kind,
        "required": item.component.required,
        "targetRelativePath": item.component.target_relative_path,
        "variant": item.variant.name,
        "peakBytes": item.variant.peak_bytes,
        "probe": item.variant.probe,
        "artifacts": [_artifact_identity(artifact) for artifact in item.variant.artifacts],
    }
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _tree_files(root: Path) -> set[str] | None:
    files: set[str] = set()
    try:
        for current, directories, names in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            for directory in directories:
                if (current_path / directory).is_symlink():
                    return None
            for name in names:
                path = current_path / name
                if path.is_symlink() or not path.is_file():
                    return None
                files.add(str(path.relative_to(root)).replace(os.sep, "\\"))
    except OSError:
        return None
    return files


def component_is_current(layout: ProjectLayout, item: PlannedComponent, record: object) -> bool:
    """Verify a prior component record before allowing an idempotent skip."""
    if not isinstance(record, dict):
        return False
    if record.get("componentId") != item.component.component_id or record.get("variant") != item.variant.name:
        return False
    if record.get("fingerprint") != component_fingerprint(item):
        return False
    if record.get("targetRelativePath") != item.component.target_relative_path:
        return False
    raw_files = record.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return False
    try:
        target = assert_within_root(layout.project_root, layout.project_root / Path(safe_relative(item.component.target_relative_path).replace("\\", os.sep)))
    except (PathSafetyError, OSError, ValueError):
        return False
    if not target.is_dir() or target.is_symlink():
        return False
    if item.runtime_id is not None:
        try:
            if any(entry.is_dir() and entry.name.casefold().endswith(".dist-info") for entry in target.iterdir()):
                return False
        except OSError:
            return False
    expected: dict[str, tuple[int, str]] = {}
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"relativePath", "sizeBytes", "sha256"}:
            return False
        try:
            relative = safe_relative(raw["relativePath"])
        except (PathSafetyError, TypeError, ValueError):
            return False
        key = relative.casefold()
        if key in {path.casefold() for path in expected}:
            return False
        if type(raw["sizeBytes"]) is not int or raw["sizeBytes"] < 0 or not isinstance(raw["sha256"], str) or len(raw["sha256"]) != 64:
            return False
        expected[relative] = (raw["sizeBytes"], raw["sha256"])
    actual = _tree_files(target)
    if actual is None or {path.casefold() for path in actual} != {path.casefold() for path in expected}:
        return False
    actual_by_case = {path.casefold(): path for path in actual}
    for relative, (size, digest) in expected.items():
        path = target / Path(actual_by_case[relative.casefold()].replace("\\", os.sep))
        try:
            if _digest(path) != (size, digest):
                return False
        except OSError:
            return False
    if item.runtime_id is not None:
        manifest_relative = record.get("runtimeManifestRelativePath")
        if not isinstance(manifest_relative, str):
            return False
        try:
            manifest_path = assert_within_root(
                layout.project_root,
                layout.project_root / Path(safe_relative(manifest_relative).replace("\\", os.sep)),
            )
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError, PathSafetyError):
            return False
        if not isinstance(value, dict) or not isinstance(value.get("runtime"), dict) or value["runtime"].get("runtimeId") != item.runtime_id:
            return False
        dependency_lock_sha256 = value["runtime"].get("dependencyLockSha256")
        canonical_lock = layout.runtime_root / "manifests" / "requirements" / f"{item.runtime_id}.lock"
        if (
            not isinstance(dependency_lock_sha256, str)
            or len(dependency_lock_sha256) != 64
            or not canonical_lock.is_file()
            or canonical_lock.is_symlink()
        ):
            return False
        try:
            if _digest(canonical_lock)[1] != dependency_lock_sha256:
                return False
        except OSError:
            return False
    return True


def build_install_state(
    manifest: InstallManifest,
    plan: InstallationPlan,
    records: Mapping[str, object],
    *,
    completed_at_utc: str | None = None,
) -> dict[str, object]:
    """Build the completion state only after every selected component has a record."""
    expected = {item.component.component_id: item for item in plan.components}
    if set(records) != set(expected):
        missing = sorted(set(expected) - set(records))
        extra = sorted(set(records) - set(expected))
        detail = ", ".join([*(f"missing {item}" for item in missing), *(f"unexpected {item}" for item in extra)])
        raise AssemblyError(f"required component state is incomplete: {detail}")
    normalized: dict[str, object] = {}
    for component_id, item in expected.items():
        record = records[component_id]
        if not isinstance(record, dict):
            raise AssemblyError(f"component state is invalid: {component_id}")
        if (
            record.get("componentId") != component_id
            or record.get("variant") != item.variant.name
            or record.get("fingerprint") != component_fingerprint(item)
            or record.get("targetRelativePath") != item.component.target_relative_path
        ):
            raise AssemblyError(f"component state identity is invalid: {component_id}")
        normalized[component_id] = json.loads(json.dumps(record, ensure_ascii=False, sort_keys=True))
    completed = completed_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schemaVersion": 1,
        "sourceCommit": manifest.source_commit,
        "releaseVersion": manifest.release_version,
        "installManifestSha256": manifest.fingerprint,
        "accelerator": plan.accelerator,
        "components": normalized,
        "completedAtUtc": completed,
    }


def _assert_ordinary_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise AssemblyError(f"assembly source is not an ordinary directory: {root}")
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in (*directories, *files)):
            raise AssemblyError(f"assembly source contains a link: {current_path}")


def _merge_extracted_tree(source: Path, destination: Path, seen: set[str], *, relative_prefix: str = "") -> None:
    for current, directories, files in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_directory = current_path.relative_to(source)
        for directory in directories:
            target_directory = destination / relative_directory / directory
            target_directory.mkdir(parents=True, exist_ok=True)
        for filename in files:
            source_file = current_path / filename
            relative = str(Path(relative_prefix) / source_file.relative_to(source)).replace(os.sep, "\\")
            key = relative.casefold()
            target = destination / Path(relative.replace("\\", os.sep))
            if target.exists() and filecmp.cmp(source_file, target, shallow=False):
                seen.add(key)
                continue
            if key in seen or target.exists():
                raise AssemblyError(f"duplicate wheel path: {relative}")
            seen.add(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)


def assemble_runtime(
    layout: ProjectLayout,
    item: PlannedComponent,
    *,
    base_runtime: str | Path,
    wheels: list[tuple[str | Path, str]],
    destination: str | Path,
    owner_sources: Mapping[str, str | Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    """Build one runtime in private staging without invoking pip or another builder."""
    target: Path | None = None
    try:
        layout.ensure_directories()
        base = assert_within_root(layout.project_root, base_runtime)
        target = assert_within_root(layout.staging, destination)
        if target.exists():
            raise AssemblyError(f"assembly destination already exists: {target}")
        _assert_ordinary_tree(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, target, symlinks=False)
        package_root = target / "Lib" / "site-packages"
        package_root.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        for index, (raw_wheel, manifest_relative_path) in enumerate(wheels, 1):
            if progress is not None:
                progress(
                    f"  Extracting wheel {index}/{len(wheels)}: "
                    f"{PureWindowsPath(manifest_relative_path).name}"
                )
            wheel = assert_within_root(layout.project_root, raw_wheel)
            manifest_name = PureWindowsPath(safe_relative(manifest_relative_path)).name
            if not wheel.is_file() or wheel.is_symlink() or not manifest_name.lower().endswith(".whl"):
                raise AssemblyError(f"wheel input is invalid: {wheel}")
            extracted = layout.staging / (".wheel-" + hashlib.sha256(str(wheel).encode("utf-8")).hexdigest())
            if extracted.exists():
                raise AssemblyError(f"wheel staging path already exists: {extracted}")
            try:
                safe_extract_zip(wheel, extracted)
                _merge_extracted_tree(extracted, package_root, seen)
            except (PathSafetyError, zipfile.BadZipFile, OSError) as exc:
                raise AssemblyError(f"wheel cannot be safely assembled: {wheel}") from exc
            finally:
                if extracted.exists():
                    shutil.rmtree(extracted, ignore_errors=True)
            if progress is not None:
                progress(f"  Extracted wheel {index}/{len(wheels)}")
        for package_name, raw_source in sorted((owner_sources or {}).items()):
            if "\\" in package_name or "/" in package_name:
                raise AssemblyError(f"owner package name is invalid: {package_name}")
            source = assert_within_root(layout.project_root, raw_source)
            _assert_ordinary_tree(source)
            _merge_extracted_tree(
                source,
                package_root,
                seen,
                relative_prefix=package_name,
            )
        for helper_name in ("pip", "wheel", "pytest"):
            helper = package_root / helper_name
            if helper.exists():
                if helper.is_symlink() or not helper.is_dir():
                    raise AssemblyError(f"build helper path is unsafe: {helper}")
                shutil.rmtree(helper)
            for metadata in package_root.glob(f"{helper_name}-*.dist-info"):
                if metadata.is_symlink() or not metadata.is_dir():
                    raise AssemblyError(f"build helper metadata path is unsafe: {metadata}")
                shutil.rmtree(metadata)
        files = _tree_files(target)
        if files is None:
            raise AssemblyError("assembled runtime contains an unsafe entry")
        return tuple(sorted(files))
    except Exception:
        if target is not None and target.exists() and target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        raise


def assemble_resource(
    layout: ProjectLayout,
    item: PlannedComponent,
    *,
    artifact_paths: Mapping[str, str | Path],
    destination: str | Path,
) -> tuple[str, ...]:
    """Copy a selected resource package into private staging without following links."""
    target: Path | None = None
    try:
        layout.ensure_directories()
        if item.component.kind == "runtime":
            raise AssemblyError(f"resource assembly received runtime component: {item.component.component_id}")
        target = assert_within_root(layout.staging, destination)
        if target.exists():
            raise AssemblyError(f"assembly destination already exists: {target}")
        target.mkdir(parents=True)
        seen: set[str] = set()
        for artifact in item.variant.artifacts:
            try:
                source = assert_within_root(layout.project_root, artifact_paths[artifact.artifact_id])
            except KeyError as exc:
                raise AssemblyError(f"resource artifact is unavailable: {artifact.artifact_id}") from exc
            relative = safe_relative(artifact.relative_path)
            key = relative.casefold()
            destination_file = target / Path(relative.replace("\\", os.sep))
            if key in seen or destination_file.exists():
                raise AssemblyError(f"duplicate resource path: {relative}")
            if not source.is_file() or source.is_symlink():
                raise AssemblyError(f"resource artifact input is invalid: {source}")
            seen.add(key)
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination_file)
        files = _tree_files(target)
        if files is None or not files:
            raise AssemblyError("assembled resource contains no ordinary files")
        return tuple(sorted(files))
    except Exception:
        if target is not None and target.exists() and target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        raise


def component_record(
    layout: ProjectLayout,
    item: PlannedComponent,
    *,
    runtime_manifest_relative_path: str | None = None,
) -> dict[str, object]:
    """Capture every published component file for a future idempotency check."""
    target = assert_within_root(
        layout.project_root,
        layout.project_root / Path(safe_relative(item.component.target_relative_path).replace("\\", os.sep)),
    )
    files = _tree_files(target)
    if files is None or not files:
        raise AssemblyError(f"published component is incomplete: {item.component.component_id}")
    records: list[dict[str, object]] = []
    for relative in sorted(files, key=str.casefold):
        path = target / Path(relative.replace("\\", os.sep))
        size, digest = _digest(path)
        records.append({"relativePath": relative, "sizeBytes": size, "sha256": digest})
    value: dict[str, object] = {
        "componentId": item.component.component_id,
        "variant": item.variant.name,
        "fingerprint": component_fingerprint(item),
        "targetRelativePath": item.component.target_relative_path,
        "files": records,
    }
    if item.runtime_id is not None:
        if runtime_manifest_relative_path is None:
            runtime_manifest_relative_path = f".runtime-build\\manifests\\runtimes\\{item.runtime_id}.json"
        value["runtimeManifestRelativePath"] = safe_relative(runtime_manifest_relative_path)
    return value


def read_install_state(layout: ProjectLayout) -> dict[str, object] | None:
    path = layout.runtime_root / "manifests" / "install-state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_install_state(layout: ProjectLayout, state: Mapping[str, object]) -> Path:
    """Atomically write a complete top-level state after all components are published."""
    layout.ensure_directories()
    directory = assert_within_root(layout.project_root, layout.runtime_root / "manifests")
    directory.mkdir(parents=True, exist_ok=True)
    target = assert_within_root(layout.project_root, directory / "install-state.json")
    temporary = assert_within_root(layout.project_root, directory / "install-state.json.tmp")
    payload = json.dumps(dict(state), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    except OSError as exc:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise AssemblyError("install state cannot be written") from exc
    return target


def _select_variant(component: Component, accelerator: Literal["cpu", "nvidia"]) -> tuple[_VARIANT, ComponentVariant] | None:
    preferred: tuple[_VARIANT, ...] = ("cuda", "cpu", "shared") if accelerator == "nvidia" else ("cpu", "shared")
    for name in preferred:
        variant = component.variants.get(name)
        if variant is not None:
            return name, variant
    if component.required:
        raise ManifestError(f"component has no usable variant: {component.component_id}")
    return None


def _runtime_id(component: Component) -> str | None:
    if component.kind != "runtime":
        return None
    parts = PureWindowsPath(component.target_relative_path).parts
    normalized = tuple(part.casefold() for part in parts)
    try:
        index = normalized.index("runtimes")
    except ValueError as exc:
        raise ManifestError(f"runtime target is invalid: {component.component_id}") from exc
    if index != len(parts) - 2 or not parts[-1]:
        raise ManifestError(f"runtime target is invalid: {component.component_id}")
    return parts[-1]


def _lock_name(component: Component, variant: _VARIANT) -> str | None:
    if component.kind != "runtime":
        return None
    try:
        return _LOCK_NAMES[component.component_id][variant]
    except KeyError as exc:
        raise ManifestError(f"runtime lock mapping is invalid: {component.component_id}:{variant}") from exc


def installation_plan(
    manifest: InstallManifest,
    *,
    accelerator: str,
    optional_components: set[str] | frozenset[str] = frozenset(),
) -> InstallationPlan:
    """Choose the exact CPU/NVIDIA component set without resolving dependencies."""
    if accelerator not in {"cpu", "nvidia"}:
        raise ManifestError("accelerator is invalid")
    unknown_optional = set(optional_components) - _OPTIONAL_COMPONENT_IDS
    if unknown_optional:
        raise ManifestError("unknown optional component: " + ", ".join(sorted(unknown_optional)))
    selected: list[PlannedComponent] = []
    for component in manifest.components:
        if component.component_id in _OPTIONAL_COMPONENT_IDS and not component.required and component.component_id not in optional_components:
            continue
        result = _select_variant(component, accelerator)
        if result is None:
            continue
        variant_name, variant = result
        selected.append(
            PlannedComponent(
                component=component,
                variant=variant,
                runtime_id=_runtime_id(component),
                lock_name=_lock_name(component, variant_name),
            )
        )
    return InstallationPlan(accelerator, tuple(selected))
