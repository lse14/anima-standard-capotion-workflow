"""Standard-library orchestration for the source bootstrap installer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

_INSTALLER_ROOT = Path(__file__).resolve().parent
if str(_INSTALLER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_ROOT))

from assemble import (
    AssemblyError,
    InstallationPlan,
    PlannedComponent,
    assemble_resource,
    assemble_runtime,
    build_install_state,
    component_is_current,
    component_record,
    installation_plan,
    read_install_state,
    write_install_state,
)
from download import ManualDownloadRequired, download_verified
from manifest import Artifact, InstallManifest, ManifestError, load_manifest_path, sha256_bytes
from paths import (
    ProjectLayout,
    assert_within_root,
    cleanup_failure,
    cleanup_success,
    publish_directories,
    recover_transactions,
    safe_relative,
)


ArtifactFetcher = Callable[[Artifact], Path]
Probe = Callable[[PlannedComponent, Path], bool]
RuntimeManifestWriter = Callable[[PlannedComponent, ProjectLayout], str]
OptionalOcrModelImporter = Callable[[Path], object]
ProgressReporter = Callable[[str], None]

_MANUAL_OCR_ARCHIVE_FILENAMES = (
    "PP-OCRv5_server_det_infer.tar",
    "PP-OCRv5_server_rec_infer.tar",
    "PP-LCNet_x1_0_textline_ori_infer.tar",
)
_OCR_RESOURCE_RELATIVE = (
    Path("resource-library")
    / "ocr-models"
    / "ocr-ppocrv5-server-paddle-v1"
)
_NL_PROMPT_RESOURCE_IDS = (
    "nl-default-prompt-v1",
    "nl-default-prompt-v2",
    "nl-default-prompt-v3",
)

_RUNTIME_SOURCES: dict[str, dict[str, str]] = {
    "core": {"anima_core": "core/src/anima_core", "anima_caption_format": "shared/anima_caption_format/anima_caption_format"},
    "caption-e621": {"anima_caption_worker": "workers/caption/src/anima_caption_worker"},
    "classify-e621": {"anima_classify_worker": "workers/classify/src/anima_classify_worker"},
    "replace-e621": {"anima_replace_worker": "workers/replace/src/anima_replace_worker"},
    "nl": {"anima_nl_worker": "workers/nl/src/anima_nl_worker"},
    "policy": {"anima_policy_worker": "workers/policy/src/anima_policy_worker"},
    "export": {"anima_export_worker": "workers/export/src/anima_export_worker", "anima_caption_format": "shared/anima_caption_format/anima_caption_format"},
    "token-budget": {"anima_token_budget_worker": "workers/token_budget/src/anima_token_budget_worker", "anima_caption_format": "shared/anima_caption_format/anima_caption_format"},
    "ocr-paddle": {"anima_ocr_worker": "workers/ocr/src/anima_ocr_worker"},
    "ocr-paddle-gpu": {"anima_ocr_worker": "workers/ocr/src/anima_ocr_worker"},
}

_CUDA_PROBE_COMPANIONS = {
    "caption-e621": "e621-tagger",
    "policy": "policy",
}
_FUNCTIONAL_PROBE_GROUPS = (
    ("caption-e621", "e621-tagger"),
    ("token-budget", "qwen3-tokenizer"),
)
_CAPTION_RUNTIME_CPU_FALLBACK = "caption-e621"


@dataclass(frozen=True)
class InstallResult:
    installed_component_ids: tuple[str, ...]
    skipped_component_ids: tuple[str, ...]
    state_path: Path
    messages: tuple[str, ...]


def _owner_sources(source_root: Path, runtime_id: str) -> dict[str, Path]:
    try:
        values = _RUNTIME_SOURCES[runtime_id]
    except KeyError as exc:
        raise AssemblyError(f"runtime source mapping is missing: {runtime_id}") from exc
    result = {name: source_root / relative for name, relative in values.items()}
    missing = [str(path) for path in result.values() if not path.is_dir()]
    if missing:
        raise AssemblyError("runtime source is missing: " + ", ".join(missing))
    return result


def _runtime_owner_sources_are_current(
    source_root: Path,
    layout: ProjectLayout,
    item: PlannedComponent,
) -> bool:
    """Require source-owned Python modules to match before skipping a runtime."""
    if item.runtime_id is None:
        return True
    source_packages = _owner_sources(source_root, item.runtime_id)
    target = _published_target(layout, item) / "Lib" / "site-packages"
    try:
        for package_name, source_package in source_packages.items():
            installed_package = target / package_name
            if not installed_package.is_dir() or installed_package.is_symlink():
                return False
            source_files = {
                path.relative_to(source_package): path
                for path in source_package.rglob("*.py")
                if path.is_file() and not path.is_symlink()
            }
            installed_files = {
                path.relative_to(installed_package): path
                for path in installed_package.rglob("*.py")
                if path.is_file() and not path.is_symlink()
            }
            if set(source_files) != set(installed_files):
                return False
            for relative, source_file in source_files.items():
                if hashlib.sha256(source_file.read_bytes()).digest() != hashlib.sha256(
                    installed_files[relative].read_bytes()
                ).digest():
                    return False
    except OSError:
        return False
    return True


def _state_records(manifest: InstallManifest, plan: InstallationPlan, state: object) -> dict[str, object]:
    if not isinstance(state, dict):
        return {}
    if (
        state.get("schemaVersion") != 1
        or state.get("sourceCommit") != manifest.source_commit
        or state.get("releaseVersion") != manifest.release_version
        or state.get("installManifestSha256") != manifest.fingerprint
        or state.get("accelerator") != plan.accelerator
        or not isinstance(state.get("components"), dict)
    ):
        return {}
    return state["components"]


def _load_runtime_manifest_generator(source_root: Path):
    path = source_root / "packaging" / "scripts" / "generate_runtime_manifests.py"
    spec = importlib.util.spec_from_file_location("_source_bootstrap_runtime_manifest_generator", path)
    if spec is None or spec.loader is None:
        raise AssemblyError(f"runtime manifest generator is unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_runtime_manifest_at(source_root: Path, item: PlannedComponent, runtime_root: Path) -> Path:
    if item.runtime_id is None or item.lock_name is None:
        raise AssemblyError("runtime manifest requires a runtime and lock identity")
    generator = _load_runtime_manifest_generator(source_root)
    specs = generator.runtime_specs(
        include_ocr_paddle=item.runtime_id == "ocr-paddle",
        include_ocr_paddle_gpu=item.runtime_id == "ocr-paddle-gpu",
        lock_names={item.runtime_id: item.lock_name},
    )
    try:
        owner, entry_module, _lock_name, dll_paths = specs[item.runtime_id]
    except KeyError as exc:
        raise AssemblyError(f"runtime manifest identity is missing: {item.runtime_id}") from exc
    specs[item.runtime_id] = (owner, entry_module, item.lock_name, dll_paths)
    requirements_root = source_root / "packaging" / "requirements"
    value = generator.manifest(runtime_root, requirements_root, item.runtime_id, specs)
    manifest_path = runtime_root / "manifests" / "runtimes" / f"{item.runtime_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    source_lock = requirements_root / f"{item.lock_name}.lock"
    if not source_lock.is_file():
        raise AssemblyError(f"runtime lock is missing: {source_lock}")
    lock_target = runtime_root / "manifests" / "requirements" / f"{item.runtime_id}.lock"
    lock_target.parent.mkdir(parents=True, exist_ok=True)
    lock_target.write_bytes(source_lock.read_bytes())
    return manifest_path


def _default_runtime_manifest_writer(source_root: Path, item: PlannedComponent, layout: ProjectLayout) -> str:
    manifest_path = _write_runtime_manifest_at(source_root, item, layout.runtime_root)
    return str(manifest_path.relative_to(layout.project_root)).replace("/", "\\")


def _load_ocr_resource_module(source_root: Path):
    path = source_root / "packaging" / "scripts" / "ocr_resource.py"
    spec = importlib.util.spec_from_file_location("_source_bootstrap_ocr_resource", path)
    if spec is None or spec.loader is None:
        raise AssemblyError(f"OCR model importer is unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _load_nl_prompt_resource_module(source_root: Path):
    path = source_root / "packaging" / "scripts" / "assemble_nl_prompt_resource.py"
    spec = importlib.util.spec_from_file_location("_source_bootstrap_nl_prompt_resource", path)
    if spec is None or spec.loader is None:
        raise AssemblyError(f"NL prompt resource assembler is unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        raise AssemblyError("NL prompt resource assembly failed") from None
    return module


def _assemble_nl_prompt_resources(source_root: Path, install_root: Path) -> None:
    module = _load_nl_prompt_resource_module(source_root)
    resource_root = source_root / "packaging" / "resources"
    try:
        for resource_id in _NL_PROMPT_RESOURCE_IDS:
            module.assemble(resource_root / f"{resource_id}.txt", install_root)
        module.assemble_v4(resource_root, install_root)
        module.assemble_moss(resource_root, install_root)
    except Exception as exc:
        raise AssemblyError(f"NL prompt resource assembly failed: {exc}") from exc


def _complete_manual_ocr_archives(project_root: Path) -> bool:
    archive_root = project_root / "ocr-model-archives"
    return all((archive_root / filename).is_file() for filename in _MANUAL_OCR_ARCHIVE_FILENAMES)


def _has_optional_ocr_model_input(project_root: Path) -> bool:
    return _complete_manual_ocr_archives(project_root) or (
        project_root / _OCR_RESOURCE_RELATIVE
    ).exists()


def _has_published_ocr_cpu_runtime(project_root: Path) -> bool:
    runtime_python = project_root / ".runtime-build" / "runtimes" / "ocr-paddle" / "python.exe"
    try:
        return runtime_python.is_file() and not runtime_python.is_symlink()
    except OSError:
        return False


def _import_available_ocr_models(source_root: Path, project_root: Path) -> object | None:
    module = _load_ocr_resource_module(source_root)
    try:
        return module.import_available_local_model_resource(project_root)
    except Exception as exc:
        raise AssemblyError(f"OCR model import failed: {exc}") from exc


def _published_target(layout: ProjectLayout, item: PlannedComponent) -> Path:
    relative = safe_relative(item.component.target_relative_path)
    return layout.project_root / Path(relative.replace("\\", os.sep))


def _stage_target(stage_root: Path, item: PlannedComponent) -> Path:
    relative = safe_relative(item.component.target_relative_path).split("\\")
    if item.runtime_id is not None:
        if relative[:2] != [".runtime-build", "runtimes"] or len(relative) != 3:
            raise AssemblyError(f"runtime staging target is invalid: {item.component.component_id}")
        return stage_root / "runtimes" / item.runtime_id
    if not relative or relative[0].casefold() != "resource-library":
        raise AssemblyError(f"resource staging target is invalid: {item.component.component_id}")
    return stage_root.joinpath(*relative)


def _source_tree_artifact_path(source_root: Path, artifact: Artifact) -> Path:
    if artifact.delivery != "source-tree" or artifact.source_relative_path is None:
        raise AssemblyError(f"source-tree artifact identity is invalid: {artifact.artifact_id}")
    try:
        relative = safe_relative(artifact.source_relative_path)
        path = assert_within_root(
            source_root,
            source_root / Path(relative.replace("\\", os.sep)),
        )
    except Exception as exc:
        raise AssemblyError(f"source-tree artifact path is unsafe: {artifact.artifact_id}") from exc
    if not path.is_file() or path.is_symlink():
        raise AssemblyError(f"source-tree artifact is missing or unsafe: {artifact.artifact_id}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise AssemblyError(f"source-tree artifact is unreadable: {artifact.artifact_id}") from exc
    if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
        raise AssemblyError(f"source-tree artifact identity does not match: {artifact.artifact_id}")
    return path


def _resolve_artifact(source_root: Path, artifact: Artifact, fetch: ArtifactFetcher) -> Path:
    if artifact.delivery == "source-tree":
        return _source_tree_artifact_path(source_root, artifact)
    return fetch(artifact)


def _remove_staged_component(stage: Path) -> None:
    if not stage.exists():
        return
    if stage.is_symlink() or not stage.is_dir():
        raise AssemblyError(f"staged component is unsafe: {stage}")
    shutil.rmtree(stage)


def _assemble_component(
    *,
    layout: ProjectLayout,
    source_root: Path,
    item: PlannedComponent,
    stage: Path,
    base_runtime: str | Path,
    fetch: ArtifactFetcher,
    progress: ProgressReporter | None = None,
) -> None:
    artifact_paths: dict[str, Path] = {}
    for index, artifact in enumerate(item.variant.artifacts, 1):
        if progress is not None:
            progress(
                f"  Preparing artifact {index}/{len(item.variant.artifacts)}: "
                f"{artifact.artifact_id}"
            )
        artifact_paths[artifact.artifact_id] = _resolve_artifact(source_root, artifact, fetch)
    if item.runtime_id is not None:
        wheels = [
            (artifact_paths[artifact.artifact_id], artifact.relative_path)
            for artifact in item.variant.artifacts
        ]
        assemble_runtime(
            layout,
            item,
            base_runtime=base_runtime,
            wheels=wheels,
            destination=stage,
            owner_sources=_owner_sources(source_root, item.runtime_id),
            progress=progress,
        )
    else:
        assemble_resource(layout, item, artifact_paths=artifact_paths, destination=stage)


def _custom_probe_results(
    pending: list[PlannedComponent],
    targets: Mapping[str, Path],
    probe: Probe,
) -> dict[str, bool]:
    return {
        item.component.component_id: bool(probe(item, targets[item.component.component_id]))
        for item in pending
    }


def _default_probe_results(
    *,
    source_root: Path,
    pending: list[PlannedComponent],
    components: Sequence[PlannedComponent],
    targets: Mapping[str, Path],
    ocr_resource_target: Path | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, bool | None]:
    for item in pending:
        if item.runtime_id is not None:
            _write_runtime_manifest_at(source_root, item, _stage_target_root(targets[item.component.component_id], item))
    from probes import run_offline_probes

    selected_ids = {item.component.component_id for item in pending}
    for group in _FUNCTIONAL_PROBE_GROUPS:
        if selected_ids.intersection(group):
            selected_ids.update(group)
    probe_components = [
        item for item in components if item.component.component_id in selected_ids
    ]
    return run_offline_probes(
        probe_components,
        component_targets=targets,
        ocr_resource_target=ocr_resource_target,
        progress=progress,
    )


def _stage_target_root(target: Path, item: PlannedComponent) -> Path:
    if item.runtime_id is None or target.parent.name.casefold() != "runtimes":
        raise AssemblyError(f"runtime staging target is invalid: {item.component.component_id}")
    return target.parent.parent


def _cpu_fallback_items(manifest: InstallManifest) -> dict[str, PlannedComponent]:
    fallback_manifest = replace(
        manifest,
        components=tuple(
            component
            for component in manifest.components
            if component.component_id in _CUDA_PROBE_COMPANIONS
        ),
    )
    return {
        item.component.component_id: item
        for item in installation_plan(fallback_manifest, accelerator="cpu", optional_components={"policy"}).components
    }


def _fallback_group_component_ids(fallback_ids: set[str]) -> set[str]:
    return fallback_ids | {
        companion
        for component_id, companion in _CUDA_PROBE_COMPANIONS.items()
        if component_id in fallback_ids
    }


def _classify_probe_failures(
    pending: list[PlannedComponent],
    results: Mapping[str, bool | None],
) -> tuple[set[str], set[str], list[str]]:
    """Separate CUDA fallbacks from fatal probe failures."""
    items = {item.component.component_id: item for item in pending}
    fallback_ids = {
        component_id
        for component_id, item in items.items()
        if component_id in _CUDA_PROBE_COMPANIONS
        and item.variant.name == "cuda"
        and results.get(component_id) is not True
    }
    deferred = _fallback_group_component_ids(fallback_ids) & set(items)
    discarded_gpu = {
        component_id
        for component_id, item in items.items()
        if component_id == "ocr-gpu"
        and item.variant.name == "cuda"
        and results.get(component_id) is False
    }
    failures = sorted(
        component_id
        for component_id in items
        if results.get(component_id) is not True
        and not (
            component_id in {"ocr-cpu", "ocr-gpu"}
            and results.get(component_id) is None
        )
        and component_id not in deferred | discarded_gpu
    )
    return fallback_ids, discarded_gpu, failures


def install_project(
    *,
    project_root: str | Path,
    source_root: str | Path,
    manifest: InstallManifest,
    accelerator: str,
    optional_components: set[str] | frozenset[str] = frozenset(),
    base_runtime: str | Path,
    fetch_artifact: ArtifactFetcher | None = None,
    probe_component: Probe | None = None,
    write_runtime_manifest: RuntimeManifestWriter | None = None,
    require_mandatory_e621: bool = True,
    import_optional_ocr_models: OptionalOcrModelImporter | None = None,
    preserve_bootstrap_on_success: bool = False,
    progress: ProgressReporter | None = None,
) -> InstallResult:
    """Assemble, probe, publish and record a manifest-selected installation."""
    layout = ProjectLayout.create(project_root)
    source = Path(source_root)
    if require_mandatory_e621:
        from assemble import validate_mandatory_e621_components

        validate_mandatory_e621_components(manifest)
    plan = installation_plan(manifest, accelerator=accelerator, optional_components=optional_components)
    report = progress or (lambda _message: None)
    fetch = fetch_artifact or (lambda artifact: download_verified(artifact, layout.cache, progress=report))
    writer = write_runtime_manifest or (lambda item, current_layout: _default_runtime_manifest_writer(source, item, current_layout))
    try:
        layout.ensure_directories()
        recover_transactions(layout)
        if any(item.runtime_id == "nl" for item in plan.components):
            _assemble_nl_prompt_resources(source, layout.runtime_root)
        existing = _state_records(manifest, plan, read_install_state(layout))
        skipped: list[PlannedComponent] = []
        pending: list[PlannedComponent] = []
        for item in plan.components:
            if (
                component_is_current(layout, item, existing.get(item.component.component_id))
                and _runtime_owner_sources_are_current(source, layout, item)
            ):
                skipped.append(item)
            else:
                pending.append(item)
        stage_root = layout.staging / ("install-" + uuid.uuid4().hex)
        staged: dict[str, Path] = {}
        messages: list[str] = []
        targets: dict[str, Path] = {
            item.component.component_id: _published_target(layout, item)
            for item in skipped
        }
        for index, item in enumerate(pending, 1):
            report(
                f"Installing component {index}/{len(pending)}: "
                f"{item.component.component_id} ({item.variant.name})"
            )
            stage = _stage_target(stage_root, item)
            _assemble_component(
                layout=layout,
                source_root=source,
                item=item,
                stage=stage,
                base_runtime=base_runtime,
                fetch=fetch,
                progress=report,
            )
            staged[item.component.target_relative_path] = stage
            targets[item.component.component_id] = stage
            report(f"Prepared component: {item.component.component_id}")

        if probe_component is not None:
            results = _custom_probe_results(pending, targets, probe_component)
        elif pending:
            ocr_resource_target = layout.project_root / _OCR_RESOURCE_RELATIVE
            results = _default_probe_results(
                source_root=source,
                pending=pending,
                components=plan.components,
                targets=targets,
                ocr_resource_target=ocr_resource_target if ocr_resource_target.is_dir() else None,
                progress=report,
            )
        else:
            results = {}

        fallback_ids, discard_gpu, failures = _classify_probe_failures(pending, results)
        if failures:
            raise AssemblyError("offline probe failed: " + ", ".join(sorted(failures)))

        final_components = list(plan.components)
        final_pending = list(pending)
        rebuild_ids: set[str] = set()
        if fallback_ids:
            cpu_items = _cpu_fallback_items(manifest)
            rebuild_ids = fallback_ids - {_CAPTION_RUNTIME_CPU_FALLBACK}
            if _CAPTION_RUNTIME_CPU_FALLBACK in fallback_ids:
                messages.append(
                    "Caption CUDA offline probe failed; keeping the CUDA runtime so the worker can retry CPU if CUDA initialization fails"
                )
            for index, item in enumerate(tuple(final_components)):
                component_id = item.component.component_id
                if component_id not in rebuild_ids:
                    continue
                try:
                    fallback = cpu_items[component_id]
                except KeyError as exc:
                    raise AssemblyError(f"CPU fallback is unavailable: {component_id}") from exc
                stage = targets[component_id]
                _remove_staged_component(stage)
                staged.pop(item.component.target_relative_path, None)
                fallback_stage = _stage_target(stage_root, fallback)
                _assemble_component(
                    layout=layout,
                    source_root=source,
                    item=fallback,
                    stage=fallback_stage,
                    base_runtime=base_runtime,
                    fetch=fetch,
                    progress=report,
                )
                staged[fallback.component.target_relative_path] = fallback_stage
                targets[component_id] = fallback_stage
                final_components[index] = fallback
                final_pending[final_pending.index(item)] = fallback
                messages.append(f"{component_id} CUDA offline probe failed; rebuilding the CPU variant")
        if discard_gpu:
            for item in tuple(final_components):
                if item.component.component_id not in discard_gpu:
                    continue
                stage = targets.pop(item.component.component_id)
                _remove_staged_component(stage)
                staged.pop(item.component.target_relative_path, None)
                final_components.remove(item)
                final_pending.remove(item)
                messages.append("OCR GPU offline probe failed; GPU runtime was not published and OCR CPU remains selected")
        elif any(
            item.component.component_id == "ocr-gpu" and results.get("ocr-gpu") is None
            for item in pending
        ):
            messages.append("OCR GPU model functionality was not verified; GPU runtime remains installed")

        final_plan = InstallationPlan(plan.accelerator, tuple(final_components))
        if rebuild_ids:
            retry_component_ids = _fallback_group_component_ids(rebuild_ids)
            retry_items = [
                item
                for item in final_plan.components
                if item.component.component_id in retry_component_ids
            ]
            report(
                "Retrying offline probes after CPU fallback: "
                + ", ".join(item.component.component_id for item in retry_items)
            )
            if probe_component is not None:
                retry_results = _custom_probe_results(
                    retry_items,
                    targets,
                    probe_component,
                )
            else:
                retry_results = _default_probe_results(
                    source_root=source,
                    pending=retry_items,
                    components=final_plan.components,
                    targets=targets,
                    progress=report,
                )
            _retry_fallbacks, _retry_discarded_gpu, retry_failures = _classify_probe_failures(
                retry_items,
                retry_results,
            )
            if retry_failures:
                raise AssemblyError("offline probe failed after CPU fallback: " + ", ".join(sorted(retry_failures)))

        publish_directories(layout, staged)
        for item in final_pending:
            if item.runtime_id is not None:
                writer(item, layout)
        records: dict[str, object] = {}
        for item in final_plan.components:
            records[item.component.component_id] = component_record(layout, item)
            if not component_is_current(layout, item, records[item.component.component_id]):
                raise AssemblyError(f"published component cannot be verified: {item.component.component_id}")
        state = build_install_state(manifest, final_plan, records)
        state_path = write_install_state(layout, state)
        cleanup_success(layout, preserve_bootstrap=preserve_bootstrap_on_success)
        if import_optional_ocr_models is not None:
            if _complete_manual_ocr_archives(layout.project_root):
                import_optional_ocr_models(layout.project_root)
                messages.append("OCR model import completed")
            else:
                messages.append(
                    "OCR models are not installed; see OCR_MODEL_DOWNLOAD.md and place the exact archives in ocr-model-archives"
                )
        elif _has_optional_ocr_model_input(layout.project_root):
            if (
                _complete_manual_ocr_archives(layout.project_root)
                and not _has_published_ocr_cpu_runtime(layout.project_root)
            ):
                messages.append(
                    "OCR model archives are present, but the optional OCR CPU runtime is not installed; "
                    "install the optional OCR component before importing models"
                )
            else:
                optional_result = _import_available_ocr_models(source, layout.project_root)
                if optional_result is None:
                    messages.append(
                        "OCR models are not installed; see OCR_MODEL_DOWNLOAD.md and place the exact archives in ocr-model-archives"
                    )
                elif isinstance(optional_result, Mapping) and optional_result.get("resource") == "idempotent":
                    messages.append("OCR model resource is already verified")
                else:
                    messages.append("OCR model import completed")
        else:
            messages.append(
                "OCR models are not installed; see OCR_MODEL_DOWNLOAD.md and place the exact archives in ocr-model-archives"
            )
        return InstallResult(
            tuple(item.component.component_id for item in final_pending),
            tuple(item.component.component_id for item in skipped),
            state_path,
            tuple(messages),
        )
    except Exception:
        cleanup_failure(layout, preserve_bootstrap=True, preserve_cache=True)
        raise


def _bootstrap_runtime_from_argument(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_dir() or not (path / "python.exe").is_file():
        raise AssemblyError(f"bootstrap runtime is invalid: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--accelerator", choices=("cpu", "nvidia"), required=True)
    parser.add_argument("--include-policy", action="store_true")
    parser.add_argument("--include-quality-stack", action="store_true")
    parser.add_argument("--bootstrap-runtime", type=Path, required=True)
    arguments = parser.parse_args()
    raw_manifest = arguments.manifest.read_bytes()
    actual_sha256 = sha256_bytes(raw_manifest)
    if actual_sha256 != arguments.manifest_sha256:
        parser.error("install manifest SHA-256 does not match the PowerShell bootstrap identity")
    try:
        manifest = load_manifest_path(arguments.manifest)
        result = install_project(
            project_root=arguments.project_root,
            source_root=arguments.project_root,
            manifest=manifest,
            accelerator=arguments.accelerator,
            optional_components=frozenset(
                component_id
                for component_id, enabled in (
                    ("policy", arguments.include_policy),
                    ("quality-stack", arguments.include_quality_stack),
                )
                if enabled
            ),
            base_runtime=_bootstrap_runtime_from_argument(arguments.bootstrap_runtime),
            preserve_bootstrap_on_success=True,
            progress=print,
        )
    except (AssemblyError, ManifestError, ManualDownloadRequired, OSError) as exc:
        print(f"source bootstrap failed: {exc}", file=sys.stderr)
        return 1
    for message in result.messages:
        print(message)
    print("installed components: " + ", ".join(result.installed_component_ids))
    print("skipped components: " + ", ".join(result.skipped_component_ids))
    print(f"install state: {result.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
