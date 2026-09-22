from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .path_safety import PathSafetyError, canonicalize
from .resource_catalog_package import ResourceFile, ResourcePackage
from .resource_catalog_validation import (
    KIND_LAYOUT,
    MAX_MANIFEST_BYTES,
    OCR_MODEL_DIRECTORY,
    TOKENIZER_DIRECTORY,
    ResourceCatalogError,
    ResourceKind,
    _CACHE_LOCK,
    _text,
)


DEFAULT_KEYS = frozenset(layout[1] for layout in KIND_LAYOUT.values())
DEFAULT_KIND = {layout[1]: kind for kind, layout in KIND_LAYOUT.items()}
_COMPATIBILITY_CACHE: set[tuple[str, str]] = set()
DANBOORU_TAGGER_SOURCE_URLS = {
    "caption-danbooru-cl-tagger-v2-00": "https://huggingface.co/cella110n/cl_tagger_v2/tree/main/v2_00",
    "caption-danbooru-wd-eva02-large-v3": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3",
}
DANBOORU_CLASSIFICATION_FALLBACK_RESOURCE_ID = "classify-e621-20260724-v1"
RESOURCE_PLACEHOLDERS: tuple[dict[str, object], ...] = (
    {
        "schemaVersion": 2,
        "kind": "tagging-model",
        "resourceId": "caption-danbooru-cl-tagger-v2-00",
        "resourceVersion": "cella110n/cl_tagger_v2:v2_00",
        "profile": "danbooru",
        "displayName": {"zh-CN": "CL Tagger v2（v2_00）", "en": "CL Tagger v2 (v2_00)"},
        "description": {
            "zh-CN": "Danbooru 默认 Tagger；受许可证约束，需从官方仓库手动安装。",
            "en": "Default Danbooru tagger; manual installation from the official gated repository is required.",
        },
        "runtimeFormat": "cl-tagger-v2-onnx-v1",
        "distribution": {
            "mode": "local-only",
            "sourceUrl": "https://huggingface.co/cella110n/cl_tagger_v2/tree/main/v2_00",
            "licenseUrl": "https://huggingface.co/cella110n/cl_tagger_v2/blob/main/LICENSE.md",
        },
        "adjustableCategories": ["general", "character", "copyright"],
        "excludedCategories": ["meta", "rating", "quality"],
        "defaultThresholds": {"general": 0.55, "character": 0.55, "copyright": 0.55},
    },
    {
        "schemaVersion": 2,
        "kind": "tagging-model",
        "resourceId": "caption-danbooru-wd-eva02-large-v3",
        "resourceVersion": "SmilingWolf/wd-eva02-large-tagger-v3",
        "profile": "danbooru",
        "displayName": {"zh-CN": "WD EVA02-Large v3", "en": "WD EVA02-Large v3"},
        "description": {
            "zh-CN": "可选 Danbooru Tagger；按项目策略从官方仓库手动安装。",
            "en": "Optional Danbooru tagger; project policy requires manual installation from the official repository.",
        },
        "runtimeFormat": "wd-eva02-large-tagger-v3-onnx-v1",
        "distribution": {
            "mode": "local-only",
            "sourceUrl": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3",
            "licenseUrl": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        "adjustableCategories": ["general", "character"],
        "excludedCategories": ["rating"],
        "defaultThresholds": {"general": 0.5296, "character": 0.5296},
    },
    {
        "schemaVersion": 2,
        "kind": "classification-index",
        "resourceId": "danbooru-classify-20260727-v1",
        "resourceVersion": "20260727-v1",
        "profile": "danbooru",
        "displayName": {
            "zh-CN": "Danbooru 分类与 Count 索引",
            "en": "Danbooru Classification and Count Index",
        },
        "description": {
            "zh-CN": "由已审计公开快照生成；当前正式资源尚未安装。",
            "en": "Built from audited public snapshots; the production resource is not installed yet.",
        },
        "runtimeFormat": "danbooru-classification-index-v1",
        "distribution": {"mode": "bundled"},
        "adjustableCategories": [],
        "excludedCategories": [],
        "defaultThresholds": {},
    },
)


def danbooru_resource_install_message(kind: ResourceKind, resource_id: str) -> str:
    """Actionable fatal text for a missing Danbooru resource; never implies fallback."""
    if kind == "tagging-model":
        source = DANBOORU_TAGGER_SOURCE_URLS.get(resource_id)
        source_text = f" from {source}" if source is not None else " from its official source"
        return (
            "manual_install_required: selected Danbooru tagger "
            f"{resource_id} is not installed in the top-level resource-library; "
            f"install it manually{source_text}. Automatic download and model fallback are disabled."
        )
    if kind == "classification-index":
        if resource_id == DANBOORU_CLASSIFICATION_FALLBACK_RESOURCE_ID:
            return (
                "resource_install_required: temporary E621 classification fallback "
                f"{resource_id} is unavailable in the top-level resource-library"
            )
        return (
            "resource_install_required: selected Danbooru classification resource "
            f"{resource_id} is unavailable in the top-level resource-library; install or generate it before "
            "starting the task. E621 classification fallback is disabled."
        )
    return f"resource_install_required: selected Danbooru {kind} is unavailable: {resource_id}"


def is_danbooru_e621_classification_fallback(resource_id: str, resource_profile: str) -> bool:
    return (
        resource_id == DANBOORU_CLASSIFICATION_FALLBACK_RESOURCE_ID
        and resource_profile == "e621"
    )


@dataclass(frozen=True)
class InvalidResource:
    relative_path: str
    reason: str

    def api_dict(self) -> dict[str, str]:
        return {"relativePath": self.relative_path, "reason": self.reason}


@dataclass(frozen=True)
class ResourceCatalogSnapshot:
    defaults_schema_version: int
    defaults: dict[str, str]
    packages: tuple[ResourcePackage, ...]
    invalid: tuple[InvalidResource, ...]

    def defaults_for(self, profile: str | None = None) -> dict[str, str]:
        return dict(self.defaults)

    def package(
        self,
        kind: ResourceKind,
        resource_id: str,
        *,
        verify_hashes: bool,
        profile: str | None = None,
    ) -> ResourcePackage:
        matches = [
            item for item in self.packages
            if item.kind == kind
            and item.resource_id == resource_id
            and (
                profile is None
                or kind == "dropout-model"
                or item.profile == profile
                or (
                    kind == "classification-index"
                    and profile == "danbooru"
                    and is_danbooru_e621_classification_fallback(item.resource_id, item.profile)
                )
            )
        ]
        if len(matches) != 1:
            raise ResourceCatalogError(f"selected {kind} is unavailable: {resource_id}")
        matches[0].verify_files(verify_hashes=verify_hashes)
        return matches[0]

    def missing_defaults(self, profile: str | None = None) -> tuple[dict[str, str], ...]:
        defaults = self.defaults_for(profile)
        missing: list[dict[str, str]] = []
        for default_key, resource_id in sorted(defaults.items()):
            kind = DEFAULT_KIND[default_key]
            matches = [
                item for item in self.packages
                if item.kind == kind and item.resource_id == resource_id
            ]
            if len(matches) != 1:
                missing.append({"kind": default_key, "resourceId": resource_id})
        return tuple(missing)

    def _compatibility(self, package: ResourcePackage) -> dict[str, str]:
        if package.kind != "tagging-model":
            return {"status": "not_applicable"}
        classification_id = self.defaults.get("classificationIndex")
        if classification_id is None:
            return {"status": "unavailable", "reason": "profile classification default is unavailable"}
        matches = [
            item for item in self.packages
            if item.kind == "classification-index"
            and item.resource_id == classification_id
        ]
        if len(matches) != 1:
            return {
                "status": "unavailable",
                "classificationResourceId": classification_id,
                "reason": "profile classification default is not installed",
            }
        try:
            verify_tagger_dictionary_compatibility(package, matches[0])
        except ResourceCatalogError as exc:
            return {
                "status": "incompatible",
                "classificationResourceId": classification_id,
                "reason": str(exc),
            }
        return {"status": "compatible", "classificationResourceId": classification_id}

    def api_dict(self) -> dict[str, object]:
        installed = {
            package.resource_id: package.api_dict(
                default=package.resource_id in self.defaults.values(),
                compatibility=self._compatibility(package),
            )
            for package in self.packages
        }
        placeholders = {
            str(definition["resourceId"]): {
                **definition,
                "fingerprint": None,
                "metadata": {},
                "compatibility": {"status": "unavailable", "reason": "not_installed"},
                "available": False,
                "default": definition["resourceId"] in self.defaults.values(),
            }
            for definition in RESOURCE_PLACEHOLDERS
            if definition["resourceId"] not in installed
        }
        return {
            "schemaVersion": 3,
            "defaultsSchemaVersion": self.defaults_schema_version,
            "defaults": self.defaults,
            "resources": sorted(
                (*installed.values(), *placeholders.values()),
                key=lambda item: (str(item["kind"]), str(item["resourceId"])),
            ),
            "invalidResources": [item.api_dict() for item in self.invalid],
        }


class ResourceCatalog:
    def __init__(self, root: str | Path) -> None:
        try:
            self.root = canonicalize(Path(root), must_exist=True, directory=True).value
        except PathSafetyError as exc:
            raise ResourceCatalogError(f"resource library root is unusable: {exc}") from exc

    def _defaults(self) -> tuple[int, dict[str, str]]:
        path = self.root / "defaults.json"
        try:
            data = path.read_bytes()
            value = json.loads(data.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ResourceCatalogError("defaults.json is unreadable or invalid") from exc
        if len(data) > MAX_MANIFEST_BYTES or not isinstance(value, dict) or set(value) != {"schemaVersion", "defaults"}:
            raise ResourceCatalogError("defaults.json fields are invalid")
        defaults = value["defaults"]
        schema_version = value["schemaVersion"]
        if schema_version != 3 or not isinstance(defaults, dict) or set(defaults) != DEFAULT_KEYS:
            raise ResourceCatalogError("defaults.json mapping is invalid")
        normalized = {
            key: _text(defaults[key], f"defaults.{key}", max_bytes=128)
            for key in sorted(defaults)
        }
        if len(set(normalized.values())) != len(normalized):
            raise ResourceCatalogError("default resource IDs must be unique")
        return 3, normalized

    def scan(
        self,
        *,
        include_tokenizers: bool = True,
        require_quality_stack: bool = False,
    ) -> ResourceCatalogSnapshot:
        defaults_schema_version, defaults = self._defaults()
        loaded: list[ResourcePackage] = []
        invalid: list[InvalidResource] = []
        for kind, (directory, _, _, _) in KIND_LAYOUT.items():
            category = self.root / directory
            if not category.is_dir():
                invalid.append(InvalidResource(directory, "resource category directory is missing"))
                continue
            for package_root in sorted(category.iterdir(), key=lambda path: path.name.lower()):
                if package_root.name.startswith("_"):
                    continue
                relative = str(package_root.relative_to(self.root)).replace("/", "\\")
                if not package_root.is_dir():
                    invalid.append(InvalidResource(relative, "resource package must be a directory"))
                    continue
                try:
                    loaded.append(ResourcePackage.load(self.root, package_root / "resource.json", kind))
                except (OSError, ResourceCatalogError) as exc:
                    invalid.append(InvalidResource(relative, str(exc)))
        ocr_category = self.root / OCR_MODEL_DIRECTORY
        if ocr_category.exists():
            if not ocr_category.is_dir():
                invalid.append(InvalidResource(OCR_MODEL_DIRECTORY, "resource category directory is missing"))
            else:
                for package_root in sorted(ocr_category.iterdir(), key=lambda path: path.name.lower()):
                    if package_root.name.startswith("_"):
                        continue
                    relative = str(package_root.relative_to(self.root)).replace("/", "\\")
                    if not package_root.is_dir():
                        invalid.append(InvalidResource(relative, "resource package must be a directory"))
                        continue
                    try:
                        loaded.append(
                            ResourcePackage.load(self.root, package_root / "resource.json", "ocr-model")
                        )
                    except (OSError, ResourceCatalogError) as exc:
                        invalid.append(InvalidResource(relative, str(exc)))
        if include_tokenizers:
            tokenizer_category = self.root / TOKENIZER_DIRECTORY
            if tokenizer_category.exists():
                if not tokenizer_category.is_dir():
                    invalid.append(InvalidResource(TOKENIZER_DIRECTORY, "resource category directory is missing"))
                else:
                    for package_root in sorted(tokenizer_category.iterdir(), key=lambda path: path.name.lower()):
                        if package_root.name.startswith("_"):
                            continue
                        relative = str(package_root.relative_to(self.root)).replace("/", "\\")
                        if not package_root.is_dir():
                            invalid.append(InvalidResource(relative, "resource package must be a directory"))
                            continue
                        try:
                            loaded.append(
                                ResourcePackage.load(self.root, package_root / "resource.json", "tokenizer")
                            )
                        except (OSError, ResourceCatalogError) as exc:
                            invalid.append(InvalidResource(relative, str(exc)))
        by_id: dict[str, list[ResourcePackage]] = {}
        for package in loaded:
            by_id.setdefault(package.resource_id, []).append(package)
        duplicates = {resource_id for resource_id, packages in by_id.items() if len(packages) > 1}
        if duplicates:
            for package in loaded:
                if package.resource_id in duplicates:
                    invalid.append(InvalidResource(package.manifest_relative_path, f"duplicate resourceId: {package.resource_id}"))
            loaded = [package for package in loaded if package.resource_id not in duplicates]
        snapshot = ResourceCatalogSnapshot(defaults_schema_version, defaults, tuple(loaded), tuple(invalid))
        missing_required = [
            record["resourceId"]
            for record in snapshot.missing_defaults()
            if require_quality_stack or record["kind"] != "dropoutModel"
        ]
        if missing_required:
            raise ResourceCatalogError("default resources are unavailable: " + ", ".join(sorted(missing_required)))
        return snapshot


def verify_tagger_dictionary_compatibility(
    tagging_model: ResourcePackage,
    classification_index: ResourcePackage,
) -> None:
    if tagging_model.kind != "tagging-model" or classification_index.kind != "classification-index":
        raise ResourceCatalogError("tagging compatibility requires a tagging model and classification index")
    if tagging_model.profile != classification_index.profile:
        if (
            tagging_model.profile == "danbooru"
            and is_danbooru_e621_classification_fallback(
                classification_index.resource_id, classification_index.profile,
            )
        ):
            return
        raise ResourceCatalogError("tagging model and classification index profiles do not match")
    key = (tagging_model.fingerprint, classification_index.fingerprint)
    with _CACHE_LOCK:
        if key in _COMPATIBILITY_CACHE:
            return
    if tagging_model.profile == "danbooru":
        vocabulary_fingerprint = tagging_model.metadata.get("vocabularyFingerprint")
        supported = classification_index.metadata.get("supportedVocabularyFingerprints")
        if (
            not isinstance(vocabulary_fingerprint, str)
            or not isinstance(supported, list)
            or vocabulary_fingerprint not in supported
        ):
            raise ResourceCatalogError("tagging vocabulary fingerprint is not supported by the classification index")
        with _CACHE_LOCK:
            _COMPATIBILITY_CACHE.add(key)
        return
    try:
        tags_value = json.loads(tagging_model.entrypoint("tags").read_text(encoding="utf-8"))
        dictionary_value = json.loads(classification_index.entrypoint("dictionary").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResourceCatalogError("tagging model or classification dictionary is unreadable") from exc
    tag_names = tags_value.get("tag_names") if isinstance(tags_value, dict) else None
    entries = dictionary_value.get("entries") if isinstance(dictionary_value, dict) else None
    if not isinstance(tag_names, list) or not tag_names or not isinstance(entries, dict) or not entries:
        raise ResourceCatalogError("tagging model or classification dictionary structure is invalid")
    if len(tag_names) != tagging_model.metadata["tagCount"] or any(not isinstance(tag, str) for tag in tag_names):
        raise ResourceCatalogError("tagging model tag count or labels are invalid")
    missing = [tag for tag in tag_names if tag not in entries]
    if missing:
        raise ResourceCatalogError(
            f"tagging model and classification index are incompatible: {len(missing)} labels are missing"
        )
    with _CACHE_LOCK:
        _COMPATIBILITY_CACHE.add(key)


def default_resource_library_root(install_root: str | Path | None = None) -> Path:
    configured = os.environ.get("ANIMA_RESOURCE_ROOT")
    if configured:
        return Path(configured)
    candidates: list[Path] = []
    if install_root is not None:
        install = Path(install_root)
        candidates.extend((install / "resource-library", install.parent / "resource-library"))
    current = Path.cwd()
    candidates.extend((current / "resource-library", current.parent / "resource-library"))
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "resource-library")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "defaults.json").is_file():
            return candidate
    raise ResourceCatalogError("project-local resource-library could not be located")
