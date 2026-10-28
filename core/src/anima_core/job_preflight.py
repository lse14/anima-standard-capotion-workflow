from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .contracts import (
    CaptionFormatPolicy,
    ImageDecodePolicy,
    JobConfig,
    job_config_supports_caption_input_txt_mode,
    job_config_supports_nl_v4,
    job_config_supports_ocr,
    job_config_supports_ocr_device,
    job_config_supports_token_budget,
    utc_now,
    validate_job_config,
)
from .custom_replace_index import CustomReplaceIndex, CustomReplaceIndexError, inspect_custom_replace_index
from .custom_classification_resource import (
    CustomClassificationResource,
    CustomClassificationResourceError,
    freeze_custom_classification_resource,
    inspect_custom_classification_resource,
)
from .db import StateDatabase, assert_database_outside_datasets
from .locks import DatasetLock, DatasetLockError, release_completed_dataset_lock
from .manifest import ManifestBuilder, ManifestCancelled, ManifestError
from .nl_runner import nl_http_attempt_budget
from .nl_length import character_name
from .nl_profiles import NlApiProfileStore, NlProfileError
from .launcher import WorkerLaunchError, WorkerLauncher
from .ocr_runtime_binding import (
    OcrExecutionRequestV1,
    read_execution_request,
    write_execution_request,
)
from .overlay import OverlayLayout
from .path_safety import (
    CanonicalPath,
    PathSafetyError,
    ensure_within,
    file_fingerprint,
    safe_relative_path,
    validate_source_output,
    windows_key,
)
from .resource_catalog import (
    ResourceCatalog,
    ResourceCatalogError,
    ResourceCatalogSnapshot,
    ResourceKind,
    ResourcePackage,
    default_resource_library_root,
    verify_tagger_dictionary_compatibility,
)
from .resource_catalog_validation import OCR_MODEL_RESOURCE_ID
from .workspace import prepare_dataset


class JobPreflightError(ValueError):
    pass


# Fallback per-annotation size used only when a dataset has no annotation yet.
DEFAULT_ANNOTATION_BYTES = 1024


def _nl_api_work_enabled(config: JobConfig) -> bool:
    return (
        config.nl.get("enabled") is True
        and config.nl.get("apiEnabled") is True
        and not (
            job_config_supports_caption_input_txt_mode(config.schemaVersion)
            and config.caption.get("inputTxtMode") == "nl"
        )
    )


@dataclass(frozen=True)
class PreflightSummary:
    jobId: str
    sampleCount: int
    inScopeCount: int
    outOfScopeCount: int
    nonblankTxtCount: int
    nonblankJsonCount: int
    configHash: str
    replaceIndex: dict[str, object] | None = None
    resources: dict[str, object] = field(default_factory=dict)
    blankTxtCount: int = 0
    blankJsonCount: int = 0
    annotationKeyCollisionCount: int = 0
    imageIssueCount: int = 0
    projection: dict[str, object] = field(default_factory=dict)
    estimate: dict[str, int] = field(default_factory=dict)
    api: dict[str, int] = field(default_factory=dict)


def config_from_dict(value: object) -> JobConfig:
    if not isinstance(value, dict):
        raise JobPreflightError("JobConfig must be an object")
    try:
        caption_format = value["captionFormat"]
        image_decode = value["imageDecode"]
        if not isinstance(caption_format, dict) or not isinstance(image_decode, dict):
            raise TypeError("nested policy is invalid")
        schema_version = value.get("schemaVersion")
        if schema_version != 10:
            raise JobPreflightError(
                "legacy JobConfig is incompatible; reinitialize the state database and create a new task"
            )
        allowed_fields = {
            "schemaVersion", "workMode", "overwriteMode", "sourceRoot", "outputRoot",
            "annotationBackup", "recursive", "captionFormat", "imageDecode", "caption",
            "classify", "replace", "ocr", "nl", "countReview", "dropout", "tokenBudget", "export", "moduleBatchSize",
        }
        if set(value) - allowed_fields:
            raise JobPreflightError("JobConfig shape is invalid; task profile is not supported by schema v9")
        if not job_config_supports_ocr(schema_version) and "ocr" in value:
            raise JobPreflightError("OCR is only supported by JobConfig v5 through v9")
        count_review = value.get("countReview")
        ocr = value.get("ocr")
        config = JobConfig(
            workMode=value["workMode"], overwriteMode=value["overwriteMode"], sourceRoot=value["sourceRoot"],
            outputRoot=value.get("outputRoot"), annotationBackup=value.get("annotationBackup", "required"),
            recursive=value["recursive"], captionFormat=CaptionFormatPolicy(**{**caption_format, "triggerTerms": tuple(caption_format.get("triggerTerms", ())) }),
            imageDecode=ImageDecodePolicy(**image_decode), caption=dict(value["caption"]), classify=dict(value["classify"]),
            replace=dict(value["replace"]), ocr=dict(ocr) if ocr is not None else {}, nl=dict(value["nl"]),
            countReview=dict(count_review) if count_review is not None else None,
            dropout=dict(value["dropout"]), export=dict(value["export"]), schemaVersion=schema_version,
            tokenBudget=dict(value["tokenBudget"]) if value.get("tokenBudget") is not None else None,
            moduleBatchSize=dict(value["moduleBatchSize"]),
        )
    except (KeyError, TypeError) as exc:
        raise JobPreflightError("JobConfig shape is invalid") from exc
    try:
        validate_job_config(config)
    except ValueError as exc:
        raise JobPreflightError(str(exc)) from exc
    if _nl_api_work_enabled(config) and not isinstance(config.nl.get("systemPrompt"), str):
        raise JobPreflightError("NL system prompt is invalid")
    if _nl_api_work_enabled(config) and not str(config.nl.get("systemPrompt", "")).strip():
        raise JobPreflightError("API-enabled NL requires a non-empty system prompt")
    if (
        _nl_api_work_enabled(config)
        and job_config_supports_nl_v4(config.schemaVersion)
        and isinstance(config.nl.get("systemPrompt"), str)
        and len(config.nl["systemPrompt"].encode("utf-8")) > 16_384
    ):
        raise JobPreflightError("v6 NL user supplement exceeds 16 KiB")
    return config


class JobPreparationService:
    """Creates immutable manifest drafts before any overlay/copy/worker activity."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        resource_catalog: ResourceCatalog | None = None,
        install_root: str | Path | None = None,
        profile_store: NlApiProfileStore | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.resource_catalog = resource_catalog or ResourceCatalog(default_resource_library_root())
        self.install_root = Path(install_root) if install_root is not None else None
        self.profile_store = profile_store
        self._locks: dict[str, DatasetLock] = {}
        self._preflight_lock = threading.Lock()
        self._preflight_threads: dict[str, threading.Thread] = {}
        self._preflight_cancel: dict[str, threading.Event] = {}

    def _preflight_ocr_request_path(self, job_id: str) -> Path:
        if not job_id or not job_id.isalnum():
            raise JobPreflightError("OCR execution request job identity is invalid")
        return self.database_path.parent / f".{self.database_path.name}.ocr-requests" / job_id / "resources" / "ocr-execution-request-v1.json"

    @staticmethod
    def _rebind_full_copy_manifest(database: StateDatabase, job_id: str, dataset_root: Path) -> None:
        """Bind immutable image checks to the copied dataset, not the source inode."""
        rows = list(database.connection.execute(
            "SELECT sample_id,relative_image_path FROM samples WHERE job_id=?",
            (job_id,),
        ))
        with database.transaction(immediate=True):
            for row in rows:
                relative = safe_relative_path(str(row["relative_image_path"]))
                image = ensure_within(
                    dataset_root,
                    dataset_root / Path(relative.replace("\\", os.sep)),
                )
                fingerprint = file_fingerprint(image)
                database.connection.execute(
                    """UPDATE samples SET image_file_id=?,image_size=?,image_mtime_ns=?
                       WHERE job_id=? AND sample_id=?""",
                    (
                        fingerprint["file_id"],
                        fingerprint["size"],
                        fingerprint["mtime_ns"],
                        job_id,
                        int(row["sample_id"]),
                    ),
                )

    @staticmethod
    def _reject_client_frozen_resources(value: object) -> None:
        if not isinstance(value, dict):
            return
        sections = [value.get(name) for name in ("caption", "classify", "replace")]
        sections.extend((value.get("ocr"), value.get("tokenBudget")))
        dropout = value.get("dropout")
        if isinstance(dropout, dict):
            sections.append(dropout.get("quality"))
        if any(
            isinstance(section, dict)
            and ({
                "resourceManifestRelativePath", "resourceFingerprint", "contextLimit",
                "wikiDataSourceId", "dictionaryEntryCount", "resourceProfile",
                "customResourceContentSha256",
            } & set(section))
            for section in sections
        ):
            raise JobPreflightError("resource path and fingerprint are assigned by preflight")

    @staticmethod
    def _resource_id(section: dict[str, object], default_id: str, field: str) -> str:
        value = section.get("resourceId", default_id)
        if not isinstance(value, str):
            raise ResourceCatalogError(f"{field} resourceId is invalid")
        return value

    def _restore_missing_bundled_resource_ids(self, raw_config: object) -> object:
        """Restore catalog defaults only for a current v10 ready task."""
        if not isinstance(raw_config, dict) or raw_config.get("schemaVersion") != 10:
            return raw_config
        token_budget = raw_config.get("tokenBudget")
        include_tokenizers = isinstance(token_budget, dict) and token_budget.get("enabled") is True
        defaults = self.resource_catalog.scan(include_tokenizers=include_tokenizers).defaults_for()

        def restore(section_name: str, default_key: str) -> None:
            section = raw_config.get(section_name)
            if isinstance(section, dict) and "resourceId" not in section:
                section["resourceId"] = defaults[default_key]

        restore("caption", "taggingModel")
        classify = raw_config.get("classify")
        if isinstance(classify, dict) and classify.get("indexMode") == "bundled" and "resourceId" not in classify:
            classify["resourceId"] = defaults["classificationIndex"]
        replace = raw_config.get("replace")
        if isinstance(replace, dict) and replace.get("indexMode") == "bundled" and "resourceId" not in replace:
            replace["resourceId"] = defaults["replacementIndex"]
        dropout = raw_config.get("dropout")
        quality = dropout.get("quality") if isinstance(dropout, dict) else None
        if isinstance(quality, dict) and "resourceId" not in quality:
            quality["resourceId"] = defaults["dropoutModel"]
        ocr = raw_config.get("ocr")
        if isinstance(ocr, dict) and ocr.get("enabled") is True and "resourceId" not in ocr:
            ocr["resourceId"] = OCR_MODEL_RESOURCE_ID
        return raw_config

    @staticmethod
    def _freeze_reference(section: dict[str, object], package: ResourcePackage) -> None:
        section["resourceId"] = package.resource_id
        section["resourceManifestRelativePath"] = package.manifest_relative_path
        section["resourceFingerprint"] = package.fingerprint

    def _validate_ocr_runtime(self, config: JobConfig) -> None:
        if self.install_root is None or not job_config_supports_ocr_device(config.schemaVersion):
            return
        ocr = config.ocr
        if not isinstance(ocr, dict) or ocr.get("enabled") is not True:
            return
        device = ocr.get("device", "auto")
        if device not in {"auto", "cuda", "cpu"}:
            raise JobPreflightError("OCR device is invalid")
        runtime_ids = ["ocr-paddle"]
        if device == "cuda":
            runtime_ids.append("ocr-paddle-gpu")
        launcher = WorkerLauncher.from_install_root(
            self.install_root,
            resource_root=self.resource_catalog.root,
        )
        try:
            for runtime_id in runtime_ids:
                launcher.resolve(runtime_id, expected_owner="ocr", verify_interpreter=False)
        except (WorkerLaunchError, OSError) as exc:
            raise JobPreflightError(
                "ocr_runtime_install_required: selected OCR runtime "
                f"{runtime_ids[-1]} is unavailable; install the optional OCR component "
                "before enabling OCR"
            ) from exc

    def _validate_policy_runtime(self, config: JobConfig) -> None:
        """Check the policy worker only when the Dropout worker will run."""
        if self.install_root is None or config.dropout.get("enabled") is not True:
            return
        children = (config.dropout.get("artist"), config.dropout.get("quality"), config.dropout.get("appearanceNl"))
        if not any(isinstance(child, dict) and child.get("enabled") is True for child in children):
            return
        launcher = WorkerLauncher.from_install_root(
            self.install_root,
            resource_root=self.resource_catalog.root,
        )
        try:
            launcher.resolve("policy", expected_owner="policy", verify_interpreter=False)
        except (WorkerLaunchError, OSError) as exc:
            raise JobPreflightError(
                "policy_runtime_install_required: policy runtime is unavailable; "
                "install the optional policy component before enabling Dropout"
            ) from exc

    def _freeze_api_runtime(self, config: JobConfig) -> None:
        """Copy connection-only Profile data into the immutable task config."""
        if self.profile_store is None or not _nl_api_work_enabled(config):
            return
        profile_id = config.nl.get("apiProfileId")
        if not isinstance(profile_id, str):
            return
        try:
            profile = next(item for item in self.profile_store.load_all() if item.profileId == profile_id)
        except (StopIteration, NlProfileError):
            return
        config.nl["apiRuntime"] = {
            "revision": profile.revision,
            "protocol": profile.protocol,
            "providerLabel": profile.providerLabel,
            "endpoint": profile.endpoint,
            "model": profile.model,
            "backupModel": profile.backupModel,
            "credentialRef": profile.apiCredentialRef,
            "jsonMode": profile.jsonMode,
            "policy": profile.policy,
            "appliedAt": profile.updatedAt or utc_now(),
        }

    def _resolve_resources(
        self,
        config: JobConfig,
        *,
        freeze: bool,
    ) -> tuple[ResourceCatalogSnapshot, dict[str, ResourcePackage]]:
        token_budget_enabled = (
            job_config_supports_token_budget(config.schemaVersion)
            and isinstance(config.tokenBudget, dict)
            and config.tokenBudget.get("enabled") is True
        )
        snapshot = self.resource_catalog.scan(include_tokenizers=token_budget_enabled)
        defaults = snapshot.defaults_for()
        selected: dict[str, ResourcePackage] = {}

        def resolve(
            name: str,
            kind: ResourceKind,
            section: dict[str, object],
            default_key: str,
            active: bool,
        ) -> ResourcePackage:
            resource_id = self._resource_id(section, defaults[default_key], name)
            package = snapshot.package(kind, resource_id, verify_hashes=active)
            frozen_path = section.get("resourceManifestRelativePath")
            frozen_fingerprint = section.get("resourceFingerprint")
            if freeze or (frozen_path is None and frozen_fingerprint is None):
                self._freeze_reference(section, package)
            else:
                if (
                    frozen_path != package.manifest_relative_path
                    or frozen_fingerprint != package.fingerprint
                ):
                    raise ResourceCatalogError(f"frozen {name} resource no longer matches the resource library")
            selected[name] = package
            return package

        caption = resolve(
            "caption", "tagging-model", config.caption, "taggingModel", config.caption.get("enabled") is True,
        )
        if config.classify.get("indexMode") == "custom":
            custom_path = config.classify.get("customResourcePath")
            if not isinstance(custom_path, str):
                raise ResourceCatalogError("custom classification resource path is invalid")
            inspected = inspect_custom_classification_resource(custom_path)
            classify = inspected.package
            expected_identity = {
                "resourceId": classify.resource_id,
                "resourceFingerprint": classify.fingerprint,
                "resourceProfile": classify.profile,
                "dictionaryEntryCount": classify.metadata["dictionaryEntryCount"],
                "wikiDataSourceId": classify.metadata["wikiDataSourceId"],
                "customResourceContentSha256": inspected.content_sha256,
            }
            if freeze:
                config.classify.update(expected_identity)
            elif any(config.classify.get(key) != value for key, value in expected_identity.items()):
                raise ResourceCatalogError(
                    "custom classification resource changed after preflight; run preflight again"
                )
            selected["classify"] = classify
        else:
            classify = resolve(
                "classify", "classification-index", config.classify, "classificationIndex",
                config.classify.get("enabled") is True or config.caption.get("enabled") is True,
            )
            config.classify["wikiDataSourceId"] = classify.metadata["wikiDataSourceId"]
            config.classify["dictionaryEntryCount"] = classify.metadata["dictionaryEntryCount"]
            config.classify["resourceProfile"] = classify.profile
        if config.caption.get("enabled") is True:
            caption.verify_files(verify_hashes=True)
            classify.verify_files(verify_hashes=True)
            verify_tagger_dictionary_compatibility(caption, classify)
        if job_config_supports_ocr(config.schemaVersion) and config.ocr.get("enabled") is True:
            resource_id = config.ocr.get("resourceId")
            if resource_id != OCR_MODEL_RESOURCE_ID:
                raise ResourceCatalogError(
                    f"ocr resourceId must be {OCR_MODEL_RESOURCE_ID}"
                )
            try:
                ocr_package = snapshot.package(
                    "ocr-model",
                    resource_id,
                    verify_hashes=True,
                    profile="shared",
                )
            except ResourceCatalogError as exc:
                raise ResourceCatalogError(
                    "ocr_resource_install_required: selected OCR resource "
                    f"{OCR_MODEL_RESOURCE_ID} is unavailable ({exc}); download the exact archives "
                    "listed in OCR_MODEL_DOWNLOAD.md into <project-root>/ocr-model-archives and "
                    "run Install-WebUI.bat"
                ) from exc
            frozen_path = config.ocr.get("resourceManifestRelativePath")
            frozen_fingerprint = config.ocr.get("resourceFingerprint")
            if freeze or (frozen_path is None and frozen_fingerprint is None):
                self._freeze_reference(config.ocr, ocr_package)
            elif (
                frozen_path != ocr_package.manifest_relative_path
                or frozen_fingerprint != ocr_package.fingerprint
            ):
                raise ResourceCatalogError("frozen ocr resource no longer matches the resource library")
            selected["ocr"] = ocr_package
        if token_budget_enabled:
            token_budget = config.tokenBudget
            assert isinstance(token_budget, dict)
            resource_id = token_budget.get("resourceId")
            if not isinstance(resource_id, str):
                raise ResourceCatalogError("tokenBudget resourceId is invalid")
            try:
                tokenizer_package = snapshot.package(
                    "tokenizer", resource_id, verify_hashes=True, profile="shared",
                )
            except ResourceCatalogError as exc:
                if "SHA-256 mismatch" in str(exc):
                    raise
                raise ResourceCatalogError(
                    "tokenizer_resource_install_required: selected tokenizer resource "
                    f"{resource_id} is unavailable in resource-library/tokenizers; "
                    "run Import-TokenizerResources.bat before enabling Token Budget"
                ) from exc
            context_limit = tokenizer_package.context_limit
            if context_limit is None:
                raise ResourceCatalogError("tokenizer resource contextLimit is unavailable")
            frozen_path = token_budget.get("resourceManifestRelativePath")
            frozen_fingerprint = token_budget.get("resourceFingerprint")
            frozen_context_limit = token_budget.get("contextLimit")
            if freeze or (
                frozen_path is None and frozen_fingerprint is None and frozen_context_limit is None
            ):
                self._freeze_reference(token_budget, tokenizer_package)
                token_budget["contextLimit"] = context_limit
            elif (
                frozen_path != tokenizer_package.manifest_relative_path
                or frozen_fingerprint != tokenizer_package.fingerprint
                or frozen_context_limit != context_limit
            ):
                raise ResourceCatalogError("frozen tokenBudget resource no longer matches the resource library")
            if token_budget["maxTokens"] > context_limit:
                raise ResourceCatalogError("tokenBudget maxTokens must not exceed contextLimit")
            selected["tokenBudget"] = tokenizer_package
        validate_job_config(config, adjustable_categories=caption.adjustable_categories)
        if config.replace.get("indexMode") == "bundled":
            resolve(
                "replace", "replacement-index", config.replace, "replacementIndex",
                config.replace.get("enabled") is True,
            )
        quality = config.dropout.get("quality")
        if not isinstance(quality, dict):
            raise ResourceCatalogError("dropout quality configuration is invalid")
        quality_active = config.dropout.get("enabled") is True and quality.get("enabled") is True
        if quality_active:
            try:
                resolve("dropout", "dropout-model", quality, "dropoutModel", True)
            except ResourceCatalogError as exc:
                raise ResourceCatalogError(
                    "quality_stack_install_required: selected LSE14 quality model "
                    "lse14-scorer-5k-v1 is unavailable or invalid; install the optional "
                    "quality-stack component before enabling quality scoring"
                ) from exc
        elif quality.get("resourceManifestRelativePath") is None and quality.get("resourceFingerprint") is None:
            # Keep legacy reference migration when the optional package is present,
            # while allowing disabled Dropout to proceed when it is absent.
            try:
                resolve("dropout", "dropout-model", quality, "dropoutModel", False)
            except ResourceCatalogError:
                pass
        return snapshot, selected

    @staticmethod
    def _resource_summary(selected: dict[str, ResourcePackage]) -> dict[str, object]:
        return {
            name: {
                "resourceId": package.resource_id,
                "resourceVersion": package.resource_version,
                "manifestRelativePath": package.manifest_relative_path,
                "fingerprint": package.fingerprint,
            }
            for name, package in sorted(selected.items())
        }

    @staticmethod
    def _job_row(job_id: str, config: JobConfig, dataset_root: Path) -> dict[str, object]:
        return {
            "job_id": job_id, "config_schema_version": config.schemaVersion, "config_json": json.dumps(config.to_dict(), ensure_ascii=False),
            "config_hash": config.config_hash, "work_mode": config.workMode,
            "overwrite_mode": config.overwriteMode, "source_root": config.sourceRoot, "output_root": config.outputRoot,
            "dataset_root": str(dataset_root), "dataset_root_key": windows_key(dataset_root), "manifest_schema_version": 1,
            "recursive": int(config.recursive), "sample_count": 0, "manifest_generated_at": None, "status": "preflighting",
            "current_module_id": None, "last_event_id": 0, "pinned": 0, "api_budget_extra": 0, "api_budget_revision": 0,
            "overlay_root": None, "commit_journal_path": None, "resume_status": None, "created_at": utc_now(),
            "started_at": None, "cancel_requested_at": None, "finished_at": None,
        }

    @staticmethod
    def _estimates(builder: ManifestBuilder, projection: dict[str, object]) -> dict[str, int]:
        """Backup and incremental-space estimates from measured annotation sizes."""
        average = (
            builder.annotation_bytes // builder.annotation_files
            if builder.annotation_files
            else DEFAULT_ANNOTATION_BYTES
        )
        written = sum(int(projection[key]) for key in ("jsonCreate", "jsonOverwrite", "txtCreate", "txtOverwrite"))
        return {
            "existingAnnotationFiles": builder.annotation_files,
            "existingAnnotationBytes": builder.annotation_bytes,
            "averageAnnotationBytes": average,
            # The ZIP is deflated, so the raw sum is an upper bound.
            "backupUpperBoundBytes": builder.annotation_bytes,
            # Export hard-links images, so only rewritten annotations grow.
            "incrementalWriteBytes": written * average,
        }

    @staticmethod
    def _http_attempt_budget(config: JobConfig, candidate_count: int) -> int:
        if not _nl_api_work_enabled(config):
            return 0
        policy = config.nl.get("apiPolicy", {})
        if not isinstance(policy, dict):
            raise JobPreflightError("NL apiPolicy must be an object")
        explicit = policy.get("maxHttpAttempts")
        if explicit is not None:
            if type(explicit) is not int or not 1 <= explicit <= 10_000_000:
                raise JobPreflightError("NL maxHttpAttempts must be between 1 and 10000000")
            return explicit
        derived = max(1, nl_http_attempt_budget(candidate_count))
        if derived > 10_000_000:
            raise JobPreflightError("derived NL HTTP attempt budget exceeds 10000000")
        return derived

    @classmethod
    def _api_bounds(cls, config: JobConfig, candidate_count: int, image_bytes: int) -> dict[str, int]:
        """ROADMAP.md:830-833 request bounds and the frozen HTTP attempt budget."""
        if not _nl_api_work_enabled(config):
            candidate_count = 0
            image_bytes = 0
        elif config.nl.get("useImage") is not True:
            image_bytes = 0
        return {
            "candidateCount": candidate_count,
            "minRequests": candidate_count,
            "maxPrimaryRequests": candidate_count * 3,
            "maxWithBackupRequests": candidate_count * 5,
            "httpAttemptBudget": cls._http_attempt_budget(config, candidate_count),
            "estimatedUploadBytes": image_bytes,
        }

    @classmethod
    def _freeze_nl_attempt_budget(
        cls,
        database: StateDatabase,
        job_id: str,
        config: JobConfig,
        candidate_count: int,
    ) -> JobConfig:
        if not _nl_api_work_enabled(config):
            return config
        policy = config.nl.get("apiPolicy", {})
        if not isinstance(policy, dict):
            raise JobPreflightError("NL apiPolicy must be an object")
        if "maxHttpAttempts" in policy:
            cls._http_attempt_budget(config, candidate_count)
            return config
        previous_hash = config.config_hash
        config.nl["apiPolicy"] = {
            **policy,
            "maxHttpAttempts": cls._http_attempt_budget(config, candidate_count),
        }
        database.update_preflight_config(
            job_id,
            expected_config_hash=previous_hash,
            config_json=json.dumps(config.to_dict(), ensure_ascii=False),
            config_hash=config.config_hash,
        )
        return config

    @staticmethod
    def _validate_named_folder_manifest(
        database: StateDatabase,
        job_id: str,
        *,
        feature_name: str,
        name_placeholder: str,
        flat_root_name: str | None = None,
    ) -> None:
        invalid_count = 0
        examples: list[str] = []
        rows = database.connection.execute(
            "SELECT relative_image_path FROM samples WHERE job_id=? AND in_processing_scope=1 ORDER BY sample_id",
            (job_id,),
        ).fetchall()
        for row in rows:
            relative = str(row["relative_image_path"]).replace("\\", "/")
            try:
                candidate = f"{flat_root_name}/{relative}" if flat_root_name is not None and "/" not in relative else relative
                character_name(candidate)
            except ValueError:
                invalid_count += 1
                if len(examples) < 8:
                    examples.append(relative)
        if invalid_count:
            remaining = invalid_count - len(examples)
            suffix = f"; {remaining} more" if remaining else ""
            raise JobPreflightError(
                f"{feature_name} requires every in-scope image to use <digits>_<{name_placeholder}>; "
                + ", ".join(examples)
                + suffix
            )

    def _validate_preflight_input(
        self,
        raw_config: object,
    ) -> tuple[JobConfig, dict[str, ResourcePackage], CustomReplaceIndex | None, CanonicalPath, CanonicalPath | None]:
        """Validate and freeze non-manifest inputs before creating a job row."""
        try:
            self._reject_client_frozen_resources(raw_config)
            config = config_from_dict(raw_config)
            self._freeze_api_runtime(config)
            _, selected_resources = self._resolve_resources(config, freeze=True)
            self._validate_ocr_runtime(config)
            self._validate_policy_runtime(config)
            validate_job_config(config)
            replace_index: CustomReplaceIndex | None = None
            if config.replace["indexMode"] == "custom":
                if "customIndexSha256" in config.replace or "customIndexRuleCount" in config.replace:
                    raise JobPreflightError("custom replace index metadata is assigned by preflight")
                replace_index = inspect_custom_replace_index(str(config.replace["customIndexPath"]))
                config.replace["customIndexSha256"] = replace_index.sha256
                config.replace["customIndexRuleCount"] = replace_index.rule_count
            source, output = validate_source_output(config.sourceRoot, config.outputRoot, config.workMode)
            assert_database_outside_datasets(self.database_path, [source.value, *( [output.value] if output else [] )])
        except (ValueError, PathSafetyError, CustomReplaceIndexError) as exc:
            raise JobPreflightError(str(exc)) from exc
        return config, selected_resources, replace_index, source, output

    def _create_preflight_job(
        self,
        config: JobConfig,
        source: CanonicalPath,
        *,
        ocr_execution: OcrExecutionRequestV1 | None,
        total_files: int | None = None,
    ) -> str:
        """Persist the immutable job header and (for async scans) its progress row."""
        job_id = uuid.uuid4().hex
        database = StateDatabase.open(self.database_path)
        try:
            database.insert_job(self._job_row(job_id, config, source.value))
            if job_config_supports_ocr_device(config.schemaVersion) and config.ocr.get("enabled") is True:
                write_execution_request(
                    self._preflight_ocr_request_path(job_id),
                    ocr_execution or OcrExecutionRequestV1.auto(),
                )
            if total_files is not None:
                database.create_preflight_progress(job_id, total_files=total_files)
        except Exception:
            # No user dataset is touched here; remove only the just-created
            # control row if initialization itself fails.
            try:
                database.delete_job_control_record(job_id)
            except Exception:
                pass
            raise
        finally:
            database.close()
        return job_id

    @staticmethod
    def _preflight_summary_json(summary: PreflightSummary) -> str:
        return json.dumps(asdict(summary), ensure_ascii=False, separators=(",", ":"))

    def _execute_preflight(
        self,
        job_id: str,
        config: JobConfig,
        selected_resources: dict[str, ResourcePackage],
        replace_index: CustomReplaceIndex | None,
        source: CanonicalPath,
        *,
        total_files: int | None = None,
        persist_progress: bool = False,
        retain_failed_job: bool = False,
        cancel_check: Callable[[], bool] | None = None,
    ) -> PreflightSummary:
        """Run the bounded manifest pass for either sync or async callers."""
        database = StateDatabase.open(self.database_path)
        try:
            if cancel_check is not None and cancel_check():
                raise ManifestCancelled()
            if persist_progress:
                database.update_preflight_progress(job_id, status="preflighting")
            builder = ManifestBuilder(
                source.value, recursive=config.recursive, profile=str(config.classify["resourceProfile"]),
                invalid_image_action=config.imageDecode.invalidImageAction,
            )
            on_progress = (
                lambda checked, total: database.update_preflight_progress(
                    job_id, status="preflighting", checked_files=checked, total_files=total,
                )
            ) if persist_progress else None
            builder.scan_into(
                database,
                job_id,
                total_files=total_files,
                on_progress=on_progress,
                cancel_check=cancel_check,
            )
            if cancel_check is not None and cancel_check():
                raise ManifestCancelled()
            counts = database.connection.execute(
                """SELECT COUNT(*) AS samples,COALESCE(SUM(in_processing_scope),0) AS scoped,
                   COALESCE(SUM(original_txt_state='nonblank'),0) AS txt,COALESCE(SUM(original_json_state='nonblank'),0) AS json,
                   COALESCE(SUM(CASE WHEN in_processing_scope=1 THEN image_size END),0) AS image_bytes
                   FROM samples WHERE job_id=?""", (job_id,),
            ).fetchone()
            projection = dict(database.preflight_projection_counts(job_id))
            samples, scoped = int(counts["samples"]), int(counts["scoped"])
            if job_config_supports_nl_v4(config.schemaVersion) and config.nl.get("captionPreset") in {"character", "moss-character"}:
                self._validate_named_folder_manifest(
                    database, job_id, feature_name="character preset", name_placeholder="character"
                )
            if config.dropout.get("enabled") is True and config.dropout.get("artist", {}).get("enabled") is True:
                self._validate_named_folder_manifest(
                    database,
                    job_id,
                    feature_name="Dropout artist",
                    name_placeholder="name",
                    flat_root_name=source.value.name,
                )
            if cancel_check is not None and cancel_check():
                raise ManifestCancelled()
            config = self._freeze_nl_attempt_budget(database, job_id, config, scoped)
            if cancel_check is not None and cancel_check():
                raise ManifestCancelled()
            # Serialize the final ready/completed transition with a cancellation
            # request.  Exactly one side wins: cancellation either observes an
            # active job and the worker rolls it back, or observes completed and
            # returns ``False`` without publishing a misleading cancellation.
            finalization_guard = self._preflight_lock if persist_progress else nullcontext()
            with finalization_guard:
                if cancel_check is not None and cancel_check():
                    raise ManifestCancelled()
                database.set_job_status(job_id, "ready")
                replace_summary = replace_index.summary() if replace_index else {
                    "mode": "bundled",
                    "resourceId": selected_resources["replace"].resource_id,
                    "sha256": selected_resources["replace"].fingerprint,
                    "ruleCount": selected_resources["replace"].metadata["ruleCount"],
                }
                summary = PreflightSummary(
                    job_id, samples, scoped, samples - scoped, int(counts["txt"]), int(counts["json"]), config.config_hash,
                    replace_summary,
                    resources=self._resource_summary(selected_resources),
                    blankTxtCount=samples - int(counts["txt"]), blankJsonCount=samples - int(counts["json"]),
                    # A duplicate annotationKey is fatal above, so a summary always reports zero.
                    annotationKeyCollisionCount=0, imageIssueCount=builder.image_issue_count, projection=projection,
                    estimate=self._estimates(builder, projection),
                    api=self._api_bounds(config, scoped, int(counts["image_bytes"])),
                )
                if persist_progress:
                    database.update_preflight_progress(
                        job_id,
                        status="completed",
                        checked_files=int(total_files or 0),
                        total_files=int(total_files or 0),
                        summary_json=self._preflight_summary_json(summary),
                    )
            return summary
        except ManifestCancelled as exc:
            if persist_progress:
                try:
                    current = database.get_preflight_progress(job_id)
                    database.update_preflight_progress(
                        job_id,
                        status="cancelled",
                        checked_files=int(current["checked_files"]),
                        total_files=int(current["total_files"]),
                        error=str(exc),
                    )
                    database.clear_manifest_rows(job_id)
                    database.set_job_status(job_id, "draft")
                except Exception:
                    pass
            raise JobPreflightError(str(exc)) from exc
        except (ManifestError, OSError, ValueError) as exc:
            if persist_progress:
                try:
                    current = database.get_preflight_progress(job_id)
                    database.update_preflight_progress(
                        job_id,
                        status="failed",
                        checked_files=int(current["checked_files"]),
                        total_files=int(current["total_files"]),
                        error=str(exc)[:16_384],
                    )
                except Exception:
                    pass
                try:
                    if database.get_job(job_id)["status"] != "failed":
                        database.set_job_status(job_id, "failed", current_module_id="workspace")
                except Exception:
                    pass
            elif not retain_failed_job:
                database.delete_job_control_record(job_id)
            raise JobPreflightError(str(exc)) from exc
        finally:
            database.close()

    def preflight(
        self,
        raw_config: object,
        *,
        ocr_execution: OcrExecutionRequestV1 | None = None,
    ) -> PreflightSummary:
        config, selected_resources, replace_index, source, _output = self._validate_preflight_input(raw_config)
        job_id = self._create_preflight_job(config, source, ocr_execution=ocr_execution)
        return self._execute_preflight(
            job_id,
            config,
            selected_resources,
            replace_index,
            source,
            retain_failed_job=False,
        )

    def _async_preflight_worker(
        self,
        job_id: str,
        config: JobConfig,
        selected_resources: dict[str, ResourcePackage],
        replace_index: CustomReplaceIndex | None,
        source: CanonicalPath,
    ) -> None:
        with self._preflight_lock:
            cancel_event = self._preflight_cancel.get(job_id)
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise ManifestCancelled()
            builder = ManifestBuilder(
                source.value,
                recursive=config.recursive,
                profile=str(config.classify["resourceProfile"]),
                invalid_image_action=config.imageDecode.invalidImageAction,
            )
            total_files = builder.count_candidate_images(
                cancel_check=cancel_event.is_set if cancel_event is not None else None,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise ManifestCancelled()
            database = StateDatabase.open(self.database_path)
            try:
                database.update_preflight_progress(
                    job_id, status="preflighting", checked_files=0, total_files=total_files,
                )
            finally:
                database.close()
            self._execute_preflight(
                job_id,
                config,
                selected_resources,
                replace_index,
                source,
                total_files=total_files,
                persist_progress=True,
                retain_failed_job=True,
                cancel_check=cancel_event.is_set if cancel_event is not None else None,
            )
        except ManifestCancelled as exc:
            self._persist_preflight_cancelled(job_id, str(exc))
        except Exception as exc:
            # A cancellation request owns the terminal transition.  Do not
            # turn a cooperative cancellation observed while unwinding into a
            # failed preflight.
            if cancel_event is not None and cancel_event.is_set():
                self._persist_preflight_cancelled(job_id, str(exc))
                return
            # _execute_preflight persists expected failures.  Keep a final
            # fail-safe for injected/unexpected exceptions so polling never
            # remains stuck in queued/preflighting forever.
            database = StateDatabase.open(self.database_path)
            try:
                try:
                    current = database.get_preflight_progress(job_id)
                except KeyError:
                    current = None
                if current is not None and current["status"] not in {"completed", "failed", "cancelled"}:
                    database.update_preflight_progress(
                        job_id,
                        status="failed",
                        checked_files=int(current["checked_files"]),
                        total_files=int(current["total_files"]),
                        error=f"{type(exc).__name__}: {exc}"[:16_384],
                    )
                    try:
                        database.set_job_status(job_id, "failed", current_module_id="workspace")
                    except Exception:
                        pass
            finally:
                database.close()
        finally:
            with self._preflight_lock:
                self._preflight_threads.pop(job_id, None)
                self._preflight_cancel.pop(job_id, None)

    def start_preflight(
        self,
        raw_config: object,
        *,
        ocr_execution: OcrExecutionRequestV1 | None = None,
    ) -> str:
        """Queue an image preflight and return its durable job identity."""
        config, selected_resources, replace_index, source, _output = self._validate_preflight_input(raw_config)
        job_id = self._create_preflight_job(
            config,
            source,
            ocr_execution=ocr_execution,
            # The candidate walk itself belongs to the background worker.  A
            # zero denominator is the explicit "not counted yet" state.
            total_files=0,
        )
        thread = threading.Thread(
            target=self._async_preflight_worker,
            args=(job_id, config, selected_resources, replace_index, source),
            daemon=True,
            name=f"anima-preflight-{job_id[:12]}",
        )
        with self._preflight_lock:
            self._preflight_threads[job_id] = thread
            self._preflight_cancel[job_id] = threading.Event()
        try:
            thread.start()
        except Exception:
            with self._preflight_lock:
                self._preflight_threads.pop(job_id, None)
                self._preflight_cancel.pop(job_id, None)
            database = StateDatabase.open(self.database_path)
            try:
                database.update_preflight_progress(
                    job_id, status="failed", error="preflight worker could not start",
                )
                database.set_job_status(job_id, "failed", current_module_id="workspace")
            finally:
                database.close()
            raise
        return job_id

    def cancel_preflight(self, job_id: str) -> bool:
        """Request cancellation of a queued or running asynchronous preflight."""
        with self._preflight_lock:
            event = self._preflight_cancel.get(job_id)
            thread = self._preflight_threads.get(job_id)
            if event is not None:
                event.set()
        database = StateDatabase.open(self.database_path)
        try:
            progress = database.get_preflight_progress(job_id)
            if progress["status"] == "cancelled":
                return True
            if progress["status"] in {"completed", "failed"}:
                return False
            if event is None or thread is None:
                raise JobPreflightError("preflight worker is not active")
            # The worker owns cleanup and the terminal transition.  Publishing
            # a terminal row here would race with an in-flight manifest write
            # and could leave samples/issues behind after cancellation.
            return True
        finally:
            database.close()

    def _persist_preflight_cancelled(self, job_id: str, error: str) -> None:
        database = StateDatabase.open(self.database_path)
        try:
            try:
                progress = database.get_preflight_progress(job_id)
            except KeyError:
                return
            if progress["status"] in {"completed", "failed", "cancelled"}:
                return
            # Clear rows first while the progress row is still mutable.  This
            # ordering also makes cancellation safe if a scan was interrupted
            # between a batch insert and its next callback.
            database.clear_manifest_rows(job_id)
            database.update_preflight_progress(
                job_id,
                status="cancelled",
                checked_files=int(progress["checked_files"]),
                total_files=int(progress["total_files"]),
                error=error[:16_384] or "preflight cancelled",
            )
            if database.get_job(job_id)["status"] == "preflighting":
                database.set_job_status(job_id, "draft")
        finally:
            database.close()

    def confirm_workspace(self, job_id: str, *, confirmed: bool, confirmed_rebuild: bool) -> dict[str, object]:
        if not confirmed:
            raise JobPreflightError("workspace preparation requires explicit confirmation")
        database = StateDatabase.open(self.database_path)
        acquired: DatasetLock | None = None
        first_lock_acquired = False
        retain_database = False
        try:
            job = database.get_job(job_id)
            if job["status"] != "ready":
                raise JobPreflightError("only a ready preflight job can prepare a workspace")
            raw_config = json.loads(str(job["config_json"]))
            raw_config = self._restore_missing_bundled_resource_ids(raw_config)
            config = config_from_dict(raw_config)
            previous_hash = str(job["config_hash"])
            preservation = database.freeze_incremental_preservation(
                job_id,
                overwrite_mode=config.overwriteMode,
                overwrite_json=bool(config.classify.get("overwriteJson")),
                overwrite_txt=bool(config.caption.get("overwriteTxt")),
                overwrite_count=bool(config.classify.get("overwriteCount")),
            )
            try:
                self._resolve_resources(config, freeze=False)
            except CustomClassificationResourceError as exc:
                raise JobPreflightError(
                    "custom classification resource changed after preflight; run preflight again"
                ) from exc
            if config.config_hash != previous_hash:
                database.update_preflight_config(
                    job_id,
                    expected_config_hash=previous_hash,
                    config_json=json.dumps(config.to_dict(), ensure_ascii=False),
                    config_hash=config.config_hash,
                )
            if database.count_unresolved_blocking_issues(job_id):
                raise JobPreflightError("preflight found unusable images; fix them or choose skip and run preflight again")
            # Incremental-preserved samples never reach a mutating worker or
            # Token Budget, so they must not inflate the frozen API/budget
            # candidate count. Explicit overwrite decisions are already
            # represented by incremental_preserve=0.
            candidate_count = int(database.connection.execute(
                "SELECT COUNT(*) FROM samples WHERE job_id=? AND in_processing_scope=1 AND incremental_preserve=0",
                (job_id,),
            ).fetchone()[0])
            config = self._freeze_nl_attempt_budget(database, job_id, config, candidate_count)
            if config.overwriteMode == "rebuild" and not confirmed_rebuild:
                raise JobPreflightError("rebuild workspace preparation requires second confirmation")
            source, _ = validate_source_output(config.sourceRoot, config.outputRoot, config.workMode)
            self._release_succeeded_dataset_claim(database, source.value)
            acquired = DatasetLock.acquire(database, source.value, job_id)
            first_lock_acquired = True
            dataset = prepare_dataset(source.value, config.outputRoot, config.workMode, job_id).datasetRoot
            if config.workMode == "full_copy":
                self._rebind_full_copy_manifest(database, job_id, dataset)
            if config.workMode == "full_copy":
                acquired.release(recovery_complete=True)
                acquired = DatasetLock.acquire(database, dataset, job_id)
            layout = OverlayLayout.create(dataset, job_id)
            if job_config_supports_ocr_device(config.schemaVersion) and config.ocr.get("enabled") is True:
                request = read_execution_request(self._preflight_ocr_request_path(job_id))
                write_execution_request(layout.resource_path("ocr-execution-request-v1.json"), request)
            if config.classify.get("indexMode") == "custom":
                current = inspect_custom_classification_resource(str(config.classify["customResourcePath"]))
                if (
                    current.package.fingerprint != config.classify.get("resourceFingerprint")
                    or current.content_sha256 != config.classify.get("customResourceContentSha256")
                ):
                    raise JobPreflightError(
                        "custom classification resource changed after preflight; run preflight again"
                    )
                manifest_relative_path, frozen_package = freeze_custom_classification_resource(layout, current)
                expected_hash = config.config_hash
                config.classify.pop("customResourcePath", None)
                config.classify.pop("customResourceContentSha256", None)
                config.classify.update({
                    "resourceId": frozen_package.resource_id,
                    "resourceManifestRelativePath": manifest_relative_path,
                    "resourceFingerprint": frozen_package.fingerprint,
                    "wikiDataSourceId": frozen_package.metadata["wikiDataSourceId"],
                    "dictionaryEntryCount": frozen_package.metadata["dictionaryEntryCount"],
                    "resourceProfile": frozen_package.profile,
                })
                validate_job_config(config)
                database.update_preflight_config(
                    job_id,
                    expected_config_hash=expected_hash,
                    config_json=json.dumps(config.to_dict(), ensure_ascii=False),
                    config_hash=config.config_hash,
                )
            if config.replace["indexMode"] == "custom":
                current = inspect_custom_replace_index(str(config.replace["customIndexPath"]))
                if current.sha256 != config.replace["customIndexSha256"] or current.rule_count != config.replace["customIndexRuleCount"]:
                    raise JobPreflightError("custom replace index changed after preflight; run preflight again")
                frozen = layout.write_resource("replace\\custom-index.csv", current.content)
                if frozen.stat().st_size != len(current.content):
                    raise JobPreflightError("custom replace index could not be frozen")
            database.set_workspace_metadata(job_id, dataset_root=str(dataset), dataset_root_key=windows_key(dataset), overlay_root=str(layout.root))
            database.set_job_status(job_id, "preparing_workspace", current_module_id="workspace")
            self._locks[job_id] = acquired
            acquired = None
            retain_database = True
            return {"jobId": job_id, "status": "preparing_workspace", "datasetRoot": str(dataset), "overlayRoot": str(layout.root)}
        except DatasetLockError:
            if not first_lock_acquired:
                raise
            if acquired is not None:
                acquired.release(recovery_complete=True)
            database.set_job_status(job_id, "failed", current_module_id="workspace")
            raise
        except Exception:
            if acquired is not None:
                acquired.release(recovery_complete=True)
            database.set_job_status(job_id, "failed", current_module_id="workspace")
            raise
        finally:
            if not retain_database:
                database.close()

    def _release_succeeded_dataset_claim(self, database: StateDatabase, dataset: Path) -> None:
        owner = database.connection.execute(
            """SELECT c.job_id,j.status FROM dataset_claims AS c
               JOIN jobs AS j ON j.job_id=c.job_id WHERE c.dataset_root=?""",
            (str(dataset),),
        ).fetchone()
        if owner is None or owner["status"] not in {"succeeded", "succeeded_with_issues"}:
            return
        claiming_job_id = str(owner["job_id"])
        lock = self._locks.get(claiming_job_id)
        if lock is not None:
            lock.release(recovery_complete=True)
            lock.database.close()
            del self._locks[claiming_job_id]
            return
        with database.transaction(immediate=True):
            current = database.connection.execute(
                """SELECT c.job_id,j.status FROM dataset_claims AS c
                   JOIN jobs AS j ON j.job_id=c.job_id WHERE c.dataset_root=?""",
                (str(dataset),),
            ).fetchone()
            if current is not None and current["status"] in {"succeeded", "succeeded_with_issues"}:
                database.connection.execute(
                    "DELETE FROM dataset_claims WHERE dataset_root=? AND job_id=?",
                    (str(dataset), str(current["job_id"])),
                )

    def close(self) -> None:
        with self._preflight_lock:
            preflight_threads = tuple(self._preflight_threads.values())
            for event in self._preflight_cancel.values():
                event.set()
        for thread in preflight_threads:
            thread.join(timeout=5)
        for lock in tuple(self._locks.values()):
            try:
                lock.release(recovery_complete=True)
            finally:
                lock.database.close()
        self._locks.clear()

    def release_lock_for_discard(self, job_id: str) -> bool:
        """Release this process's live handle after the job is durably discarded."""
        lock = self._locks.get(job_id)
        if lock is None:
            return False
        if lock.database.get_job(job_id)["status"] != "discarded":
            raise JobPreflightError("only a discarded task can release its live dataset lock")
        lock.release(recovery_complete=True)
        lock.database.close()
        del self._locks[job_id]
        return True

    def release_lock_for_history_deletion(self, job_id: str) -> bool:
        return release_completed_dataset_lock(self._locks, self.database_path, job_id)

    def release_lock_for_repair(self, job_id: str) -> bool:
        """Hand a terminal task's live dataset lock to a forthcoming repair task."""
        lock = self._locks.get(job_id)
        if lock is None:
            database = StateDatabase.open(self.database_path)
            try:
                claim = database.connection.execute(
                    "SELECT lock_path FROM dataset_claims WHERE job_id=?", (job_id,)
                ).fetchone()
                # A task with no persisted claim has no lock to hand off.  This
                # is the normal path for an already-restarted fixture (and for
                # a task whose claim was released by startup recovery); only a
                # live claim requires the terminal-state and journal guards.
                if claim is None:
                    return False
                job = database.get_job(job_id)
                if job["status"] not in {"reviewing", "failed", "succeeded"}:
                    raise JobPreflightError("only a reviewed, failed, or succeeded task can release its lock for repair")
                overlay_root = job["overlay_root"]
                if isinstance(overlay_root, str) and overlay_root:
                    try:
                        state = OverlayLayout.open_existing(overlay_root, job_id).journal_state()
                    except Exception as exc:
                        raise JobPreflightError("task commit journal cannot be safely inspected") from exc
                    if state not in {None, "committed", "rolled_back"}:
                        raise JobPreflightError("task lock cannot be released before commit journal recovery")
                database.release_dataset_claim(job_id)
                lock_path = claim["lock_path"]
            finally:
                database.close()
            if isinstance(lock_path, str) and lock_path:
                try:
                    Path(lock_path).unlink()
                except FileNotFoundError:
                    pass
            return True
        job = lock.database.get_job(job_id)
        if job["status"] not in {"reviewing", "failed", "succeeded"}:
            raise JobPreflightError("only a reviewed, failed, or succeeded task can release its lock for repair")
        overlay_root = job["overlay_root"]
        if isinstance(overlay_root, str) and overlay_root:
            try:
                state = OverlayLayout.open_existing(overlay_root, job_id).journal_state()
            except Exception as exc:
                raise JobPreflightError("task commit journal cannot be safely inspected") from exc
            if state not in {None, "committed", "rolled_back"}:
                raise JobPreflightError("task lock cannot be released before commit journal recovery")
        lock.release(recovery_complete=True)
        lock.database.close()
        del self._locks[job_id]
        return True

    def restore_lock_after_repair_failure(self, job_id: str) -> None:
        if job_id in self._locks:
            return
        database = StateDatabase.open(self.database_path)
        try:
            job = database.get_job(job_id)
            dataset = Path(str(job["dataset_root"]))
            self._locks[job_id] = DatasetLock.acquire(database, dataset, job_id)
        except Exception:
            database.close()
            raise
