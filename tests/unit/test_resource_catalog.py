from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import validate_resource_library


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))
sys.path.insert(0, str(ROOT / "packaging" / "scripts"))

from anima_core.resource_catalog import ResourceCatalog, ResourceCatalogError, default_resource_library_root
from copy_resource_library import assert_no_local_only_leaks, copy_distributable


KINDS = {
    "replacement-index": ("replacement-indexes", "e621-replacement-csv-v1", {"index": "index.csv"}),
    "classification-index": (
        "classification-indexes",
        "e621-classification-index-v1",
        {"dictionary": "dictionary.json", "countDatabase": "count.sqlite3"},
    ),
    "tagging-model": (
        "tagging-models",
        "e621-eva02-onnx-v1",
        {
            "model": "model.onnx",
            "modelData": "model.onnx.data",
            "preprocess": "preprocess.json",
            "tags": "tags.json",
            "thresholds": "thresholds.json",
        },
    ),
    "dropout-model": (
        "dropout-models",
        "lse14-scorer-5k-v1",
        {
            "clip": "clip/model.pt",
            "fusion": "fusion/model.safetensors",
            "jtp3": "jtp3/model.safetensors",
            "waifu": "waifu/model.safetensors",
        },
    ),
}

OCR_RESOURCE_ID = "ocr-ppocrv5-server-paddle-v1"
OCR_ENTRYPOINTS = {
    "detection": r"detection\inference.json",
    "recognition": r"recognition\inference.json",
    "textlineOrientation": r"textline-orientation\inference.json",
}
OCR_MODEL_FILES = {
    r"detection\inference.json": b'{"model":"det"}',
    r"detection\model.pdiparams": b"det-params",
    r"recognition\inference.json": b'{"model":"rec"}',
    r"recognition\model.pdiparams": b"rec-params",
    r"textline-orientation\inference.json": b'{"model":"orientation"}',
    r"textline-orientation\model.pdiparams": b"orientation-params",
}
OCR_MODEL_METADATA = {
    "models": {
        "detection": "PP-OCRv5_server_det",
        "recognition": "PP-OCRv5_server_rec",
        "textlineOrientation": "PP-LCNet_x1_0_textline_ori",
    },
    "inference": {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": True,
        "textRecScoreThresh": 0,
        "textDetLimitSideLen": 1920,
        "textDetLimitType": "max",
    },
}
TOKENIZER_RESOURCES = {
    "tokenizer-qwen3-0.6b-anima-v1": "Qwen/Qwen3-0.6B",
    "tokenizer-qwen3-vl-4b-krea2-v1": "Qwen/Qwen3-VL-4B-Instruct",
}
TOKENIZER_FILES = {
    "config.json": b'{"max_position_embeddings":1}',
    "tokenizer.json": b'{"version":"1.0"}',
}


def _metadata(kind: str) -> dict[str, object]:
    if kind == "replacement-index":
        return {
            "ruleCount": 1,
            "actionCounts": {"keep": 1, "replace": 0, "drop": 0},
            "pipeReplacementCount": 0,
            "literalKeepPipeCount": 0,
        }
    if kind == "classification-index":
        return {
            "dictionaryEntryCount": 1,
            "wikiDataSourceId": "wiki-test-v1",
            "wikiApplicationId": 1,
            "wikiSchemaVersion": 1,
            "wikiSchemaFingerprint": "a" * 64,
            "wikiPageTitles": ["solo"],
        }
    if kind == "tagging-model":
        return {"tagCount": 1, "categories": ["general", "character", "species", "rating"]}
    return {}


class ResourceCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "resource-library"
        self.root.mkdir()
        self.ids = {
            "replacementIndex": "replace-default",
            "classificationIndex": "classify-default",
            "taggingModel": "tagger-default",
            "dropoutModel": "dropout-default",
        }
        (self.root / "defaults.json").write_text(
            json.dumps({"schemaVersion": 3, "defaults": self.ids}), encoding="utf-8"
        )
        self._write_package("replacement-index", "default", self.ids["replacementIndex"], documentation=True)
        self._write_package("classification-index", "default", self.ids["classificationIndex"])
        self._write_package("tagging-model", "default", self.ids["taggingModel"])
        self._write_package("dropout-model", "default", self.ids["dropoutModel"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_resource_validator_accepts_profileless_flat_defaults(self) -> None:
        result = validate_resource_library.validate(self.root)
        self.assertEqual(["e621"], result["compatibleProfiles"])

    def _write_package(
        self,
        kind: str,
        directory_name: str,
        resource_id: str,
        *,
        documentation: bool = False,
    ) -> Path:
        category, runtime_format, entrypoints = KINDS[kind]
        package = self.root / category / directory_name
        package.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, object]] = {}
        for index, relative in enumerate(entrypoints.values(), start=1):
            target = package / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == "tags.json":
                content = b'{"tag_names":["tag_a"]}'
            elif relative == "thresholds.json":
                content = b'{"general":0.6,"character":0.65,"species":0.6,"rating":0.65}'
            elif relative == "dictionary.json":
                content = b'{"entries":{"tag_a":{}}}'
            else:
                content = f"file-{index}".encode("ascii")
            target.write_bytes(content)
            files[relative.replace("/", "\\")] = {
                "sizeBytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        docs: list[dict[str, str]] = []
        if documentation:
            (package / "manual.txt").write_text("documentation-v1", encoding="utf-8")
            docs.append({"path": "manual.txt", "language": "en", "title": "Manual"})
        manifest = {
            "schemaVersion": 1,
            "kind": kind,
            "resourceId": resource_id,
            "resourceVersion": "test-v1",
            "profile": "e621",
            "displayName": {"zh-CN": "Test", "en": "Test"},
            "description": {"zh-CN": "Test resource", "en": "Test resource"},
            "runtimeFormat": runtime_format,
            "entrypoints": entrypoints,
            "files": files,
            "metadata": _metadata(kind),
            "documentation": docs,
        }
        path = package / "resource.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_v2_tagger(self, resource_id: str) -> Path:
        package = self.root / "tagging-models" / resource_id
        package.mkdir(parents=True, exist_ok=True)
        entrypoints = {
            "model": "model.onnx",
            "modelData": "model.onnx.data",
            "metadata": "model_metadata.json",
            "vocabulary": "model_vocabulary.json",
            "thresholds": "thresholds.json",
        }
        files: dict[str, dict[str, object]] = {}
        for index, relative in enumerate(entrypoints.values(), start=1):
            content = (
                b'{"general":0.55,"character":0.55,"copyright":0.55}'
                if relative == "thresholds.json"
                else f"v2-file-{index}".encode("ascii")
            )
            (package / relative).write_bytes(content)
            files[relative] = {
                "sizeBytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        manifest = {
            "schemaVersion": 2,
            "kind": "tagging-model",
            "resourceId": resource_id,
            "resourceVersion": "cella110n/cl_tagger_v2:v2_00",
            "profile": "danbooru",
            "displayName": {"zh-CN": "CL Tagger v2", "en": "CL Tagger v2"},
            "description": {"zh-CN": "测试资源", "en": "Test resource"},
            "runtimeFormat": "cl-tagger-v2-onnx-v1",
            "distribution": {
                "mode": "local-only",
                "sourceUrl": "https://huggingface.co/cella110n/cl_tagger_v2/tree/main/v2_00",
                "licenseUrl": "https://huggingface.co/cella110n/cl_tagger_v2/blob/main/LICENSE.md",
            },
            "entrypoints": entrypoints,
            "files": files,
            "metadata": {
                "tagCount": 106536,
                "modelCategories": ["General", "Character", "Copyright", "Meta", "Rating", "Quality"],
                "adjustableCategories": ["general", "character", "copyright"],
                "excludedCategories": ["meta", "rating", "quality"],
                "vocabularyFingerprint": files[entrypoints["vocabulary"]]["sha256"],
            },
            "documentation": [],
        }
        path = package / "resource.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_ocr_model(
        self,
        resource_id: str = OCR_RESOURCE_ID,
        *,
        package_name: str | None = None,
        profile: str = "shared",
        runtime_format: str = "ppocrv5-server-paddle-v1",
        entrypoints: dict[str, str] | None = None,
        metadata: dict[str, object] | None = None,
        distribution: dict[str, str] | None = None,
    ) -> Path:
        package = self.root / "ocr-models" / (package_name or resource_id)
        package.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, object]] = {}
        for relative, content in OCR_MODEL_FILES.items():
            target = package / Path(relative.replace("\\", os.sep))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            files[relative] = {
                "sizeBytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        manifest = {
            "schemaVersion": 2,
            "kind": "ocr-model",
            "resourceId": resource_id,
            "resourceVersion": "ppocrv5-server-paddle-v1",
            "profile": profile,
            "displayName": {"zh-CN": "OCR test", "en": "OCR test"},
            "description": {"zh-CN": "Test OCR resource", "en": "Test OCR resource"},
            "runtimeFormat": runtime_format,
            "distribution": distribution or {"mode": "bundled"},
            "entrypoints": dict(entrypoints or OCR_ENTRYPOINTS),
            "files": files,
            "metadata": json.loads(json.dumps(metadata or OCR_MODEL_METADATA)),
            "documentation": [],
        }
        path = package / "resource.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_tokenizer(
        self,
        resource_id: str,
        *,
        package_name: str | None = None,
        context_limit: int = 1,
    ) -> Path:
        package = self.root / "tokenizers" / (package_name or resource_id)
        package.mkdir(parents=True, exist_ok=True)
        files = []
        for relative, content in sorted(TOKENIZER_FILES.items()):
            target = package / relative
            target.write_bytes(content)
            files.append({
                "path": relative,
                "sizeBytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        manifest = {
            "schemaVersion": 3,
            "kind": "tokenizer",
            "resourceId": resource_id,
            "owner": "token-budget",
            "profile": "shared",
            "resourceVersion": "test-v1",
            "officialModelId": TOKENIZER_RESOURCES[resource_id],
            "revision": "a" * 40,
            "tokenizerFamily": "qwen3",
            "contextLimit": context_limit,
            "rootRelativePath": f"tokenizers\\{resource_id}",
            "files": files,
            "distribution": {
                "mode": "local-only",
                "sourceUrl": f"https://huggingface.co/{TOKENIZER_RESOURCES[resource_id]}",
                "licenseStatus": "unverified",
            },
        }
        manifest["fingerprint"] = hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        path = package / "resource.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return path

    def _use_current_defaults(self) -> None:
        (self.root / "defaults.json").write_text(
            json.dumps({"schemaVersion": 3, "defaults": self.ids}),
            encoding="utf-8",
        )

    def test_valid_catalog_and_documentation_do_not_change_runtime_fingerprint(self) -> None:
        catalog = ResourceCatalog(self.root)
        first = catalog.scan()
        self.assertEqual(4, len(first.packages))
        replacement = first.package("replacement-index", self.ids["replacementIndex"], verify_hashes=True)
        fingerprint = replacement.fingerprint
        (replacement.package_root / "manual.txt").write_text("documentation-v2", encoding="utf-8")
        manifest = json.loads((replacement.package_root / "resource.json").read_text(encoding="utf-8"))
        manifest["documentation"][0]["title"] = "Updated manual"
        (replacement.package_root / "resource.json").write_text(json.dumps(manifest), encoding="utf-8")
        second = catalog.scan().package("replacement-index", self.ids["replacementIndex"], verify_hashes=False)
        self.assertEqual(fingerprint, second.fingerprint)

    def test_schema3_defaults_are_flat_and_profile_independent(self) -> None:
        self._use_current_defaults()
        snapshot = ResourceCatalog(self.root).scan()
        api = snapshot.api_dict()

        self.assertEqual(self.ids, snapshot.defaults)
        self.assertEqual(self.ids, snapshot.defaults_for("e621"))
        self.assertEqual(self.ids, snapshot.defaults_for("danbooru"))
        self.assertEqual(self.ids, api["defaults"])
        self.assertNotIn("profiles", api)

    def test_scan_is_size_only_but_preflight_hash_verification_rejects_tampering(self) -> None:
        catalog = ResourceCatalog(self.root)
        package = catalog.scan().package("replacement-index", self.ids["replacementIndex"], verify_hashes=False)
        target = package.entrypoint("index")
        target.write_bytes(b"x" * target.stat().st_size)
        refreshed = catalog.scan()
        with self.assertRaisesRegex(ResourceCatalogError, "SHA-256 mismatch"):
            refreshed.package("replacement-index", self.ids["replacementIndex"], verify_hashes=True)

    def test_invalid_paths_sizes_and_duplicate_ids_are_reported_but_not_selectable(self) -> None:
        unsafe = self._write_package("replacement-index", "unsafe", "replace-unsafe")
        value = json.loads(unsafe.read_text(encoding="utf-8"))
        value["entrypoints"]["index"] = "..\\outside.csv"
        unsafe.write_text(json.dumps(value), encoding="utf-8")

        wrong_size = self._write_package("replacement-index", "wrong-size", "replace-wrong-size")
        value = json.loads(wrong_size.read_text(encoding="utf-8"))
        value["files"]["index.csv"]["sizeBytes"] += 1
        wrong_size.write_text(json.dumps(value), encoding="utf-8")

        self._write_package("replacement-index", "duplicate-a", "replace-duplicate")
        self._write_package("replacement-index", "duplicate-b", "replace-duplicate")
        snapshot = ResourceCatalog(self.root).scan()
        reasons = "\n".join(item.reason for item in snapshot.invalid)
        self.assertIn("unsafe", reasons)
        self.assertIn("size mismatch", reasons)
        self.assertEqual(2, reasons.count("duplicate resourceId: replace-duplicate"))
        with self.assertRaises(ResourceCatalogError):
            snapshot.package("replacement-index", "replace-duplicate", verify_hashes=False)

    def test_an_unavailable_default_fails_the_entire_catalog(self) -> None:
        defaults = json.loads((self.root / "defaults.json").read_text(encoding="utf-8"))
        defaults["defaults"]["taggingModel"] = "missing-model"
        (self.root / "defaults.json").write_text(json.dumps(defaults), encoding="utf-8")
        with self.assertRaisesRegex(ResourceCatalogError, "default resources are unavailable"):
            ResourceCatalog(self.root).scan()

    def test_missing_quality_stack_is_allowed_unless_explicitly_required(self) -> None:
        (self.root / "dropout-models" / "default").rename(self.root / "dropout-models" / "removed")
        defaults = json.loads((self.root / "defaults.json").read_text(encoding="utf-8"))
        defaults["defaults"]["dropoutModel"] = "missing-quality"
        (self.root / "defaults.json").write_text(json.dumps(defaults), encoding="utf-8")

        snapshot = ResourceCatalog(self.root).scan()
        self.assertEqual(
            [{"kind": "dropoutModel", "resourceId": "missing-quality"}],
            list(snapshot.missing_defaults()),
        )
        with self.assertRaisesRegex(ResourceCatalogError, "default resources are unavailable"):
            ResourceCatalog(self.root).scan(require_quality_stack=True)

    def test_schema3_api_publishes_optional_danbooru_placeholders_without_profiles(self) -> None:
        self._use_current_defaults()
        snapshot = ResourceCatalog(self.root).scan()
        api = snapshot.api_dict()
        self.assertEqual(3, api["schemaVersion"])
        self.assertEqual(self.ids, api["defaults"])
        self.assertNotIn("profiles", api)
        placeholders = {
            item["resourceId"]: item
            for item in api["resources"]
            if item["profile"] == "danbooru" and not item["available"]
        }
        self.assertEqual(
            {
                "caption-danbooru-cl-tagger-v2-00",
                "caption-danbooru-wd-eva02-large-v3",
                "danbooru-classify-20260727-v1",
            },
            set(placeholders),
        )
        self.assertIsNone(placeholders["caption-danbooru-cl-tagger-v2-00"]["fingerprint"])
        self.assertEqual(
            "local-only",
            placeholders["caption-danbooru-wd-eva02-large-v3"]["distribution"]["mode"],
        )
        self.assertEqual(self.ids, snapshot.defaults_for())

    def test_manifest_v2_exposes_distribution_categories_and_compatibility(self) -> None:
        resource_id = "caption-danbooru-cl-tagger-v2-00"
        manifest_path = self._write_v2_tagger(resource_id)
        self._use_current_defaults()
        snapshot = ResourceCatalog(self.root).scan()
        package = snapshot.package("tagging-model", resource_id, verify_hashes=True, profile="danbooru")
        first_fingerprint = package.fingerprint
        self.assertEqual(("general", "character", "copyright"), package.adjustable_categories)
        self.assertEqual("local-only", package.distribution["mode"])

        resource = next(item for item in snapshot.api_dict()["resources"] if item["resourceId"] == resource_id)
        self.assertEqual("incompatible", resource["compatibility"]["status"])
        self.assertEqual(
            self.ids["classificationIndex"],
            resource["compatibility"]["classificationResourceId"],
        )
        self.assertEqual(["meta", "rating", "quality"], resource["excludedCategories"])
        self.assertEqual(
            {"general": 0.55, "character": 0.55, "copyright": 0.55},
            resource["defaultThresholds"],
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["distribution"]["sourceUrl"] = "https://huggingface.co/cella110n/cl_tagger_v2/tree/main/v2_00?revision=stable"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        changed = ResourceCatalog(self.root).scan().package(
            "tagging-model", resource_id, verify_hashes=False, profile="danbooru",
        )
        self.assertNotEqual(first_fingerprint, changed.fingerprint)

    def test_manifest_v2_rejects_category_partition_and_non_https_sources(self) -> None:
        category_path = self._write_v2_tagger("tagger-invalid-categories")
        category_manifest = json.loads(category_path.read_text(encoding="utf-8"))
        category_manifest["metadata"]["excludedCategories"].remove("quality")
        category_path.write_text(json.dumps(category_manifest), encoding="utf-8")

        source_path = self._write_v2_tagger("tagger-invalid-source")
        source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
        source_manifest["distribution"]["sourceUrl"] = "http://example.test/model"
        source_path.write_text(json.dumps(source_manifest), encoding="utf-8")

        bundled_path = self._write_v2_tagger("tagger-invalid-bundled")
        bundled_manifest = json.loads(bundled_path.read_text(encoding="utf-8"))
        bundled_manifest["distribution"] = {"mode": "bundled"}
        bundled_path.write_text(json.dumps(bundled_manifest), encoding="utf-8")

        vocabulary_path = self._write_v2_tagger("tagger-invalid-vocabulary")
        vocabulary_manifest = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        vocabulary_manifest["metadata"]["vocabularyFingerprint"] = "f" * 64
        vocabulary_path.write_text(json.dumps(vocabulary_manifest), encoding="utf-8")
        snapshot = ResourceCatalog(self.root).scan()
        reasons = "\n".join(item.reason for item in snapshot.invalid)
        self.assertIn("partition modelCategories", reasons)
        self.assertIn("HTTPS URL", reasons)
        self.assertIn("must be local-only", reasons)
        self.assertIn("vocabularyFingerprint", reasons)

    def test_release_copy_excludes_local_only_packages_and_detects_leaks(self) -> None:
        resource_id = "caption-danbooru-cl-tagger-v2-00"
        self._write_v2_tagger(resource_id)
        self._use_current_defaults()
        destination = Path(self.temporary.name) / "release-resource-library"
        result = copy_distributable(self.root, destination)
        self.assertEqual([resource_id], result["excludedLocalOnlyResourceIds"])
        self.assertFalse((destination / "tagging-models" / resource_id).exists())

        released = ResourceCatalog(destination).scan()
        self.assertEqual(set(self.ids.values()), {package.resource_id for package in released.packages})
        self.assertEqual(self.ids, released.defaults)
        self.assertEqual((), released.missing_defaults())
        self.assertNotIn("profiles", released.api_dict())
        assert_no_local_only_leaks(destination)

        leaked = destination / "tagging-models" / resource_id / "model.onnx"
        leaked.parent.mkdir()
        leaked.write_bytes(b"forbidden")
        with self.assertRaisesRegex(ValueError, "local-only resource path"):
            assert_no_local_only_leaks(destination)

    def test_release_copy_keeps_the_optional_ocr_category_without_a_model(self) -> None:
        destination = Path(self.temporary.name) / "release-resource-library"
        result = copy_distributable(self.root, destination)
        self.assertTrue((destination / "ocr-models").is_dir())
        self.assertNotIn(OCR_RESOURCE_ID, result["copiedResourceIds"])
        self.assertNotIn(OCR_RESOURCE_ID, result["excludedLocalOnlyResourceIds"])
        self.assertEqual(
            (self.root / "defaults.json").read_bytes(),
            (destination / "defaults.json").read_bytes(),
        )

    def test_missing_ocr_category_is_optional_and_defaults_stay_exact(self) -> None:
        snapshot = ResourceCatalog(self.root).scan()
        self.assertEqual(self.ids, snapshot.defaults_for("e621"))
        self.assertEqual(4, len(snapshot.packages))
        self.assertFalse(any(item.relative_path == "ocr-models" for item in snapshot.invalid))

    def test_existing_resource_kinds_keep_their_exact_entrypoint_layout(self) -> None:
        for kind in KINDS:
            with self.subTest(kind=kind):
                manifest_path = self._write_package(kind, f"unexpected-{kind}", f"{kind}-unexpected")
                package = manifest_path.parent
                extra = package / "extra.bin"
                extra.write_bytes(b"extra")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["files"]["extra.bin"] = {
                    "sizeBytes": 5,
                    "sha256": hashlib.sha256(b"extra").hexdigest(),
                }
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                reasons = "\n".join(item.reason for item in ResourceCatalog(self.root).scan().invalid)
                self.assertIn("entrypoints must reference every runtime file exactly once", reasons)

    def test_ocr_model_v2_is_shared_and_verifies_every_model_file(self) -> None:
        self._write_ocr_model()
        snapshot = ResourceCatalog(self.root).scan()
        packages = {package.resource_id: package for package in snapshot.packages}
        self.assertIn(OCR_RESOURCE_ID, packages)
        package = packages[OCR_RESOURCE_ID]
        package.verify_files(verify_hashes=True)
        self.assertEqual("shared", package.profile)
        self.assertEqual("ppocrv5-server-paddle-v1", package.runtime_format)
        self.assertEqual(OCR_ENTRYPOINTS, package.entrypoints)
        self.assertEqual(set(OCR_MODEL_FILES), set(package.files))
        self.assertEqual(OCR_MODEL_METADATA, package.metadata)
        self.assertEqual(self.ids, snapshot.defaults_for("e621"))

    def test_only_ocr_model_accepts_unverified_local_only_distribution(self) -> None:
        distribution = {
            "mode": "local-only",
            "sourceUrl": "https://www.paddleocr.ai/latest/en/version3.x/model_list.html",
            "licenseStatus": "unverified",
        }
        self._write_ocr_model(distribution=distribution)
        snapshot = ResourceCatalog(self.root).scan()
        packages = {package.resource_id: package for package in snapshot.packages}
        self.assertIn(
            OCR_RESOURCE_ID,
            packages,
            "\n".join(item.reason for item in snapshot.invalid),
        )
        package = snapshot.package("ocr-model", OCR_RESOURCE_ID, verify_hashes=True, profile="shared")
        self.assertEqual(distribution, package.distribution)

        destination = Path(self.temporary.name) / "release-with-local-ocr"
        copied = copy_distributable(self.root, destination)
        self.assertIn(OCR_RESOURCE_ID, copied["excludedLocalOnlyResourceIds"])
        self.assertFalse((destination / "ocr-models" / OCR_RESOURCE_ID).exists())

        invalid_id = "caption-danbooru-unverified-license"
        manifest_path = self._write_v2_tagger(invalid_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["distribution"] = distribution
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reasons = "\n".join(item.reason for item in ResourceCatalog(self.root).scan().invalid)
        self.assertIn("local-only distribution fields are invalid", reasons)

    def test_ocr_model_rejects_invalid_identity_runtime_metadata_and_layout(self) -> None:
        cases = (
            ("ocr-wrong-profile", {"profile": "e621"}, "ocr model profile"),
            ("ocr-wrong-id", {"resourceId": "ocr-other"}, "ocr model resourceId"),
            ("ocr-wrong-runtime", {"runtimeFormat": "other"}, "ocr model runtime"),
            ("ocr-wrong-metadata", {"metadata": {"models": {}}}, "ocr model metadata"),
            (
                "ocr-wrong-entrypoints",
                {"entrypoints": {**OCR_ENTRYPOINTS, "extra": r"extra\inference.json"}},
                "ocr model entrypoints",
            ),
            (
                "ocr-wrong-entrypoint-name",
                {"entrypoints": {**OCR_ENTRYPOINTS, "detection": r"detection\model.json"}},
                "ocr model entrypoints",
            ),
            (
                "ocr-duplicate-root",
                {"entrypoints": {**OCR_ENTRYPOINTS, "recognition": r"detection\inference.json"}},
                "ocr model roots",
            ),
        )
        for resource_id, change, expected in cases:
            with self.subTest(resource_id=resource_id):
                manifest_path = self._write_ocr_model(package_name=resource_id)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(change)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                reasons = "\n".join(item.reason for item in ResourceCatalog(self.root).scan().invalid)
                self.assertIn(expected, reasons)

    def test_ocr_model_rejects_files_outside_roots_and_file_set_mismatches(self) -> None:
        extra_path = self._write_ocr_model(package_name="ocr-extra")
        extra_package = extra_path.parent
        (extra_package / "detection" / "extra.bin").write_bytes(b"extra")

        missing_path = self._write_ocr_model(package_name="ocr-missing")
        (missing_path.parent / "recognition" / "model.pdiparams").unlink()

        size_path = self._write_ocr_model(package_name="ocr-wrong-size")
        size_manifest = json.loads(size_path.read_text(encoding="utf-8"))
        size_manifest["files"][r"detection\model.pdiparams"]["sizeBytes"] += 1
        size_path.write_text(json.dumps(size_manifest), encoding="utf-8")

        outside_path = self._write_ocr_model(package_name="ocr-outside")
        outside_manifest = json.loads(outside_path.read_text(encoding="utf-8"))
        outside_content = b"outside"
        (outside_path.parent / "outside.bin").write_bytes(outside_content)
        outside_manifest["files"]["outside.bin"] = {
            "sizeBytes": len(outside_content),
            "sha256": hashlib.sha256(outside_content).hexdigest(),
        }
        outside_path.write_text(json.dumps(outside_manifest), encoding="utf-8")

        escape_path = self._write_ocr_model(package_name="ocr-escape")
        escape_manifest = json.loads(escape_path.read_text(encoding="utf-8"))
        record = escape_manifest["files"].pop(r"detection\model.pdiparams")
        escape_manifest["files"][r"..\escape.bin"] = record
        escape_path.write_text(json.dumps(escape_manifest), encoding="utf-8")

        reasons = "\n".join(item.reason for item in ResourceCatalog(self.root).scan().invalid)
        self.assertIn("resource package contains missing or unlisted files", reasons)
        self.assertIn("resource file size mismatch", reasons)
        self.assertIn("ocr model files must be under entrypoint model roots", reasons)
        self.assertIn("files path is unsafe", reasons)

    def test_ocr_model_hash_tampering_is_rejected_when_hashes_are_requested(self) -> None:
        manifest_path = self._write_ocr_model()
        target = manifest_path.parent / "recognition" / "model.pdiparams"
        target.write_bytes(b"rec-paramx")
        snapshot = ResourceCatalog(self.root).scan()
        with self.assertRaisesRegex(ResourceCatalogError, "SHA-256 mismatch"):
            snapshot.package("ocr-model", OCR_RESOURCE_ID, verify_hashes=True, profile="shared")

    def test_ocr_model_reparse_tree_is_rejected_when_supported(self) -> None:
        manifest_path = self._write_ocr_model()
        package = manifest_path.parent
        target = package / "target-directory"
        target.mkdir()
        link = package / "detection" / "linked-directory"
        try:
            os.symlink(target, link, target_is_directory=True)
        except (NotImplementedError, OSError):
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                self.skipTest("current Windows account cannot create a symlink or junction")
        reasons = "\n".join(item.reason for item in ResourceCatalog(self.root).scan().invalid)
        self.assertIn("reparse", reasons)

    def test_tokenizer_resources_are_shared_and_freeze_their_exact_identity(self) -> None:
        for resource_id in TOKENIZER_RESOURCES:
            self._write_tokenizer(resource_id)
        snapshot = ResourceCatalog(self.root).scan()
        for resource_id, official_model_id in TOKENIZER_RESOURCES.items():
            with self.subTest(resource_id=resource_id):
                package = snapshot.package("tokenizer", resource_id, verify_hashes=True, profile="shared")
                self.assertEqual("shared", package.profile)
                self.assertEqual(official_model_id, package.official_model_id)
                self.assertEqual("a" * 40, package.revision)
                self.assertEqual("qwen3", package.tokenizer_family)
                self.assertEqual(1, package.context_limit)
                self.assertEqual(f"tokenizers\\{resource_id}\\resource.json", package.manifest_relative_path)

    def test_tokenizer_api_entries_publish_context_limit_and_official_model_id(self) -> None:
        """The browser must receive the manifest-backed cap instead of a UI constant."""
        self._write_tokenizer("tokenizer-qwen3-0.6b-anima-v1", context_limit=40960)
        snapshot = ResourceCatalog(self.root).scan()
        entry = next(item for item in snapshot.api_dict()["resources"] if item["kind"] == "tokenizer")

        self.assertEqual("Qwen/Qwen3-0.6B", entry["officialModelId"])
        self.assertEqual(40960, entry["contextLimit"])
        self.assertIsInstance(entry["contextLimit"], int)
        self.assertGreater(entry["contextLimit"], 0)
        self.assertNotIn("packageRoot", entry)
        self.assertNotIn("manifestPath", entry)

    def test_tokenizer_scan_can_be_disabled_without_hiding_existing_resource_kinds(self) -> None:
        self._write_tokenizer("tokenizer-qwen3-0.6b-anima-v1")
        snapshot = ResourceCatalog(self.root).scan(include_tokenizers=False)
        self.assertFalse(any(package.kind == "tokenizer" for package in snapshot.packages))
        self.assertEqual(4, len(snapshot.packages))

    def test_tokenizer_manifest_rejects_weights_unknown_files_duplicate_paths_and_mutable_revisions(self) -> None:
        cases = (
            ("tokenizer-weight", lambda value: value["files"].append({"path": "model.safetensors", "sizeBytes": 1, "sha256": "b" * 64}), "allowlist"),
            ("tokenizer-bin", lambda value: value["files"].append({"path": "pytorch_model.bin", "sizeBytes": 1, "sha256": "b" * 64}), "allowlist"),
            ("tokenizer-onnx", lambda value: value["files"].append({"path": "model.onnx", "sizeBytes": 1, "sha256": "b" * 64}), "allowlist"),
            ("tokenizer-unknown", lambda value: value["files"].append({"path": "unknown.txt", "sizeBytes": 1, "sha256": "b" * 64}), "allowlist"),
            ("tokenizer-duplicate", lambda value: value["files"].append(dict(value["files"][0])), "paths must be sorted and unique"),
            ("tokenizer-revision", lambda value: value.__setitem__("revision", "main"), "revision"),
        )
        for package_name, mutate, expected in cases:
            with self.subTest(package_name=package_name):
                path = self._write_tokenizer("tokenizer-qwen3-0.6b-anima-v1", package_name=package_name)
                manifest = json.loads(path.read_text(encoding="utf-8"))
                mutate(manifest)
                unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
                manifest["fingerprint"] = hashlib.sha256(
                    json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
                reasons = "\n".join(item.reason for item in ResourceCatalog(self.root).scan().invalid)
                self.assertIn(expected, reasons)

    def test_tokenizer_manifest_rejects_bad_fingerprint_and_missing_required_files(self) -> None:
        fingerprint_path = self._write_tokenizer("tokenizer-qwen3-0.6b-anima-v1", package_name="bad-fingerprint")
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        fingerprint["fingerprint"] = "f" * 64
        fingerprint_path.write_text(json.dumps(fingerprint, ensure_ascii=False), encoding="utf-8")

        missing_path = self._write_tokenizer("tokenizer-qwen3-vl-4b-krea2-v1", package_name="missing-tokenizer")
        missing = json.loads(missing_path.read_text(encoding="utf-8"))
        missing["files"] = [record for record in missing["files"] if record["path"] != "tokenizer.json"]
        unsigned = {key: value for key, value in missing.items() if key != "fingerprint"}
        missing["fingerprint"] = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        missing_path.write_text(json.dumps(missing, ensure_ascii=False), encoding="utf-8")

        reasons = "\n".join(item.reason for item in ResourceCatalog(self.root).scan().invalid)
        self.assertIn("fingerprint", reasons)
        self.assertIn("tokenizer.json", reasons)

    def test_project_default_catalog_preserves_existing_fingerprints_and_includes_ocr_resource(self) -> None:
        expected_existing = {
            "replace-e621-20260726-v2": "3cabbeeffd379a893a0b53d427c3dbb26ea6c587f474ae761b21afde4ee4c47b",
            "classify-e621-20260724-v1": "530323a5d1ca5c3f903c0d57b04d6f1014cdcc0ca01b8de5dc0a41e27e1d2baf",
            "caption-e621-eva02-large-full-v1": "ba31816d7e8283ab13f8127419fdb5ea9f322344fc88bb01f6d3a64afab62ec3",
            "lse14-scorer-5k-v1": "1281c8365e0a2d9bc62b5cd8953665cf8d6f5ce32f41c4ec10a347c673b128ba",
            "tokenizer-qwen3-0.6b-anima-v1": "274dac06b71d9cd4f531a85808874507768ef80ba47d3ebdb28c9a4ac7d1299d",
            "tokenizer-qwen3-vl-4b-krea2-v1": "6b3c3b9fc34439b3667f60fdb50253eb101df7a7025fe909ba51bc9185f1eeac",
        }
        root = default_resource_library_root(ROOT / ".runtime-build")
        self.assertEqual(ROOT / "resource-library", root)
        snapshot = ResourceCatalog(root).scan()
        fingerprints = {package.resource_id: package.fingerprint for package in snapshot.packages}
        self.assertEqual(expected_existing, {resource_id: fingerprints[resource_id] for resource_id in expected_existing})
        if OCR_RESOURCE_ID not in fingerprints:
            self.assertEqual(
                [(r"ocr-models\ocr-ppocrv5-server-paddle-v1", "resource.json cannot be read")],
                [
                    (item.relative_path, item.reason)
                    for item in snapshot.invalid
                    if item.relative_path == r"ocr-models\ocr-ppocrv5-server-paddle-v1"
                ],
            )
            self.skipTest("existing OCR resource is blocked by Windows ACL")
        self.assertEqual("368c31b8af0e96cc61239097688a457a050dfcc1205d054d4e631bd20529c9ca", fingerprints[OCR_RESOURCE_ID])
        self.assertEqual(set(expected_existing) | {OCR_RESOURCE_ID}, set(fingerprints))
        self.assertEqual(3, snapshot.defaults_schema_version)
        self.assertEqual(
            {
                "replacementIndex": "replace-e621-20260726-v2",
                "taggingModel": "caption-e621-eva02-large-full-v1",
                "classificationIndex": "classify-e621-20260724-v1",
                "dropoutModel": "lse14-scorer-5k-v1",
            },
            snapshot.defaults,
        )
        self.assertNotIn("profiles", snapshot.api_dict())
        self.assertEqual(
            {
                "replacementIndex": "replace-e621-20260726-v2",
                "taggingModel": "caption-e621-eva02-large-full-v1",
                "classificationIndex": "classify-e621-20260724-v1",
                "dropoutModel": "lse14-scorer-5k-v1",
            },
            snapshot.defaults_for("danbooru"),
        )
        fallback = snapshot.package(
            "classification-index", "classify-e621-20260724-v1", verify_hashes=False, profile="danbooru",
        )
        self.assertEqual("e621", fallback.profile)


if __name__ == "__main__":
    unittest.main()
