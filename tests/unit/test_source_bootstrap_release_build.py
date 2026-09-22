from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "packaging" / "requirements"
BOOTSTRAP_BUILDER = ROOT / "packaging" / "scripts" / "build_bootstrap_runtime.ps1"
BOOTSTRAP_ASSET_VERIFIER = ROOT / "packaging" / "scripts" / "Test-BootstrapRuntimeAsset.ps1"
MANIFEST_BUILDER = ROOT / "packaging" / "scripts" / "build_install_manifest.py"
INVENTORY = ROOT / "packaging" / "installer" / "source-bootstrap.inventory.json"


def _load_manifest_builder():
    if not MANIFEST_BUILDER.is_file():
        return None
    name = "source_bootstrap_manifest_builder_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MANIFEST_BUILDER)
    if spec is None or spec.loader is None:
        raise AssertionError("manifest builder module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(artifact_id: str, relative_path: str, sha256: str) -> dict[str, object]:
    return {
        "id": artifact_id,
        "url": f"https://downloads.example.test/{relative_path}",
        "allowedHosts": ["downloads.example.test"],
        "sizeBytes": 1,
        "sha256": sha256,
        "relativePath": relative_path,
    }


def _inventory() -> dict[str, object]:
    wheel_hash = "a" * 64
    bootstrap_hash = "b" * 64
    return {
        "schemaVersion": 1,
        "manifest": {
            "schemaVersion": 1,
            "releaseVersion": "source-bootstrap-v1",
            "sourceCommit": "c" * 40,
            "allowedHosts": ["downloads.example.test"],
            "bootstrap": {
                "artifact": _artifact("cpython311-base", "bootstrap/cpython311-base.zip", bootstrap_hash),
                "entryRelativePath": "python.exe",
                "peakBytes": 1,
            },
            "components": [
                {
                    "componentId": "caption-e621",
                    "kind": "runtime",
                    "required": True,
                    "targetRelativePath": "runtimes/caption-e621",
                    "variants": {
                        "cpu": {
                            "artifacts": [_artifact("wheel-numpy", "wheels/numpy-1.0-py3-none-any.whl", wheel_hash)],
                            "peakBytes": 1,
                            "probe": "caption-offline",
                        }
                    },
                }
            ],
            "cleanup": {"successRelativePaths": [".runtime-build/source-bootstrap"]},
        },
        "releaseArtifacts": {
            "schemaVersion": 1,
            "releaseVersion": "source-bootstrap-v1",
            "publicationState": "published",
            "artifacts": [
                {
                    "id": "cpython311-base",
                    "publishedUrl": "https://downloads.example.test/bootstrap/cpython311-base.zip",
                    "sizeBytes": 1,
                    "sha256": bootstrap_hash,
                }
            ],
        },
        "variantLocks": {"caption-e621:cpu": "caption-e621-cpu"},
    }


class SourceBootstrapReleaseBuildTests(unittest.TestCase):
    def test_production_inventory_covers_the_required_e621_component_and_lock_surface(self) -> None:
        self.assertTrue(INVENTORY.is_file(), "source-bootstrap inventory must exist")
        value = json.loads(INVENTORY.read_text(encoding="utf-8"))
        self.assertEqual(1, value["schemaVersion"])
        self.assertEqual("published", value["releaseArtifacts"]["publicationState"])
        self.assertEqual(
            {
                "core", "caption-e621", "classify-e621", "replace-e621", "nl", "policy",
                "export", "token-budget", "ocr-cpu", "e621-indexes", "e621-tagger",
                "e621-replacement-indexes", "qwen3-tokenizer", "quality-stack", "ocr-gpu",
            },
            {item["componentId"] for item in value["manifest"]["components"]},
        )
        components = {item["componentId"]: item for item in value["manifest"]["components"]}
        self.assertFalse(components["ocr-cpu"]["required"])
        self.assertFalse(components["ocr-gpu"]["required"])
        self.assertEqual(
            {
                "core:cpu", "caption-e621:cpu", "caption-e621:cuda", "classify-e621:cpu",
                "replace-e621:cpu", "nl:cpu", "policy:cpu", "policy:cuda", "export:cpu",
                "token-budget:cpu", "ocr-cpu:cpu", "ocr-gpu:cuda",
            },
            set(value["variantLocks"]),
        )
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        self.assertEqual("source-bootstrap-e621-v8", module.audit_inventory(value, REQUIREMENTS)["releaseVersion"])

    def test_candidate_inventory_keeps_e621_indexes_in_independent_resource_packages(self) -> None:
        self.assertTrue(INVENTORY.is_file(), "source-bootstrap inventory must exist")
        value = json.loads(INVENTORY.read_text(encoding="utf-8"))
        components = {item["componentId"]: item for item in value["manifest"]["components"]}

        expected = {
            "e621-indexes": {
                "target": "resource-library/classification-indexes/e621-classify-20260724-v1",
                "files": {"resource.json", "e621_tag_dictionary.json", "e621_count_wiki.sqlite3"},
            },
            "e621-replacement-indexes": {
                "target": "resource-library/replacement-indexes/e621-replace-20260726-v2",
                "files": {"resource.json", "e621_tag_replacement_index.csv", "e621_tag_replacement_index_manual_zh.docx"},
            },
        }
        for component_id, contract in expected.items():
            with self.subTest(component_id=component_id):
                component = components[component_id]
                self.assertEqual(contract["target"], component["targetRelativePath"])
                artifacts = component["variants"]["shared"]["artifacts"]
                self.assertEqual(contract["files"], {artifact["relativePath"] for artifact in artifacts})
                self.assertTrue(all(artifact["delivery"] == "source-tree" for artifact in artifacts))

    def test_manifest_builder_rejects_required_ocr_components(self) -> None:
        self.assertTrue(INVENTORY.is_file(), "source-bootstrap inventory must exist")
        value = json.loads(INVENTORY.read_text(encoding="utf-8"))
        value["manifest"]["components"] = [
            dict(component) for component in value["manifest"]["components"]
        ]
        next(component for component in value["manifest"]["components"] if component["componentId"] == "ocr-cpu")["required"] = True
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        with self.assertRaisesRegex(module.ManifestBuildError, "optional component"):
            module.audit_inventory(value, REQUIREMENTS)

    def test_production_source_tree_identities_match_the_tracked_git_blobs(self) -> None:
        value = json.loads(INVENTORY.read_text(encoding="utf-8"))
        artifacts = [
            artifact
            for component in value["manifest"]["components"]
            for variant in component["variants"].values()
            for artifact in variant.get("artifacts", [])
            if artifact.get("delivery") == "source-tree"
        ]

        for artifact in artifacts:
            with self.subTest(artifact=artifact["id"]):
                payload = subprocess.check_output(
                    ["git", "show", f"HEAD:{artifact['sourceRelativePath']}"],
                    cwd=ROOT,
                )
                self.assertEqual(artifact["sizeBytes"], len(payload))
                self.assertEqual(artifact["sha256"], hashlib.sha256(payload).hexdigest())

    def test_huggingface_resolve_urls_request_the_download_response(self) -> None:
        value = json.loads(INVENTORY.read_text(encoding="utf-8"))
        artifacts = [
            artifact
            for component in value["manifest"]["components"]
            for variant in component["variants"].values()
            for artifact in variant.get("artifacts", [])
            if artifact.get("url", "").startswith("https://huggingface.co/")
        ]

        self.assertTrue(artifacts)
        for artifact in artifacts:
            url = artifact["url"]
            with self.subTest(url=url):
                self.assertTrue(url.endswith("?download=true"), url)
                self.assertEqual(
                    ["huggingface.co", "us.aws.cdn.hf.co"],
                    artifact["allowedHosts"],
                )

        hydra = next(item for item in artifacts if item["id"] == "quality-stack-jtp3-jtp-3-hydra-safetensors")
        self.assertIn("/models/jtp-3-hydra.safetensors?download=true", hydra["url"])

    def test_source_bootstrap_defaults_are_e621_only_and_tracked_for_the_installer(self) -> None:
        defaults = ROOT / "resource-library" / "defaults.json"
        self.assertTrue(defaults.is_file(), "source-bootstrap defaults must exist")
        value = json.loads(defaults.read_text(encoding="utf-8"))
        self.assertEqual(3, value["schemaVersion"])
        self.assertEqual({"replacementIndex", "classificationIndex", "taggingModel", "dropoutModel"}, set(value["defaults"]))
        self.assertEqual(
            {
                "replacementIndex": "replace-e621-20260726-v2",
                "classificationIndex": "classify-e621-20260724-v1",
                "taggingModel": "caption-e621-eva02-large-full-v1",
                "dropoutModel": "lse14-scorer-5k-v1",
            },
            value["defaults"],
        )

    def test_production_inventory_records_the_published_bootstrap_identity(self) -> None:
        self.assertTrue(INVENTORY.is_file(), "source-bootstrap inventory must exist")
        value = json.loads(INVENTORY.read_text(encoding="utf-8"))
        records = value["releaseArtifacts"]["artifacts"]
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("cpython311-base", record["id"])
        self.assertEqual(
            "https://github.com/lse14/e621-standard-capotion-workflow/releases/download/source-bootstrap-e621-v8/cpython-3.11.15-win-amd64-572501a.zip",
            record["publishedUrl"],
        )
        self.assertEqual(20565968, record["sizeBytes"])
        self.assertEqual("3ab496658760f8bbf90b6593231ba1f4de90d4bb732e7ce19f25683382e1424a", record["sha256"])

    def test_cpu_variants_have_no_cuda_distribution(self) -> None:
        required = {
            "caption-e621-cpu.in", "caption-e621-cpu.lock", "caption-e621-cuda.in", "caption-e621-cuda.lock",
            "policy-cpu.in", "policy-cpu.lock", "policy-cuda.in", "policy-cuda.lock",
        }
        missing = sorted(name for name in required if not (REQUIREMENTS / name).is_file())
        self.assertEqual([], missing, f"missing explicit source-bootstrap variant inputs: {missing}")
        for name in ("caption-e621-cpu.in", "caption-e621-cpu.lock", "policy-cpu.in", "policy-cpu.lock"):
            with self.subTest(path=name):
                text = (REQUIREMENTS / name).read_text(encoding="utf-8").lower()
                self.assertNotIn("onnxruntime-gpu", text)
                self.assertNotIn("+cu", text)
                self.assertNotIn("cuda", text)
        self.assertIn("onnxruntime-gpu==1.26.0", (REQUIREMENTS / "caption-e621-cuda.in").read_text(encoding="utf-8"))
        self.assertIn("torch==2.9.1+cu128", (REQUIREMENTS / "policy-cuda.in").read_text(encoding="utf-8"))
        self.assertIn("torch==2.9.1+cpu", (REQUIREMENTS / "policy-cpu.in").read_text(encoding="utf-8"))

    def test_core_lock_excludes_unused_pywebview_and_the_sdist_only_proxy_tools(self) -> None:
        core_input = (REQUIREMENTS / "core.in").read_text(encoding="utf-8").lower()
        core_lock = (REQUIREMENTS / "core.lock").read_text(encoding="utf-8").lower()
        self.assertNotIn("pywebview", core_input)
        self.assertNotIn("pywebview", core_lock)
        self.assertNotIn("proxy-tools", core_lock)

    def test_bootstrap_runtime_builder_has_an_offline_stdlib_contract_and_parses(self) -> None:
        self.assertTrue(BOOTSTRAP_BUILDER.is_file(), "bootstrap runtime builder must exist")
        source = BOOTSTRAP_BUILDER.read_text(encoding="utf-8")
        self.assertIn("python311.dll", source)
        self.assertIn("python311._pth", source)
        self.assertIn("-B -I -c", source)
        self.assertIn("sys.version_info[:3] == (3, 11, 15)", source)
        self.assertIn("Lib\\site-packages", source)
        self.assertIn("Bootstrap base site-packages must be empty", source)
        self.assertIn("Add-Type -AssemblyName System.IO.Compression\n", source)
        self.assertIn("System.IO.Compression", source)
        self.assertNotIn("pip install", source.lower())
        command = (
            "$tokens=$null; $errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "(Resolve-Path 'packaging\\scripts\\build_bootstrap_runtime.ps1'),[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_bootstrap_asset_verifier_rejects_tampering(self) -> None:
        self.assertTrue(BOOTSTRAP_ASSET_VERIFIER.is_file(), "bootstrap asset verifier must exist")
        test_root = ROOT / ".test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temporary_name:
            temporary = Path(temporary_name)
            asset = temporary / "cpython-3.11.15-win-amd64.zip"
            provenance = temporary / "cpython-3.11.15-win-amd64.provenance.json"
            source_commit = "a" * 40
            asset.write_bytes(b"verified bootstrap fixture")
            provenance.write_text(
                json.dumps({
                    "schemaVersion": 1,
                    "releaseVersion": "source-bootstrap-test",
                    "sourceCommit": source_commit,
                    "pythonVersion": "3.11.15",
                    "assetFileName": asset.name,
                    "assetSizeBytes": asset.stat().st_size,
                    "assetSha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                    "buildScriptSha256": hashlib.sha256(BOOTSTRAP_BUILDER.read_bytes()).hexdigest(),
                    "offlineProbe": "bootstrap-stdlib-ok",
                }),
                encoding="utf-8",
            )

            with asset.open("r+b") as target:
                target.seek(-1, 2)
                original = target.read(1)
                target.seek(-1, 2)
                target.write(bytes([original[0] ^ 0x01]))
            tampered = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BOOTSTRAP_ASSET_VERIFIER),
                    "-ProjectRoot", str(ROOT),
                    "-AssetZip", str(asset),
                    "-Provenance", str(provenance),
                    "-ExpectedSourceCommit", source_commit,
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, tampered.returncode)
            self.assertIn("SHA-256", tampered.stdout + tampered.stderr)

    def test_bootstrap_asset_verifier_rejects_backslash_parent_entry(self) -> None:
        self.assertTrue(BOOTSTRAP_ASSET_VERIFIER.is_file(), "bootstrap asset verifier must exist")
        test_root = ROOT / ".test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temporary_name:
            temporary = Path(temporary_name)
            asset = temporary / "cpython-3.11.15-win-amd64.zip"
            provenance = temporary / "cpython-3.11.15-win-amd64.provenance.json"
            with zipfile.ZipFile(asset, "w") as archive:
                archive.writestr("python.exe", b"not an executable")
                archive.writestr("python311.dll", b"not a dll")
                archive.writestr("python311._pth", b"Lib\nLib/site-packages\n")
                archive.writestr("Lib/os.py", b"pass\n")
                archive.writestr("Lib\\..\\outside.txt", b"unsafe")
            asset_bytes = asset.read_bytes()
            provenance.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "releaseVersion": "source-bootstrap-test",
                        "sourceCommit": "a" * 40,
                        "pythonVersion": "3.11.15",
                        "assetFileName": asset.name,
                        "assetSizeBytes": len(asset_bytes),
                        "assetSha256": hashlib.sha256(asset_bytes).hexdigest(),
                        "buildScriptSha256": hashlib.sha256(BOOTSTRAP_BUILDER.read_bytes()).hexdigest(),
                        "offlineProbe": "bootstrap-stdlib-ok",
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BOOTSTRAP_ASSET_VERIFIER),
                    "-ProjectRoot", str(ROOT),
                    "-AssetZip", str(asset),
                    "-Provenance", str(provenance),
                    "-ExpectedSourceCommit", "a" * 40,
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertIn("unsafe entry", completed.stdout + completed.stderr)

    def test_manifest_builder_rejects_missing_artifact_identity_and_lock_mismatch(self) -> None:
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        with tempfile.TemporaryDirectory() as temporary_name:
            requirements_root = Path(temporary_name)
            (requirements_root / "caption-e621-cpu.lock").write_text(
                "numpy==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="ascii"
            )
            inventory = _inventory()
            manifest = module.build_manifest(inventory, requirements_root)
            self.assertEqual("source-bootstrap-v1", manifest["releaseVersion"])

            missing_identity = copy.deepcopy(inventory)
            del missing_identity["manifest"]["components"][0]["variants"]["cpu"]["artifacts"][0]["sizeBytes"]
            with self.assertRaisesRegex(module.ManifestBuildError, "artifact fields"):
                module.build_manifest(missing_identity, requirements_root)

            mismatched_lock = copy.deepcopy(inventory)
            mismatched_lock["manifest"]["components"][0]["variants"]["cpu"]["artifacts"][0]["sha256"] = "d" * 64
            with self.assertRaisesRegex(module.ManifestBuildError, "does not match wheel lock"):
                module.build_manifest(mismatched_lock, requirements_root)

            incomplete_release = copy.deepcopy(inventory)
            del incomplete_release["releaseArtifacts"]["artifacts"][0]["publishedUrl"]
            with self.assertRaisesRegex(module.ManifestBuildError, "release artifact record"):
                module.build_manifest(incomplete_release, requirements_root)

    def test_candidate_inventory_is_audit_only_and_cannot_build_manifest(self) -> None:
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        with tempfile.TemporaryDirectory() as temporary_name:
            requirements_root = Path(temporary_name)
            (requirements_root / "caption-e621-cpu.lock").write_text(
                "numpy==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="ascii"
            )
            candidate = _inventory()
            candidate["releaseArtifacts"]["publicationState"] = "candidate"
            candidate["releaseArtifacts"]["artifacts"] = [{
                "id": "cpython311-base",
                "candidatePath": ".release-candidate/bootstrap/cpython.zip",
                "candidateSizeBytes": 1,
                "candidateSha256": "b" * 64,
            }]
            with self.assertRaisesRegex(module.ManifestBuildError, "published release identity"):
                module.build_manifest(candidate, requirements_root)

    def test_candidate_inventory_can_be_audited_without_a_published_release(self) -> None:
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        with tempfile.TemporaryDirectory() as temporary_name:
            requirements_root = Path(temporary_name)
            (requirements_root / "caption-e621-cpu.lock").write_text(
                "numpy==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="ascii"
            )
            candidate = _inventory()
            candidate["releaseArtifacts"]["publicationState"] = "candidate"
            candidate["releaseArtifacts"]["artifacts"] = [{
                "id": "cpython311-base",
                "candidatePath": ".release-candidate/bootstrap/cpython.zip",
                "candidateSizeBytes": 1,
                "candidateSha256": "b" * 64,
            }]

            self.assertEqual("source-bootstrap-v1", module.audit_inventory(candidate, requirements_root)["releaseVersion"])

    def test_candidate_inventory_accepts_a_safe_forward_slash_candidate_path(self) -> None:
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        with tempfile.TemporaryDirectory() as temporary_name:
            requirements_root = Path(temporary_name)
            (requirements_root / "caption-e621-cpu.lock").write_text(
                "numpy==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="ascii"
            )
            candidate = _inventory()
            candidate["manifest"]["bootstrap"]["artifact"] = {
                "id": "cpython311-base",
                "delivery": "candidate-release",
                "candidatePath": ".release-candidate/bootstrap/cpython.zip",
                "sizeBytes": 1,
                "sha256": "b" * 64,
                "relativePath": "bootstrap/cpython.zip",
            }
            candidate["releaseArtifacts"]["publicationState"] = "candidate"
            candidate["releaseArtifacts"]["artifacts"] = [{
                "id": "cpython311-base",
                "candidatePath": ".release-candidate/bootstrap/cpython.zip",
                "candidateSizeBytes": 1,
                "candidateSha256": "b" * 64,
            }]

            self.assertEqual("source-bootstrap-v1", module.audit_inventory(candidate, requirements_root)["releaseVersion"])

    def test_published_asset_validation_rejects_size_or_digest_mismatch(self) -> None:
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        with tempfile.TemporaryDirectory() as temporary_name:
            asset = Path(temporary_name) / "asset.zip"
            asset.write_bytes(b"published-bytes")
            declared = {
                "id": "bootstrap",
                "publishedUrl": "https://github.com/example/project/releases/download/v1/asset.zip",
                "sizeBytes": asset.stat().st_size,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }
            self.assertEqual(declared["sha256"], module.validate_published_asset(asset, declared)["sha256"])
            declared["sha256"] = "c" * 64
            with self.assertRaisesRegex(module.ManifestBuildError, "does not match"):
                module.validate_published_asset(asset, declared)

    def test_empty_source_only_lock_is_valid(self) -> None:
        module = _load_manifest_builder()
        self.assertIsNotNone(module, "source bootstrap manifest builder must exist")
        with tempfile.TemporaryDirectory() as temporary_name:
            lock = Path(temporary_name) / "classify-e621.lock"
            lock.write_text("# No third-party production dependencies for this runtime.\n", encoding="ascii")
            self.assertEqual({}, module.parse_lock(lock))


if __name__ == "__main__":
    unittest.main()
