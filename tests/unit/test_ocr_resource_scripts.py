"""Fixture contracts for the preview-first OCR resource tooling."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "packaging" / "scripts"
CORE_SOURCE = ROOT / "core" / "src"
CORE_PYTHON = ROOT / ".runtime-build" / "runtimes" / "core" / "python.exe"
IMPORT_SCRIPT = SCRIPTS / "Import-OcrResource.ps1"
RESET_SCRIPT = SCRIPTS / "Reset-OcrRuntime.ps1"
CLEAN_SCRIPT = SCRIPTS / "Clean-OcrArtifacts.ps1"
IMPORT_BAT = ROOT / "Import-OcrResource.bat"
INSTALL_BAT = ROOT / "Install-Ocr.bat"
RESET_BAT = ROOT / "Reset-OcrRuntime.bat"
CLEAN_BAT = ROOT / "Clean-OcrArtifacts.bat"
DOWNLOAD_GUIDE = ROOT / "OCR_MODEL_DOWNLOAD.md"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(CORE_SOURCE))

from anima_core.resource_catalog_package import ResourcePackage


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _powershell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )


class OcrResourceScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        requirements = self.root / "packaging" / "requirements"
        requirements.mkdir(parents=True)
        (requirements / "ocr-paddle.in").write_text(
            "paddlepaddle==3.2.2\n"
            "paddleocr==3.7.0\n"
            "paddlex[ocr-core]==3.7.2\n",
            encoding="utf-8",
        )
        (self.root / "workers" / "ocr" / "src" / "anima_ocr_worker").mkdir(parents=True)
        (self.root / "workers" / "ocr" / "src" / "anima_ocr_worker" / "entry.py").write_text(
            "# fixture worker\n", encoding="utf-8",
        )
        self.model_roots = self._create_model_roots("models-a")
        self.artifacts = tuple(
            {
                "name": name,
                "url": f"https://example.invalid/{name}.tar",
                "size": len(payload),
                "sha256": _sha256(payload),
                "payload": payload,
            }
            for name, payload in (
                ("PP-OCRv5_server_det", b"det-archive"),
                ("PP-OCRv5_server_rec", b"rec-archive"),
                ("PP-LCNet_x1_0_textline_ori", b"orientation-archive"),
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _module(self):
        try:
            return importlib.import_module("ocr_resource")
        except ModuleNotFoundError:
            self.fail("missing preview-first OCR resource module")

    def _create_model_roots(self, name: str) -> dict[str, Path]:
        base = self.root / name
        roles = {
            "detection": "PP-OCRv5_server_det",
            "recognition": "PP-OCRv5_server_rec",
            "textlineOrientation": "PP-LCNet_x1_0_textline_ori",
        }
        result: dict[str, Path] = {}
        for role, directory in roles.items():
            root = base / directory
            root.mkdir(parents=True)
            (root / "inference.json").write_text("{}\n", encoding="utf-8")
            (root / "inference.pdiparams").write_bytes(f"{role}-weights".encode("ascii"))
            result[role] = root
        return result

    @staticmethod
    def _tree(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    def _prepared_package(self, module, *, model_roots: dict[str, Path] | None = None, name: str = "stage") -> Path:
        return module.prepare_resource_package(
            self.root / name,
            model_roots or self.model_roots,
            source_url=self.artifacts[0]["url"],
            license_status="unverified",
        )

    def test_default_plan_is_read_only_and_reports_exact_inputs_and_unresolved_dependencies(self) -> None:
        module = self._module()
        before = self._tree(self.root)

        plan = module.plan_import(self.root, artifacts=self.artifacts)

        self.assertEqual("preview", plan["mode"])
        self.assertEqual(
            ["paddlepaddle==3.2.2", "paddleocr==3.7.0", "paddlex[ocr-core]==3.7.2"],
            plan["requirements"],
        )
        self.assertEqual(sum(record["size"] for record in self.artifacts), plan["modelBytes"])
        self.assertEqual("unresolved", plan["dependencyBytes"])
        self.assertEqual("local-only", plan["distribution"]["mode"])
        self.assertEqual("unverified", plan["distribution"]["licenseStatus"])
        self.assertNotIn("licenseUrl", plan["distribution"])
        self.assertEqual(3, len(plan["models"]))
        self.assertTrue(all(record["cacheHit"] is False for record in plan["models"]))
        self.assertTrue(
            all(
                Path(record["cachePath"]).parent
                == self.root / "ocr-model-archives"
                for record in plan["models"]
            )
        )
        self.assertEqual(before, self._tree(self.root))
        self.assertFalse((self.root / "resource-library" / "ocr-models" / module.RESOURCE_ID / "resource.json").exists())

    def test_only_complete_matching_cache_files_are_reported_as_reusable(self) -> None:
        module = self._module()
        first = module.plan_import(self.root, artifacts=self.artifacts)
        for record, artifact in zip(first["models"], self.artifacts, strict=True):
            cache_file = Path(record["cachePath"])
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(artifact["payload"])

        second = module.plan_import(self.root, artifacts=self.artifacts)

        self.assertTrue(all(record["cacheHit"] is True for record in second["models"]))
        Path(second["models"][0]["cachePath"]).write_bytes(b"tampered")
        third = module.plan_import(self.root, artifacts=self.artifacts)
        self.assertFalse(third["models"][0]["cacheHit"])
        self.assertTrue(all(record["cacheHit"] for record in third["models"][1:]))

    def test_apply_resolves_only_complete_user_downloaded_model_archives(self) -> None:
        module = self._module()
        with self.assertRaisesRegex(module.OcrResourceError, "OCR_MODEL_DOWNLOAD.md"):
            module.resolve_manual_model_archives(self.root, artifacts=self.artifacts)

        plan = module.plan_import(self.root, artifacts=self.artifacts)
        for record, artifact in zip(plan["models"], self.artifacts, strict=True):
            path = Path(record["cachePath"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact["payload"])

        archives = module.resolve_manual_model_archives(self.root, artifacts=self.artifacts)
        self.assertEqual(
            {artifact["name"] for artifact in self.artifacts},
            set(archives),
        )
        self.assertTrue(all(path.is_file() for path in archives.values()))

        Path(plan["models"][0]["cachePath"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(module.OcrResourceError, "invalid"):
            module.resolve_manual_model_archives(self.root, artifacts=self.artifacts)

    def test_explicit_model_root_helpers_verify_archives_and_stage_resource(self) -> None:
        module = self._module()
        model_root = self.root / "application-model-archives"
        model_root.mkdir()
        artifacts = []
        for name in ("PP-OCRv5_server_det", "PP-OCRv5_server_rec", "PP-LCNet_x1_0_textline_ori"):
            source = self.root / (name + "-source")
            source.mkdir()
            (source / "inference.json").write_text("{}\n", encoding="ascii")
            (source / "inference.pdiparams").write_bytes(name.encode("ascii"))
            archive = model_root / (name + ".tar")
            with tarfile.open(archive, "w") as output:
                output.add(source, arcname=name)
            payload = archive.read_bytes()
            artifacts.append({
                "name": name,
                "url": f"https://example.invalid/{archive.name}",
                "size": len(payload),
                "sha256": _sha256(payload),
            })

        resolved = module.resolve_model_archives(model_root, artifacts=artifacts)
        staged = module.stage_model_resource(
            model_root, self.root / "stage-library", artifacts=artifacts,
        )

        self.assertEqual(set(record["name"] for record in artifacts), set(resolved))
        package = ResourcePackage.load(self.root / "stage-library", staged / "resource.json", "ocr-model")
        package.verify_files(verify_hashes=True)

    def test_model_only_import_uses_the_published_runtime_without_rebuilding_it(self) -> None:
        module = self._module()
        model_root = self.root / "ocr-model-archives"
        model_root.mkdir()
        runtime = self.root / ".runtime-build" / "runtimes" / module.RUNTIME_ID
        runtime.mkdir(parents=True)
        (runtime / "python.exe").write_bytes(b"fixture runtime")
        staging_root = self.root / ".runtime-build" / "ocr-model-import" / "staging"
        observed: dict[str, Path] = {}

        def stage_model_resource(model_input, stage_library, *, artifacts):
            observed["modelRoot"] = model_input
            observed["stageLibrary"] = stage_library
            package = stage_library / "ocr-models" / module.RESOURCE_ID
            package.mkdir(parents=True)
            return package

        with (
            mock.patch.object(module, "resolve_model_archives", return_value={}) as resolver,
            mock.patch.object(module, "stage_model_resource", side_effect=stage_model_resource),
            mock.patch.object(module, "_offline_probe", return_value={"device": "cpu"}) as probe,
            mock.patch.object(module, "install_resource_package", return_value="installed") as publisher,
            mock.patch.object(module, "_build_environment") as build_environment,
            mock.patch.object(module, "_resolve_and_stage_runtime") as rebuild_runtime,
        ):
            result = module.import_local_model_resource(
                self.root,
                model_root=model_root,
                staging_root=staging_root,
                runtime=runtime,
                artifacts=self.artifacts,
            )

        resolver.assert_called_once_with(model_root, artifacts=self.artifacts)
        self.assertEqual(model_root, observed["modelRoot"])
        self.assertEqual(runtime, probe.call_args.args[0])
        self.assertEqual("installed", result["resource"])
        publisher.assert_called_once()
        build_environment.assert_not_called()
        rebuild_runtime.assert_not_called()
        self.assertFalse(staging_root.exists())

    @unittest.skipUnless(CORE_PYTHON.is_file(), "embedded Core runtime has not been installed")
    def test_cli_apply_reports_missing_manual_models_without_a_traceback(self) -> None:
        completed = subprocess.run(
            [
                str(CORE_PYTHON),
                "-B",
                "-I",
                str(SCRIPTS / "ocr_resource.py"),
                "import",
                "--project-root",
                str(self.root),
                "--apply",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(2, completed.returncode)
        self.assertEqual("", completed.stdout)
        error = json.loads(completed.stderr)
        self.assertEqual("ocr_manual_models_unavailable", error["error"])
        self.assertEqual("OCR_MODEL_DOWNLOAD.md", error["guide"])
        self.assertNotIn("Traceback", completed.stderr)

    def test_staged_package_lists_every_model_file_and_verifies_full_hashes(self) -> None:
        module = self._module()
        staged = self._prepared_package(module)

        package = ResourcePackage.load(self.root / "stage", staged / "resource.json", "ocr-model")
        package.verify_files(verify_hashes=True)

        self.assertEqual(module.RESOURCE_ID, package.resource_id)
        self.assertEqual("shared", package.profile)
        self.assertEqual(
            {
                "detection": "detection\\inference.json",
                "recognition": "recognition\\inference.json",
                "textlineOrientation": "textline-orientation\\inference.json",
            },
            package.entrypoints,
        )
        self.assertEqual(6, len(package.files))

    def test_same_fingerprint_is_idempotent_and_different_fingerprint_refuses_overwrite(self) -> None:
        module = self._module()
        first = self._prepared_package(module, name="stage-first")

        self.assertEqual("installed", module.install_resource_package(self.root, first))
        second = self._prepared_package(module, name="stage-second")
        self.assertEqual("idempotent", module.install_resource_package(self.root, second))

        changed_roots = self._create_model_roots("models-changed")
        (changed_roots["recognition"] / "inference.pdiparams").write_bytes(b"different-recognition-weights")
        conflicting = self._prepared_package(module, model_roots=changed_roots, name="stage-conflict")
        with self.assertRaises(module.OcrResourceError):
            module.install_resource_package(self.root, conflicting)

        installed = self.root / "resource-library" / "ocr-models" / module.RESOURCE_ID
        package = ResourcePackage.load(self.root / "resource-library", installed / "resource.json", "ocr-model")
        package.verify_files(verify_hashes=True)
        self.assertNotEqual(
            ResourcePackage.load(self.root / "stage-conflict", conflicting / "resource.json", "ocr-model").fingerprint,
            package.fingerprint,
        )

    def test_transaction_rolls_back_runtime_lock_wheelhouse_manifest_and_resource_on_failure(self) -> None:
        module = self._module()
        staged_resource = self._prepared_package(module, name="stage-transaction")
        staging = self.root / "transaction-input"
        runtime = staging / "runtime" / "ocr-paddle"
        (runtime / "Lib" / "site-packages").mkdir(parents=True)
        (runtime / "python.exe").write_bytes(b"fixture-runtime")
        lock = staging / "requirements" / "ocr-paddle.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("fixture==1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
        wheelhouse = staging / "wheelhouse" / "ocr-paddle"
        wheelhouse.mkdir(parents=True)
        (wheelhouse / "fixture-1-py3-none-any.whl").write_bytes(b"fixture-wheel")
        manifest = staging / "manifest" / "ocr-paddle.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")

        def fail_on_manifest(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
            if Path(destination).name == "ocr-paddle.json":
                raise OSError("fixture manifest publication failure")
            os.replace(source, destination)

        with self.assertRaises(module.OcrResourceError):
            module.publish_install_transaction(
                self.root,
                staged_resource=staged_resource,
                staged_runtime=runtime,
                staged_lock=lock,
                staged_wheelhouse=wheelhouse,
                staged_manifest=manifest,
                replacer=fail_on_manifest,
            )

        self.assertFalse((self.root / ".runtime-build" / "runtimes" / "ocr-paddle").exists())
        self.assertFalse((self.root / "packaging" / "requirements" / "ocr-paddle.lock").exists())
        self.assertFalse((self.root / "packaging" / "wheelhouse" / "ocr-paddle").exists())
        self.assertFalse((self.root / ".runtime-build" / "manifests" / "runtimes" / "ocr-paddle.json").exists())
        self.assertFalse((self.root / "resource-library" / "ocr-models" / module.RESOURCE_ID / "resource.json").exists())

    def test_transaction_publishes_runtime_manifest_lock_with_the_runtime_manifest(self) -> None:
        module = self._module()
        staged_resource = self._prepared_package(module, name="stage-manifest-lock")
        staging = self.root / "manifest-lock-input"
        runtime = staging / "runtime" / "ocr-paddle"
        (runtime / "Lib" / "site-packages").mkdir(parents=True)
        (runtime / "python.exe").write_bytes(b"fixture-runtime")
        lock = staging / "requirements" / "ocr-paddle.lock"
        lock.parent.mkdir(parents=True)
        lock_bytes = ("fixture==1 --hash=sha256:" + "b" * 64 + "\n").encode("ascii")
        lock.write_bytes(lock_bytes)
        wheelhouse = staging / "wheelhouse" / "ocr-paddle"
        wheelhouse.mkdir(parents=True)
        (wheelhouse / "fixture-1-py3-none-any.whl").write_bytes(b"fixture-wheel")
        manifest = staging / "install" / "manifests" / "runtimes" / "ocr-paddle.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        manifest_lock = staging / "install" / "manifests" / "requirements" / "ocr-paddle.lock"
        manifest_lock.parent.mkdir(parents=True)
        manifest_lock.write_bytes(lock_bytes)

        module.publish_install_transaction(
            self.root,
            staged_resource=staged_resource,
            staged_runtime=runtime,
            staged_lock=lock,
            staged_wheelhouse=wheelhouse,
            staged_manifest=manifest,
        )

        published = self.root / ".runtime-build" / "manifests" / "requirements" / "ocr-paddle.lock"
        self.assertTrue(published.is_file())
        self.assertEqual(lock_bytes, published.read_bytes())

    def test_reset_targets_only_the_ocr_runtime_and_default_is_read_only(self) -> None:
        module = self._module()
        runtime = self.root / ".runtime-build" / "runtimes" / "ocr-paddle"
        runtime.mkdir(parents=True)
        (runtime / "sentinel.txt").write_bytes(b"delete-me")
        protected = self.root / ".runtime-build" / "runtimes" / "core" / "sentinel.txt"
        protected.parent.mkdir(parents=True)
        protected.write_bytes(b"keep-me")

        preview = module.reset_ocr_runtime(self.root)
        self.assertEqual("preview", preview["mode"])
        self.assertTrue(runtime.exists())
        self.assertEqual(b"keep-me", protected.read_bytes())

        applied = module.reset_ocr_runtime(self.root, apply=True)
        self.assertEqual("apply", applied["mode"])
        self.assertFalse(runtime.exists())
        self.assertEqual(b"keep-me", protected.read_bytes())

    def test_cleanup_touches_only_named_ocr_cache_paths_and_preserves_protected_sentinels(self) -> None:
        module = self._module()
        for relative in module.CLEANABLE_CACHE_RELATIVES:
            target = self.root / Path(relative)
            target.mkdir(parents=True)
            (target / "cache.bin").write_bytes(relative.encode("ascii"))
        protected_relatives = (
            "resource-library/ocr-models/protected/resource.json",
            ".runtime-build/ocr-import/evidence/protected.json",
            "workers/ocr/src/anima_ocr_worker/entry.py",
            "config/protected.json",
            "datasets/protected.txt",
            "output/protected.txt",
            ".toolchains/protected.txt",
            "ocr_annotations/protected.ocr.json",
        )
        protected: list[Path] = []
        for relative in protected_relatives:
            target = self.root / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"keep")
            protected.append(target)

        preview = module.clean_ocr_artifacts(self.root)
        self.assertEqual("preview", preview["mode"])
        self.assertTrue(all((self.root / Path(relative)).exists() for relative in module.CLEANABLE_CACHE_RELATIVES))
        self.assertTrue(all(path.read_bytes() == b"keep" for path in protected))

        applied = module.clean_ocr_artifacts(self.root, apply=True)
        self.assertEqual("apply", applied["mode"])
        self.assertTrue(all(not (self.root / Path(relative)).exists() for relative in module.CLEANABLE_CACHE_RELATIVES))
        self.assertTrue(all(path.read_bytes() == b"keep" for path in protected))

    def test_cleanup_preview_reports_an_uninspectable_cache_but_apply_fails_closed(self) -> None:
        module = self._module()
        target = self.root / Path(module.CLEANABLE_CACHE_RELATIVES[-1])
        target.mkdir(parents=True)
        original = module._assert_tree_safe

        def block_only_target(path: Path) -> None:
            if path == target:
                raise module.OcrResourceError("fixture access denied")
            original(path)

        with mock.patch.object(module, "_assert_tree_safe", side_effect=block_only_target):
            preview = module.clean_ocr_artifacts(self.root)
            self.assertEqual("unavailable", preview["targets"][-1]["bytes"])
            self.assertEqual("uninspectable", preview["targets"][-1]["inspection"])
            with self.assertRaises(module.OcrResourceError):
                module.clean_ocr_artifacts(self.root, apply=True)

    def test_escape_and_reparse_paths_fail_closed_before_any_write(self) -> None:
        module = self._module()
        outside = self.root.parent / (self.root.name + "-outside")
        outside.mkdir(exist_ok=True)
        with self.assertRaises(module.OcrResourceError):
            module.assert_project_path(self.root, outside)

        cache_parent = self.root / ".runtime-build"
        cache_parent.mkdir()
        junction = cache_parent / "ocr-import"
        completed = _powershell(
            "New-Item -ItemType Junction -Path '"
            + str(junction).replace("'", "''")
            + "' -Target '"
            + str(outside).replace("'", "''")
            + "' | Out-Null"
        )
        if completed.returncode:
            self.skipTest("junction creation is unavailable for this Windows test process")
        try:
            with self.assertRaises(module.OcrResourceError):
                module.plan_import(self.root, artifacts=self.artifacts)
        finally:
            if junction.exists() or junction.is_symlink():
                junction.rmdir()
            shutil.rmtree(outside, ignore_errors=True)

    def test_power_shell_and_batch_wrappers_default_to_preview_and_expose_apply_only_explicitly(self) -> None:
        for script in (IMPORT_SCRIPT, RESET_SCRIPT, CLEAN_SCRIPT):
            self.assertTrue(script.is_file(), f"missing OCR wrapper: {script}")
            source = script.read_text(encoding="utf-8")
            self.assertIn("[switch]$Apply", source)
            self.assertIn("-B", source)
            self.assertIn("-I", source)
        for wrapper in (IMPORT_BAT, RESET_BAT, CLEAN_BAT):
            self.assertTrue(wrapper.is_file(), f"missing OCR batch wrapper: {wrapper}")
            source = wrapper.read_text(encoding="ascii")
            self.assertIn("powershell.exe -NoProfile", source)
            self.assertIn("%*", source)

        import_source = IMPORT_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("LicenseUrl", import_source)
        self.assertTrue(INSTALL_BAT.is_file(), "missing one-click OCR installer")
        install_source = INSTALL_BAT.read_text(encoding="ascii")
        self.assertIn("ocr_component.py", install_source)
        self.assertIn("--mode cpu", install_source)
        self.assertIn("--model-root", install_source)
        self.assertNotIn("PIP_FIND_LINKS", install_source)
        self.assertNotIn("Import-OcrResource.bat", install_source)
        importer_source = (SCRIPTS / "ocr_resource.py").read_text(encoding="utf-8")
        self.assertIn("source_wheelhouse = assert_project_path(layout.project_root, layout.wheelhouse_target)", importer_source)
        self.assertNotIn('layout.manual_model_downloads.parent / "wheels"', importer_source)
        self.assertIn('"--source-wheelhouse"', importer_source)
        self.assertIn('"-ReuseExistingBuild"', importer_source)
        self.assertIn('"--runtime-id", RUNTIME_ID', importer_source)
        self.assertIn('"-KeepSetuptools"', importer_source)
        resolver_source = (SCRIPTS / "resolve_wheels.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--source-wheelhouse", type=Path)', resolver_source)
        self.assertIn('"--no-index", "--find-links", str(arguments.source_wheelhouse)', resolver_source)
        self.assertIn('arguments.python, "-B", "-I", "-m", "pip", "download"', resolver_source)
        assembler_source = (SCRIPTS / "assemble_runtime.ps1").read_text(encoding="utf-8")
        self.assertIn("& $builder -B -I -m pip install", assembler_source)
        self.assertTrue(DOWNLOAD_GUIDE.is_file(), "missing manual OCR model download guide")
        guide = DOWNLOAD_GUIDE.read_text(encoding="utf-8")
        self.assertIn("ocr-model-archives", guide)
        self.assertIn("Install-WebUI.bat", guide)
        self.assertNotIn("Install-WebUI.bat -OcrMode", guide)
        self.assertIn("paddle-model-ecology.bj.bcebos.com", guide)
        self.assertIn("88340480", guide)
        self.assertIn("22a33e0ba6a21425ea4192da03bf4395c9a0c67902bd924b7328fc859073045d", guide)
        self.assertIn("d99be2ffd348943ab52876179168be4fb5b14f5f0812f2ae4c76d89ec2ea750a", guide)
        self.assertIn("6171f69605215a85624d650e9079fa45f7c3eaf944296181bcc5395bf3ddc7f6", guide)
        for artifact in self.artifacts:
            self.assertIn(str(artifact["name"]), guide)

    def test_cpu_offline_probe_passes_explicit_device_to_engine_factory(self) -> None:
        source = self._module()._OFFLINE_PROBE
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_paddle_engine"
        ]

        self.assertEqual(1, len(calls))
        device = next((keyword.value for keyword in calls[0].keywords if keyword.arg == "device"), None)
        self.assertIsInstance(device, ast.Constant)
        self.assertEqual("cpu", device.value)

    @unittest.skipUnless(CORE_PYTHON.is_file(), "embedded Core runtime has not been installed")
    def test_root_batch_wrappers_run_their_default_read_only_previews(self) -> None:
        expected_actions = {
            IMPORT_BAT: "ImportOcrResource",
            RESET_BAT: "ResetOcrRuntime",
            CLEAN_BAT: "CleanOcrArtifacts",
        }
        for wrapper, action in expected_actions.items():
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", str(wrapper)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stderr.decode("cp936", errors="replace"))
            preview = json.loads(completed.stdout.decode("cp936"))
            self.assertEqual(action, preview["action"])
            self.assertEqual("preview", preview["mode"])


if __name__ == "__main__":
    unittest.main()
