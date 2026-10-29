"""Safety contracts for project-local engineering PowerShell scripts."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = ROOT / "packaging" / "scripts" / "Sync-CoreRuntime.ps1"
NL_SYNC_SCRIPT = ROOT / "packaging" / "scripts" / "Sync-NlRuntime.ps1"
EXPORT_SYNC_SCRIPT = ROOT / "packaging" / "scripts" / "Sync-ExportRuntime.ps1"
TOKEN_BUDGET_SYNC_SCRIPT = ROOT / "packaging" / "scripts" / "Sync-TokenBudgetRuntime.ps1"
OCR_SYNC_SCRIPT = ROOT / "packaging" / "scripts" / "Sync-OcrRuntimes.ps1"
OCR_GPU_REQUIREMENTS = ROOT / "packaging" / "requirements" / "ocr-paddle-gpu.in"
OCR_GPU_DRIVER = ROOT / "packaging" / "scripts" / "ocr_gpu_resource.py"
OCR_GPU_INSTALL = ROOT / "Install-OcrGpu.bat"
OCR_GPU_RESET = ROOT / "Reset-OcrGpuRuntime.bat"
OCR_GPU_CLEAN = ROOT / "Clean-OcrGpuArtifacts.bat"
CLEAN_SCRIPT = ROOT / "packaging" / "scripts" / "Clean-LocalArtifacts.ps1"
BROWSER_SCRIPT = ROOT / "packaging" / "scripts" / "Install-FrontendBrowser.ps1"
PROJECT_NODE = ROOT / ".runtime-build" / "node-v24.18.0-win-x64" / "node.exe"
VERIFY_SCRIPT = ROOT / "packaging" / "scripts" / "Verify-Project.ps1"
CORE_PYTHON = ROOT / ".runtime-build" / "runtimes" / "core" / "python.exe"
TOOLCHAIN_PYTHON = CORE_PYTHON
PROJECT_NPM = ROOT / ".runtime-build" / "node-v24.18.0-win-x64" / "npm.cmd"
README = ROOT / "README.md"
RULES = ROOT / "RULES.md"
MODELS_README = ROOT / "models" / "README.md"
THIRD_PARTY_NOTICES = ROOT / "docs" / "THIRD_PARTY_NOTICES.md"
PLAYWRIGHT_CONFIG = ROOT / "frontend" / "playwright.config.ts"
VITE_CONFIG = ROOT / "frontend" / "vite.config.ts"
E2E_MOCK_API = ROOT / "frontend" / "tests" / "e2e" / "mockApi.ts"
E2E_GLOBAL_SETUP = ROOT / "frontend" / "tests" / "e2e" / "globalSetup.ts"
BUILD_DISTRIBUTION_SCRIPT = ROOT / "packaging" / "scripts" / "build_distribution.ps1"
RUNTIME_MANIFEST_GENERATOR = ROOT / "packaging" / "scripts" / "generate_runtime_manifests.py"
RESOLVE_WHEELS_SCRIPT = ROOT / "packaging" / "scripts" / "resolve_wheels.py"
BUILD_CPYTHON_RUNTIME_SCRIPT = ROOT / "packaging" / "scripts" / "build_cpython311_runtime.ps1"
ASSEMBLE_RUNTIME_SCRIPT = ROOT / "packaging" / "scripts" / "assemble_runtime.ps1"
EMBEDDED_SUITE = ROOT / "tests" / "run_embedded_suite.py"


def _ps_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _powershell(command: str) -> subprocess.CompletedProcess[str]:
    command = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); " + command
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResolveWheelsTests(unittest.TestCase):
    def test_metadata_uses_only_the_wheel_top_level_dist_info(self) -> None:
        resolver = _load_module(RESOLVE_WHEELS_SCRIPT, "resolve_wheels_for_test")
        with tempfile.TemporaryDirectory() as temporary_name:
            wheel = Path(temporary_name) / "fixture-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("fixture-1.0.dist-info/METADATA", "Name: fixture\nVersion: 1.0\n")
                archive.writestr("fixture/_vendor/other-2.0.dist-info/METADATA", "Name: other\nVersion: 2.0\n")
            self.assertEqual(("fixture", "1.0"), resolver.metadata_from_wheel(wheel))


class BuildCpythonRuntimeScriptTests(unittest.TestCase):
    @staticmethod
    def _prebuilt_source(root: Path) -> Path:
        source = root / "Python-3.11.15"
        built = source / "PCbuild" / "amd64"
        include = source / "Include"
        library = source / "Lib"
        built.mkdir(parents=True)
        include.mkdir()
        library.mkdir()
        (include / "patchlevel.h").write_text('#define PY_VERSION "3.11.15"\n', encoding="ascii")
        (built / "python.exe").write_bytes(b"prebuilt-python")
        (built / "python311.dll").write_bytes(b"prebuilt-dll")
        (built / "_example.pyd").write_bytes(b"prebuilt-extension")
        (library / "os.py").write_text("name = 'nt'\n", encoding="ascii")
        return source

    @staticmethod
    def _run_reuse_build(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return _powershell(
            "$ErrorActionPreference='Stop'; "
            f"& {_ps_literal(BUILD_CPYTHON_RUNTIME_SCRIPT)} -PythonSourceRoot {_ps_literal(source)} "
            f"-OutputRoot {_ps_literal(output)} -ReuseExistingBuild"
        )

    def test_reuse_existing_build_copies_prebuilt_artifacts_without_build_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = self._prebuilt_source(root)
            output = root / "output"

            completed = self._run_reuse_build(source, output)

            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            base = output / "runtimes" / "_base"
            self.assertEqual(b"prebuilt-python", (base / "python.exe").read_bytes())
            self.assertEqual(b"prebuilt-dll", (base / "python311.dll").read_bytes())
            self.assertEqual(b"prebuilt-extension", (base / "_example.pyd").read_bytes())
            self.assertEqual("name = 'nt'\n", (base / "Lib" / "os.py").read_text(encoding="ascii"))

    def test_reuse_existing_build_allows_safe_existing_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = self._prebuilt_source(root)
            output = root / "output"
            (output / "packaging" / "requirements").mkdir(parents=True)
            (output / "packaging" / "requirements" / "fixture.in").write_text("fixture==1\n", encoding="ascii")

            completed = self._run_reuse_build(source, output)

            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            self.assertEqual(b"prebuilt-python", (output / "runtimes" / "_base" / "python.exe").read_bytes())

    def test_default_build_still_rejects_existing_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = self._prebuilt_source(root)
            (source / "PCbuild" / "build.bat").write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")
            output = root / "output"
            output.mkdir()

            completed = _powershell(
                "$ErrorActionPreference='Stop'; "
                f"& {_ps_literal(BUILD_CPYTHON_RUNTIME_SCRIPT)} -PythonSourceRoot {_ps_literal(source)} "
                f"-OutputRoot {_ps_literal(output)}"
            )

            self.assertNotEqual(0, completed.returncode)
            self.assertIn("Output root already exists", completed.stdout + completed.stderr)

    def test_reuse_existing_build_rejects_unsafe_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = self._prebuilt_source(root)
            cases: list[tuple[str, Path, str]] = []

            output_file = root / "output-file"
            output_file.write_text("not a directory\n", encoding="ascii")
            cases.append(("file", output_file, "must be an ordinary directory"))

            target = root / "junction-target"
            target.mkdir()
            junction = root / "output-junction"
            created = _powershell(
                "$ErrorActionPreference='Stop'; "
                f"New-Item -ItemType Junction -Path {_ps_literal(junction)} -Target {_ps_literal(target)} | Out-Null"
            )
            self.assertEqual(0, created.returncode, created.stdout + created.stderr)
            cases.append(("junction", junction, "must not be a reparse point"))

            incomplete_base = root / "output-incomplete" / "runtimes" / "_base"
            incomplete_base.mkdir(parents=True)
            (incomplete_base / "partial.txt").write_text("partial\n", encoding="ascii")
            cases.append(("incomplete base", incomplete_base.parents[1], "Base runtime output already exists"))

            conflicting_base = root / "output-conflict" / "runtimes" / "_base"
            conflicting_base.parent.mkdir(parents=True)
            conflicting_base.write_text("conflict\n", encoding="ascii")
            cases.append(("conflicting base", conflicting_base.parents[1], "Base runtime output already exists"))

            for label, output, expected in cases:
                with self.subTest(output=label):
                    completed = self._run_reuse_build(source, output)
                    self.assertNotEqual(0, completed.returncode)
                    self.assertIn(expected, completed.stdout + completed.stderr)


class AssembleRuntimeScriptTests(unittest.TestCase):
    def test_ocr_keep_setuptools_removes_pip_but_preserves_setuptools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            base = root / "base"
            packages = base / "Lib" / "site-packages"
            (packages / "setuptools").mkdir(parents=True)
            (packages / "pip").mkdir()
            (packages / "setuptools" / "__init__.py").write_text("VALUE = 1\n", encoding="ascii")
            (packages / "pip" / "__init__.py").write_text("VALUE = 1\n", encoding="ascii")
            (base / "python.exe").write_bytes(b"base-python")
            lock = root / "ocr-paddle.lock"
            lock.write_text("# fixture\n", encoding="ascii")
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            owner = root / "anima_ocr_worker"
            owner.mkdir()
            (owner / "entry.py").write_text("VALUE = 1\n", encoding="ascii")
            builder = root / "builder.cmd"
            builder.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")
            destination = root / "ocr-paddle"

            completed = _powershell(
                "$ErrorActionPreference='Stop'; "
                f"& {_ps_literal(ASSEMBLE_RUNTIME_SCRIPT)} -BaseRuntime {_ps_literal(base)} "
                f"-DestinationRuntime {_ps_literal(destination)} -RequirementsLock {_ps_literal(lock)} "
                f"-Wheelhouse {_ps_literal(wheelhouse)} -OwnerSource {_ps_literal(owner)} "
                f"-BuildPython {_ps_literal(builder)} -KeepSetuptools"
            )

            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            self.assertTrue((destination / "Lib" / "site-packages" / "setuptools").is_dir())
            self.assertFalse((destination / "Lib" / "site-packages" / "pip").exists())
            self.assertTrue((destination / "Lib" / "site-packages" / "anima_ocr_worker" / "entry.py").is_file())


class RuntimeManifestGenerationTests(unittest.TestCase):
    def test_core_manifest_hashes_the_controlled_baseline_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install = root / "install"
            runtime = install / "runtimes" / "core"
            package = runtime / "Lib" / "site-packages" / "anima_core"
            package.mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (runtime / filename).write_bytes(filename.encode("ascii"))
            (package / "__main__.py").write_text("VALUE = 1\n", encoding="ascii")
            baseline = package / "benchmark_baseline_v1.json"
            baseline.write_text('{"schemaVersion":1}\n', encoding="ascii")
            requirements = root / "requirements"
            requirements.mkdir()
            (requirements / "core.lock").write_text("# fixture\n", encoding="ascii")

            completed = subprocess.run(
                [
                    str(CORE_PYTHON), "-B", "-I", str(RUNTIME_MANIFEST_GENERATOR),
                    "--install-root", str(install), "--requirements-root", str(requirements),
                    "--runtime-id", "core",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            manifest = json.loads(
                (install / "manifests" / "runtimes" / "core.json").read_text(encoding="utf-8")
            )
            relative = "runtimes\\core\\Lib\\site-packages\\anima_core\\benchmark_baseline_v1.json"
            critical = manifest["runtime"]["criticalFilesSha256"]
            self.assertEqual(hashlib.sha256(baseline.read_bytes()).hexdigest(), critical.get(relative))

    def test_explicit_ocr_runtime_selection_writes_only_ocr_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install = root / "install"
            runtime = install / "runtimes" / "ocr-paddle"
            worker = runtime / "Lib" / "site-packages" / "anima_ocr_worker"
            worker.mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (runtime / filename).write_bytes(filename.encode("ascii"))
            (worker / "entry.py").write_text("VALUE = 1\n", encoding="ascii")
            requirements = root / "requirements"
            requirements.mkdir()
            (requirements / "ocr-paddle.lock").write_text("# fixture\n", encoding="ascii")

            completed = subprocess.run(
                [
                    str(CORE_PYTHON), "-B", "-I", str(RUNTIME_MANIFEST_GENERATOR),
                    "--install-root", str(install), "--requirements-root", str(requirements),
                    "--include-ocr-paddle", "--runtime-id", "ocr-paddle",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            manifest = install / "manifests" / "runtimes" / "ocr-paddle.json"
            self.assertTrue(manifest.is_file())
            self.assertEqual("ocr-paddle", json.loads(manifest.read_text(encoding="utf-8"))["runtime"]["runtimeId"])
            self.assertTrue((install / "manifests" / "requirements" / "ocr-paddle.lock").is_file())
            self.assertFalse((install / "manifests" / "runtimes" / "core.json").exists())

    def test_gpu_manifest_generation_is_an_explicit_all_or_none_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install = root / "install"
            requirements = root / "requirements"
            requirements.mkdir()
            missing = subprocess.run(
                [
                    str(CORE_PYTHON), "-B", "-I", str(RUNTIME_MANIFEST_GENERATOR),
                    "--install-root", str(install), "--requirements-root", str(requirements),
                    "--include-ocr-paddle-gpu", "--runtime-id", "ocr-paddle-gpu",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertNotEqual(0, missing.returncode)
            self.assertIn("GPU runtime inputs are incomplete", missing.stderr)
            self.assertFalse((install / "manifests").exists())

            worker = install / "runtimes" / "ocr-paddle-gpu" / "Lib" / "site-packages" / "anima_ocr_worker"
            worker.mkdir(parents=True)
            runtime = worker.parents[2]
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (runtime / filename).write_bytes(filename.encode("ascii"))
            (worker / "entry.py").write_text("VALUE = 1\n", encoding="ascii")
            (requirements / "ocr-paddle-gpu.lock").mkdir()
            partial = subprocess.run(
                [
                    str(CORE_PYTHON), "-B", "-I", str(RUNTIME_MANIFEST_GENERATOR),
                    "--install-root", str(install), "--requirements-root", str(requirements),
                    "--include-ocr-paddle-gpu", "--runtime-id", "ocr-paddle-gpu",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertNotEqual(0, partial.returncode)
            self.assertIn("GPU runtime inputs are incomplete", partial.stderr)
            self.assertFalse((install / "manifests").exists())
            shutil.rmtree(requirements / "ocr-paddle-gpu.lock")
            (requirements / "ocr-paddle-gpu.lock").write_text("# fixture\n", encoding="ascii")
            complete = subprocess.run(
                [
                    str(CORE_PYTHON), "-B", "-I", str(RUNTIME_MANIFEST_GENERATOR),
                    "--install-root", str(install), "--requirements-root", str(requirements),
                    "--include-ocr-paddle-gpu", "--runtime-id", "ocr-paddle-gpu",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertEqual(0, complete.returncode, complete.stdout + complete.stderr)
            self.assertTrue((install / "manifests" / "runtimes" / "ocr-paddle-gpu.json").is_file())
            self.assertTrue((install / "manifests" / "requirements" / "ocr-paddle-gpu.lock").is_file())
            launch = json.loads(
                (install / "manifests" / "runtimes" / "ocr-paddle-gpu.json").read_text(encoding="utf-8")
            )["launch"]
            self.assertEqual(
                [
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\cublas\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\cuda_runtime\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\cudnn\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\cufft\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\curand\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\cusolver\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\cusparse\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\nvidia\\nvjitlink\\bin",
                    "runtimes\\ocr-paddle-gpu\\Lib\\site-packages\\paddle\\libs",
                ],
                launch["dllDirectoriesRelative"],
            )


class DistributionPythonIsolationTests(unittest.TestCase):
    def test_distribution_builder_invocations_use_explicit_isolated_flags(self) -> None:
        contents = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")
        invocations = [line.strip() for line in contents.splitlines() if line.strip().startswith("& $builder")]
        self.assertTrue(invocations)
        self.assertTrue(all("-B -I" in line for line in invocations), invocations)


class SyncCoreRuntimeScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SYNC_SCRIPT.is_file(), f"missing production script: {SYNC_SCRIPT}")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "core" / "src" / "anima_core"
        self.target = self.root / ".runtime-build" / "runtimes" / "core" / "Lib" / "site-packages" / "anima_core"
        self.source.mkdir(parents=True)
        self.target.mkdir(parents=True)
        (self.source / "kept.py").write_text("KEPT = 'source'\n", encoding="utf-8")
        (self.source / "changed.py").write_text("CHANGED = 'source'\n", encoding="utf-8")
        (self.source / "benchmark_baseline_v1.json").write_text('{"baselineVersion":"fixture"}\n', encoding="utf-8")
        (self.target / "changed.py").write_text("CHANGED = 'stale'\n", encoding="utf-8")
        (self.target / "stale.py").write_text("STALE = True\n", encoding="utf-8")
        self.shared_source = self.root / "shared" / "anima_caption_format" / "anima_caption_format"
        self.shared_target = self.root / ".runtime-build" / "runtimes" / "core" / "Lib" / "site-packages" / "anima_caption_format"
        self.shared_source.mkdir(parents=True)
        self.shared_target.mkdir(parents=True)
        (self.shared_source / "format.py").write_text("FORMAT = 'source'\n", encoding="utf-8")
        (self.shared_target / "format.py").write_text("FORMAT = 'stale'\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _invoke(self, *, apply: bool = False) -> subprocess.CompletedProcess[str]:
        apply_switch = " -Apply" if apply else ""
        command = (
            "$ErrorActionPreference='Stop'; "
            f"$records = & {_ps_literal(SYNC_SCRIPT)} -ProjectRoot {_ps_literal(self.root)} "
            f"-SkipManifestRefresh{apply_switch} 6>$null; @($records) | ConvertTo-Json -Depth 4 -Compress"
        )
        return _powershell(command)

    def test_sync_script_parses_in_windows_powershell(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile({_ps_literal(SYNC_SCRIPT)},[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = _powershell(command)
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_preview_reports_only_add_update_remove_records_without_writing(self) -> None:
        completed = self._invoke()

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        records = json.loads(completed.stdout)
        self.assertEqual({"Add", "Update", "Remove"}, {record["Action"] for record in records})
        self.assertTrue(all(set(record) == {"Action", "Source", "Target", "Bytes"} for record in records))
        self.assertEqual("CHANGED = 'stale'\n", (self.target / "changed.py").read_text(encoding="utf-8"))
        self.assertFalse((self.target / "kept.py").exists())
        self.assertTrue((self.target / "stale.py").exists())

    def test_direct_preview_reports_summary_without_writing(self) -> None:
        command = (
            "$ErrorActionPreference='Stop'; "
            f"& {_ps_literal(SYNC_SCRIPT)} -ProjectRoot {_ps_literal(self.root)} -SkipManifestRefresh"
        )
        completed = _powershell(command)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertIn("Core runtime sync plan: 2 add, 2 update, 1 remove,", completed.stdout)
        self.assertEqual("CHANGED = 'stale'\n", (self.target / "changed.py").read_text(encoding="utf-8"))
        self.assertFalse((self.target / "kept.py").exists())
        self.assertTrue((self.target / "stale.py").exists())

    def test_apply_copies_changed_and_added_modules_and_removes_only_stale_module(self) -> None:
        completed = self._invoke(apply=True)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertEqual((self.source / "kept.py").read_bytes(), (self.target / "kept.py").read_bytes())
        self.assertEqual((self.source / "changed.py").read_bytes(), (self.target / "changed.py").read_bytes())
        self.assertEqual(
            (self.source / "benchmark_baseline_v1.json").read_bytes(),
            (self.target / "benchmark_baseline_v1.json").read_bytes(),
        )
        self.assertFalse((self.target / "stale.py").exists())

    def test_apply_syncs_the_shared_formatter_required_by_token_budget_protocol(self) -> None:
        completed = self._invoke(apply=True)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertEqual((self.shared_source / "format.py").read_bytes(), (self.shared_target / "format.py").read_bytes())

    def test_reparse_point_target_is_rejected_before_preview_or_apply(self) -> None:
        shutil.rmtree(self.target)
        outside = self.root / "outside"
        outside.mkdir()
        junction_command = (
            f"New-Item -ItemType Junction -Path {_ps_literal(self.target)} -Target {_ps_literal(outside)} | Out-Null"
        )
        created = _powershell(junction_command)
        if created.returncode:
            self.skipTest(f"junction creation is unavailable: {created.stdout}{created.stderr}")

        completed = self._invoke()

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("reparse", (completed.stdout + completed.stderr).lower())

    def test_skip_manifest_refresh_is_rejected_for_the_real_project(self) -> None:
        completed = _powershell(f"& {_ps_literal(SYNC_SCRIPT)} -SkipManifestRefresh")

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("SkipManifestRefresh", completed.stdout + completed.stderr)


class SyncNlRuntimeScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(NL_SYNC_SCRIPT.is_file(), f"missing production script: {NL_SYNC_SCRIPT}")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "workers" / "nl" / "src" / "anima_nl_worker"
        self.target = self.root / ".runtime-build" / "runtimes" / "nl" / "Lib" / "site-packages" / "anima_nl_worker"
        self.source.mkdir(parents=True)
        self.target.mkdir(parents=True)
        (self.source / "kept.py").write_text("KEPT = 'source'\n", encoding="utf-8")
        (self.source / "changed.py").write_text("CHANGED = 'source'\n", encoding="utf-8")
        (self.target / "changed.py").write_text("CHANGED = 'stale'\n", encoding="utf-8")
        (self.target / "stale.py").write_text("STALE = True\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _invoke(self, *, apply: bool = False) -> subprocess.CompletedProcess[str]:
        apply_switch = " -Apply" if apply else ""
        command = (
            "$ErrorActionPreference='Stop'; "
            f"$records = & {_ps_literal(NL_SYNC_SCRIPT)} -ProjectRoot {_ps_literal(self.root)} "
            f"-SkipManifestRefresh{apply_switch} 6>$null; @($records) | ConvertTo-Json -Depth 4 -Compress"
        )
        return _powershell(command)

    def test_sync_script_parses_in_windows_powershell(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile({_ps_literal(NL_SYNC_SCRIPT)},[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = _powershell(command)
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_preview_reports_only_add_update_remove_records_without_writing(self) -> None:
        completed = self._invoke()

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        records = json.loads(completed.stdout)
        self.assertEqual({"Add", "Update", "Remove"}, {record["Action"] for record in records})
        self.assertTrue(all(set(record) == {"Action", "Source", "Target", "Bytes"} for record in records))
        self.assertEqual("CHANGED = 'stale'\n", (self.target / "changed.py").read_text(encoding="utf-8"))
        self.assertFalse((self.target / "kept.py").exists())
        self.assertTrue((self.target / "stale.py").exists())

    def test_direct_preview_reports_summary_without_writing(self) -> None:
        command = (
            "$ErrorActionPreference='Stop'; "
            f"& {_ps_literal(NL_SYNC_SCRIPT)} -ProjectRoot {_ps_literal(self.root)} -SkipManifestRefresh"
        )
        completed = _powershell(command)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertIn("NL runtime sync plan: 1 add, 1 update, 1 remove,", completed.stdout)
        self.assertEqual("CHANGED = 'stale'\n", (self.target / "changed.py").read_text(encoding="utf-8"))
        self.assertFalse((self.target / "kept.py").exists())
        self.assertTrue((self.target / "stale.py").exists())

    def test_apply_copies_changed_and_added_modules_and_removes_only_stale_module(self) -> None:
        completed = self._invoke(apply=True)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertEqual((self.source / "kept.py").read_bytes(), (self.target / "kept.py").read_bytes())
        self.assertEqual((self.source / "changed.py").read_bytes(), (self.target / "changed.py").read_bytes())
        self.assertFalse((self.target / "stale.py").exists())

    def test_reparse_point_target_is_rejected_before_preview_or_apply(self) -> None:
        shutil.rmtree(self.target)
        outside = self.root / "outside"
        outside.mkdir()
        junction_command = (
            f"New-Item -ItemType Junction -Path {_ps_literal(self.target)} -Target {_ps_literal(outside)} | Out-Null"
        )
        created = _powershell(junction_command)
        if created.returncode:
            self.skipTest(f"junction creation is unavailable: {created.stdout}{created.stderr}")

        completed = self._invoke()

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("reparse", (completed.stdout + completed.stderr).lower())

    def test_skip_manifest_refresh_is_rejected_for_the_real_project(self) -> None:
        completed = _powershell(f"& {_ps_literal(NL_SYNC_SCRIPT)} -SkipManifestRefresh")

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("SkipManifestRefresh", completed.stdout + completed.stderr)

    def test_real_apply_scopes_assembled_drift_to_the_nl_owner(self) -> None:
        source = NL_SYNC_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("${env:ANIMA_DRIFT_RUNTIME_IDS} = 'nl'", source)
        self.assertIn("Remove-Item -LiteralPath Env:ANIMA_DRIFT_RUNTIME_IDS", source)


class SyncExportRuntimeScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(EXPORT_SYNC_SCRIPT.is_file(), f"missing production script: {EXPORT_SYNC_SCRIPT}")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        trees = (
            ("workers/export/src/anima_export_worker", ".runtime-build/runtimes/export/Lib/site-packages/anima_export_worker"),
            ("shared/anima_caption_format/anima_caption_format", ".runtime-build/runtimes/export/Lib/site-packages/anima_caption_format"),
        )
        self.sources: list[Path] = []
        self.targets: list[Path] = []
        for source_relative, target_relative in trees:
            source = self.root / source_relative
            target = self.root / target_relative
            source.mkdir(parents=True)
            target.mkdir(parents=True)
            (source / "kept.py").write_text("KEPT = 'source'\n", encoding="utf-8")
            (source / "changed.py").write_text("CHANGED = 'source'\n", encoding="utf-8")
            (target / "changed.py").write_text("CHANGED = 'stale'\n", encoding="utf-8")
            (target / "stale.py").write_text("STALE = True\n", encoding="utf-8")
            self.sources.append(source)
            self.targets.append(target)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _invoke(self, *, apply: bool = False) -> subprocess.CompletedProcess[str]:
        apply_switch = " -Apply" if apply else ""
        command = (
            "$ErrorActionPreference='Stop'; "
            f"$records = & {_ps_literal(EXPORT_SYNC_SCRIPT)} -ProjectRoot {_ps_literal(self.root)} "
            f"-SkipManifestRefresh{apply_switch} 6>$null; @($records) | ConvertTo-Json -Depth 4 -Compress"
        )
        return _powershell(command)

    def test_preview_and_apply_handle_owner_and_shared_trees(self) -> None:
        preview = self._invoke()
        self.assertEqual(0, preview.returncode, preview.stdout + preview.stderr)
        records = json.loads(preview.stdout)
        self.assertEqual({"Add", "Update", "Remove"}, {record["Action"] for record in records})
        self.assertTrue(all(set(record) == {"Action", "Source", "Target", "Bytes"} for record in records))
        self.assertTrue(all(not (target / "kept.py").exists() for target in self.targets))
        applied = self._invoke(apply=True)
        self.assertEqual(0, applied.returncode, applied.stdout + applied.stderr)
        for source, target in zip(self.sources, self.targets):
            self.assertEqual((source / "kept.py").read_bytes(), (target / "kept.py").read_bytes())
            self.assertEqual((source / "changed.py").read_bytes(), (target / "changed.py").read_bytes())
            self.assertFalse((target / "stale.py").exists())

    def test_preview_handles_missing_shared_target_as_additions(self) -> None:
        shutil.rmtree(self.targets[1])
        preview = self._invoke()
        self.assertEqual(0, preview.returncode, preview.stdout + preview.stderr)
        records = json.loads(preview.stdout)
        shared_records = [record for record in records if "anima_caption_format" in record["Target"]]
        self.assertEqual({"Add"}, {record["Action"] for record in shared_records})
        self.assertEqual(2, len(shared_records))
        self.assertFalse(self.targets[1].exists())

    def test_reparse_target_and_real_skip_manifest_refresh_are_rejected(self) -> None:
        shutil.rmtree(self.targets[1])
        outside = self.root / "outside"
        outside.mkdir()
        created = _powershell(
            f"New-Item -ItemType Junction -Path {_ps_literal(self.targets[1])} -Target {_ps_literal(outside)} | Out-Null"
        )
        if created.returncode:
            self.skipTest(f"junction creation is unavailable: {created.stdout}{created.stderr}")
        reparse = self._invoke()
        self.assertNotEqual(0, reparse.returncode)
        self.assertIn("reparse", (reparse.stdout + reparse.stderr).lower())
        real = _powershell(f"& {_ps_literal(EXPORT_SYNC_SCRIPT)} -SkipManifestRefresh")
        self.assertNotEqual(0, real.returncode)
        self.assertIn("SkipManifestRefresh", real.stdout + real.stderr)


class SyncOcrRuntimesScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(OCR_SYNC_SCRIPT.is_file(), f"missing production script: {OCR_SYNC_SCRIPT}")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "workers" / "ocr" / "src" / "anima_ocr_worker"
        self.target = self.root / ".runtime-build" / "runtimes" / "ocr-paddle" / "Lib" / "site-packages" / "anima_ocr_worker"
        self.source.mkdir(parents=True)
        self.target.mkdir(parents=True)
        (self.source / "kept.py").write_text("KEPT = 'source'\n", encoding="utf-8")
        (self.source / "changed.py").write_text("CHANGED = 'source'\n", encoding="utf-8")
        (self.target / "changed.py").write_text("CHANGED = 'stale'\n", encoding="utf-8")
        (self.target / "stale.py").write_text("STALE = True\n", encoding="utf-8")
        (self.target / "dependency.txt").write_text("must remain\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _invoke(self, *, apply: bool = False) -> subprocess.CompletedProcess[str]:
        apply_switch = " -Apply" if apply else ""
        command = (
            "$ErrorActionPreference='Stop'; "
            f"$records = & {_ps_literal(OCR_SYNC_SCRIPT)} -ProjectRoot {_ps_literal(self.root)} "
            f"-SkipManifestRefresh{apply_switch} 6>$null; @($records) | ConvertTo-Json -Depth 4 -Compress"
        )
        return _powershell(command)

    def test_preview_and_apply_only_sync_python_owner_files(self) -> None:
        preview = self._invoke()
        self.assertEqual(0, preview.returncode, preview.stdout + preview.stderr)
        records = json.loads(preview.stdout)
        self.assertEqual({"Add", "Update", "Remove"}, {record["Action"] for record in records})
        self.assertTrue(all(record["Target"].endswith(".py") for record in records))
        self.assertEqual("CHANGED = 'stale'\n", (self.target / "changed.py").read_text(encoding="utf-8"))
        self.assertTrue((self.target / "dependency.txt").is_file())

        applied = self._invoke(apply=True)
        self.assertEqual(0, applied.returncode, applied.stdout + applied.stderr)
        self.assertEqual((self.source / "kept.py").read_bytes(), (self.target / "kept.py").read_bytes())
        self.assertEqual((self.source / "changed.py").read_bytes(), (self.target / "changed.py").read_bytes())
        self.assertFalse((self.target / "stale.py").exists())
        self.assertEqual("must remain\n", (self.target / "dependency.txt").read_text(encoding="utf-8"))

    def test_reparse_and_real_project_skip_manifest_refresh_are_rejected(self) -> None:
        shutil.rmtree(self.target)
        outside = self.root / "outside"
        outside.mkdir()
        created = _powershell(
            f"New-Item -ItemType Junction -Path {_ps_literal(self.target)} -Target {_ps_literal(outside)} | Out-Null"
        )
        if created.returncode:
            self.skipTest(f"junction creation is unavailable: {created.stdout}{created.stderr}")
        rejected = self._invoke()
        self.assertNotEqual(0, rejected.returncode)
        self.assertIn("reparse", (rejected.stdout + rejected.stderr).lower())
        real = _powershell(f"& {_ps_literal(OCR_SYNC_SCRIPT)} -SkipManifestRefresh")
        self.assertNotEqual(0, real.returncode)
        self.assertIn("SkipManifestRefresh", real.stdout + real.stderr)

    def test_real_apply_refreshes_only_cpu_manifest_and_restores_drift_filter(self) -> None:
        source = OCR_SYNC_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("'--runtime-id', 'ocr-paddle'", source)
        self.assertIn("$runtimeIds = @('ocr-paddle')", source)
        self.assertIn("${env:ANIMA_DRIFT_RUNTIME_IDS} = ($runtimeIds -join ',')", source)
        self.assertIn("Remove-Item -LiteralPath Env:ANIMA_DRIFT_RUNTIME_IDS", source)
        self.assertIn("ocr-paddle-gpu", source)
        self.assertIn("Test-CompleteGpuRuntime", source)
        self.assertIn(".runtime-build\\manifests\\requirements\\ocr-paddle-gpu.lock", source)
        self.assertIn("GPU runtime artifacts are partial", source)
        self.assertNotIn("Remove-Item -LiteralPath $targetPackage", source)

    def test_complete_gpu_artifacts_without_transient_wheelhouse_sync_gpu_preview(self) -> None:
        gpu_runtime = self.root / ".runtime-build" / "runtimes" / "ocr-paddle-gpu"
        gpu_runtime.mkdir(parents=True)
        for artifact in (
            self.root / ".runtime-build" / "manifests" / "runtimes" / "ocr-paddle-gpu.json",
            self.root / ".runtime-build" / "manifests" / "requirements" / "ocr-paddle-gpu.lock",
            self.root / "packaging" / "requirements" / "ocr-paddle-gpu.lock",
        ):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("fixture\n", encoding="ascii")
        self.assertFalse((self.root / "packaging" / "wheelhouse" / "ocr-paddle-gpu").exists())

        preview = self._invoke()

        self.assertEqual(0, preview.returncode, preview.stdout + preview.stderr)
        records = json.loads(preview.stdout)
        self.assertTrue(any(str(gpu_runtime) in record["Target"] for record in records))

    def test_missing_gpu_artifacts_keep_sync_preview_cpu_only_without_creating_a_target(self) -> None:
        preview = self._invoke()
        self.assertEqual(0, preview.returncode, preview.stdout + preview.stderr)
        self.assertFalse((self.root / ".runtime-build" / "runtimes" / "ocr-paddle-gpu").exists())


class SyncTokenBudgetRuntimeScriptTests(unittest.TestCase):
    def test_sync_script_owns_only_the_worker_and_shared_caption_formatter(self) -> None:
        self.assertTrue(TOKEN_BUDGET_SYNC_SCRIPT.is_file(), f"missing production script: {TOKEN_BUDGET_SYNC_SCRIPT}")
        source = TOKEN_BUDGET_SYNC_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("workers\\token_budget\\src\\anima_token_budget_worker", source)
        self.assertIn("runtimes\\token-budget\\Lib\\site-packages\\anima_token_budget_worker", source)
        self.assertIn("shared\\anima_caption_format\\anima_caption_format", source)
        self.assertIn("runtimes\\token-budget\\Lib\\site-packages\\anima_caption_format", source)
        self.assertIn("--runtime-id token-budget", source)
        self.assertIn("ANIMA_DRIFT_RUNTIME_IDS", source)
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile({_ps_literal(TOKEN_BUDGET_SYNC_SCRIPT)},[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = _powershell(command)
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)


class CleanLocalArtifactsScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(CLEAN_SCRIPT.is_file(), f"missing production script: {CLEAN_SCRIPT}")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scan_roots = [
            self.root / "core",
            self.root / "tests",
            self.root / "workers",
            self.root / "packaging",
            self.root / ".runtime-build" / "runtimes",
        ]
        for root in self.scan_roots:
            root.mkdir(parents=True)

        self.cache_dirs: list[Path] = []
        self.bytecode_files: list[Path] = []
        self.expected_roots: dict[Path, Path] = {}
        for index, root in enumerate(self.scan_roots):
            cache_dir = root / "nested" / "__pycache__"
            cache_dir.mkdir(parents=True)
            (cache_dir / f"cached_{index}.pyc").write_bytes(f"cache-{index}".encode("ascii"))
            (cache_dir / f"metadata_{index}.bin").write_bytes(f"metadata-{index}".encode("ascii"))
            self.cache_dirs.append(cache_dir)
            self.expected_roots[cache_dir] = root

            bytecode_file = root / f"standalone_{index}.pyc"
            bytecode_file.write_bytes(f"bytecode-{index}".encode("ascii"))
            self.bytecode_files.append(bytecode_file)
            self.expected_roots[bytecode_file] = root

        self.kept_source = self.root / "core" / "kept.py"
        self.kept_source.write_text("KEPT = True\n", encoding="utf-8")
        self.sentinels: list[Path] = []
        for relative_root in (
            "output",
            "resource-library",
            ".toolchains",
            "frontend/node_modules",
            "frontend/.playwright-browsers",
        ):
            sentinel = self.root / relative_root / "__pycache__" / "protected.pyc"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b"do-not-remove")
            self.sentinels.append(sentinel)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _invoke(self, *, apply: bool = False) -> subprocess.CompletedProcess[str]:
        apply_switch = " -Apply" if apply else ""
        command = (
            "$ErrorActionPreference='Stop'; "
            f"$records = & {_ps_literal(CLEAN_SCRIPT)} -ProjectRoot {_ps_literal(self.root)}"
            f"{apply_switch} 6>$null; @($records) | ConvertTo-Json -Depth 4 -Compress"
        )
        return _powershell(command)

    def _files(self) -> dict[str, bytes]:
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def _expected_records(self) -> dict[str, tuple[str, int, str]]:
        expected: dict[str, tuple[str, int, str]] = {}
        for cache_dir in self.cache_dirs:
            expected[str(cache_dir)] = (
                "Directory",
                sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file()),
                str(self.expected_roots[cache_dir]),
            )
        for bytecode_file in self.bytecode_files:
            expected[str(bytecode_file)] = (
                "File",
                bytecode_file.stat().st_size,
                str(self.expected_roots[bytecode_file]),
            )
        return expected

    def test_cleanup_script_parses_in_windows_powershell(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile({_ps_literal(CLEAN_SCRIPT)},[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = _powershell(command)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_preview_lists_only_whitelisted_caches_without_writing(self) -> None:
        before = self._files()
        completed = self._invoke()

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        records = json.loads(completed.stdout)
        self.assertTrue(all(set(record) == {"Path", "Type", "Bytes", "Root"} for record in records))
        actual = {
            record["Path"]: (record["Type"], record["Bytes"], record["Root"])
            for record in records
        }
        self.assertEqual(self._expected_records(), actual)
        self.assertEqual(before, self._files())
        self.assertTrue(all(sentinel.is_file() for sentinel in self.sentinels))

    def test_direct_preview_reports_counts_and_bytes_without_writing(self) -> None:
        before = self._files()
        command = (
            "$ErrorActionPreference='Stop'; "
            f"& {_ps_literal(CLEAN_SCRIPT)} -ProjectRoot {_ps_literal(self.root)}"
        )
        completed = _powershell(command)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertIn("Local artifact cleanup plan: 5 cache directories, 5 bytecode files,", completed.stdout)
        self.assertEqual(before, self._files())

    def test_apply_removes_only_whitelisted_cache_artifacts(self) -> None:
        completed = self._invoke(apply=True)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertTrue(all(not cache_dir.exists() for cache_dir in self.cache_dirs))
        self.assertTrue(all(not bytecode_file.exists() for bytecode_file in self.bytecode_files))
        self.assertEqual("KEPT = True\n", self.kept_source.read_text(encoding="utf-8"))
        self.assertTrue(all(sentinel.is_file() for sentinel in self.sentinels))

    def test_reparse_point_under_scan_root_is_rejected_without_traversal(self) -> None:
        outside = self.root / "outside"
        outside_cache = outside / "__pycache__"
        outside_cache.mkdir(parents=True)
        outside_file = outside_cache / "outside.pyc"
        outside_file.write_bytes(b"outside")
        junction = self.root / "workers" / "outside-link"
        created = _powershell(
            f"New-Item -ItemType Junction -Path {_ps_literal(junction)} -Target {_ps_literal(outside)} | Out-Null"
        )
        if created.returncode:
            self.skipTest(f"junction creation is unavailable: {created.stdout}{created.stderr}")

        completed = self._invoke()

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("reparse", (completed.stdout + completed.stderr).lower())
        self.assertTrue(outside_file.is_file())


class FrontendBrowserScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(BROWSER_SCRIPT.is_file(), f"missing production script: {BROWSER_SCRIPT}")
        self.node_modules = ROOT / "frontend" / "node_modules"
        self.browser_directory = ROOT / "frontend" / ".playwright-browsers"

    @staticmethod
    def _state(path: Path) -> tuple[bool, bool, int]:
        if not path.exists():
            return False, False, 0
        stat = path.stat()
        return True, path.is_dir(), stat.st_mtime_ns

    def _invoke(
        self,
        project_root: Path = ROOT,
        *,
        reset: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        reset_switch = " -Reset" if reset else ""
        command = (
            "$ErrorActionPreference='Stop'; "
            f"$records = & {_ps_literal(BROWSER_SCRIPT)} -ProjectRoot {_ps_literal(project_root)}"
            f"{reset_switch} 6>$null; @($records) | ConvertTo-Json -Depth 4 -Compress"
        )
        return _powershell(command)

    def test_browser_script_parses_in_windows_powershell(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile({_ps_literal(BROWSER_SCRIPT)},[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = _powershell(command)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_preview_resolves_fixed_project_toolchain_and_targets_without_writing(self) -> None:
        before = (self._state(self.node_modules), self._state(self.browser_directory))
        completed = self._invoke()

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        record = json.loads(completed.stdout)
        self.assertEqual(
            {
                "Action",
                "Node",
                "NodeVersion",
                "Npm",
                "NpmVersion",
                "NodeModules",
                "BrowserDirectory",
                "Reset",
            },
            set(record),
        )
        self.assertEqual("InstallFrontendBrowser", record["Action"])
        self.assertEqual(str(PROJECT_NODE), record["Node"])
        self.assertEqual("v24.18.0", record["NodeVersion"])
        self.assertEqual(
            str(PROJECT_NPM),
            record["Npm"],
        )
        self.assertEqual("11.16.0", record["NpmVersion"])
        self.assertEqual(str(self.node_modules), record["NodeModules"])
        self.assertEqual(str(self.browser_directory), record["BrowserDirectory"])
        self.assertFalse(record["Reset"])
        self.assertEqual(before, (self._state(self.node_modules), self._state(self.browser_directory)))

    def test_reset_preview_is_read_only(self) -> None:
        before = (self._state(self.node_modules), self._state(self.browser_directory))
        completed = self._invoke(reset=True)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        record = json.loads(completed.stdout)
        self.assertTrue(record["Reset"])
        self.assertEqual(before, (self._state(self.node_modules), self._state(self.browser_directory)))

    def test_missing_local_toolchain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            project = Path(temporary)
            (project / "frontend").mkdir()
            completed = self._invoke(project)

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("node", (completed.stdout + completed.stderr).lower())

    def test_wrong_local_npm_version_is_rejected_and_no_bare_tool_commands_are_used(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            project = Path(temporary)
            toolchain = project / ".runtime-build" / "node-v24.18.0-win-x64"
            toolchain.mkdir(parents=True)
            os.link(PROJECT_NODE, toolchain / "node.exe")
            (toolchain / "npm.cmd").write_text("@echo 0.0.0\r\n", encoding="ascii")
            (project / "frontend").mkdir()

            completed = self._invoke(project)

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("npm version", (completed.stdout + completed.stderr).lower())
        source = BROWSER_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("& node", source)
        self.assertNotIn("& npm", source)
        self.assertNotIn("& npx", source)

    def test_apply_scopes_npm_and_playwright_children_to_project_node(self) -> None:
        source = BROWSER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Invoke-WithProjectNodePath $nodeDirectory", source)


class VerifyProjectScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(VERIFY_SCRIPT.is_file(), f"missing production script: {VERIFY_SCRIPT}")

    def _gates(self, level: str) -> list[dict[str, object]]:
        command = (
            "$ErrorActionPreference='Stop'; "
            f". {_ps_literal(VERIFY_SCRIPT)} -ProjectRoot {_ps_literal(ROOT)}; "
            f"@((Get-Gates -Level {level})) | ConvertTo-Json -Depth 4 -Compress"
        )
        completed = _powershell(command)
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        decoded = json.loads(completed.stdout)
        return decoded if isinstance(decoded, list) else [decoded]

    def test_verify_script_parses_in_windows_powershell(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile({_ps_literal(VERIFY_SCRIPT)},[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = _powershell(command)

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_isolated_release_install_root_selects_its_embedded_core_and_ocr_state(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            release_root = Path(temporary) / "release"
            shutil.copytree(
                ROOT / ".runtime-build" / "runtimes" / "core",
                release_root / "runtimes" / "core",
                copy_function=os.link,
            )
            (release_root / "resource-library").mkdir()
            command = (
                "$ErrorActionPreference='Stop'; "
                f". {_ps_literal(VERIFY_SCRIPT)} -ProjectRoot {_ps_literal(ROOT)} -InstallRoot {_ps_literal(release_root)}; "
                "$gate = @(Get-Gates -Level Fast)[0]; "
                "[pscustomobject]@{runtimeRoot=$script:runtimeRoot; corePython=$script:corePython; ocrMode=$script:ocrMode; gateExecutable=$gate.Executable; gateArguments=@($gate.Arguments)} | ConvertTo-Json -Compress"
            )
            completed = _powershell(command)

            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            actual = json.loads(completed.stdout)
            expected_core = str(release_root / "runtimes" / "core" / "python.exe")
            self.assertEqual(str(release_root), actual["runtimeRoot"])
            self.assertEqual(expected_core, actual["corePython"])
            self.assertEqual("None", actual["ocrMode"])
            self.assertEqual(expected_core, actual["gateExecutable"])
            self.assertEqual(
                [
                    "-B", "-I", str(EMBEDDED_SUITE), "--level", "fast", "--ocr-mode", "none",
                    "--install-root", str(release_root),
                ],
                actual["gateArguments"],
            )

    def test_ocr_runtime_lifecycle_reports_assembled_without_adding_a_gate(self) -> None:
        source = VERIFY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("[ValidateSet('Auto', 'None', 'Cpu', 'Gpu')][string]$OcrMode = 'Auto'", source)
        self.assertIn("--ocr-mode", source)
        self.assertTrue(EMBEDDED_SUITE.is_file())
        runner = EMBEDDED_SUITE.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--ocr-mode", choices=("auto", "none", "cpu", "gpu")', runner)
        self.assertIn("inspect_optional_ocr_runtime_state", runner)
        command = (
            "$ErrorActionPreference='Stop'; "
            f". {_ps_literal(VERIFY_SCRIPT)} -ProjectRoot {_ps_literal(ROOT)}; "
            "$script:ocrMode"
        )
        completed = _powershell(command)
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertEqual("Gpu", completed.stdout.strip())
        self.assertNotIn("ocr-paddle", "\n".join(
            " ".join(str(argument) for argument in gate["Arguments"])
            for gate in self._gates("Fast")
        ))


    # OCR release assembly remains an explicit packaging concern; Fast stays launch-free.
    def test_release_assembly_has_an_explicit_ocr_runtime_opt_in(self) -> None:
        self.assertTrue(BUILD_DISTRIBUTION_SCRIPT.is_file())
        self.assertTrue(RUNTIME_MANIFEST_GENERATOR.is_file())
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")
        generator = RUNTIME_MANIFEST_GENERATOR.read_text(encoding="utf-8")

        self.assertIn("[ValidateSet('None', 'Cpu', 'Gpu')][string]$OcrComponents = 'None'", distribution)
        self.assertIn("build_optional_ocr_components.py", distribution)
        self.assertIn("$OcrComponents -eq 'None'", distribution)
        self.assertIn("--include-ocr-paddle", generator)
        self.assertIn("ASSEMBLED_OCR_RUNTIME", generator)

    def test_release_assembly_copies_the_one_click_control_plane(self) -> None:
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("$releaseRootFiles", distribution)
        for filename in (
            "Install-WebUI.bat",
            "Start-WebUI.bat",
            "Stop-WebUI.bat",
            "README.md",
            "OCR_MODEL_DOWNLOAD.md",
        ):
            self.assertIn(f"'{filename}'", distribution)
        self.assertIn("$releaseControlScripts", distribution)
        for filename in (
            "desktop_control.ps1",
            "ocr_component.py",
            "ocr_resource.py",
            "ocr_compatibility.py",
        ):
            self.assertIn(f"'{filename}'", distribution)

    def test_release_assembly_copies_optional_ocr_entrypoints_and_wrappers(self) -> None:
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")

        for filename in (
            "Install-Ocr.bat",
            "Install-OcrGpu.bat",
            "Import-OcrResource.bat",
            "Reset-OcrRuntime.bat",
            "Reset-OcrGpuRuntime.bat",
            "Clean-OcrArtifacts.bat",
            "Clean-OcrGpuArtifacts.bat",
        ):
            self.assertIn(f"'{filename}'", distribution)
        for filename in (
            "Import-OcrResource.ps1",
            "Install-OcrGpu.ps1",
            "Reset-OcrRuntime.ps1",
            "Reset-OcrGpuRuntime.ps1",
            "Clean-OcrArtifacts.ps1",
            "Clean-OcrGpuArtifacts.ps1",
            "ocr_gpu_resource.py",
        ):
            self.assertIn(f"'{filename}'", distribution)

    def test_ocr_install_wrappers_use_verified_optional_component_loader(self) -> None:
        cpu = (ROOT / "Install-Ocr.bat").read_text(encoding="ascii")
        gpu = (ROOT / "Install-OcrGpu.bat").read_text(encoding="ascii")

        self.assertIn("ocr_component.py", cpu)
        self.assertIn(" install ", cpu)
        self.assertIn("--mode cpu", cpu)
        self.assertNotIn("ocr_resource.py", cpu)
        self.assertNotIn("Import-OcrResource.bat", cpu)

        self.assertIn("ocr_component.py", gpu)
        self.assertIn(" install ", gpu)
        self.assertIn("--mode gpu", gpu)
        self.assertNotIn("ocr_gpu_resource.py", gpu)

    def test_release_assembly_preserves_the_frontend_dist_layout(self) -> None:
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("$frontendOutput = Join-Path $install 'frontend\\dist'", distribution)
        self.assertIn("-Destination $frontendOutput -Recurse", distribution)

    def test_release_assembly_has_an_all_or_none_gpu_opt_in(self) -> None:
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")
        generator = RUNTIME_MANIFEST_GENERATOR.read_text(encoding="utf-8")

        self.assertTrue(OCR_GPU_REQUIREMENTS.is_file())
        self.assertTrue(OCR_GPU_DRIVER.is_file())
        self.assertTrue(OCR_GPU_INSTALL.is_file())
        self.assertTrue(OCR_GPU_RESET.is_file())
        self.assertTrue(OCR_GPU_CLEAN.is_file())
        self.assertIn("IncludeOcrPaddleGpu", distribution)
        self.assertIn("ambiguous", distribution)
        self.assertIn("$OcrComponents = 'Gpu'", distribution)
        self.assertIn("ocr-paddle-gpu.lock", distribution)
        self.assertIn("ocr-paddle-gpu", distribution)
        self.assertIn("--include-ocr-paddle-gpu", generator)
        self.assertIn("ASSEMBLED_OCR_GPU_RUNTIME", generator)

    def test_release_assembly_declares_the_isolated_token_budget_owner_and_shared_formatter(self) -> None:
        self.assertTrue(BUILD_DISTRIBUTION_SCRIPT.is_file())
        self.assertTrue(RUNTIME_MANIFEST_GENERATOR.is_file())
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")
        generator = RUNTIME_MANIFEST_GENERATOR.read_text(encoding="utf-8")

        self.assertIn("token-budget", distribution)
        self.assertIn("anima_token_budget_worker", distribution)
        self.assertIn("anima_caption_format", distribution)
        self.assertIn("token-budget", generator)
        self.assertIn("anima_token_budget_worker.entry", generator)
        self.assertIn("anima_caption_format", generator)

    def test_release_assembly_keeps_policy_runtime_optional_with_explicit_variants(self) -> None:
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("[ValidateSet('None', 'Cpu', 'Cuda')][string]$PolicyComponents = 'None'", distribution)
        self.assertIn("Cpu = 'policy-cpu'", distribution)
        self.assertIn("Cuda = 'policy-cuda'", distribution)
        self.assertIn("if ($PolicyComponents -ne 'None')", distribution)
        self.assertIn("--runtime-id", distribution)
        self.assertIn("policy-cpu", distribution)
        self.assertIn("policy-cuda", distribution)

    def test_release_assembly_keeps_quality_resource_optional(self) -> None:
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("[switch]$IncludeQualityStack", distribution)
        self.assertIn("dropout-models\\lse14-scorer-5k-v1", distribution)
        self.assertIn("if (-not $IncludeQualityStack)", distribution)
        self.assertIn("--require-quality-stack", distribution)

    def test_release_assembly_accepts_an_explicit_wheelhouse_root(self) -> None:
        distribution = BUILD_DISTRIBUTION_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("[string]$WheelhouseRoot", distribution)
        self.assertIn("Resolve-Path -LiteralPath $WheelhouseRoot", distribution)

    def test_gate_inventory_freezes_fast_full_and_release_commands(self) -> None:
        frontend = ROOT / "frontend"
        runner = ROOT / "tests" / "run_embedded_suite.py"
        drift = ROOT / "tests" / "contract" / "test_assembled_tree_drift.py"
        ocr_integration = ROOT / "tests" / "integration" / "test_ocr_end_to_end.py"
        resource_validator = ROOT / "packaging" / "scripts" / "validate_resource_library.py"
        resource_root = ROOT / "resource-library"
        ocr_mode_arguments = ["--ocr-mode", "gpu"]
        embedded_ocr_mode_arguments = [*ocr_mode_arguments, "--install-root", str(ROOT / ".runtime-build")]
        expected = {
            "Fast": [
                ("embedded-fast", str(CORE_PYTHON), ["-B", "-I", str(runner), "--level", "fast", *embedded_ocr_mode_arguments], None),
                ("frontend-typecheck", str(PROJECT_NPM), ["--prefix", str(frontend), "run", "typecheck"], None),
            ],
            "Full": [
                ("embedded-full", str(CORE_PYTHON), ["-B", "-I", str(runner), "--level", "full", *embedded_ocr_mode_arguments], None),
                ("frontend-typecheck", str(PROJECT_NPM), ["--prefix", str(frontend), "run", "typecheck"], None),
                ("frontend-build", str(PROJECT_NPM), ["--prefix", str(frontend), "run", "build"], None),
                ("ocr-integration", str(CORE_PYTHON), ["-B", "-I", "-m", "unittest", "discover", "-s", str(ocr_integration.parent), "-t", str(ROOT), "-p", ocr_integration.name, "-v"], None),
            ],
            "Release": [
                ("embedded-full", str(CORE_PYTHON), ["-B", "-I", str(runner), "--level", "full", *embedded_ocr_mode_arguments], None),
                ("frontend-typecheck", str(PROJECT_NPM), ["--prefix", str(frontend), "run", "typecheck"], None),
                ("frontend-build", str(PROJECT_NPM), ["--prefix", str(frontend), "run", "build"], None),
                ("ocr-integration", str(CORE_PYTHON), ["-B", "-I", "-m", "unittest", "discover", "-s", str(ocr_integration.parent), "-t", str(ROOT), "-p", ocr_integration.name, "-v"], None),
                ("assembled-drift", str(CORE_PYTHON), ["-B", "-I", str(drift), *ocr_mode_arguments], None),
                ("frontend-e2e", str(PROJECT_NPM), ["--prefix", str(frontend), "run", "test:e2e"], str(frontend / ".playwright-browsers")),
                ("resource-validation", str(TOOLCHAIN_PYTHON), ["-B", "-I", str(resource_validator), "--root", str(resource_root)], None),
            ],
        }

        for level, expected_gates in expected.items():
            actual = [
                (gate["Name"], gate["Executable"], gate["Arguments"], gate["BrowserPath"])
                for gate in self._gates(level)
            ]
            self.assertEqual(expected_gates, actual)
            for _, executable, arguments, _ in actual:
                if executable.lower().endswith("python.exe"):
                    self.assertEqual(["-B", "-I"], arguments[:2])
                if executable.lower().endswith("npm.cmd"):
                    self.assertEqual(str(PROJECT_NPM), executable)

    def test_invoke_gate_propagates_fixture_exit_code_without_running_later_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            marker = fixture / "marker.txt"
            failing = fixture / "fail.cmd"
            later = fixture / "later.cmd"
            failing.write_text(f"@echo fail>>\"{marker}\"\r\n@exit /b 7\r\n", encoding="ascii")
            later.write_text(f"@echo later>>\"{marker}\"\r\n@exit /b 0\r\n", encoding="ascii")
            command = (
                "$ErrorActionPreference='Stop'; "
                f". {_ps_literal(VERIFY_SCRIPT)} -ProjectRoot {_ps_literal(ROOT)}; "
                f"Invoke-Gate -Name 'fixture-fail' -Executable {_ps_literal(failing)} -Arguments @(); "
                f"& {_ps_literal(later)}"
            )
            completed = _powershell(command)
            marker_contents = marker.read_text(encoding="utf-8")

        self.assertEqual(7, completed.returncode, completed.stdout + completed.stderr)
        self.assertEqual("fail\n", marker_contents)

    def test_npm_gate_prefixes_child_path_with_the_project_node_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            node_directory = fixture / "project-node"
            node_directory.mkdir()
            marker = fixture / "path.txt"
            fake_npm = fixture / "npm.cmd"
            fake_npm.write_text(f"@echo %PATH%>\"{marker}\"\r\n@exit /b 0\r\n", encoding="ascii")
            command = (
                "$ErrorActionPreference='Stop'; "
                f". {_ps_literal(VERIFY_SCRIPT)} -ProjectRoot {_ps_literal(ROOT)}; "
                f"$script:npm={_ps_literal(fake_npm)}; "
                f"$script:nodeDirectory={_ps_literal(node_directory)}; "
                "$before=[System.Environment]::GetEnvironmentVariable('PATH','Process'); "
                f"Invoke-Gate -Name 'fixture-npm' -Executable {_ps_literal(fake_npm)} -Arguments @() 6>$null; "
                "$after=[System.Environment]::GetEnvironmentVariable('PATH','Process'); "
                "[ordered]@{before=$before;after=$after} | ConvertTo-Json -Compress"
            )
            completed = _powershell(command)
            captured_path = marker.read_text(encoding="utf-8").strip()

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        values = json.loads(completed.stdout)
        self.assertEqual(values["before"], values["after"])
        self.assertTrue(captured_path.lower().startswith(str(node_directory).lower() + ";"))

    def test_npm_browser_gate_keeps_the_project_browser_path_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            node_directory = fixture / "project-node"
            node_directory.mkdir()
            browser_directory = fixture / "browsers"
            browser_directory.mkdir()
            marker = fixture / "browser-path.txt"
            fake_npm = fixture / "npm.cmd"
            fake_npm.write_text(
                f"@echo %PLAYWRIGHT_BROWSERS_PATH%>\"{marker}\"\r\n@exit /b 0\r\n",
                encoding="ascii",
            )
            command = (
                "$ErrorActionPreference='Stop'; "
                f". {_ps_literal(VERIFY_SCRIPT)} -ProjectRoot {_ps_literal(ROOT)}; "
                f"$script:npm={_ps_literal(fake_npm)}; "
                f"$script:nodeDirectory={_ps_literal(node_directory)}; "
                "$hadBrowserPath=Test-Path -LiteralPath 'Env:PLAYWRIGHT_BROWSERS_PATH'; "
                "$previousBrowserPath=[System.Environment]::GetEnvironmentVariable('PLAYWRIGHT_BROWSERS_PATH','Process'); "
                "try { "
                "[System.Environment]::SetEnvironmentVariable('PLAYWRIGHT_BROWSERS_PATH','outside-browser-path','Process'); "
                "$before=[System.Environment]::GetEnvironmentVariable('PLAYWRIGHT_BROWSERS_PATH','Process'); "
                f"Invoke-Gate -Name 'fixture-npm-browser' -Executable {_ps_literal(fake_npm)} -Arguments @() -BrowserPath {_ps_literal(browser_directory)} 6>$null; "
                "$after=[System.Environment]::GetEnvironmentVariable('PLAYWRIGHT_BROWSERS_PATH','Process'); "
                "[ordered]@{before=$before;after=$after} | ConvertTo-Json -Compress "
                "} finally { "
                "if ($hadBrowserPath) { [System.Environment]::SetEnvironmentVariable('PLAYWRIGHT_BROWSERS_PATH',$previousBrowserPath,'Process') } "
                "else { [System.Environment]::SetEnvironmentVariable('PLAYWRIGHT_BROWSERS_PATH',$null,'Process') } "
                "}"
            )
            completed = _powershell(command)
            captured_browser_path = marker.read_text(encoding="utf-8").strip()

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        values = json.loads(completed.stdout)
        self.assertEqual("outside-browser-path", values["before"])
        self.assertEqual(values["before"], values["after"])
        self.assertEqual(str(browser_directory), captured_browser_path)

    def test_e2e_npm_gate_scopes_an_available_port_and_restores_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            node_directory = fixture / "project-node"
            node_directory.mkdir()
            marker = fixture / "e2e-port.txt"
            fake_npm = fixture / "npm.cmd"
            fake_npm.write_text(
                f"@echo %ANIMA_E2E_PORT%>\"{marker}\"\r\n@exit /b 0\r\n",
                encoding="ascii",
            )
            command = (
                "$ErrorActionPreference='Stop'; "
                f". {_ps_literal(VERIFY_SCRIPT)} -ProjectRoot {_ps_literal(ROOT)}; "
                f"$script:npm={_ps_literal(fake_npm)}; "
                f"$script:nodeDirectory={_ps_literal(node_directory)}; "
                "$hadE2ePort=Test-Path -LiteralPath 'Env:ANIMA_E2E_PORT'; "
                "$previousE2ePort=[System.Environment]::GetEnvironmentVariable('ANIMA_E2E_PORT','Process'); "
                "try { "
                "[System.Environment]::SetEnvironmentVariable('ANIMA_E2E_PORT','outside-e2e-port','Process'); "
                "$before=[System.Environment]::GetEnvironmentVariable('ANIMA_E2E_PORT','Process'); "
                f"Invoke-Gate -Name 'frontend-e2e' -Executable {_ps_literal(fake_npm)} -Arguments @() 6>$null; "
                "$after=[System.Environment]::GetEnvironmentVariable('ANIMA_E2E_PORT','Process'); "
                "[ordered]@{before=$before;after=$after} | ConvertTo-Json -Compress "
                "} finally { "
                "if ($hadE2ePort) { [System.Environment]::SetEnvironmentVariable('ANIMA_E2E_PORT',$previousE2ePort,'Process') } "
                "else { [System.Environment]::SetEnvironmentVariable('ANIMA_E2E_PORT',$null,'Process') } "
                "}"
            )
            completed = _powershell(command)
            captured_port = marker.read_text(encoding="utf-8").strip()

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        values = json.loads(completed.stdout)
        self.assertEqual("outside-e2e-port", values["before"])
        self.assertEqual(values["before"], values["after"])
        self.assertNotEqual(values["before"], captured_port)
        self.assertGreater(int(captured_port), 0)


class E2ePortConfigurationTests(unittest.TestCase):
    def test_frontend_build_normalizes_generated_html_line_endings(self) -> None:
        self.assertTrue(VITE_CONFIG.is_file(), f"missing Vite config: {VITE_CONFIG}")
        config = VITE_CONFIG.read_text(encoding="utf-8")

        self.assertIn('name: "normalize-html-line-endings"', config)
        self.assertIn("transformIndexHtml: {", config)
        self.assertIn("return html.replace(/\\r\\n?/g, \"\\n\")", config)

    def test_e2e_config_and_mock_share_the_optional_port_override(self) -> None:
        self.assertTrue(PLAYWRIGHT_CONFIG.is_file(), f"missing Playwright config: {PLAYWRIGHT_CONFIG}")
        self.assertTrue(E2E_MOCK_API.is_file(), f"missing E2E mock API: {E2E_MOCK_API}")
        self.assertTrue(E2E_GLOBAL_SETUP.is_file(), f"missing E2E global setup: {E2E_GLOBAL_SETUP}")
        config = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
        mock_api = E2E_MOCK_API.read_text(encoding="utf-8")
        setup = E2E_GLOBAL_SETUP.read_text(encoding="utf-8")

        self.assertIn("process.env.ANIMA_E2E_PORT", config)
        self.assertIn("http://127.0.0.1:${e2ePort}", config)
        self.assertIn("process.env.ANIMA_E2E_PORT", setup)
        self.assertIn("process.env.ANIMA_E2E_PORT", mock_api)

    def test_e2e_vite_lifecycle_closes_in_process_without_windows_taskkill(self) -> None:
        self.assertTrue(PLAYWRIGHT_CONFIG.is_file(), f"missing Playwright config: {PLAYWRIGHT_CONFIG}")
        self.assertTrue(E2E_GLOBAL_SETUP.is_file(), f"missing E2E global setup: {E2E_GLOBAL_SETUP}")
        config = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
        setup = E2E_GLOBAL_SETUP.read_text(encoding="utf-8")

        self.assertIn('globalSetup: "./tests/e2e/globalSetup.ts"', config)
        self.assertNotIn("webServer:", config)
        self.assertIn("createServer", setup)
        self.assertIn("await server.listen()", setup)
        self.assertIn("await server.close()", setup)
        self.assertIn("process.env.ANIMA_E2E_PORT", setup)
        self.assertIn("process.env.ANIMA_E2E_REUSE_EXISTING_SERVER", setup)
        self.assertNotIn("taskkill", setup.lower())


class ReadmeDocumentationTests(unittest.TestCase):
    def test_root_readme_documents_local_operations_and_current_architecture(self) -> None:
        self.assertTrue(README.is_file(), f"missing root README: {README}")
        contents = README.read_text(encoding="utf-8")
        required_text = (
            r".runtime-build\runtimes\core\python.exe",
            r".toolchains\Python-3.11.15\PCbuild\amd64\python.exe",
            r".runtime-build\node-v24.18.0-win-x64\node.exe",
            r".runtime-build\node-v24.18.0-win-x64\npm.cmd",
            "Install-WebUI.bat",
            "Start-WebUI.bat",
            "Stop-WebUI.bat",
            r".\packaging\scripts\Verify-Project.ps1 -Level Fast",
            r".\packaging\scripts\Verify-Project.ps1 -Level Full",
            r".\packaging\scripts\Verify-Project.ps1 -Level Release",
            r".\packaging\scripts\Sync-CoreRuntime.ps1",
            r".\packaging\scripts\Sync-CoreRuntime.ps1 -Apply",
            r".\packaging\scripts\Clean-LocalArtifacts.ps1",
            r".\packaging\scripts\Clean-LocalArtifacts.ps1 -Apply",
            r".\packaging\scripts\Install-FrontendBrowser.ps1",
            r".\packaging\scripts\Install-FrontendBrowser.ps1 -Apply",
            "-Reset",
            "ANIMA_E2E_PORT",
            "core/src/anima_core/api.py",
            "core/src/anima_core/db.py",
            "core/src/anima_core/db_schema.py",
            "core/src/anima_core/pipeline.py",
            "core/src/anima_core/pipeline_dispatch.py",
            "core/src/anima_core/resource_catalog.py",
            "core/src/anima_core/resource_catalog_package.py",
            "core/src/anima_core/count_review_service.py",
            "workers/caption/src/anima_caption_worker/",
            "frontend/src/App.tsx",
            "formal Danbooru CL/WD resources and real model acceptance remain unavailable",
        )

        for expected in required_text:
            self.assertIn(expected, contents)

    def test_root_readme_documents_manual_ocr_model_bootstrap_boundaries(self) -> None:
        self.assertTrue(README.is_file(), f"missing root README: {README}")
        contents = README.read_text(encoding="utf-8")
        required_text = (
            "OCR is disabled by default",
            "ocr_annotations/<relative-image-path-with-extension>.ocr.json",
            "OCR_MODEL_DOWNLOAD.md",
            "ocr-model-archives",
            "only OCR-enabled jobs are blocked",
            "offline CPU OCR probe",
        )

        for expected in required_text:
            self.assertIn(expected, contents)
        self.assertNotIn("Install-WebUI.bat -OcrMode", contents)

    def test_ocr_model_download_documents_optional_install_contract(self) -> None:
        guide = ROOT / "OCR_MODEL_DOWNLOAD.md"
        self.assertTrue(guide.is_file(), f"missing OCR model guide: {guide}")
        contents = guide.read_text(encoding="utf-8")
        required_text = (
            "ocr-model-archives",
            "local-only",
            "OCR is disabled by default",
            "only a job where OCR is explicitly enabled is blocked",
            "Install-WebUI.bat",
            "88340480",
            "22a33e0ba6a21425ea4192da03bf4395c9a0c67902bd924b7328fc859073045d",
            "84869120",
            "d99be2ffd348943ab52876179168be4fb5b14f5f0812f2ae4c76d89ec2ea750a",
            "6871040",
            "6171f69605215a85624d650e9079fa45f7c3eaf944296181bcc5395bf3ddc7f6",
        )
        for expected in required_text:
            self.assertIn(expected, contents)
        self.assertNotIn("Install-WebUI.bat -OcrMode", contents)

    def test_manual_ocr_model_contract_is_synced_across_user_documents(self) -> None:
        for document in (README, MODELS_README, THIRD_PARTY_NOTICES):
            self.assertTrue(document.is_file(), f"missing OCR documentation: {document}")
            contents = document.read_text(encoding="utf-8")
            self.assertIn("OCR_MODEL_DOWNLOAD.md", contents)
            self.assertIn("ocr-model-archives", contents)

    def test_root_readme_documents_v6_token_budget_operations_and_review_boundaries(self) -> None:
        self.assertTrue(README.is_file(), f"missing root README: {README}")
        contents = README.read_text(encoding="utf-8")
        required_text = (
            r".\Import-TokenizerResources.bat",
            r".\Import-TokenizerResources.bat -Apply",
            "preview by default",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Token Budget validation is enabled by default",
            "maxTokens defaults to `490`",
            "`1..selected resource.contextLimit`",
            "not linked to `nl.apiPolicy.maxTokens`",
            "Disabling Token Budget validation",
            "does not guarantee the training token limit",
            "`general`, `style`, and `character`",
            "stable short/medium/long selection",
            "`2-3`, `4-5`, and `6-8` sentences",
            "overflow review page",
            "edit, recount, `rewrite-short`, and `apply`",
            "explicit user action",
            "may incur NL API usage",
            "proposal does not change the final JSON until `apply`",
            "never automatically loops rewrites",
        )

        for expected in required_text:
            self.assertIn(expected, contents)


if __name__ == "__main__":
    unittest.main()
