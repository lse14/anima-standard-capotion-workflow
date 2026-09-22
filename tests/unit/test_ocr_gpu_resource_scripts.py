"""Preview-only safety contracts for the future OCR Paddle GPU lifecycle."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CORE_PYTHON = ROOT / ".runtime-build" / "runtimes" / "core" / "python.exe"
DRIVER = ROOT / "packaging" / "scripts" / "ocr_gpu_resource.py"
GPU_REQUIREMENTS = ROOT / "packaging" / "requirements" / "ocr-paddle-gpu.in"
ROOT_ENTRYPOINTS = {
    "install": ROOT / "Install-OcrGpu.bat",
    "reset": ROOT / "Reset-OcrGpuRuntime.bat",
    "clean": ROOT / "Clean-OcrGpuArtifacts.bat",
}
OFFICIAL_WHEEL_URL = (
    "https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/"
    "paddlepaddle_gpu-3.3.0-cp311-cp311-win_amd64.whl"
)


def _load_driver():
    spec = importlib.util.spec_from_file_location("ocr_gpu_resource_for_test", DRIVER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load GPU lifecycle driver: {DRIVER}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


class OcrGpuResourceScriptTests(unittest.TestCase):
    def _run_driver(self, project: Path, action: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(CORE_PYTHON), "-B", "-I", str(DRIVER),
                "--project-root", str(project), "--action", action, *extra,
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=False,
        )

    def test_lifecycle_files_and_direct_requirements_are_exact(self) -> None:
        self.assertTrue(DRIVER.is_file(), f"missing lifecycle driver: {DRIVER}")
        self.assertTrue(GPU_REQUIREMENTS.is_file(), f"missing GPU requirements input: {GPU_REQUIREMENTS}")
        self.assertEqual(
            [
                OFFICIAL_WHEEL_URL,
                "paddleocr==3.7.0",
                "paddlex[ocr-core]==3.7.2",
            ],
            [line.strip() for line in GPU_REQUIREMENTS.read_text(encoding="utf-8").splitlines() if line.strip()],
        )
        for name, entrypoint in ROOT_ENTRYPOINTS.items():
            self.assertTrue(entrypoint.is_file(), f"missing {name} preview entrypoint: {entrypoint}")
            source = entrypoint.read_text(encoding="utf-8")
            self.assertIn(".runtime-build\\runtimes\\core\\python.exe", source)
            self.assertIn("-B -I", source)
            if name == "install":
                self.assertIn("ocr_component.py", source)
                self.assertIn("--mode gpu", source)
                self.assertIn("--model-root", source)
                self.assertNotIn("ocr_gpu_resource.py", source)
            else:
                self.assertIn("ocr_gpu_resource.py", source)

    def test_cleanup_entrypoints_default_to_deterministic_preview_without_writes(self) -> None:
        install_source = ROOT_ENTRYPOINTS["install"].read_text(encoding="utf-8")
        self.assertIn("ocr_component.py", install_source)
        self.assertIn("--mode gpu", install_source)
        self.assertIn("--model-root", install_source)
        for name in ("reset", "clean"):
            entrypoint = ROOT_ENTRYPOINTS[name]
            with self.subTest(entrypoint=name):
                self.assertTrue(entrypoint.is_file(), f"missing {name} entrypoint")
                before = [target.exists() for target in self._formal_gpu_targets(ROOT)]
                first = subprocess.run(
                    ["cmd.exe", "/d", "/c", str(entrypoint)], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False,
                )
                second = subprocess.run(
                    ["cmd.exe", "/d", "/c", str(entrypoint)], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False,
                )
                self.assertEqual(0, first.returncode, first.stdout + first.stderr)
                self.assertEqual(first.stdout, second.stdout)
                preview = json.loads(first.stdout)
                self.assertEqual("preview", preview["mode"])
                self.assertEqual(name, preview["action"])
                self.assertEqual(before, [target.exists() for target in self._formal_gpu_targets(ROOT)])

    def test_preview_lists_the_full_apply_contract_and_never_creates_gpu_paths(self) -> None:
        self.assertTrue(DRIVER.is_file(), "lifecycle driver must exist before it can be invoked")
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)
            first = self._run_driver(project, "install")
            second = self._run_driver(project, "install")
            self.assertEqual(0, first.returncode, first.stdout + first.stderr)
            self.assertEqual(first.stdout, second.stdout)
            preview = json.loads(first.stdout)
            self.assertEqual("preview", preview["mode"])
            self.assertEqual("install", preview["action"])
            self.assertEqual(OFFICIAL_WHEEL_URL, preview["wheelUrl"])
            self.assertEqual(
                [
                    ".runtime-build/ocr-gpu/v1/build-environment",
                    ".runtime-build/ocr-gpu/v1/downloads",
                    ".runtime-build/ocr-gpu/v1/staging",
                    ".runtime-build/runtimes/ocr-paddle-gpu",
                    ".runtime-build/manifests/runtimes/ocr-paddle-gpu.json",
                    ".runtime-build/manifests/requirements/ocr-paddle-gpu.lock",
                    "packaging/requirements/ocr-paddle-gpu.lock",
                ],
                preview["targets"],
            )
            self.assertEqual([], preview["writes"])
            self.assertTrue(any("offline install" in gate for gate in preview["applyGates"]))
            self.assertTrue(any("three-model CUDA probe" in gate for gate in preview["applyGates"]))
            self.assertFalse(any(target.exists() for target in self._formal_gpu_targets(project)))
            self.assertFalse((project / ".runtime-build" / "ocr-gpu" / "v1").exists())

    def test_reset_and_clean_reject_protected_or_reparse_targets(self) -> None:
        self.assertTrue(DRIVER.is_file(), "lifecycle driver must exist before its path guards can be checked")
        driver = _load_driver()
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)
            allowed = project / ".runtime-build" / "ocr-gpu" / "v1" / "downloads"
            self.assertEqual(allowed, driver.safe_gpu_cache_target(project, allowed))
            for protected in (
                project,
                project / ".runtime-build" / "runtimes" / "ocr-paddle",
                project / ".runtime-build" / "manifests" / "runtimes" / "ocr-paddle.json",
                project / "packaging" / "requirements" / "ocr-paddle.lock",
                project / "packaging" / "wheelhouse" / "ocr-paddle",
                project / "resource-library" / "ocr-models" / "ocr-ppocrv5-server-paddle-v1",
                project / "dataset",
                project / "output",
                project / "ocr_annotations" / "image.png.ocr.json",
            ):
                with self.subTest(protected=protected):
                    with self.assertRaises(ValueError):
                        driver.safe_gpu_cache_target(project, protected)
            outside = project / "outside"
            outside.mkdir()
            with self.assertRaises(ValueError):
                driver.safe_gpu_cache_target(project, outside)
            reparse = project / ".runtime-build" / "ocr-gpu" / "v1" / "staging"
            reparse.parent.mkdir(parents=True)
            created = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-Command",
                    f"New-Item -ItemType Junction -Path '{reparse}' -Target '{outside}' | Out-Null",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False,
            )
            self.assertEqual(0, created.returncode, created.stdout + created.stderr)
            with self.assertRaises(ValueError):
                driver.safe_gpu_cache_target(project, reparse)

    def test_isolated_apply_transaction_publishes_all_artifacts_or_none(self) -> None:
        self.assertTrue(DRIVER.is_file(), "lifecycle driver must exist before its apply transaction can be checked")
        driver = _load_driver()
        transaction = getattr(driver, "install_transaction", None)
        self.assertTrue(callable(transaction), "GPU lifecycle needs an explicit staged install transaction")
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)
            legacy_wheelhouse = project / "packaging" / "wheelhouse" / "ocr-paddle-gpu"
            legacy_wheelhouse.mkdir(parents=True)
            (legacy_wheelhouse / "legacy.whl").write_bytes(b"legacy")
            calls: list[str] = []

            def fake_downloader(paths: object) -> None:
                calls.append("download")
                getattr(paths, "downloads").mkdir(parents=True)
                (getattr(paths, "downloads") / "paddlepaddle_gpu-3.2.2-cp311-cp311-win_amd64.whl").write_bytes(b"wheel")

            def fake_builder(paths: object) -> None:
                calls.append("build")
                runtime = getattr(paths, "staging_runtime")
                runtime.mkdir(parents=True)
                (runtime / "python.exe").write_bytes(b"python")
                manifest = getattr(paths, "staging_manifest")
                manifest.parent.mkdir(parents=True)
                manifest.write_bytes(b"manifest")
                lock = getattr(paths, "staging_lock")
                lock.parent.mkdir(parents=True)
                lock.write_bytes(b"lock\n")
                mirror = getattr(paths, "staging_manifest_lock")
                mirror.parent.mkdir(parents=True)
                mirror.write_bytes(b"lock\n")
                wheelhouse = getattr(paths, "staging_wheelhouse")
                wheelhouse.mkdir(parents=True)
                (wheelhouse / "inventory.whl").write_bytes(b"wheel")

            def fake_probe(paths: object) -> None:
                calls.append("probe")
                self.assertTrue(getattr(paths, "staging_runtime").is_dir())
                self.assertEqual(b"lock\n", getattr(paths, "staging_manifest_lock").read_bytes())

            def prepare(paths: object) -> None:
                fake_downloader(paths)
                fake_builder(paths)

            result = transaction(project, prepare=prepare, probe=fake_probe)
            self.assertEqual(["download", "build", "probe"], calls)
            self.assertEqual("apply", result["mode"])
            self.assertEqual(
                [str(target.relative_to(project)).replace("\\", "/") for target in self._formal_gpu_targets(project)],
                result["writes"],
            )
            self.assertEqual(b"lock\n", (project / "packaging" / "requirements" / "ocr-paddle-gpu.lock").read_bytes())
            self.assertEqual(
                (project / "packaging" / "requirements" / "ocr-paddle-gpu.lock").read_bytes(),
                (project / ".runtime-build" / "manifests" / "requirements" / "ocr-paddle-gpu.lock").read_bytes(),
            )
            self.assertTrue(all(target.exists() for target in self._formal_gpu_targets(project)))
            self.assertFalse((project / "packaging" / "wheelhouse" / "ocr-paddle-gpu").exists())
            self.assertFalse(driver._install_paths(project).staging_wheelhouse.exists())

        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)

            def prepare(paths: object) -> None:
                runtime = getattr(paths, "staging_runtime")
                runtime.mkdir(parents=True)
                (runtime / "python.exe").write_bytes(b"python")
                manifest = getattr(paths, "staging_manifest")
                manifest.parent.mkdir(parents=True)
                manifest.write_bytes(b"manifest")
                lock = getattr(paths, "staging_lock")
                lock.parent.mkdir(parents=True)
                lock.write_bytes(b"lock\n")
                mirror = getattr(paths, "staging_manifest_lock")
                mirror.parent.mkdir(parents=True)
                mirror.write_bytes(b"lock\n")
                wheelhouse = getattr(paths, "staging_wheelhouse")
                wheelhouse.mkdir(parents=True)
                (wheelhouse / "inventory.whl").write_bytes(b"wheel")

            with self.assertRaisesRegex(RuntimeError, "probe"):
                transaction(project, prepare=prepare, probe=lambda paths: (_ for _ in ()).throw(RuntimeError("probe failed")))
            self.assertFalse(any(target.exists() for target in self._formal_gpu_targets(project)))

    def test_wheelhouse_cleanup_failure_publishes_no_formal_artifacts(self) -> None:
        driver = _load_driver()
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)

            with mock.patch("shutil.rmtree", side_effect=OSError("wheelhouse cleanup failed")):
                with self.assertRaisesRegex(OSError, "wheelhouse cleanup failed"):
                    driver.install_transaction(
                        project,
                        prepare=self._write_complete_gpu_staging,
                        probe=lambda paths: None,
                    )

            self.assertFalse(any(target.exists() for target in self._formal_gpu_targets(project)))
            self.assertTrue(driver._install_paths(project).staging_wheelhouse.is_dir())

    def test_existing_staging_is_archived_before_a_fresh_transaction(self) -> None:
        driver = _load_driver()
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)
            paths = driver._install_paths(project)
            paths.staging.mkdir(parents=True)
            sentinel = paths.staging / "partial-sentinel.txt"
            sentinel.write_text("preserve this failed attempt\n", encoding="ascii")
            paths.downloads.mkdir()
            (paths.downloads / "download-sentinel.whl").write_bytes(b"download")
            paths.build_environment.mkdir()
            (paths.build_environment / "environment-sentinel.txt").write_text("keep\n", encoding="ascii")
            seen_staging: list[Path] = []

            def prepare(current: object) -> None:
                staging = getattr(current, "staging")
                seen_staging.append(staging)
                self.assertTrue(staging.is_dir())
                self.assertEqual([], list(staging.iterdir()))
                self._write_complete_gpu_staging(current)

            try:
                result = driver.install_transaction(project, prepare=prepare, probe=lambda current: None)
            except Exception as exc:  # The RED must surface the transaction lifecycle cause, not a fixture error.
                self.fail(f"safe existing staging must be archived before prepare: {type(exc).__name__}: {exc}")

            self.assertEqual([paths.staging], seen_staging)
            self.assertTrue(all(target.exists() for target in self._formal_gpu_targets(project)))
            archive_root = paths.staging.parent / "failed-attempts"
            archives = [item for item in archive_root.iterdir() if item.is_dir()]
            self.assertEqual(1, len(archives))
            self.assertEqual("preserve this failed attempt\n", (archives[0] / sentinel.name).read_text(encoding="ascii"))
            self.assertEqual(
                f".runtime-build/ocr-gpu/v1/failed-attempts/{archives[0].name}",
                result["archivedStaging"],
            )
            self.assertEqual(b"download", (paths.downloads / "download-sentinel.whl").read_bytes())
            self.assertEqual("keep\n", (paths.build_environment / "environment-sentinel.txt").read_text(encoding="ascii"))

    def test_existing_staging_rejects_unsafe_paths_without_mutation(self) -> None:
        driver = _load_driver()
        cases = ("file", "reparse", "reparse-ancestor", "unsafe-archive-root")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary_name:
                project = Path(temporary_name)
                staging = project / ".runtime-build" / "ocr-gpu" / "v1" / "staging"
                expected_message = ""
                if case == "file":
                    staging.parent.mkdir(parents=True)
                    staging.write_text("not a directory\n", encoding="ascii")
                    preserved = staging
                    expected_message = "GPU staging is not a safe directory"
                elif case == "reparse":
                    target = project / "outside"
                    target.mkdir()
                    preserved = target / "partial-sentinel.txt"
                    preserved.write_text("keep\n", encoding="ascii")
                    staging.parent.mkdir(parents=True)
                    created = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-Command", f"New-Item -ItemType Junction -Path '{staging}' -Target '{target}' | Out-Null"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False,
                    )
                    self.assertEqual(0, created.returncode, created.stdout + created.stderr)
                    expected_message = "reparse point"
                elif case == "reparse-ancestor":
                    target = project / "outside"
                    preserved = target / "staging" / "partial-sentinel.txt"
                    preserved.parent.mkdir(parents=True)
                    preserved.write_text("keep\n", encoding="ascii")
                    staging.parent.parent.mkdir(parents=True)
                    created = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-Command", f"New-Item -ItemType Junction -Path '{staging.parent}' -Target '{target}' | Out-Null"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False,
                    )
                    self.assertEqual(0, created.returncode, created.stdout + created.stderr)
                    expected_message = "reparse point"
                else:
                    staging.mkdir(parents=True)
                    preserved = staging / "partial-sentinel.txt"
                    preserved.write_text("keep\n", encoding="ascii")
                    archive_root = staging.parent / "failed-attempts"
                    archive_root.write_text("not a directory\n", encoding="ascii")
                    expected_message = "GPU failed-attempt archive root is not a safe directory"

                with self.assertRaises(Exception) as caught:
                    driver.install_transaction(project, prepare=lambda current: None, probe=lambda current: None)
                self.assertIsInstance(caught.exception, ValueError)
                self.assertIn(expected_message, str(caught.exception))
                self.assertTrue(preserved.exists())
                self.assertFalse(any(target.exists() for target in self._formal_gpu_targets(project)))

    def test_existing_staging_archive_collision_and_formal_artifacts_fail_closed(self) -> None:
        driver = _load_driver()
        archive_staging = getattr(driver, "_archive_existing_staging", None)
        self.assertTrue(callable(archive_staging), "GPU lifecycle needs a fail-closed staging archive helper")
        if not callable(archive_staging):
            return
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)
            paths = driver._install_paths(project)
            paths.staging.mkdir(parents=True)
            sentinel = paths.staging / "partial-sentinel.txt"
            sentinel.write_text("keep\n", encoding="ascii")
            collision = paths.staging.parent / "failed-attempts" / "attempt-fixture"
            collision.mkdir(parents=True)

            with self.assertRaises(Exception) as caught:
                archive_staging(paths, attempt_name="attempt-fixture")
            self.assertIsInstance(caught.exception, ValueError)
            self.assertIn("archive target already exists", str(caught.exception))
            self.assertEqual("keep\n", sentinel.read_text(encoding="ascii"))

        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)
            paths = driver._install_paths(project)
            paths.staging.mkdir(parents=True)
            sentinel = paths.staging / "partial-sentinel.txt"
            sentinel.write_text("keep\n", encoding="ascii")
            paths.lock.parent.mkdir(parents=True)
            paths.lock.write_text("formal\n", encoding="ascii")

            with self.assertRaises(Exception) as caught:
                driver.install_transaction(project, prepare=lambda current: None, probe=lambda current: None)
            self.assertIsInstance(caught.exception, ValueError)
            self.assertIn("formal artifacts", str(caught.exception))
            self.assertEqual("keep\n", sentinel.read_text(encoding="ascii"))
            self.assertFalse((paths.staging.parent / "failed-attempts").exists())

    def test_existing_staging_prepare_or_probe_failure_retains_new_diagnostics(self) -> None:
        driver = _load_driver()
        for phase in ("prepare", "probe"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary_name:
                project = Path(temporary_name)
                paths = driver._install_paths(project)
                paths.staging.mkdir(parents=True)
                sentinel = paths.staging / "partial-sentinel.txt"
                sentinel.write_text("keep\n", encoding="ascii")

                def prepare(current: object) -> None:
                    if phase == "prepare":
                        raise RuntimeError("prepare failed")
                    self._write_complete_gpu_staging(current)

                def probe(current: object) -> None:
                    if phase == "probe":
                        raise RuntimeError("probe failed")

                with self.assertRaises(Exception) as caught:
                    driver.install_transaction(project, prepare=prepare, probe=probe)
                self.assertIsInstance(caught.exception, RuntimeError)
                self.assertIn(f"{phase} failed", str(caught.exception))
                self.assertTrue(paths.staging.is_dir())
                archives = list((paths.staging.parent / "failed-attempts").iterdir())
                self.assertEqual(1, len(archives))
                self.assertEqual("keep\n", (archives[0] / sentinel.name).read_text(encoding="ascii"))
                self.assertFalse(any(target.exists() for target in self._formal_gpu_targets(project)))

    def test_wheel_inventory_accepts_frozen_windows_x64_compatible_tags(self) -> None:
        driver = _load_driver()
        with tempfile.TemporaryDirectory() as temporary_name:
            wheelhouse = Path(temporary_name)
            # The platform-specific tags were read from the preserved failed
            # GPU download cache. CPython 3.11 accepts these older abi3 wheels.
            for name, version, tags in (
                ("paddlepaddle-gpu", "3.3.0", ("cp311-cp311-win_amd64",)),
                ("paddleocr", "3.7.0", ("py2-none-any", "py3-none-any")),
                ("paddlex", "3.7.2", ("py3-none-win_amd64",)),
                ("protobuf", "7.35.1", ("cp310-abi3-win_amd64",)),
                ("opencv-contrib-python", "4.10.0.84", ("cp37-abi3-win_amd64",)),
                ("hf-xet", "1.6.0", ("cp38-abi3-win_amd64",)),
                ("stable-abi", "1.0", ("cp311-abi3-win_amd64",)),
                ("pure-python", "1.0", ("py311-none-any",)),
            ):
                self._write_wheel(wheelhouse, name=name, version=version, tags=tags)

            inventory = driver._validate_wheel_inventory(wheelhouse)

            self.assertEqual(8, len(inventory))
            self.assertEqual(
                [
                    "hf-xet", "opencv-contrib-python", "paddleocr", "paddlepaddle-gpu",
                    "paddlex", "protobuf", "pure-python", "stable-abi",
                ],
                [record["name"] for record in inventory],
            )

    def test_wheel_inventory_rejects_incompatible_platform_and_python_tags(self) -> None:
        driver = _load_driver()
        for invalid_tag in (
            "py3-none-manylinux_2_17_x86_64",
            "py3-none-macosx_11_0_x86_64",
            "py3-none-win32",
            "cp312-cp312-win_amd64",
            "cp312-abi3-win_amd64",
        ):
            with self.subTest(tag=invalid_tag), tempfile.TemporaryDirectory() as temporary_name:
                wheelhouse = Path(temporary_name)
                self._write_required_wheels(wheelhouse, paddlex_tags=(invalid_tag,))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    driver._validate_wheel_inventory(wheelhouse)

    def test_wheel_inventory_rejects_duplicate_normalized_package(self) -> None:
        driver = _load_driver()
        with tempfile.TemporaryDirectory() as temporary_name:
            wheelhouse = Path(temporary_name)
            self._write_required_wheels(wheelhouse)
            self._write_wheel(
                wheelhouse, name="duplicate-package", version="1.0", tags=("py3-none-any",), filename_stem="duplicate-one",
            )
            self._write_wheel(
                wheelhouse, name="duplicate_package", version="1.0", tags=("py3-none-any",), filename_stem="duplicate-two",
            )
            with self.assertRaisesRegex(ValueError, "incompatible or duplicated"):
                driver._validate_wheel_inventory(wheelhouse)

    def test_wheel_inventory_rejects_invalid_paddle_dependencies(self) -> None:
        driver = _load_driver()
        invalid_cases = (
            ("CPU paddlepaddle", (("paddlepaddle", "3.2.2"),)),
            ("wrong Paddle version", (("paddlepaddle-gpu", "3.2.2"),)),
            ("wrong PaddleOCR version", (("paddleocr", "3.7.1"),)),
            ("wrong PaddleX version", (("paddlex", "3.7.1"),)),
        )
        for label, replacements in invalid_cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary_name:
                wheelhouse = Path(temporary_name)
                self._write_required_wheels(wheelhouse, replacements=replacements)
                with self.assertRaisesRegex(ValueError, "incompatible|invalid Paddle"):
                    driver._validate_wheel_inventory(wheelhouse)

    def test_validated_existing_downloads_are_reused_without_download(self) -> None:
        driver = _load_driver()
        reuse = getattr(driver, "_download_or_reuse_inventory", None)
        self.assertTrue(callable(reuse), "GPU lifecycle needs validated existing-download reuse")
        with tempfile.TemporaryDirectory() as temporary_name:
            paths = driver._install_paths(Path(temporary_name))
            paths.downloads.mkdir(parents=True)
            self._write_required_wheels(paths.downloads)

            inventory = reuse(paths, builder=Path("unused-builder.exe"), source=Path("unused-requirements.in"))

            self.assertEqual(3, len(inventory))
            self.assertEqual(
                ["paddleocr", "paddlepaddle-gpu", "paddlex"],
                [record["name"] for record in inventory],
            )

    def test_invalid_existing_downloads_fail_closed_without_download(self) -> None:
        driver = _load_driver()
        reuse = getattr(driver, "_download_or_reuse_inventory", None)
        self.assertTrue(callable(reuse), "GPU lifecycle needs validated existing-download reuse")
        with tempfile.TemporaryDirectory() as temporary_name:
            paths = driver._install_paths(Path(temporary_name))
            paths.downloads.mkdir(parents=True)
            self._write_required_wheels(paths.downloads, paddlex_tags=("py3-none-manylinux_2_17_x86_64",))

            with self.assertRaisesRegex(ValueError, "incompatible"):
                reuse(paths, builder=Path("unused-builder.exe"), source=Path("unused-requirements.in"))

    def test_probe_rebuilds_unicode_sample_paths_without_trusting_lossy_child_paths(self) -> None:
        driver = _load_driver()
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name) / "GPU unicode 验证"
            project.mkdir()
            paths = driver._install_paths(project)
            paths.staging_runtime.mkdir(parents=True)
            paths.staging_manifest.parent.mkdir(parents=True)
            paths.staging_manifest.write_bytes(b"manifest\n")
            manifest_fingerprint = hashlib.sha256(paths.staging_manifest.read_bytes()).hexdigest()
            sample_root = paths.staging / "probe-images"
            expected_names = ("zh", "ja", "en")
            original_run = driver._run
            original_subprocess_run = driver.subprocess.run
            worker_requests: list[dict[str, object]] = []

            def fake_run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                source = command[-1]
                if "load_ocr_resource" in source:
                    return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
                if "ImageFont.truetype" not in source:
                    self.fail(f"unexpected staged command: {command}")
                sample_root.mkdir()
                records = []
                for name in expected_names:
                    image = sample_root / f"{name}.png"
                    image.write_bytes(name.encode("ascii"))
                    records.append({"name": name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
                if "ensure_ascii=True" in source:
                    report = {"fontSha256": "a" * 64, "samples": records}
                    stdout = json.dumps(report, ensure_ascii=True)
                else:
                    legacy = {
                        "fontSha256": "a" * 64,
                        "samples": [
                            {**record, "path": str(sample_root / f"{record['name']}.png")}
                            for record in records
                        ],
                    }
                    stdout = json.dumps(legacy, ensure_ascii=False).encode("gbk").decode("utf-8", errors="replace")
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            def fake_subprocess_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
                frames = [json.loads(line) for line in bytes(kwargs["input"]).splitlines()]
                worker_requests.append(frames[1]["payload"])
                evidence = {
                    "requestedDevice": "cuda",
                    "observedDevice": "cuda",
                    "runtimeId": "ocr-paddle-gpu",
                    "runtimeFingerprint": manifest_fingerprint,
                    "paddleVersion": "3.3.0",
                    "compiledWithCuda": True,
                    "cudaVersion": "13.3",
                    "gpuName": "fixture GPU",
                }
                outcomes = [
                    {"status": "success", "items": [{"text": name}]}
                    for name in expected_names
                ]
                response = b"".join(
                    json.dumps({"payload": payload}, separators=(",", ":")).encode("utf-8") + b"\n"
                    for payload in (evidence, {"items": outcomes}, {})
                )
                return subprocess.CompletedProcess(command, 0, stdout=response, stderr=b"")

            driver._run = fake_run
            driver.subprocess.run = fake_subprocess_run
            try:
                result = driver._probe_real_gpu_runtime(paths)
            finally:
                driver._run = original_run
                driver.subprocess.run = original_subprocess_run

            self.assertEqual(list(expected_names), [sample["name"] for sample in result["samples"]["samples"]])
            self.assertTrue(worker_requests)
            self.assertEqual(
                [str(sample_root / f"{name}.png") for name in expected_names],
                [item["imagePath"] for item in worker_requests[0]["items"]],
            )
            self.assertFalse(any("\ufffd" in item["imagePath"] for item in worker_requests[0]["items"]))

    def test_probe_sample_metadata_rejects_untrusted_names_and_hashes(self) -> None:
        driver = _load_driver()
        validate = getattr(driver, "_validate_probe_samples", None)
        self.assertTrue(callable(validate), "GPU probe needs fail-closed sample metadata validation")
        if not callable(validate):
            return
        with tempfile.TemporaryDirectory() as temporary_name:
            sample_root = Path(temporary_name) / "样图"
            sample_root.mkdir()
            valid_samples = []
            for name in ("zh", "ja", "en"):
                image = sample_root / f"{name}.png"
                image.write_bytes(name.encode("ascii"))
                valid_samples.append({"name": name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
            self.assertEqual(
                ["zh", "ja", "en"],
                [name for name, _, _ in validate(sample_root, {"samples": valid_samples})],
            )
            invalid_reports = {
                "missing": {"samples": valid_samples[:2]},
                "duplicate": {"samples": [valid_samples[0], valid_samples[0], valid_samples[2]]},
                "unknown": {"samples": [valid_samples[0], {"name": "other", "sha256": valid_samples[1]["sha256"]}, valid_samples[2]]},
                "path traversal": {"samples": [valid_samples[0], {"name": "../ja", "sha256": valid_samples[1]["sha256"]}, valid_samples[2]]},
                "malformed hash": {"samples": [valid_samples[0], {"name": "ja", "sha256": "A" * 64}, valid_samples[2]]},
                "hash mismatch": {"samples": [valid_samples[0], {"name": "ja", "sha256": "0" * 64}, valid_samples[2]]},
            }
            for label, report in invalid_reports.items():
                with self.subTest(case=label), self.assertRaises(ValueError):
                    validate(sample_root, report)

    def test_gpu_prepare_keeps_setuptools_when_assembling_the_runtime(self) -> None:
        driver = _load_driver()
        with tempfile.TemporaryDirectory() as temporary_name:
            project = Path(temporary_name)
            requirements = project / "packaging" / "requirements"
            requirements.mkdir(parents=True)
            (requirements / "ocr-paddle-gpu.in").write_text("\n".join(driver.DIRECT_REQUIREMENTS) + "\n", encoding="ascii")
            toolchain = project / ".toolchains" / "Python-3.11.15" / "PCbuild" / "amd64" / "python.exe"
            toolchain.parent.mkdir(parents=True)
            toolchain.write_bytes(b"toolchain")
            scripts = project / "packaging" / "scripts"
            scripts.mkdir(parents=True)
            for name in ("resolve_wheels.py", "build_cpython311_runtime.ps1", "assemble_runtime.ps1", "generate_runtime_manifests.py"):
                (scripts / name).write_text("fixture\n", encoding="ascii")
            paths = driver._install_paths(project)
            calls: list[list[str]] = []

            def fake_run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                del cwd
                calls.append(command)
                if "venv" in command:
                    builder = paths.build_environment / "Scripts" / "python.exe"
                    builder.parent.mkdir(parents=True)
                    builder.write_bytes(b"builder")
                elif str(scripts / "generate_runtime_manifests.py") in command:
                    paths.staging_manifest_lock.parent.mkdir(parents=True)
                    paths.staging_manifest_lock.write_text("lock\n", encoding="ascii")
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(driver, "_run", side_effect=fake_run),
                mock.patch.object(driver, "_download_or_reuse_inventory"),
                mock.patch.object(driver, "_validate_wheel_inventory", return_value=[]),
            ):
                driver._prepare_real_install(paths)

            assemble = next(command for command in calls if str(scripts / "assemble_runtime.ps1") in command)
            self.assertIn("-KeepSetuptools", assemble)

    @staticmethod
    def _write_complete_gpu_staging(paths: object) -> None:
        runtime = getattr(paths, "staging_runtime")
        runtime.mkdir(parents=True)
        (runtime / "python.exe").write_bytes(b"python")
        manifest = getattr(paths, "staging_manifest")
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(b"manifest")
        lock = getattr(paths, "staging_lock")
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"lock\n")
        mirror = getattr(paths, "staging_manifest_lock")
        mirror.parent.mkdir(parents=True)
        mirror.write_bytes(b"lock\n")
        wheelhouse = getattr(paths, "staging_wheelhouse")
        wheelhouse.mkdir(parents=True)
        (wheelhouse / "inventory.whl").write_bytes(b"wheel")

    @staticmethod
    def _write_required_wheels(
        wheelhouse: Path,
        *,
        paddlex_tags: tuple[str, ...] = ("py3-none-any",),
        replacements: tuple[tuple[str, str], ...] = (),
    ) -> None:
        versions = {
            "paddlepaddle-gpu": "3.3.0",
            "paddleocr": "3.7.0",
            "paddlex": "3.7.2",
        }
        for name, version in replacements:
            versions[name] = version
        for name, version in versions.items():
            tags = paddlex_tags if name == "paddlex" else ("py3-none-any",)
            OcrGpuResourceScriptTests._write_wheel(wheelhouse, name=name, version=version, tags=tags)

    @staticmethod
    def _write_wheel(
        wheelhouse: Path,
        *,
        name: str,
        version: str,
        tags: tuple[str, ...],
        filename_stem: str | None = None,
    ) -> None:
        filename = f"{filename_stem or name.replace('-', '_')}-{version}-fixture.whl"
        dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
        with zipfile.ZipFile(wheelhouse / filename, "w") as archive:
            archive.writestr(f"{dist_info}/METADATA", f"Name: {name}\nVersion: {version}\n")
            archive.writestr(
                f"{dist_info}/WHEEL",
                "Wheel-Version: 1.0\n" + "".join(f"Tag: {tag}\n" for tag in tags),
            )

    @staticmethod
    def _formal_gpu_targets(project: Path) -> tuple[Path, ...]:
        return (
            project / ".runtime-build" / "runtimes" / "ocr-paddle-gpu",
            project / ".runtime-build" / "manifests" / "runtimes" / "ocr-paddle-gpu.json",
            project / ".runtime-build" / "manifests" / "requirements" / "ocr-paddle-gpu.lock",
            project / "packaging" / "requirements" / "ocr-paddle-gpu.lock",
        )


if __name__ == "__main__":
    unittest.main()
