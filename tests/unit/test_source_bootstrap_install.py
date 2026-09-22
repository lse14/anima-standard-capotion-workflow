from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
import importlib.util
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
INSTALLER_ROOT = ROOT / "packaging" / "installer"
SOURCE_COMMIT = "2e85063591c266a14e2111da8ec6a3602139c61e"
OCR_MODEL_ARCHIVE_FILENAMES = (
    "PP-OCRv5_server_det_infer.tar",
    "PP-OCRv5_server_rec_infer.tar",
    "PP-LCNet_x1_0_textline_ori_infer.tar",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_fixture_runtime_manifest(item, layout) -> str:
    lock_payload = f"{item.runtime_id}:{item.variant.name}\n".encode("ascii")
    lock = layout.runtime_root / "manifests" / "requirements" / f"{item.runtime_id}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(lock_payload)
    target = layout.runtime_root / "manifests" / "runtimes" / f"{item.runtime_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "runtime": {
                    "runtimeId": item.runtime_id,
                    "dependencyLockSha256": _sha256(lock_payload),
                }
            }
        ),
        encoding="utf-8",
    )
    return str(target.relative_to(layout.project_root)).replace("/", "\\")


def _artifact(artifact_id: str, relative_path: str) -> dict[str, object]:
    payload = artifact_id.encode("ascii")
    return {
        "id": artifact_id,
        "url": f"https://downloads.example.test/anima/{artifact_id}",
        "allowedHosts": ["downloads.example.test"],
        "sizeBytes": len(payload),
        "sha256": _sha256(payload),
        "relativePath": relative_path,
    }


def _source_tree_artifact(
    artifact_id: str,
    source_relative_path: str,
    relative_path: str,
    payload: bytes,
) -> dict[str, object]:
    return {
        "id": artifact_id,
        "delivery": "source-tree",
        "sourceRelativePath": source_relative_path,
        "sizeBytes": len(payload),
        "sha256": _sha256(payload),
        "relativePath": relative_path,
    }


def _variant(artifact_id: str, relative_path: str, *, peak_bytes: int = 4096) -> dict[str, object]:
    return {
        "artifacts": [_artifact(artifact_id, relative_path)],
        "peakBytes": peak_bytes,
        "probe": "fixture",
    }


def fixture_manifest() -> dict[str, object]:
    runtime_components = [
        ("core", "core"),
        ("caption-e621", "caption-e621"),
        ("classify-e621", "classify-e621"),
        ("replace-e621", "replace-e621"),
        ("nl", "nl"),
        ("policy", "policy"),
        ("export", "export"),
        ("token-budget", "token-budget"),
        ("ocr-cpu", "ocr-paddle"),
    ]
    components: list[dict[str, object]] = []
    for component_id, runtime_id in runtime_components:
        variants: dict[str, object] = {
            "cpu": _variant(f"{component_id}-cpu-wheel", f"wheels/{component_id}-cpu.whl")
        }
        if component_id in {"caption-e621", "policy"}:
            variants["cuda"] = _variant(f"{component_id}-cuda-wheel", f"wheels/{component_id}-cuda.whl")
        components.append(
            {
                "componentId": component_id,
                "kind": "runtime",
                "required": component_id not in {"ocr-cpu", "policy"},
                "targetRelativePath": f".runtime-build/runtimes/{runtime_id}",
                "variants": variants,
            }
        )
    components.append(
        {
            "componentId": "ocr-gpu",
            "kind": "runtime",
            "required": False,
            "targetRelativePath": ".runtime-build/runtimes/ocr-paddle-gpu",
            "variants": {"cuda": _variant("ocr-gpu-cuda-wheel", "wheels/ocr-gpu-cuda.whl")},
        }
    )
    return {
        "schemaVersion": 1,
        "releaseVersion": "fixture-v1",
        "sourceCommit": SOURCE_COMMIT,
        "allowedHosts": ["downloads.example.test"],
        "bootstrap": {
            "artifact": _artifact("cpython311-base", "bootstrap/cpython311-base.zip"),
            "entryRelativePath": "python.exe",
            "peakBytes": 4096,
        },
        "components": components,
        "cleanup": {
            "successRelativePaths": [
                ".runtime-build/bootstrap",
                ".runtime-build/cache",
                ".runtime-build/staging",
            ]
        },
    }


def _modules():
    sys.path.insert(0, str(INSTALLER_ROOT))
    try:
        sys.modules.pop("assemble", None)
        sys.modules.pop("manifest", None)
        return importlib.import_module("assemble"), importlib.import_module("manifest")
    finally:
        sys.path.pop(0)


def _install_module():
    sys.path.insert(0, str(INSTALLER_ROOT))
    try:
        sys.modules.pop("install", None)
        return importlib.import_module("install")
    finally:
        sys.path.pop(0)


class IsolatedInstallerEntryTests(unittest.TestCase):
    def test_installer_help_loads_local_modules_in_isolated_mode(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", str(INSTALLER_ROOT / "install.py"), "--help"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertIn("--bootstrap-runtime", completed.stdout)

    def test_installer_main_reports_manual_download_details_without_a_traceback(self) -> None:
        install_module = _install_module()
        artifact = SimpleNamespace(
            artifact_id="fixture",
            url="https://downloads.example.test/fixture.bin",
            allowed_hosts=("downloads.example.test",),
            size_bytes=7,
            sha256="a" * 64,
            relative_path="fixture.bin",
        )
        error = install_module.ManualDownloadRequired(artifact, "fixture download failed")
        stderr = StringIO()

        with (
            mock.patch.object(sys, "argv", [
                "install.py", "--project-root", ".", "--manifest", "manifest.json",
                "--manifest-sha256", "a" * 64, "--accelerator", "cpu",
                "--bootstrap-runtime", ".",
            ]),
            mock.patch.object(Path, "read_bytes", return_value=b"manifest"),
            mock.patch.object(install_module, "sha256_bytes", return_value="a" * 64),
            mock.patch.object(install_module, "load_manifest_path", return_value=object()),
            mock.patch.object(install_module, "install_project", side_effect=error),
            mock.patch.object(install_module, "_bootstrap_runtime_from_argument", return_value=Path(".")),
            redirect_stderr(stderr),
        ):
            exit_code = install_module.main()

        self.assertEqual(1, exit_code)
        self.assertIn("Official URL: https://downloads.example.test/fixture.bin", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_installer_main_preserves_its_running_bootstrap_for_outer_cleanup(self) -> None:
        install_module = _install_module()
        installer = mock.Mock(return_value=SimpleNamespace(
            messages=(),
            installed_component_ids=(),
            skipped_component_ids=(),
            state_path=Path("install-state.json"),
        ))

        with (
            mock.patch.object(sys, "argv", [
                "install.py", "--project-root", ".", "--manifest", "manifest.json",
                "--manifest-sha256", "a" * 64, "--accelerator", "cpu",
                "--bootstrap-runtime", ".",
            ]),
            mock.patch.object(Path, "read_bytes", return_value=b"manifest"),
            mock.patch.object(install_module, "sha256_bytes", return_value="a" * 64),
            mock.patch.object(install_module, "load_manifest_path", return_value=object()),
            mock.patch.object(install_module, "install_project", installer),
            mock.patch.object(install_module, "_bootstrap_runtime_from_argument", return_value=Path(".")),
            mock.patch("builtins.print"),
        ):
            exit_code = install_module.main()

        self.assertEqual(0, exit_code)
        self.assertIs(True, installer.call_args.kwargs.get("preserve_bootstrap_on_success"))
        self.assertTrue(callable(installer.call_args.kwargs.get("progress")))


def _probes_module():
    sys.path.insert(0, str(INSTALLER_ROOT))
    try:
        sys.modules.pop("probes", None)
        return importlib.import_module("probes")
    finally:
        sys.path.pop(0)


def _runtime_manifest_generator():
    path = ROOT / "packaging" / "scripts" / "generate_runtime_manifests.py"
    spec = importlib.util.spec_from_file_location("runtime_manifest_generator_for_source_bootstrap_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load runtime manifest generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_install_manifest(*, include_delayed_ocr_models: bool = False) -> dict[str, object]:
    value = fixture_manifest()
    core = next(component for component in value["components"] if component["componentId"] == "core")
    resource = {
        "componentId": "fixture-resource",
        "kind": "resource",
        "required": True,
        "targetRelativePath": "resource-library/fixture-resource",
        "variants": {"shared": _variant("fixture-resource-json", "resource.json")},
    }
    value["components"] = [core, resource]
    if include_delayed_ocr_models:
        value["components"].append(
            {
                "componentId": "ocr-models",
                "kind": "resource",
                "required": False,
                "targetRelativePath": "resource-library/ocr-models/ocr-ppocrv5-server-paddle-v1",
                "variants": {"shared": _variant("ocr-models-resource", "models/ocr-models.json")},
            }
        )
    return value


def fallback_fixture_manifest(*, include_probe_companions: bool = False) -> dict[str, object]:
    value = fixture_manifest()
    selected = {"caption-e621", "policy", "ocr-cpu", "ocr-gpu"}
    value["components"] = [
        component for component in value["components"]
        if component["componentId"] in selected
    ]
    if include_probe_companions:
        value["components"].extend(
            [
                {
                    "componentId": "e621-tagger",
                    "kind": "resource",
                    "required": True,
                    "targetRelativePath": "resource-library/e621-tagger",
                    "variants": {"shared": _variant("e621-tagger-resource", "models/tagger.json")},
                },
                {
                    "componentId": "quality-stack",
                    "kind": "resource",
                    "required": False,
                    "targetRelativePath": "resource-library/quality-stack",
                    "variants": {"shared": _variant("quality-stack-resource", "models/quality.json")},
                },
            ]
        )
    return value


class SourceBootstrapInstallTests(unittest.TestCase):
    def _prepare_fixture_install(
        self,
        root: Path,
        *,
        include_delayed_ocr_models: bool = False,
    ):
        install_module = _install_module()
        _, manifest_module = _modules()
        manifest = manifest_module.load_manifest(
            fixture_install_manifest(include_delayed_ocr_models=include_delayed_ocr_models)
        )
        base = root / "bootstrap-base"
        (base / "Lib").mkdir(parents=True)
        for filename in ("python.exe", "python311.dll", "python311._pth"):
            (base / filename).write_bytes(filename.encode("ascii"))
        source = root / "core" / "src" / "anima_core"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("VALUE = 'core'\n", encoding="ascii")
        (source / "__main__.py").write_text("VALUE = 'main'\n", encoding="ascii")
        shared_source = root / "shared" / "anima_caption_format" / "anima_caption_format"
        shared_source.mkdir(parents=True)
        (shared_source / "__init__.py").write_text("VALUE = 'format'\n", encoding="ascii")
        wheel = root / "core.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("fixture_pkg/__init__.py", "VALUE = 1\n")
        resource = root / "resource.json"
        resource.write_text('{"fixture":true}\n', encoding="utf-8")
        paths = {"core-cpu-wheel": wheel, "fixture-resource-json": resource}

        def fetch(artifact):
            return paths[artifact.artifact_id]

        def probe(item, target):
            return target.is_dir() and item.component.component_id in {"core", "fixture-resource"}

        return install_module, manifest, base, fetch, probe, _write_fixture_runtime_manifest

    def test_representative_probe_rejects_import_only_and_wrong_accelerator_evidence(self) -> None:
        probes = _probes_module()
        calls: list[tuple[object, dict[str, object]]] = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(
                command,
                0,
                '{"kind":"caption","provider":"CPUExecutionProvider","tags":["alpha","beta"]}\n',
                "",
            )

        evidence = probes.run_json_probe(
            Path("C:/fixture/python.exe"),
            "print('fixture')",
            (),
            cwd=ROOT,
            environment={"HTTP_PROXY": "http://proxy.invalid", "UNCHANGED": "value"},
            runner=runner,
        )

        self.assertTrue(probes.validate_evidence("caption-e621", "cpu", evidence))
        command, kwargs = calls[0]
        self.assertIn("socket.socket.connect", command[4])
        self.assertNotIn("HTTP_PROXY", kwargs["env"])
        self.assertEqual("1", kwargs["env"]["HF_HUB_OFFLINE"])
        self.assertEqual("1", kwargs["env"]["TRANSFORMERS_OFFLINE"])
        self.assertEqual("value", kwargs["env"]["UNCHANGED"])

        with self.assertRaisesRegex(probes.ProbeError, "import-only"):
            probes.validate_evidence("caption-e621", "cpu", {"kind": "import", "module": "onnxruntime"})
        with self.assertRaisesRegex(probes.ProbeError, "CPU"):
            probes.validate_evidence(
                "caption-e621",
                "cpu",
                {"kind": "caption", "provider": "CUDAExecutionProvider", "tags": ["alpha"]},
            )
        with self.assertRaisesRegex(probes.ProbeError, "GPU"):
            probes.validate_evidence(
                "ocr-gpu",
                "cuda",
                {"kind": "ocr", "device": "cpu", "resultCount": 1, "texts": []},
            )

    def test_caption_probe_uses_the_resource_adapter(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            runtime = root / "runtimes" / "caption-e621"
            resource = root / "resource-library" / "tagging-models" / "e621"
            runtime.mkdir(parents=True)
            resource.mkdir(parents=True)
            (resource / "resource.json").write_text(
                json.dumps({"fingerprint": "a" * 64}),
                encoding="utf-8",
            )
            scripts: list[str] = []

            def runner(command, **_kwargs):
                scripts.append(command[4])
                return subprocess.CompletedProcess(
                    command,
                    0,
                    '{"kind":"caption","provider":"CPUExecutionProvider","tags":["alpha"]}\n',
                    "",
                )

            probes._probe_caption(runtime, resource, "cpu", runner=runner)

            self.assertIn("from anima_caption_worker.model import create_tagger_adapter", scripts[0])
            self.assertIn("model = create_tagger_adapter(resource)", scripts[0])
            self.assertNotIn("CaptionModel(resource.entrypoints)", scripts[0])

    def test_caption_cuda_runtime_manifest_exposes_packaged_nvidia_dll_directories(self) -> None:
        generator = _runtime_manifest_generator()

        specifications = generator.runtime_specs(
            lock_names={"caption-e621": "caption-e621-cuda"}
        )

        self.assertEqual(
            (
                "Lib/site-packages/onnxruntime/capi",
                "Lib/site-packages/nvidia/cublas/bin",
                "Lib/site-packages/nvidia/cuda_runtime/bin",
                "Lib/site-packages/nvidia/cudnn/bin",
                "Lib/site-packages/nvidia/cufft/bin",
                "Lib/site-packages/nvidia/curand/bin",
                "Lib/site-packages/nvidia/cusolver/bin",
                "Lib/site-packages/nvidia/cusparse/bin",
                "Lib/site-packages/nvidia/nvjitlink/bin",
            ),
            specifications["caption-e621"][3],
        )

    def test_caption_cuda_lock_includes_onnxruntime_gpu_dependencies(self) -> None:
        lock = (ROOT / "packaging" / "requirements" / "caption-e621-cuda.lock").read_text(
            encoding="utf-8"
        )

        self.assertIn("nvidia-cublas-cu12==12.9.0.13", lock)
        self.assertIn("nvidia-cuda-runtime-cu12==12.9.37", lock)
        self.assertIn("nvidia-cudnn-cu12==9.9.0.52", lock)
        self.assertIn("nvidia-cufft-cu12==11.4.0.6", lock)
        self.assertIn("nvidia-curand-cu12==10.3.10.19", lock)
        self.assertIn("nvidia-cusolver-cu12==11.7.4.40", lock)
        self.assertIn("nvidia-cusparse-cu12==12.5.9.5", lock)
        self.assertIn("nvidia-nvjitlink-cu12==12.9.86", lock)

    def test_source_bootstrap_assembles_all_frozen_nl_prompt_resources(self) -> None:
        install_module = _install_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            install_root = Path(temporary_name)

            install_module._assemble_nl_prompt_resources(ROOT, install_root)

            resource_ids = (
                "nl-default-prompt-v1",
                "nl-default-prompt-v2",
                "nl-default-prompt-v3",
                "nl-default-prompt-v4-base",
                "nl-default-prompt-v4-general",
                "nl-default-prompt-v4-style",
                "nl-default-prompt-v4-character",
                "nl-default-prompt-v4-short",
                "nl-default-prompt-v4-medium",
                "nl-default-prompt-v4-long",
            )
            for resource_id in resource_ids:
                with self.subTest(resource_id=resource_id):
                    self.assertTrue(
                        (install_root / "manifests" / "resources" / f"{resource_id}.json").is_file()
                    )

    def test_default_probe_rejects_gpu_ocr_result_outside_cpu_tolerance(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            targets: dict[str, Path] = {}
            for component_id, runtime_id, variant in (
                ("core", "core", "cpu"),
                ("caption-e621", "caption-e621", "cpu"),
                ("policy", "policy", "cpu"),
                ("token-budget", "token-budget", "cpu"),
                ("ocr-cpu", "ocr-paddle", "cpu"),
                ("ocr-gpu", "ocr-paddle-gpu", "cuda"),
            ):
                target = root / "runtimes" / runtime_id
                target.mkdir(parents=True)
                targets[component_id] = target
            for component_id in ("e621-tagger", "quality-stack", "qwen3-tokenizer", "ocr-models", "e621-indexes"):
                target = root / "resource-library" / component_id
                target.mkdir(parents=True)
                manifest: dict[str, object] = {"fingerprint": "a" * 64}
                if component_id == "qwen3-tokenizer":
                    manifest.update({"resourceId": "tokenizer-qwen3-0.6b-anima-v1", "contextLimit": 32768})
                (target / "resource.json").write_text(json.dumps(manifest), encoding="utf-8")
                targets[component_id] = target
            components = tuple(
                SimpleNamespace(
                    component=SimpleNamespace(component_id=component_id),
                    variant=SimpleNamespace(name=variant),
                )
                for component_id, variant in (
                    ("core", "cpu"),
                    ("caption-e621", "cpu"),
                    ("e621-tagger", "shared"),
                    ("policy", "cpu"),
                    ("quality-stack", "shared"),
                    ("token-budget", "cpu"),
                    ("qwen3-tokenizer", "shared"),
                    ("ocr-cpu", "cpu"),
                    ("ocr-gpu", "cuda"),
                    ("ocr-models", "shared"),
                    ("e621-indexes", "shared"),
                )
            )
            observed_environments: list[dict[str, str]] = []

            def runner(command, **kwargs):
                observed_environments.append(kwargs["env"])
                script = command[4]
                self.assertIn("socket.socket.connect", script)
                if 'runpy.run_module("anima_core"' in script:
                    output = "anima-core-runtime-ok\n"
                elif "anima_caption_worker.model" in script:
                    output = '{"kind":"caption","provider":"CPUExecutionProvider","tags":["alpha"]}\n'
                elif "Lse14Scorer" in script:
                    output = '{"kind":"quality","device":"cpu","loaded":["clip","fusion","jtp3","waifu"],"score":1.0}\n'
                elif "apply_policy" in script:
                    output = '{"kind":"policy","check":"ok","device":"cpu"}\n'
                elif "tokenizer_count_many" in script:
                    output = '{"kind":"tokenizer","counts":[3,4]}\n'
                elif "create_paddle_engine" in script:
                    output = (
                        '{"kind":"ocr","device":"cpu","resultCount":1,"texts":["offline"]}\n'
                        if command[-1] == "cpu"
                        else '{"kind":"ocr","device":"gpu:0","resultCount":1,"texts":["different"]}\n'
                    )
                elif "ResourcePackage.load" in script or "ResourceCatalog" in script:
                    output = '{"kind":"indexes","resourceCount":1}\n'
                else:
                    self.fail(f"unexpected probe script: {script}")
                return subprocess.CompletedProcess(command, 0, output, "")

            results = probes.run_offline_probes(components, component_targets=targets, runner=runner)

            self.assertTrue(results["ocr-cpu"])
            self.assertFalse(results["ocr-gpu"])
            self.assertTrue(all("HTTP_PROXY" not in environment for environment in observed_environments))
            self.assertTrue(all(environment["HF_HUB_OFFLINE"] == "1" for environment in observed_environments))

    def test_e621_replacement_indexes_are_probed_against_their_own_target(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            core = root / "runtimes" / "core"
            live_library = root / "live" / "resource-library"
            stage_library = root / "stage" / "resource-library"
            indexes = live_library / "classification-indexes" / "e621-classify"
            replacement = stage_library / "replacement-indexes" / "e621-replace"
            for path in (core, indexes, replacement):
                path.mkdir(parents=True)
            (core / "python.exe").write_bytes(b"python")
            probed_packages: list[str] = []

            def runner(command, **_kwargs):
                script = command[4]
                if "ResourcePackage.load" not in script:
                    self.fail(f"unexpected probe script: {script}")
                package_root = command[5]
                probed_packages.append(package_root)
                if Path(package_root) == replacement:
                    return subprocess.CompletedProcess(command, 1, "", "replacement package invalid")
                return subprocess.CompletedProcess(
                    command, 0, '{"kind":"indexes","resourceCount":1}\n', ""
                )

            components = (
                SimpleNamespace(
                    component=SimpleNamespace(component_id="core"),
                    variant=SimpleNamespace(name="cpu"),
                ),
                SimpleNamespace(
                    component=SimpleNamespace(component_id="e621-indexes"),
                    variant=SimpleNamespace(name="shared"),
                ),
                SimpleNamespace(
                    component=SimpleNamespace(component_id="e621-replacement-indexes"),
                    variant=SimpleNamespace(name="shared"),
                ),
            )

            def core_and_index_runner(command, **kwargs):
                script = command[4]
                if 'runpy.run_module("anima_core"' in script:
                    return subprocess.CompletedProcess(command, 0, "anima-core-runtime-ok\n", "")
                return runner(command, **kwargs)

            results = probes.run_offline_probes(
                components,
                component_targets={
                    "core": core,
                    "e621-indexes": indexes,
                    "e621-replacement-indexes": replacement,
                },
                runner=core_and_index_runner,
            )

            self.assertTrue(results["e621-indexes"])
            self.assertFalse(results["e621-replacement-indexes"])
            self.assertEqual([str(indexes), str(replacement)], probed_packages)

    def test_source_worker_runtimes_receive_functional_probes(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            component_ids = ("classify-e621", "replace-e621", "nl", "export")
            targets = {}
            for component_id in component_ids:
                target = root / "runtimes" / component_id
                target.mkdir(parents=True)
                targets[component_id] = target
            components = tuple(
                SimpleNamespace(component=SimpleNamespace(component_id=component_id), variant=SimpleNamespace(name="cpu"))
                for component_id in component_ids
            )
            observed = []

            def runner(command, **kwargs):
                component_id = command[-1]
                observed.append(component_id)
                output = json.dumps({"kind": "worker", "component": component_id, "check": "ok"}) + "\n"
                return subprocess.CompletedProcess(command, 0, output, "")

            results = probes.run_offline_probes(components, component_targets=targets, runner=runner)

            self.assertEqual(list(component_ids), observed)
            self.assertTrue(all(results[component_id] is True for component_id in component_ids))

    def test_resource_descriptor_calculates_catalog_fingerprint_without_a_stored_field(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            target = root / "resource-library" / "tagging-models" / "fixture"
            target.mkdir(parents=True)
            manifest = {
                "schemaVersion": 1,
                "kind": "tagging-model",
                "resourceId": "caption-e621-eva02-large-full-v1",
                "resourceVersion": "fixture-v1",
                "profile": "e621",
                "displayName": {"zh-CN": "fixture", "en": "fixture"},
                "description": {"zh-CN": "fixture", "en": "fixture"},
                "runtimeFormat": "e621-eva02-onnx-v1",
                "entrypoints": {"model": "models/model.onnx"},
                "files": {"models/model.onnx": {"sizeBytes": 1, "sha256": "a" * 64}},
                "metadata": {"tagCount": 8783, "categories": ["general", "character", "species", "rating"]},
                "documentation": [],
            }
            (target / "resource.json").write_text(json.dumps(manifest), encoding="utf-8")

            _install_root, _relative, fingerprint, _value = probes._resource_descriptor(target)

            unsigned = {
                "schemaVersion": manifest["schemaVersion"],
                "kind": manifest["kind"],
                "resourceId": manifest["resourceId"],
                "resourceVersion": manifest["resourceVersion"],
                "profile": manifest["profile"],
                "runtimeFormat": manifest["runtimeFormat"],
                "entrypoints": {"model": "models\\model.onnx"},
                "files": {"models\\model.onnx": {"sizeBytes": 1, "sha256": "a" * 64}},
                "metadata": manifest["metadata"],
            }
            expected = hashlib.sha256(
                json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self.assertEqual(expected, fingerprint)

    def test_cuda_probe_group_failure_defers_its_resource_companion_until_cpu_retry(self) -> None:
        install_module = _install_module()
        pending = [
            SimpleNamespace(component=SimpleNamespace(component_id=component_id), variant=SimpleNamespace(name=variant))
            for component_id, variant in (
                ("caption-e621", "cuda"),
                ("e621-tagger", "shared"),
                ("policy", "cuda"),
                ("quality-stack", "shared"),
                ("ocr-cpu", "cpu"),
                ("ocr-gpu", "cuda"),
            )
        ]

        fallback_ids, discarded_gpu, failures = install_module._classify_probe_failures(
            pending,
            {
                "caption-e621": False,
                "e621-tagger": False,
                "policy": False,
                "quality-stack": False,
                "ocr-cpu": True,
                "ocr-gpu": False,
            },
        )

        self.assertEqual({"caption-e621", "policy"}, fallback_ids)
        self.assertEqual({"ocr-gpu"}, discarded_gpu)
        self.assertEqual(["quality-stack"], failures)

    def test_cpu_retry_keeps_group_resource_failure_fatal(self) -> None:
        install_module = _install_module()
        pending = [
            SimpleNamespace(component=SimpleNamespace(component_id=component_id), variant=SimpleNamespace(name=variant))
            for component_id, variant in (
                ("caption-e621", "cpu"),
                ("e621-tagger", "shared"),
                ("policy", "cpu"),
                ("quality-stack", "shared"),
            )
        ]

        fallback_ids, discarded_gpu, failures = install_module._classify_probe_failures(
            pending,
            {
                "caption-e621": True,
                "e621-tagger": False,
                "policy": True,
                "quality-stack": False,
            },
        )

        self.assertEqual(set(), fallback_ids)
        self.assertEqual(set(), discarded_gpu)
        self.assertEqual(["e621-tagger", "quality-stack"], failures)

    def test_cpu_fallback_items_ignore_required_cuda_only_ocr_gpu(self) -> None:
        install_module = _install_module()
        _, manifest_module = _modules()
        manifest = manifest_module.load_manifest_path(ROOT / "packaging" / "installer" / "install-manifest.json")

        fallback_items = install_module._cpu_fallback_items(manifest)

        self.assertEqual({"caption-e621", "policy"}, set(fallback_items))
        self.assertTrue(all(item.variant.name == "cpu" for item in fallback_items.values()))

    def test_unverified_ocr_gpu_probe_does_not_discard_the_cuda_runtime(self) -> None:
        install_module = _install_module()
        pending = [
            SimpleNamespace(
                component=SimpleNamespace(component_id="ocr-gpu"),
                variant=SimpleNamespace(name="cuda"),
            )
        ]

        fallback_ids, discarded_gpu, failures = install_module._classify_probe_failures(
            pending,
            {"ocr-gpu": None},
        )

        self.assertEqual(set(), fallback_ids)
        self.assertEqual(set(), discarded_gpu)
        self.assertEqual([], failures)

    def test_base_installation_plan_never_selects_optional_ocr(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())

        cpu = assemble.installation_plan(manifest, accelerator="cpu")
        nvidia = assemble.installation_plan(manifest, accelerator="nvidia")

        self.assertEqual(set(), cpu.runtime_ids & {"ocr-paddle", "ocr-paddle-gpu"})
        self.assertEqual(set(), nvidia.runtime_ids & {"ocr-paddle", "ocr-paddle-gpu"})
        self.assertNotIn("caption-e621-cuda", cpu.lock_names)
        self.assertNotIn("policy-cuda", cpu.lock_names)
        self.assertIn("caption-e621-cuda", nvidia.lock_names)
        self.assertNotIn("policy-cuda", nvidia.lock_names)

    def test_installation_plan_excludes_optional_delayed_ocr_models(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(
            fixture_install_manifest(include_delayed_ocr_models=True)
        )

        plan = assemble.installation_plan(manifest, accelerator="cpu")

        self.assertNotIn("ocr-models", {item.component.component_id for item in plan.components})

    def test_base_installation_plan_excludes_optional_ocr_runtimes(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())

        plan = assemble.installation_plan(manifest, accelerator="cpu")

        selected = {item.component.component_id for item in plan.components}
        self.assertNotIn("ocr-cpu", selected)
        self.assertNotIn("ocr-gpu", selected)

    def test_base_validation_does_not_require_optional_ocr_runtimes(self) -> None:
        assemble, manifest_module = _modules()
        value = fixture_manifest()
        for component_id in (
            "e621-indexes",
            "e621-replacement-indexes",
            "e621-tagger",
            "quality-stack",
            "qwen3-tokenizer",
        ):
            value["components"].append(
                {
                    "componentId": component_id,
                    "kind": "resource",
                    "required": component_id != "quality-stack",
                    "targetRelativePath": f"resource-library/{component_id}",
                    "variants": {"shared": _variant(f"{component_id}-resource", f"resources/{component_id}.json")},
                }
            )
        value["components"] = [
            component
            for component in value["components"]
            if component["componentId"] not in {"ocr-cpu", "ocr-gpu"}
        ]
        manifest = manifest_module.load_manifest(value)

        assemble.validate_mandatory_e621_components(manifest)

    def test_base_validation_rejects_ocr_components_marked_required(self) -> None:
        assemble, manifest_module = _modules()
        value = fixture_manifest()
        for component_id in (
            "e621-indexes",
            "e621-replacement-indexes",
            "e621-tagger",
            "quality-stack",
            "qwen3-tokenizer",
        ):
            value["components"].append(
                {
                    "componentId": component_id,
                    "kind": "resource",
                    "required": component_id != "quality-stack",
                    "targetRelativePath": f"resource-library/{component_id}",
                    "variants": {"shared": _variant(f"{component_id}-resource", f"resources/{component_id}.json")},
                }
            )
        next(component for component in value["components"] if component["componentId"] == "ocr-cpu")["required"] = True
        manifest = manifest_module.load_manifest(value)

        with self.assertRaisesRegex(assemble.ManifestError, "optional component"):
            assemble.validate_mandatory_e621_components(manifest)

    def test_production_plan_requires_all_mandatory_e621_components(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_install_manifest())

        with self.assertRaisesRegex(assemble.ManifestError, "mandatory E621 components"):
            assemble.validate_mandatory_e621_components(manifest)

    def test_production_plan_requires_the_replacement_index_package(self) -> None:
        assemble, manifest_module = _modules()
        value = fixture_manifest()
        for component_id in ("e621-indexes", "e621-tagger", "quality-stack", "qwen3-tokenizer"):
            value["components"].append(
                {
                    "componentId": component_id,
                    "kind": "resource",
                    "required": component_id != "quality-stack",
                    "targetRelativePath": f"resource-library/{component_id}",
                    "variants": {"shared": _variant(f"{component_id}-resource", f"resources/{component_id}.json")},
                }
            )
        value["components"] = [
            component
            for component in value["components"]
            if component["componentId"] != "ocr-models"
        ]
        manifest = manifest_module.load_manifest(value)

        with self.assertRaisesRegex(assemble.ManifestError, "e621-replacement-indexes"):
            assemble.validate_mandatory_e621_components(manifest)

    def test_base_e621_validation_does_not_require_delayed_ocr_models(self) -> None:
        assemble, manifest_module = _modules()
        value = fixture_manifest()
        for component_id in (
            "e621-indexes",
            "e621-replacement-indexes",
            "e621-tagger",
            "quality-stack",
            "qwen3-tokenizer",
        ):
            value["components"].append(
                {
                    "componentId": component_id,
                    "kind": "resource",
                    "required": component_id != "quality-stack",
                    "targetRelativePath": f"resource-library/{component_id}",
                    "variants": {"shared": _variant(f"{component_id}-resource", f"resources/{component_id}.json")},
                }
            )
        value["components"] = [
            component
            for component in value["components"]
            if component["componentId"] != "ocr-models"
        ]
        manifest = manifest_module.load_manifest(value)

        assemble.validate_mandatory_e621_components(manifest)

    def test_ocr_runtime_without_a_selected_model_is_not_a_functional_probe_success(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            runtime = root / "runtimes" / "ocr-paddle"
            runtime.mkdir(parents=True)
            components = (
                SimpleNamespace(
                    component=SimpleNamespace(component_id="ocr-cpu"),
                    variant=SimpleNamespace(name="cpu"),
                ),
            )

            def fail_runner(*_args, **_kwargs):
                self.fail("OCR probe must not run without an OCR model component")

            results = probes.run_offline_probes(
                components,
                component_targets={"ocr-cpu": runtime},
                runner=fail_runner,
            )

            self.assertIsNone(results["ocr-cpu"])

    def test_existing_ocr_resource_runs_cpu_and_gpu_offline_probes(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            cpu_runtime = root / "runtimes" / "ocr-paddle"
            gpu_runtime = root / "runtimes" / "ocr-paddle-gpu"
            resource = root / "resource-library" / "ocr-models" / "fixture"
            for target in (cpu_runtime, gpu_runtime, resource):
                target.mkdir(parents=True)
            (resource / "resource.json").write_text(
                json.dumps({"fingerprint": "a" * 64}), encoding="utf-8"
            )
            components = (
                SimpleNamespace(
                    component=SimpleNamespace(component_id="ocr-cpu"),
                    variant=SimpleNamespace(name="cpu"),
                ),
                SimpleNamespace(
                    component=SimpleNamespace(component_id="ocr-gpu"),
                    variant=SimpleNamespace(name="cuda"),
                ),
            )

            def runner(command, **_kwargs):
                device = command[-1]
                output = json.dumps(
                    {
                        "kind": "ocr",
                        "device": "cpu" if device == "cpu" else "gpu:0",
                        "resultCount": 1,
                        "texts": ["offline"],
                    }
                )
                return subprocess.CompletedProcess(command, 0, output, "")

            results = probes.run_offline_probes(
                components,
                component_targets={"ocr-cpu": cpu_runtime, "ocr-gpu": gpu_runtime},
                ocr_resource_target=resource,
                runner=runner,
            )

            self.assertEqual({"ocr-cpu": True, "ocr-gpu": True}, results)

    def test_runtime_manifest_uses_selected_caption_cpu_lock(self) -> None:
        generator = _runtime_manifest_generator()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            runtime = root / "runtimes" / "caption-e621"
            worker = runtime / "Lib" / "site-packages" / "anima_caption_worker"
            worker.mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (runtime / filename).write_bytes(filename.encode("ascii"))
            (worker / "entry.py").write_text("VALUE = 1\n", encoding="ascii")
            requirements = root / "requirements"
            requirements.mkdir()
            lock = requirements / "caption-e621-cpu.lock"
            lock.write_text("caption fixture\n", encoding="ascii")

            specs = generator.runtime_specs(lock_names={"caption-e621": "caption-e621-cpu"})
            value = generator.manifest(root, requirements, "caption-e621", specs)

            self.assertEqual(_sha256(lock.read_bytes()), value["runtime"]["dependencyLockSha256"])

    def test_runtime_manifest_publishes_selected_variant_as_the_canonical_runtime_lock(self) -> None:
        install_module = _install_module()
        _, manifest_module = _modules()
        manifest = manifest_module.load_manifest_path(ROOT / "packaging" / "installer" / "install-manifest.json")
        caption = install_module._cpu_fallback_items(manifest)["caption-e621"]
        self.assertEqual("caption-e621-cpu", caption.lock_name)

        with tempfile.TemporaryDirectory() as temporary_name:
            runtime_root = Path(temporary_name) / ".runtime-build"
            runtime = runtime_root / "runtimes" / "caption-e621"
            worker = runtime / "Lib" / "site-packages" / "anima_caption_worker"
            (runtime / "Lib" / "site-packages" / "onnxruntime" / "capi").mkdir(parents=True)
            worker.mkdir(parents=True)
            (worker / "entry.py").write_text("VALUE = 1\n", encoding="ascii")
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (runtime / filename).write_bytes(filename.encode("ascii"))

            install_module._write_runtime_manifest_at(ROOT, caption, runtime_root)

            canonical = runtime_root / "manifests" / "requirements" / "caption-e621.lock"
            selected = ROOT / "packaging" / "requirements" / "caption-e621-cpu.lock"
            self.assertEqual(selected.read_bytes(), canonical.read_bytes())

    def test_skip_requires_fingerprint_all_files_and_runtime_manifest(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        plan = assemble.installation_plan(manifest, accelerator="cpu")
        core = next(item for item in plan.components if item.component.component_id == "core")

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            target = root / ".runtime-build" / "runtimes" / "core"
            target.mkdir(parents=True)
            payload = b"core fixture"
            (target / "marker.txt").write_bytes(payload)
            runtime_manifest = root / ".runtime-build" / "manifests" / "runtimes" / "core.json"
            runtime_manifest.parent.mkdir(parents=True)
            runtime_manifest.write_text(
                json.dumps({"runtime": {"runtimeId": "core"}}, sort_keys=True), encoding="utf-8"
            )
            record = {
                "componentId": "core",
                "variant": "cpu",
                "fingerprint": assemble.component_fingerprint(core),
                "targetRelativePath": ".runtime-build\\runtimes\\core",
                "files": [
                    {
                        "relativePath": "marker.txt",
                        "sizeBytes": len(payload),
                        "sha256": _sha256(payload),
                    }
                ],
                "runtimeManifestRelativePath": ".runtime-build\\manifests\\runtimes\\core.json",
            }

            self.assertFalse(assemble.component_is_current(layout, core, record))
            lock_payload = b"core fixture lock\n"
            canonical_lock = root / ".runtime-build" / "manifests" / "requirements" / "core.lock"
            canonical_lock.parent.mkdir(parents=True)
            canonical_lock.write_bytes(lock_payload)
            runtime_manifest.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "runtimeId": "core",
                            "dependencyLockSha256": _sha256(lock_payload),
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self.assertTrue(assemble.component_is_current(layout, core, record))
            (target / "marker.txt").write_bytes(b"drift")
            self.assertFalse(assemble.component_is_current(layout, core, record))
            (target / "marker.txt").write_bytes(payload)
            runtime_manifest.unlink()
            self.assertFalse(assemble.component_is_current(layout, core, record))

    def test_component_record_accepts_unchanged_zero_byte_files(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        core = next(
            item
            for item in assemble.installation_plan(manifest, accelerator="cpu").components
            if item.component.component_id == "core"
        )

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            target = root / ".runtime-build" / "runtimes" / "core"
            target.mkdir(parents=True)
            (target / "empty.py").write_bytes(b"")
            runtime_manifest = root / ".runtime-build" / "manifests" / "runtimes" / "core.json"
            runtime_manifest.parent.mkdir(parents=True)
            lock_payload = b"core zero-byte fixture lock\n"
            canonical_lock = root / ".runtime-build" / "manifests" / "requirements" / "core.lock"
            canonical_lock.parent.mkdir(parents=True)
            canonical_lock.write_bytes(lock_payload)
            runtime_manifest.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "runtimeId": "core",
                            "dependencyLockSha256": _sha256(lock_payload),
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            record = assemble.component_record(layout, core)

            self.assertEqual(0, record["files"][0]["sizeBytes"])
            self.assertTrue(assemble.component_is_current(layout, core, record))

    def test_runtime_skip_rejects_legacy_root_wheel_metadata(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        core = next(
            item
            for item in assemble.installation_plan(manifest, accelerator="cpu").components
            if item.component.component_id == "core"
        )

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            target = root / ".runtime-build" / "runtimes" / "core"
            metadata = target / "fixture_pkg-1.0.dist-info" / "METADATA"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("Name: fixture-pkg\nVersion: 1.0\n", encoding="ascii")
            _write_fixture_runtime_manifest(core, layout)

            record = assemble.component_record(layout, core)

            self.assertFalse(assemble.component_is_current(layout, core, record))

    def test_runtime_assembly_rejects_duplicate_wheel_paths_before_publish(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        item = next(
            planned
            for planned in assemble.installation_plan(manifest, accelerator="cpu").components
            if planned.component.component_id == "core"
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            base = root / "base"
            (base / "Lib").mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (base / filename).write_bytes(filename.encode("ascii"))
            wheels = []
            for index in range(2):
                wheel = root / f"fixture-{index}.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("fixture.dist-info/METADATA", "Name: fixture\nVersion: 1\n")
                    archive.writestr("shared.py", str(index))
                wheels.append(wheel)

            with self.assertRaisesRegex(assemble.AssemblyError, "duplicate wheel path"):
                assemble.assemble_runtime(
                    layout,
                    item,
                    base_runtime=base,
                    wheels=[(wheel, wheel.name) for wheel in wheels],
                    destination=layout.staging / "core",
                )
            self.assertFalse((layout.staging / "core").exists())

    def test_runtime_assembly_allows_identical_namespace_files(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        item = assemble.installation_plan(manifest, accelerator="cpu").components[0]
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            base = root / "base"
            (base / "Lib").mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (base / filename).write_bytes(filename.encode("ascii"))
            wheels = []
            for index in range(2):
                wheel = root / f"namespace-{index}.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr(f"fixture_{index}.dist-info/METADATA", f"Name: fixture-{index}\nVersion: 1\n")
                    archive.writestr("nvidia/__init__.py", b"")
                wheels.append(wheel)

            assemble.assemble_runtime(
                layout,
                item,
                base_runtime=base,
                wheels=[(wheel, wheel.name) for wheel in wheels],
                destination=layout.staging / "core",
            )

            self.assertEqual(
                b"",
                (layout.staging / "core" / "Lib" / "site-packages" / "nvidia" / "__init__.py").read_bytes(),
            )

    def test_runtime_assembly_uses_manifest_name_for_hash_named_wheel_cache(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        item = assemble.installation_plan(manifest, accelerator="cpu").components[0]
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            base = root / "base"
            (base / "Lib").mkdir(parents=True)
            (base / "python.exe").write_bytes(b"python")
            wheel_cache = root / ("a" * 64)
            with zipfile.ZipFile(wheel_cache, "w") as archive:
                archive.writestr("fixture_pkg/__init__.py", "VALUE = 1\n")

            destination = layout.staging / "core"
            assemble.assemble_runtime(
                layout,
                item,
                base_runtime=base,
                wheels=[(wheel_cache, "wheels/core/fixture-1.0-py3-none-any.whl")],
                destination=destination,
            )

            self.assertTrue((destination / "Lib" / "site-packages" / "fixture_pkg" / "__init__.py").is_file())

    def test_runtime_assembly_rejects_non_wheel_manifest_name(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        item = assemble.installation_plan(manifest, accelerator="cpu").components[0]
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            base = root / "base"
            (base / "Lib").mkdir(parents=True)
            (base / "python.exe").write_bytes(b"python")
            wheel_cache = root / ("b" * 64)
            with zipfile.ZipFile(wheel_cache, "w") as archive:
                archive.writestr("fixture_pkg/__init__.py", "VALUE = 1\n")

            with self.assertRaisesRegex(assemble.AssemblyError, "wheel input is invalid"):
                assemble.assemble_runtime(
                    layout,
                    item,
                    base_runtime=base,
                    wheels=[(wheel_cache, "wheels/core/not-a-wheel.zip")],
                    destination=layout.staging / "core",
                )

    def test_runtime_assembly_copies_owner_source_and_strips_build_helpers(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        item = next(
            planned
            for planned in assemble.installation_plan(manifest, accelerator="cpu").components
            if planned.component.component_id == "core"
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            base = root / "base"
            packages = base / "Lib" / "site-packages"
            for package in ("pip", "wheel", "pytest"):
                (packages / package).mkdir(parents=True, exist_ok=True)
                (packages / package / "__init__.py").write_text("VALUE = 1\n", encoding="ascii")
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (base / filename).write_bytes(filename.encode("ascii"))
            wheel = root / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("fixture_pkg/__init__.py", "VALUE = 2\n")
            owner = root / "owner" / "anima_core"
            owner.mkdir(parents=True)
            (owner / "__init__.py").write_text("VALUE = 'owner'\n", encoding="ascii")

            destination = layout.staging / "core"
            files = assemble.assemble_runtime(
                layout,
                item,
                base_runtime=base,
                wheels=[(wheel, wheel.name)],
                destination=destination,
                owner_sources={"anima_core": owner},
            )

            package_root = destination / "Lib" / "site-packages"
            self.assertTrue((package_root / "fixture_pkg" / "__init__.py").is_file())
            self.assertEqual("VALUE = 'owner'\n", (package_root / "anima_core" / "__init__.py").read_text(encoding="ascii"))
            self.assertFalse((package_root / "pip").exists())
            self.assertFalse((package_root / "wheel").exists())
            self.assertFalse((package_root / "pytest").exists())
            self.assertIn("Lib\\site-packages\\anima_core\\__init__.py", files)

    def test_install_state_contains_only_complete_selected_components(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        plan = assemble.installation_plan(manifest, accelerator="cpu")
        records = {
            item.component.component_id: {
                "componentId": item.component.component_id,
                "variant": item.variant.name,
                "fingerprint": assemble.component_fingerprint(item),
                "targetRelativePath": item.component.target_relative_path,
                "files": [{"relativePath": "marker.txt", "sizeBytes": 1, "sha256": "a" * 64}],
            }
            for item in plan.components
        }

        state = assemble.build_install_state(manifest, plan, records, completed_at_utc="2026-08-11T00:00:00Z")

        self.assertEqual(1, state["schemaVersion"])
        self.assertEqual(manifest.source_commit, state["sourceCommit"])
        self.assertEqual(manifest.fingerprint, state["installManifestSha256"])
        self.assertEqual("cpu", state["accelerator"])
        self.assertNotIn("ocr-gpu", state["components"])
        self.assertNotIn("policy", state["components"])
        self.assertEqual("2026-08-11T00:00:00Z", state["completedAtUtc"])

        incomplete = dict(records)
        incomplete.pop("core")
        with self.assertRaisesRegex(assemble.AssemblyError, "required component"):
            assemble.build_install_state(manifest, plan, incomplete, completed_at_utc="2026-08-11T00:00:00Z")

    def test_runtime_assembly_never_cleans_a_destination_outside_staging(self) -> None:
        assemble, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_manifest())
        item = next(
            planned
            for planned in assemble.installation_plan(manifest, accelerator="cpu").components
            if planned.component.component_id == "core"
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            container = Path(temporary_name)
            root = container / "project"
            root.mkdir()
            layout = assemble.ProjectLayout.create(root)
            layout.ensure_directories()
            base = root / "base"
            base.mkdir()
            outside = container / "outside"
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="ascii")

            with self.assertRaises(assemble.PathSafetyError):
                assemble.assemble_runtime(
                    layout,
                    item,
                    base_runtime=base,
                    wheels=[],
                    destination=outside,
                )

            self.assertEqual("keep", marker.read_text(encoding="ascii"))

    def test_fixture_install_publishes_state_then_second_run_skips_fetches(self) -> None:
        install_module = _install_module()
        _, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fixture_install_manifest())
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            base = root / "bootstrap-base"
            (base / "Lib").mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (base / filename).write_bytes(filename.encode("ascii"))
            source = root / "core" / "src" / "anima_core"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("VALUE = 'core'\n", encoding="ascii")
            (source / "__main__.py").write_text("VALUE = 'main'\n", encoding="ascii")
            shared_source = root / "shared" / "anima_caption_format" / "anima_caption_format"
            shared_source.mkdir(parents=True)
            (shared_source / "__init__.py").write_text("VALUE = 'format'\n", encoding="ascii")
            wheel = root / "core.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("fixture_pkg/__init__.py", "VALUE = 1\n")
            resource = root / "resource.json"
            resource.write_text('{"fixture":true}\n', encoding="utf-8")
            paths = {"core-cpu-wheel": wheel, "fixture-resource-json": resource}
            calls: list[str] = []

            def fetch(artifact):
                calls.append(artifact.artifact_id)
                return paths[artifact.artifact_id]

            def probe(item, target):
                return target.is_dir() and item.component.component_id in {"core", "fixture-resource"}

            first = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="cpu",
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=_write_fixture_runtime_manifest,
                require_mandatory_e621=False,
            )

            self.assertEqual(("core", "fixture-resource"), first.installed_component_ids)
            self.assertEqual(["core-cpu-wheel", "fixture-resource-json"], calls)
            state_path = root / ".runtime-build" / "manifests" / "install-state.json"
            self.assertTrue(state_path.is_file())
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual({"core", "fixture-resource"}, set(state["components"]))
            self.assertTrue((root / ".runtime-build" / "runtimes" / "core" / "python.exe").is_file())
            self.assertTrue((root / "resource-library" / "fixture-resource" / "resource.json").is_file())
            for relative in ("bootstrap", "cache", "staging", "transactions"):
                self.assertFalse((root / ".runtime-build" / "source-bootstrap" / relative).exists())

            calls.clear()
            second = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="cpu",
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=_write_fixture_runtime_manifest,
                require_mandatory_e621=False,
            )

            self.assertEqual((), second.installed_component_ids)
            self.assertEqual(("core", "fixture-resource"), second.skipped_component_ids)
            self.assertEqual([], calls)

            (source / "__init__.py").write_text("VALUE = 'changed'\n", encoding="ascii")
            third = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="cpu",
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=_write_fixture_runtime_manifest,
                require_mandatory_e621=False,
            )

            self.assertEqual(("core",), third.installed_component_ids)
            self.assertEqual(("fixture-resource",), third.skipped_component_ids)
            self.assertEqual(["core-cpu-wheel"], calls)

    def test_offline_probe_failure_preserves_exit_code_and_stderr(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            python = root / "python.exe"
            python.write_bytes(b"python")

            def runner(command, **_kwargs):
                return subprocess.CompletedProcess(
                    command,
                    17,
                    "",
                    "CUDAExecutionProvider initialization failed\nprovider unavailable",
                )

            with self.assertRaisesRegex(
                probes.ProbeError,
                r"exit code 17.*CUDAExecutionProvider initialization failed.*provider unavailable",
            ):
                probes.run_json_probe(
                    python,
                    "print('{}')",
                    (),
                    cwd=root,
                    runner=runner,
                )

    def test_offline_probe_reports_component_progress(self) -> None:
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            runtime = root / "runtimes" / "core"
            runtime.mkdir(parents=True)
            (runtime / "python.exe").write_bytes(b"python")
            events: list[str] = []

            def runner(command, **_kwargs):
                return subprocess.CompletedProcess(command, 0, "anima-core-runtime-ok\n", "")

            results = probes.run_offline_probes(
                (
                    SimpleNamespace(
                        component=SimpleNamespace(component_id="core"),
                        variant=SimpleNamespace(name="cpu"),
                    ),
                ),
                component_targets={"core": runtime},
                runner=runner,
                progress=events.append,
            )

            self.assertEqual({"core": True}, results)
            self.assertTrue(any("Offline probe started: core" in event for event in events))
            self.assertTrue(any("Offline probe passed: core" in event for event in events))

    def test_default_probe_results_adds_installed_companion_without_rewriting_it(self) -> None:
        install_module = _install_module()
        probes = _probes_module()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            target = root / "runtimes" / "core"
            target.mkdir(parents=True)
            policy = SimpleNamespace(
                component=SimpleNamespace(component_id="policy"),
                variant=SimpleNamespace(name="cuda"),
                runtime_id="policy",
                lock_name="policy",
            )
            quality = SimpleNamespace(
                component=SimpleNamespace(component_id="quality-stack"),
                variant=SimpleNamespace(name="shared"),
                runtime_id=None,
                lock_name=None,
            )
            with (
                mock.patch.object(install_module, "_write_runtime_manifest_at") as write_manifest,
                mock.patch.object(
                    probes,
                    "run_offline_probes",
                    return_value={"policy": True, "quality-stack": True},
                ) as run,
            ):
                result = install_module._default_probe_results(
                    source_root=root,
                    pending=[policy],
                    components=[policy, quality],
                    targets={"policy": target, "quality-stack": root / "quality-stack"},
                )

            self.assertEqual({"policy": True, "quality-stack": True}, result)
            self.assertEqual([policy], list(run.call_args.args[0]))
            write_manifest.assert_called_once()
            self.assertIs(policy, write_manifest.call_args.args[1])

    def test_source_tree_resource_is_verified_and_never_fetched(self) -> None:
        install_module = _install_module()
        _, manifest_module = _modules()
        source_payload = b'{"fixture":"source-tree"}\n'
        manifest_value = fixture_install_manifest()
        resource = manifest_value["components"][1]
        resource["variants"]["shared"]["artifacts"] = [
            _source_tree_artifact(
                "fixture-source-tree-resource",
                "resource-library/fixture-resource/resource.json",
                "resource.json",
                source_payload,
            )
        ]
        manifest = manifest_module.load_manifest(manifest_value)
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            base = root / "bootstrap-base"
            (base / "Lib").mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (base / filename).write_bytes(filename.encode("ascii"))
            source = root / "core" / "src" / "anima_core"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("VALUE = 'core'\n", encoding="ascii")
            (source / "__main__.py").write_text("VALUE = 'main'\n", encoding="ascii")
            shared_source = root / "shared" / "anima_caption_format" / "anima_caption_format"
            shared_source.mkdir(parents=True)
            (shared_source / "__init__.py").write_text("VALUE = 'format'\n", encoding="ascii")
            source_resource = root / "resource-library" / "fixture-resource"
            source_resource.mkdir(parents=True)
            (source_resource / "resource.json").write_bytes(source_payload)
            unrelated = root / "resource-library" / "unrelated-resource" / "keep.txt"
            unrelated.parent.mkdir()
            unrelated.write_text("preserve", encoding="ascii")
            wheel = root / "core.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("fixture_pkg/__init__.py", "VALUE = 1\n")
            fetched: list[str] = []

            def fetch(artifact):
                fetched.append(artifact.artifact_id)
                if artifact.artifact_id == "core-cpu-wheel":
                    return wheel
                raise AssertionError(f"source-tree artifact was fetched: {artifact.artifact_id}")

            def probe(item, target):
                return target.is_dir()

            result = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="cpu",
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=_write_fixture_runtime_manifest,
                require_mandatory_e621=False,
            )

            self.assertEqual(["core-cpu-wheel"], fetched)
            self.assertEqual(source_payload, (root / "resource-library" / "fixture-resource" / "resource.json").read_bytes())
            self.assertEqual("preserve", unrelated.read_text(encoding="ascii"))
            self.assertIn("fixture-resource", result.installed_component_ids)

    def test_complete_manual_archives_import_after_base_state_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install_module, manifest, base, fetch, probe, write_runtime_manifest = self._prepare_fixture_install(
                root,
                include_delayed_ocr_models=True,
            )
            archive_root = root / "ocr-model-archives"
            archive_root.mkdir()
            for filename in OCR_MODEL_ARCHIVE_FILENAMES:
                (archive_root / filename).write_bytes(b"fixture archive")
            importer_roots: list[Path] = []

            def importer(project_root: Path):
                self.assertTrue(
                    (project_root / ".runtime-build" / "manifests" / "install-state.json").is_file()
                )
                importer_roots.append(project_root)
                return {"resource": "installed"}

            result = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="cpu",
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=write_runtime_manifest,
                require_mandatory_e621=False,
                import_optional_ocr_models=importer,
            )

            self.assertEqual([root], importer_roots)
            self.assertIn("OCR model import completed", result.messages)
            self.assertNotIn("ocr-models", result.installed_component_ids)

    def test_missing_manual_archives_leave_base_install_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install_module, manifest, base, fetch, probe, write_runtime_manifest = self._prepare_fixture_install(
                root,
                include_delayed_ocr_models=True,
            )
            importer_roots: list[Path] = []

            def importer(project_root: Path):
                importer_roots.append(project_root)
                return {"resource": "installed"}

            result = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="cpu",
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=write_runtime_manifest,
                require_mandatory_e621=False,
                import_optional_ocr_models=importer,
            )

            self.assertEqual([], importer_roots)
            self.assertTrue(result.state_path.is_file())
            self.assertIn("OCR_MODEL_DOWNLOAD.md", "\n".join(result.messages))

    def test_partial_manual_archives_leave_base_install_complete(self) -> None:
        for archive_count in (1, 2):
            with self.subTest(archive_count=archive_count), tempfile.TemporaryDirectory() as temporary_name:
                root = Path(temporary_name)
                install_module, manifest, base, fetch, probe, write_runtime_manifest = self._prepare_fixture_install(root)
                archive_root = root / "ocr-model-archives"
                archive_root.mkdir()
                for filename in OCR_MODEL_ARCHIVE_FILENAMES[:archive_count]:
                    (archive_root / filename).write_bytes(b"partial")

                with mock.patch.object(install_module, "_load_ocr_resource_module") as loader:
                    result = install_module.install_project(
                        project_root=root,
                        source_root=root,
                        manifest=manifest,
                        accelerator="cpu",
                        base_runtime=base,
                        fetch_artifact=fetch,
                        probe_component=probe,
                        write_runtime_manifest=write_runtime_manifest,
                        require_mandatory_e621=False,
                    )

                loader.assert_not_called()
                self.assertTrue(result.state_path.is_file())
                self.assertIn("OCR_MODEL_DOWNLOAD.md", "\n".join(result.messages))

    def test_base_install_without_manual_ocr_inputs_does_not_load_the_importer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install_module, manifest, base, fetch, probe, write_runtime_manifest = self._prepare_fixture_install(root)

            result = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="cpu",
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=write_runtime_manifest,
                require_mandatory_e621=False,
            )

            self.assertTrue(result.state_path.is_file())
            self.assertIn("OCR_MODEL_DOWNLOAD.md", "\n".join(result.messages))

    def test_base_install_with_complete_model_archives_does_not_require_ocr_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install_module, manifest, base, fetch, probe, write_runtime_manifest = self._prepare_fixture_install(
                root,
                include_delayed_ocr_models=True,
            )
            archive_root = root / "ocr-model-archives"
            archive_root.mkdir()
            for filename in OCR_MODEL_ARCHIVE_FILENAMES:
                (archive_root / filename).write_bytes(b"fixture archive")

            with mock.patch.object(
                install_module,
                "_import_available_ocr_models",
                return_value={"resource": "installed"},
            ) as importer:
                result = install_module.install_project(
                    project_root=root,
                    source_root=root,
                    manifest=manifest,
                    accelerator="cpu",
                    base_runtime=base,
                    fetch_artifact=fetch,
                    probe_component=probe,
                    write_runtime_manifest=write_runtime_manifest,
                    require_mandatory_e621=False,
                )

            importer.assert_not_called()
            self.assertTrue(result.state_path.is_file())
            self.assertNotIn("OCR model import completed", result.messages)
            self.assertIn("optional OCR CPU runtime is not installed", "\n".join(result.messages))

    def test_ocr_model_import_error_becomes_an_installer_error(self) -> None:
        install_module = _install_module()
        importer = SimpleNamespace(
            import_available_local_model_resource=mock.Mock(
                side_effect=RuntimeError("fixture archive hash mismatch")
            )
        )
        with mock.patch.object(install_module, "_load_ocr_resource_module", return_value=importer):
            with self.assertRaisesRegex(install_module.AssemblyError, "OCR model import failed"):
                install_module._import_available_ocr_models(Path("source"), Path("project"))

    def test_failed_complete_ocr_model_import_preserves_published_base_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            install_module, manifest, base, fetch, probe, write_runtime_manifest = self._prepare_fixture_install(root)
            archive_root = root / "ocr-model-archives"
            archive_root.mkdir()
            for filename in OCR_MODEL_ARCHIVE_FILENAMES:
                (archive_root / filename).write_bytes(b"invalid")
            runtime = root / ".runtime-build" / "runtimes" / "ocr-paddle"
            runtime.mkdir(parents=True)
            (runtime / "python.exe").write_bytes(b"fixture python")
            importer = SimpleNamespace(
                import_available_local_model_resource=mock.Mock(
                    side_effect=RuntimeError("fixture archive hash mismatch")
                )
            )

            with mock.patch.object(install_module, "_load_ocr_resource_module", return_value=importer):
                with self.assertRaisesRegex(install_module.AssemblyError, "OCR model import failed"):
                    install_module.install_project(
                        project_root=root,
                        source_root=root,
                        manifest=manifest,
                        accelerator="cpu",
                        base_runtime=base,
                        fetch_artifact=fetch,
                        probe_component=probe,
                        write_runtime_manifest=write_runtime_manifest,
                        require_mandatory_e621=False,
                    )

            self.assertTrue((root / ".runtime-build" / "manifests" / "install-state.json").is_file())
            self.assertFalse(
                (root / "resource-library" / "ocr-models" / "ocr-ppocrv5-server-paddle-v1").exists()
            )

    def test_caption_cuda_probe_failure_keeps_cuda_while_policy_rebuilds_cpu(self) -> None:
        install_module = _install_module()
        _, manifest_module = _modules()
        manifest = manifest_module.load_manifest(fallback_fixture_manifest(include_probe_companions=True))
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            base = root / "bootstrap-base"
            (base / "Lib").mkdir(parents=True)
            for filename in ("python.exe", "python311.dll", "python311._pth"):
                (base / filename).write_bytes(filename.encode("ascii"))
            for relative in (
                "workers/caption/src/anima_caption_worker",
                "workers/policy/src/anima_policy_worker",
            ):
                source = root / relative
                source.mkdir(parents=True)
                (source / "__init__.py").write_text("VALUE = 'fixture'\n", encoding="ascii")
            artifact_paths: dict[str, Path] = {}
            for component in manifest.components:
                for variant in component.variants.values():
                    for artifact in variant.artifacts:
                        wheel = root / f"{artifact.artifact_id}.whl"
                        with zipfile.ZipFile(wheel, "w") as archive:
                            archive.writestr(
                                "fixture_pkg/__init__.py",
                                "VALUE = 1\n",
                            )
                        artifact_paths[artifact.artifact_id] = wheel
            probe_calls: list[tuple[str, str]] = []

            def fetch(artifact):
                return artifact_paths[artifact.artifact_id]

            def probe(item, target):
                probe_calls.append((item.component.component_id, item.variant.name))
                self.assertTrue(target.is_dir())
                if item.component.component_id in {"e621-tagger", "quality-stack"}:
                    return True
                return (item.component.component_id, item.variant.name) not in {
                    ("caption-e621", "cuda"),
                    ("policy", "cuda"),
                }

            result = install_module.install_project(
                project_root=root,
                source_root=root,
                manifest=manifest,
                accelerator="nvidia",
                optional_components={"policy", "quality-stack"},
                base_runtime=base,
                fetch_artifact=fetch,
                probe_component=probe,
                write_runtime_manifest=_write_fixture_runtime_manifest,
                require_mandatory_e621=False,
            )

            state = json.loads(result.state_path.read_text(encoding="utf-8"))
            self.assertEqual("cuda", state["components"]["caption-e621"]["variant"])
            self.assertEqual("cpu", state["components"]["policy"]["variant"])
            self.assertNotIn("ocr-cpu", state["components"])
            self.assertNotIn("ocr-gpu", state["components"])
            self.assertIn(("caption-e621", "cuda"), probe_calls)
            self.assertNotIn(("caption-e621", "cpu"), probe_calls)
            self.assertIn(("policy", "cuda"), probe_calls)
            self.assertIn(("policy", "cpu"), probe_calls)
            self.assertEqual(1, probe_calls.count(("e621-tagger", "shared")))
            self.assertEqual(1, probe_calls.count(("quality-stack", "shared")))
            self.assertTrue(any("Caption CUDA offline probe failed; keeping the CUDA runtime" in message for message in result.messages))
            self.assertTrue(any("policy CUDA offline probe failed" in message for message in result.messages))
            self.assertFalse(any(component_id.startswith("ocr-") for component_id, _ in probe_calls))

    def test_required_policy_probe_failure_never_publishes_state(self) -> None:
        install_module = _install_module()
        _, manifest_module = _modules()
        for failed_component in ("policy",):
            with self.subTest(component=failed_component), tempfile.TemporaryDirectory() as temporary_name:
                root = Path(temporary_name)
                base = root / "bootstrap-base"
                (base / "Lib").mkdir(parents=True)
                for filename in ("python.exe", "python311.dll", "python311._pth"):
                    (base / filename).write_bytes(filename.encode("ascii"))
                for relative in (
                    "workers/caption/src/anima_caption_worker",
                    "workers/policy/src/anima_policy_worker",
                ):
                    source = root / relative
                    source.mkdir(parents=True)
                    (source / "__init__.py").write_text("VALUE = 'fixture'\n", encoding="ascii")
                manifest = manifest_module.load_manifest(fallback_fixture_manifest())
                artifact_paths: dict[str, Path] = {}
                for component in manifest.components:
                    for variant in component.variants.values():
                        for artifact in variant.artifacts:
                            wheel = root / f"{artifact.artifact_id}.whl"
                            with zipfile.ZipFile(wheel, "w") as archive:
                                archive.writestr("fixture_pkg/__init__.py", "VALUE = 1\n")
                            artifact_paths[artifact.artifact_id] = wheel

                with self.assertRaisesRegex(Exception, "offline probe failed"):
                    install_module.install_project(
                        project_root=root,
                        source_root=root,
                        manifest=manifest,
                        accelerator="cpu",
                        optional_components={"policy"},
                        base_runtime=base,
                        fetch_artifact=lambda artifact: artifact_paths[artifact.artifact_id],
                        probe_component=lambda item, _target: item.component.component_id != failed_component,
                        write_runtime_manifest=lambda item, layout: str(
                            (layout.runtime_root / "manifests" / "runtimes" / f"{item.runtime_id}.json")
                            .relative_to(layout.project_root)
                        ).replace("/", "\\"),
                        require_mandatory_e621=False,
                    )
                self.assertFalse((root / ".runtime-build" / "manifests" / "install-state.json").exists())


if __name__ == "__main__":
    unittest.main()
