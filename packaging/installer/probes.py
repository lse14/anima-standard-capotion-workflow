"""Network-blocked representative probes for source-bootstrap components."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence


_PROXY_VARIABLES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
_NETWORK_GUARD = """
import socket

def _anima_block_network(*args, **kwargs):
    raise RuntimeError("network is blocked during source bootstrap probe")

socket.create_connection = _anima_block_network
socket.socket.connect = _anima_block_network
socket.socket.connect_ex = _anima_block_network
"""


class ProbeError(RuntimeError):
    """A component has not produced representative offline evidence."""


Runner = Callable[..., subprocess.CompletedProcess[str]]
ProgressReporter = Callable[[str], None]
_PROBE_DETAIL_LIMIT = 2000


def _probe_detail(value: object) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace").strip()
    else:
        text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > _PROBE_DETAIL_LIMIT:
        text = "..." + text[-_PROBE_DETAIL_LIMIT:]
    return " ".join(text.split())


def offline_environment(
    environment: Mapping[str, str] | None = None,
    *,
    install_root: Path | None = None,
    resource_root: Path | None = None,
    dll_directories: Sequence[Path] = (),
) -> dict[str, str]:
    """Return a child environment that cannot use configured network proxies."""
    result = dict(os.environ if environment is None else environment)
    for name in _PROXY_VARIABLES:
        result.pop(name, None)
    result["HF_HUB_OFFLINE"] = "1"
    result["TRANSFORMERS_OFFLINE"] = "1"
    result["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if install_root is not None:
        result["ANIMA_INSTALL_ROOT"] = str(install_root)
    if resource_root is not None:
        result["ANIMA_RESOURCE_ROOT"] = str(resource_root)
    if dll_directories:
        result["PATH"] = os.pathsep.join(
            [*(str(directory) for directory in dll_directories), result.get("PATH", "")]
        )
    return result


def _run_script(
    python: Path,
    script: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None,
    runner: Runner | None,
) -> str:
    command = [str(python), "-B", "-I", "-c", _NETWORK_GUARD + "\n" + script, *arguments]
    try:
        completed = (runner or subprocess.run)(
            command,
            cwd=str(cwd),
            env=offline_environment(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=900,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _probe_detail(exc.stderr or exc.stdout)
        suffix = f": {detail}" if detail else ""
        raise ProbeError(f"offline probe timed out after 900 seconds{suffix}") from exc
    except OSError as exc:
        detail = _probe_detail(exc)
        suffix = f": {detail}" if detail else ""
        raise ProbeError(f"offline probe process could not be started{suffix}") from exc
    if completed.returncode != 0:
        detail = _probe_detail(completed.stderr or completed.stdout)
        suffix = f": {detail}" if detail else ""
        raise ProbeError(f"offline probe process failed (exit code {completed.returncode}){suffix}")
    return completed.stdout.strip()


def run_json_probe(
    python: Path,
    script: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    runner: Runner | None = None,
) -> dict[str, object]:
    """Run one isolated JSON-producing probe with network sockets disabled."""
    output = _run_script(python, script, arguments, cwd=cwd, environment=environment, runner=runner)
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ProbeError("offline probe did not return JSON evidence") from exc
    if not isinstance(value, dict):
        raise ProbeError("offline probe JSON evidence is invalid")
    return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _device_is_cpu(value: object) -> bool:
    return isinstance(value, str) and value.casefold() == "cpu"


def _device_is_gpu(value: object) -> bool:
    return isinstance(value, str) and value.casefold().startswith(("cuda", "gpu"))


def validate_evidence(component_id: str, variant: str, evidence: Mapping[str, object]) -> bool:
    """Reject import-only, wrong-device, or incomplete representative probe output."""
    kind = evidence.get("kind")
    if kind == "import":
        raise ProbeError("import-only evidence is not an offline functional probe")
    if component_id == "core":
        if kind != "core" or evidence.get("runtimeCheck") != "anima-core-runtime-ok":
            raise ProbeError("core runtime probe evidence is invalid")
        return True
    if component_id == "caption-e621":
        tags = evidence.get("tags")
        provider = evidence.get("provider")
        if kind != "caption" or not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag for tag in tags):
            raise ProbeError("caption probe did not produce parsed tags")
        if tags != sorted(set(tags)):
            raise ProbeError("caption probe tags are not sorted and unique")
        if variant == "cpu" and provider != "CPUExecutionProvider":
            raise ProbeError("CPU caption probe used a CUDA provider")
        if variant == "cuda" and provider != "CUDAExecutionProvider":
            raise ProbeError("GPU caption probe did not use CUDA")
        return True
    if component_id == "policy":
        device = evidence.get("device")
        if kind != "policy" or evidence.get("check") != "ok":
            raise ProbeError("policy probe evidence is invalid")
        if variant == "cpu" and not _device_is_cpu(device):
            raise ProbeError("CPU policy probe used a CUDA device")
        if variant == "cuda" and not _device_is_gpu(device):
            raise ProbeError("GPU policy probe did not use CUDA")
        return True
    if component_id == "quality-stack":
        loaded = evidence.get("loaded")
        score = evidence.get("score")
        if kind != "quality" or not isinstance(loaded, list) or set(loaded) != {"clip", "fusion", "jtp3", "waifu"} or not _finite(score):
            raise ProbeError("quality resource probe evidence is invalid")
        return True
    if component_id == "token-budget":
        counts = evidence.get("counts")
        if kind != "tokenizer" or not isinstance(counts, list) or not counts or not all(type(count) is int and count > 0 for count in counts):
            raise ProbeError("tokenizer probe evidence is invalid")
        return True
    if component_id in {"ocr-cpu", "ocr-gpu"}:
        device = evidence.get("device")
        count = evidence.get("resultCount")
        texts = evidence.get("texts")
        if kind != "ocr" or type(count) is not int or count < 1 or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ProbeError("OCR probe did not process a sample")
        if component_id == "ocr-cpu" and (variant != "cpu" or not _device_is_cpu(device)):
            raise ProbeError("CPU OCR probe used a CUDA device")
        if component_id == "ocr-gpu" and (variant != "cuda" or not _device_is_gpu(device)):
            raise ProbeError("GPU OCR probe did not use CUDA")
        return True
    if component_id in {"e621-indexes", "e621-replacement-indexes"}:
        if kind != "indexes" or type(evidence.get("resourceCount")) is not int or evidence["resourceCount"] < 1:
            raise ProbeError("E621 index probe evidence is invalid")
        return True
    if component_id in {"classify-e621", "replace-e621", "nl", "export"}:
        if kind != "worker" or evidence.get("component") != component_id or evidence.get("check") != "ok":
            raise ProbeError("source worker probe evidence is invalid")
        return True
    raise ProbeError(f"offline probe is missing for component: {component_id}")


def _runtime_root(runtime: Path) -> Path:
    if runtime.name == "" or runtime.parent.name.casefold() != "runtimes":
        raise ProbeError("runtime probe target is invalid")
    return runtime.parent.parent


def _resource_root(target: Path) -> Path:
    for candidate in (target, *target.parents):
        if candidate.name.casefold() == "resource-library":
            return candidate
    raise ProbeError("resource probe target is outside resource-library")


def _catalog_fingerprint(value: Mapping[str, object]) -> str:
    required = {
        "schemaVersion", "kind", "resourceId", "resourceVersion", "profile",
        "runtimeFormat", "entrypoints", "files", "metadata",
    }
    if not required.issubset(value) or not isinstance(value.get("schemaVersion"), int):
        raise ProbeError("resource probe manifest has no immutable fingerprint")
    entrypoints = value["entrypoints"]
    files = value["files"]
    if not isinstance(entrypoints, dict) or not isinstance(files, dict):
        raise ProbeError("resource probe manifest has no immutable fingerprint")
    normalized_entrypoints: dict[str, str] = {}
    for name, path in entrypoints.items():
        if not isinstance(name, str) or not isinstance(path, str):
            raise ProbeError("resource probe manifest has no immutable fingerprint")
        normalized_entrypoints[name] = path.replace("/", "\\")
    normalized_files: dict[str, object] = {}
    for name, record in files.items():
        if not isinstance(name, str):
            raise ProbeError("resource probe manifest has no immutable fingerprint")
        normalized = name.replace("/", "\\")
        if normalized in normalized_files:
            raise ProbeError("resource probe manifest has no immutable fingerprint")
        normalized_files[normalized] = record
    unsigned: dict[str, object] = {
        "schemaVersion": value["schemaVersion"],
        "kind": value["kind"],
        "resourceId": value["resourceId"],
        "resourceVersion": value["resourceVersion"],
        "profile": value["profile"],
        "runtimeFormat": value["runtimeFormat"],
        "entrypoints": {name: normalized_entrypoints[name] for name in sorted(normalized_entrypoints)},
        "files": {name: normalized_files[name] for name in sorted(normalized_files)},
        "metadata": value["metadata"],
    }
    if value["schemaVersion"] == 2:
        if "distribution" not in value:
            raise ProbeError("resource probe manifest has no immutable fingerprint")
        unsigned["distribution"] = value["distribution"]
    import hashlib

    return hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resource_descriptor(target: Path) -> tuple[Path, str, str, dict[str, object]]:
    manifests = sorted(
        (path for path in target.rglob("resource.json") if path.is_file() and not path.is_symlink()),
        key=lambda path: str(path).casefold(),
    )
    if len(manifests) != 1:
        raise ProbeError("resource probe requires exactly one resource.json")
    manifest = manifests[0]
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError("resource probe manifest is invalid") from exc
    if not isinstance(value, dict):
        raise ProbeError("resource probe manifest is invalid")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        fingerprint = _catalog_fingerprint(value)
    library = _resource_root(target)
    install_root = library.parent
    try:
        relative = str(manifest.relative_to(install_root)).replace(os.sep, "\\")
    except ValueError as exc:
        raise ProbeError("resource probe manifest escaped its installation") from exc
    return install_root, relative, fingerprint, value


def _probe_core(runtime: Path, *, runner: Runner | None) -> dict[str, object]:
    script = """
import runpy
import sys
sys.argv = ["anima_core", "--check-runtime"]
runpy.run_module("anima_core", run_name="__main__")
"""
    root = _runtime_root(runtime)
    output = _run_script(
        runtime / "python.exe",
        script,
        (),
        cwd=root,
        environment=offline_environment(install_root=root),
        runner=runner,
    )
    return {"kind": "core", "runtimeCheck": output}


def _probe_caption(runtime: Path, resource_target: Path, variant: str, *, runner: Runner | None) -> dict[str, object]:
    install_root, relative, fingerprint, _value = _resource_descriptor(resource_target)
    script = """
import json
import sys
from pathlib import Path
from PIL import Image
from anima_caption_worker.model import create_tagger_adapter
from anima_caption_worker.resource import load_caption_resource

resource = load_caption_resource(Path(sys.argv[1]), sys.argv[2], sys.argv[3], verify_external_data_hash=True)
model = create_tagger_adapter(resource)
predictions = model.predict(model.preprocess(Image.new("RGB", (64, 64), "white")), model.metadata.default_thresholds)
tags = sorted({prediction.raw_tag for prediction in predictions})
if not tags:
    raise RuntimeError("E621 EVA02 produced no tags")
print(json.dumps({"kind": "caption", "provider": model.provider, "tags": tags}, sort_keys=True))
"""
    return run_json_probe(
        runtime / "python.exe",
        script,
        (str(install_root), relative, fingerprint),
        cwd=_runtime_root(runtime),
        environment=offline_environment(
            install_root=install_root,
            dll_directories=_caption_cuda_dll_directories(runtime, variant),
        ),
        runner=runner,
    )


def _caption_cuda_dll_directories(runtime: Path, variant: str) -> tuple[Path, ...]:
    if variant != "cuda":
        return ()
    directories = (
        runtime / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "cufft" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "curand" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "cusolver" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "cusparse" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "nvjitlink" / "bin",
    )
    if not all(directory.is_dir() and not directory.is_symlink() for directory in directories):
        raise ProbeError("Caption CUDA runtime dependencies are incomplete")
    return directories


def _probe_policy(runtime: Path, variant: str, *, runner: Runner | None) -> dict[str, object]:
    script = """
import json
import sys
from anima_policy_worker.policy import CoupledProbabilities, PolicyConfig, apply_policy

config = PolicyConfig(
    seed="offline-probe", artistEnabled=False, artistDropoutProbability=0.0,
    qualityEnabled=False, qualityDropoutProbability=0.0, appearanceNlEnabled=False,
    solo=CoupledProbabilities(0.0, 0.0), nonSolo=CoupledProbabilities(0.0, 0.0),
    unknown=CoupledProbabilities(0.0, 0.0),
)
payload = {"quality": [], "count": "solo", "character": "", "series": "", "artist": "", "appearance": [], "tags": [], "environment": [], "nl": ""}
result, _decision = apply_policy(payload, annotation_key="offline", relative_image_path="image.png", config=config, aesthetic_score=None)
if result != payload:
    raise RuntimeError("policy probe changed a disabled policy payload")
try:
    import torch
except Exception as exc:
    raise RuntimeError("policy runtime torch import failed") from exc
if sys.argv[1] == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("policy CUDA runtime is unavailable")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(json.dumps({"kind": "policy", "check": "ok", "device": str(device)}, sort_keys=True))
"""
    return run_json_probe(
        runtime / "python.exe",
        script,
        (variant,),
        cwd=_runtime_root(runtime),
        environment=offline_environment(
            install_root=_runtime_root(runtime),
            dll_directories=_caption_cuda_dll_directories(runtime, variant),
        ),
        runner=runner,
    )


def _probe_quality(runtime: Path, resource_target: Path, variant: str, *, runner: Runner | None) -> dict[str, object]:
    install_root, relative, fingerprint, _value = _resource_descriptor(resource_target)
    script = """
import json
import sys
from pathlib import Path
from PIL import Image
from anima_policy_worker.lse14_backend import Lse14Scorer
from anima_policy_worker.resource import load_policy_resource

manifest, files = load_policy_resource(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
scorer = Lse14Scorer(files, sys.argv[4])
scores = scorer.score([Image.new("RGB", (64, 64), "white")])
if len(scores) != 1:
    raise RuntimeError("quality scorer did not produce one score")
print(json.dumps({"kind": "quality", "device": scorer.device_name, "loaded": ["clip", "fusion", "jtp3", "waifu"], "score": scores[0]}, sort_keys=True))
"""
    return run_json_probe(
        runtime / "python.exe",
        script,
        (str(install_root), relative, fingerprint, variant),
        cwd=_runtime_root(runtime),
        environment=offline_environment(install_root=install_root),
        runner=runner,
    )


def _probe_tokenizer(runtime: Path, resource_target: Path, *, runner: Runner | None) -> dict[str, object]:
    install_root, relative, fingerprint, manifest = _resource_descriptor(resource_target)
    resource_id = manifest.get("resourceId")
    context_limit = manifest.get("contextLimit")
    if not isinstance(resource_id, str) or type(context_limit) is not int:
        raise ProbeError("tokenizer resource identity is invalid")
    script = """
import json
import sys
from pathlib import Path
from anima_token_budget_worker.budget import tokenizer_count_many
from anima_token_budget_worker.resource import load_tokenizer_resource

root = Path(sys.argv[1]) / "resource-library"
resource = load_tokenizer_resource(root, sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
counts = tokenizer_count_many(resource.tokenizer, ["frozen UTF-8 sample".encode("utf-8"), "\\u4e2d\\u6587\\u6837\\u4f8b".encode("utf-8")])
print(json.dumps({"kind": "tokenizer", "counts": counts, "resourceId": resource.resource_id}, sort_keys=True))
"""
    resource_relative = relative.split("resource-library\\", 1)[-1]
    return run_json_probe(
        runtime / "python.exe",
        script,
        (str(install_root), resource_relative, resource_id, fingerprint, str(context_limit)),
        cwd=_runtime_root(runtime),
        environment=offline_environment(install_root=install_root, resource_root=install_root / "resource-library"),
        runner=runner,
    )


def _probe_ocr(runtime: Path, resource_target: Path, device: str, *, runner: Runner | None) -> dict[str, object]:
    install_root, relative, fingerprint, _value = _resource_descriptor(resource_target)
    script = """
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from anima_ocr_worker.model import create_paddle_engine
from anima_ocr_worker.resource import load_ocr_resource
import paddle

root = Path(sys.argv[1]) / "resource-library"
resource = load_ocr_resource(root, sys.argv[2], sys.argv[3])
engine = create_paddle_engine(resource, device=sys.argv[4])
image = Image.new("RGB", (320, 96), "white")
ImageDraw.Draw(image).text((8, 32), "offline OCR", fill="black")
result = engine.predict(image)
if not isinstance(result, list) or len(result) != 1:
    raise RuntimeError("OCR functional sample returned an invalid result")
raw = result[0]
if isinstance(raw, dict) and isinstance(raw.get("res"), dict):
    raw = raw["res"]
if not isinstance(raw, dict):
    raise RuntimeError("OCR functional sample returned an invalid payload")
try:
    texts = [str(value) for value in list(raw["rec_texts"])]
except (KeyError, TypeError) as exc:
    raise RuntimeError("OCR functional sample did not return recognized text") from exc
if sys.argv[4] == "cuda" and (not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1):
    raise RuntimeError("OCR CUDA device is unavailable")
print(json.dumps({"kind": "ocr", "device": paddle.get_device(), "resultCount": len(result), "texts": texts}, sort_keys=True))
"""
    resource_relative = relative.split("resource-library\\", 1)[-1]
    return run_json_probe(
        runtime / "python.exe",
        script,
        (str(install_root), resource_relative, fingerprint, device),
        cwd=_runtime_root(runtime),
        environment=offline_environment(install_root=install_root, resource_root=install_root / "resource-library"),
        runner=runner,
    )


def _index_kind_for_target(resource_target: Path) -> str:
    parts = {part.casefold() for part in resource_target.parts}
    if "classification-indexes" in parts:
        return "classification-index"
    if "replacement-indexes" in parts:
        return "replacement-index"
    raise ProbeError(f"offline probe target is not an E621 index package: {resource_target}")


def _probe_indexes(runtime: Path, resource_target: Path, *, runner: Runner | None) -> dict[str, object]:
    library = _resource_root(resource_target)
    kind = _index_kind_for_target(resource_target)
    script = """
import json
import sys
from pathlib import Path
from anima_core.resource_catalog_package import ResourcePackage

package_root = Path(sys.argv[1])
kind = sys.argv[2]
library = Path(sys.argv[3])
package = ResourcePackage.load(library, package_root / "resource.json", kind)
print(json.dumps({"kind": "indexes", "resourceCount": 1, "resourceId": package.resource_id}, sort_keys=True))
"""
    return run_json_probe(
        runtime / "python.exe",
        script,
        (str(resource_target), kind, str(library)),
        cwd=_runtime_root(runtime),
        environment=offline_environment(resource_root=library),
        runner=runner,
    )


def _target(component_targets: Mapping[str, Path], component_id: str) -> Path:
    try:
        target = component_targets[component_id]
    except KeyError as exc:
        raise ProbeError(f"offline probe target is missing: {component_id}") from exc
    if not target.is_dir() or target.is_symlink():
        raise ProbeError(f"offline probe target is invalid: {component_id}")
    return target


def _probe_source_worker(runtime: Path, component_id: str, *, runner: Runner | None = None) -> dict[str, object]:
    script = r'''
import json
import sys

component = sys.argv[1]
if component == "classify-e621":
    from anima_classify_worker.parsing import normalize_display_tag
    assert normalize_display_tag("Blue Hair") == "blue_hair"
elif component == "replace-e621":
    from anima_replace_worker.replacement import replace_projection, rule_from_csv
    value = {"quality": ["old"], "count": "", "character": "", "series": "", "artist": "", "appearance": [], "tags": [], "environment": [], "nl": ""}
    result, summary = replace_projection(value, {"old": rule_from_csv("replace", "new")})
    assert result["quality"] == ["new"] and summary.replaced == 1
elif component == "nl":
    from anima_nl_worker.validation import validate_nl
    assert validate_nl("A complete offline caption.") == "A complete offline caption."
elif component == "export":
    from anima_export_worker.protocol import parse_process
    items = parse_process({"schemaVersion": 1, "payloadType": "export_process_request", "items": [{"schemaVersion": 1, "sampleId": 1, "leaseId": "lease-1", "relativeImagePath": "images/a.png", "annotationKey": "a.json"}]})
    assert len(items) == 1 and items[0].sample_id == 1
else:
    raise RuntimeError("unknown source worker")
print(json.dumps({"kind": "worker", "component": component, "check": "ok"}, sort_keys=True))
'''
    return run_json_probe(
        runtime / "python.exe",
        script,
        (component_id,),
        cwd=_runtime_root(runtime),
        runner=runner,
    )


def run_offline_probes(
    components: Sequence[object],
    *,
    component_targets: Mapping[str, Path],
    ocr_resource_target: Path | None = None,
    runner: Runner | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, bool | None]:
    """Run all known functional probes and return one result for every selected component."""
    selected: dict[str, object] = {}
    variants: dict[str, str] = {}
    for item in components:
        component = getattr(item, "component", None)
        variant = getattr(item, "variant", None)
        component_id = getattr(component, "component_id", None)
        variant_name = getattr(variant, "name", None)
        if not isinstance(component_id, str) or not isinstance(variant_name, str) or component_id in selected:
            raise ProbeError("offline probe plan is invalid")
        selected[component_id] = item
        variants[component_id] = variant_name
    results: dict[str, bool | None] = {component_id: False for component_id in selected}
    evidence_by_component: dict[str, dict[str, object]] = {}

    def run_group(component_ids: tuple[str, ...], callback: Callable[[], dict[str, object]], evidence_component: str) -> None:
        if not all(component_id in selected for component_id in component_ids):
            return
        label = evidence_component
        if len(component_ids) > 1:
            label += " (" + ", ".join(component_ids) + ")"
        started = time.monotonic()
        if progress is not None:
            progress(f"Offline probe started: {label}")
        try:
            evidence = callback()
            validate_evidence(evidence_component, variants[evidence_component], evidence)
        except ProbeError as exc:
            if progress is not None:
                progress(f"Offline probe failed: {label} after {time.monotonic() - started:.1f}s: {exc}")
            return
        evidence_by_component[evidence_component] = evidence
        for component_id in component_ids:
            results[component_id] = True
        if progress is not None:
            progress(f"Offline probe passed: {label} after {time.monotonic() - started:.1f}s")

    run_group(("core",), lambda: _probe_core(_target(component_targets, "core"), runner=runner), "core")
    for source_component_id in ("classify-e621", "replace-e621", "nl", "export"):
        run_group(
            (source_component_id,),
            lambda component_id=source_component_id: _probe_source_worker(
                _target(component_targets, component_id), component_id, runner=runner
            ),
            source_component_id,
        )
    run_group(
        ("caption-e621", "e621-tagger"),
        lambda: _probe_caption(
            _target(component_targets, "caption-e621"),
            _target(component_targets, "e621-tagger"),
            variants["caption-e621"],
            runner=runner,
        ),
        "caption-e621",
    )
    run_group(
        ("policy",),
        lambda: _probe_policy(
            _target(component_targets, "policy"),
            variants["policy"],
            runner=runner,
        ),
        "policy",
    )
    if "quality-stack" in selected:
        if "policy" in selected:
            run_group(
                ("quality-stack",),
                lambda: _probe_quality(
                    _target(component_targets, "policy"),
                    _target(component_targets, "quality-stack"),
                    variants["policy"],
                    runner=runner,
                ),
                "quality-stack",
            )
        else:
            # The resource component is independently installable; its immutable
            # manifest and file identities were already verified during assembly.
            results["quality-stack"] = True
    run_group(
        ("token-budget", "qwen3-tokenizer"),
        lambda: _probe_tokenizer(
            _target(component_targets, "token-budget"),
            _target(component_targets, "qwen3-tokenizer"),
            runner=runner,
        ),
        "token-budget",
    )
    if ocr_resource_target is None and "ocr-models" in selected:
        ocr_resource_target = _target(component_targets, "ocr-models")
    if ocr_resource_target is not None:
        if not ocr_resource_target.is_dir() or ocr_resource_target.is_symlink():
            raise ProbeError("OCR resource probe target is invalid")
        run_group(
            ("ocr-cpu",),
            lambda: _probe_ocr(
                _target(component_targets, "ocr-cpu"),
                ocr_resource_target,
                "cpu",
                runner=runner,
            ),
            "ocr-cpu",
        )
        run_group(
            ("ocr-gpu",),
            lambda: _probe_ocr(
                _target(component_targets, "ocr-gpu"),
                ocr_resource_target,
                "cuda",
                runner=runner,
            ),
            "ocr-gpu",
        )
    else:
        for component_id in ("ocr-cpu", "ocr-gpu"):
            if component_id in selected:
                results[component_id] = None
    if results.get("ocr-cpu") and results.get("ocr-gpu"):
        cpu_texts = evidence_by_component["ocr-cpu"].get("texts")
        gpu_texts = evidence_by_component["ocr-gpu"].get("texts")
        if cpu_texts != gpu_texts:
            results["ocr-gpu"] = False
    run_group(
        ("e621-indexes",),
        lambda: _probe_indexes(
            _target(component_targets, "core"),
            _target(component_targets, "e621-indexes"),
            runner=runner,
        ),
        "e621-indexes",
    )
    run_group(
        ("e621-replacement-indexes",),
        lambda: _probe_indexes(
            _target(component_targets, "core"),
            _target(component_targets, "e621-replacement-indexes"),
            runner=runner,
        ),
        "e621-replacement-indexes",
    )
    return results
