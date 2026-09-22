from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workers" / "policy" / "src"))

from anima_policy_worker.protocol import parse_process
from anima_policy_worker.resource import load_policy_resource
from anima_policy_worker.worker import PolicyWorker


def _hello(
    dataset: Path,
    overlay: Path,
    *,
    drop_nl: float,
    drop_appearance: float,
    artist_enabled: bool = True,
    appearance_nl_enabled: bool = True,
    artist_root_name: str = "dataset",
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "payloadType": "policy_hello_request",
        "jobId": "job-policy",
        "configHash": "a" * 64,
        "datasetRoot": str(dataset),
        "overlayRoot": str(overlay),
        "artistRootName": artist_root_name,
        "resourceManifestRelativePath": None,
        "resourceFingerprint": None,
        "policy": {
            "policyVersion": "dataset-batch-policy-v1",
            "seed": "worker-test-seed",
            "artist": {"enabled": artist_enabled, "dropoutProbability": 0.0},
            "quality": {
                "enabled": False,
                "dropoutProbability": 0.0,
                "device": "auto",
                "batchSize": 4,
                "resourceId": "lse14-scorer-5k-v1",
            },
            "appearanceNl": {
                "enabled": appearance_nl_enabled,
                "solo": {"dropNl": drop_nl, "dropAppearance": drop_appearance},
                "nonSolo": {"dropNl": drop_nl, "dropAppearance": drop_appearance},
                "unknown": {"dropNl": drop_nl, "dropAppearance": drop_appearance},
            },
        },
    }


def _business(*, appearance: list[str], nl: str) -> dict[str, object]:
    return {
        "quality": [],
        "count": "solo",
        "character": "amy_rose",
        "series": "",
        "artist": "",
        "appearance": appearance,
        "tags": ["smile"],
        "environment": ["outdoors"],
        "nl": nl,
    }


class PolicyWorkerTests(unittest.TestCase):
    def test_catalog_resource_returns_its_verified_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "dropout-models" / "test-model"
            entrypoints = {
                "clip": "clip/model.pt",
                "fusion": "fusion/model.safetensors",
                "jtp3": "jtp3/model.safetensors",
                "waifu": "waifu/model.safetensors",
            }
            files: dict[str, dict[str, object]] = {}
            for role, relative in entrypoints.items():
                target = package / Path(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = f"{role}-model".encode("ascii")
                target.write_bytes(content)
                normalized = relative.replace("/", "\\")
                files[normalized] = {
                    "sizeBytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            manifest = {
                "schemaVersion": 1,
                "kind": "dropout-model",
                "resourceId": "test-model",
                "resourceVersion": "test-v1",
                "profile": "e621",
                "displayName": {"zh-CN": "Test", "en": "Test"},
                "description": {"zh-CN": "Test", "en": "Test"},
                "runtimeFormat": "lse14-scorer-5k-v1",
                "entrypoints": {role: relative.replace("/", "\\") for role, relative in entrypoints.items()},
                "files": files,
                "metadata": {},
                "documentation": [],
            }
            unsigned = {
                key: manifest[key]
                for key in (
                    "schemaVersion", "kind", "resourceId", "resourceVersion", "profile",
                    "runtimeFormat", "entrypoints", "files", "metadata",
                )
            }
            fingerprint = hashlib.sha256(
                json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            (package / "resource.json").write_text(json.dumps(manifest), encoding="utf-8")

            loaded, paths = load_policy_resource(
                root, r"dropout-models\test-model\resource.json", fingerprint
            )
            self.assertEqual(fingerprint, loaded["fingerprint"])
            self.assertEqual(4, len(paths))

    def _run(self, payload: dict[str, object], *, drop_nl: float, drop_appearance: float) -> tuple[dict[str, object], dict[str, object]]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            artist = dataset / "1_Crow (Siranui)"
            artist.mkdir()
            image = artist / "image.png"
            image.write_bytes(b"fingerprint-only")
            source_json = artist / "image.json"
            source_json.write_text(json.dumps(payload), encoding="utf-8")
            overlay = root / ".anima-job-policy"
            overlay.mkdir()
            (overlay / "overlay-manifest.json").write_text(
                json.dumps({"schemaVersion": 1, "jobId": "job-policy", "datasetRoot": str(dataset)}),
                encoding="utf-8",
            )
            worker = PolicyWorker()
            initialized = worker.initialize(
                _hello(dataset, overlay, drop_nl=drop_nl, drop_appearance=drop_appearance),
                install_root=root,
            )
            self.assertEqual(0, initialized["modelLoadCount"])
            info = image.stat()
            items = parse_process({
                "schemaVersion": 1,
                "payloadType": "policy_process_request",
                "items": [{
                    "schemaVersion": 1,
                    "sampleId": 1,
                    "leaseId": "lease-policy-1",
                    "relativeImagePath": r"1_Crow (Siranui)\image.png",
                    "annotationKey": r"1_Crow (Siranui)\image",
                    "imageSize": info.st_size,
                    "imageMtimeNs": info.st_mtime_ns,
                    "imageFileId": None,
                }],
            })
            outcome = worker.process(items)["outcomes"][0]
            self.assertEqual("prepared", outcome["status"])
            prepared = overlay / Path(str(outcome["preparedRelativePath"]).replace("\\", "/"))
            result = json.loads(prepared.read_text(encoding="utf-8"))
            self.assertEqual(payload, json.loads(source_json.read_text(encoding="utf-8")))
            return result, outcome

    def test_hundred_percent_nl_dropout_preserves_appearance(self) -> None:
        result, outcome = self._run(
            _business(appearance=["white hair"], nl="A person smiles."),
            drop_nl=1.0,
            drop_appearance=0.0,
        )
        self.assertEqual("drop_nl", outcome["decision"]["appearanceNlAction"])
        self.assertEqual(["white hair"], result["appearance"])
        self.assertEqual(["smile"], result["tags"])
        self.assertEqual("", result["nl"])

    def test_hundred_percent_appearance_dropout_preserves_nl(self) -> None:
        result, outcome = self._run(
            _business(appearance=["white hair"], nl="A person smiles."),
            drop_nl=0.0,
            drop_appearance=1.0,
        )
        self.assertEqual("drop_appearance", outcome["decision"]["appearanceNlAction"])
        self.assertEqual([], result["appearance"])
        self.assertEqual(["smile"], result["tags"])
        self.assertEqual("A person smiles.", result["nl"])

    def test_empty_appearance_protects_existing_nl(self) -> None:
        result, outcome = self._run(
            _business(appearance=[], nl="A person smiles."),
            drop_nl=1.0,
            drop_appearance=0.0,
        )
        self.assertEqual("unchanged", outcome["decision"]["appearanceNlAction"])
        self.assertEqual([], result["appearance"])
        self.assertEqual("A person smiles.", result["nl"])

    def test_disabled_quality_does_not_open_an_invalid_image(self) -> None:
        result, outcome = self._run(
            _business(appearance=["white hair"], nl="A person smiles."),
            drop_nl=0.0,
            drop_appearance=0.0,
        )
        self.assertEqual("prepared", outcome["status"])
        self.assertEqual([], result["quality"])

    def test_dropout_uses_preflight_validated_artist_path_without_revalidating(self) -> None:
        with patch(
            "anima_policy_worker.policy.artist_from_image_path",
            side_effect=AssertionError("artist folder must only be validated during import preflight"),
        ):
            result, outcome = self._run(
                _business(appearance=["white hair"], nl="A person smiles."),
                drop_nl=0.0,
                drop_appearance=0.0,
            )
        self.assertEqual("prepared", outcome["status"])
        self.assertEqual("@Crow (Siranui)", result["artist"])

    def test_disabled_artist_does_not_require_an_artist_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            image = dataset / "image.png"
            image.write_bytes(b"not-read-when-quality-is-disabled")
            source_json = dataset / "image.json"
            source_json.write_text(json.dumps(_business(appearance=["white hair"], nl="A person smiles.")), encoding="utf-8")
            overlay = root / ".anima-job-policy"
            overlay.mkdir()
            (overlay / "overlay-manifest.json").write_text(
                json.dumps({"schemaVersion": 1, "jobId": "job-policy", "datasetRoot": str(dataset)}),
                encoding="utf-8",
            )
            worker = PolicyWorker()
            worker.initialize(
                _hello(dataset, overlay, drop_nl=0.0, drop_appearance=0.0, artist_enabled=False),
                install_root=root,
            )
            info = image.stat()
            item = parse_process({
                "schemaVersion": 1,
                "payloadType": "policy_process_request",
                "items": [{
                    "schemaVersion": 1,
                    "sampleId": 1,
                    "leaseId": "lease-policy-1",
                    "relativeImagePath": "image.png",
                    "annotationKey": "image",
                    "imageSize": info.st_size,
                    "imageMtimeNs": info.st_mtime_ns,
                    "imageFileId": None,
                }],
            })
            outcome = worker.process(item)["outcomes"][0]
            self.assertEqual("prepared", outcome["status"])

    def test_flat_dataset_uses_validated_dataset_root_name_for_artist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "10_uiokv"
            dataset.mkdir()
            image = dataset / "image.png"
            image.write_bytes(b"fingerprint-only")
            (dataset / "image.json").write_text(
                json.dumps(_business(appearance=["white hair"], nl="A person smiles.")),
                encoding="utf-8",
            )
            overlay = root / ".anima-job-policy"
            overlay.mkdir()
            (overlay / "overlay-manifest.json").write_text(
                json.dumps({"schemaVersion": 1, "jobId": "job-policy", "datasetRoot": str(dataset)}),
                encoding="utf-8",
            )
            try:
                worker = PolicyWorker()
                worker.initialize(
                    _hello(
                        dataset,
                        overlay,
                        drop_nl=0.0,
                        drop_appearance=0.0,
                        artist_root_name="10_uiokv",
                    ),
                    install_root=root,
                )
                info = image.stat()
                items = parse_process({
                    "schemaVersion": 1,
                    "payloadType": "policy_process_request",
                    "items": [{
                        "schemaVersion": 1,
                        "sampleId": 1,
                        "leaseId": "lease-policy-flat",
                        "relativeImagePath": "image.png",
                        "annotationKey": "image",
                        "imageSize": info.st_size,
                        "imageMtimeNs": info.st_mtime_ns,
                        "imageFileId": None,
                    }],
                })
                outcome = worker.process(items)["outcomes"][0]
            except Exception as exc:
                self.fail(f"flat named datasets must be accepted: {exc}")
            self.assertEqual("prepared", outcome["status"])
            prepared = overlay / Path(str(outcome["preparedRelativePath"]).replace("\\", "/"))
            self.assertEqual("@uiokv", json.loads(prepared.read_text(encoding="utf-8"))["artist"])


if __name__ == "__main__":
    unittest.main()
