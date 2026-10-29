from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from .model import AestheticScorer, load_lse14_scorer
from .policy import PolicyError, apply_policy, lift_aesthetic_score
from .protocol import MAX_BATCH_SIZE, PolicyHelloV1, PolicyWorkItemV1, parse_hello
from .resource import PolicyResourceError, load_policy_resource


MAX_JSON_BYTES = 1_048_576
OVERLAY_MARKER = "overlay-manifest.json"
SAFE_LEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class PolicyWorkerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _within(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative.replace("\\", os.sep))).resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise PolicyWorkerError("policy_path_invalid", "policy path escaped its root")
    return candidate


def _image_file_id(information: os.stat_result) -> str:
    return f"{getattr(information, 'st_dev', 0)}:{getattr(information, 'st_ino', 0)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(path)


class PolicyWorker:
    def __init__(
        self,
        *,
        scorer_loader: Callable[[dict[str, Path], str], AestheticScorer] = load_lse14_scorer,
    ) -> None:
        self.hello: PolicyHelloV1 | None = None
        self.dataset_root: Path | None = None
        self.overlay_root: Path | None = None
        self.scorer: AestheticScorer | None = None
        self.scorer_loader = scorer_loader
        self.resource_fingerprint: str | None = None

    def initialize(self, payload: object, *, install_root: Path) -> dict[str, object]:
        if self.hello is not None:
            raise PolicyWorkerError("policy_protocol_violation", "policy worker was initialized twice")
        try:
            hello = parse_hello(payload)
            dataset_root = Path(hello.datasetRoot).resolve(strict=True)
            overlay_root = Path(hello.overlayRoot).resolve(strict=True)
            marker = json.loads((overlay_root / OVERLAY_MARKER).read_text(encoding="utf-8"))
            if marker.get("schemaVersion") != 1 or marker.get("jobId") != hello.jobId:
                raise PolicyWorkerError("policy_overlay_invalid", "policy overlay marker does not match the job")
            if Path(str(marker.get("datasetRoot", ""))).resolve(strict=True) != dataset_root:
                raise PolicyWorkerError("policy_overlay_invalid", "policy overlay does not match the dataset")
            if os.path.commonpath((str(dataset_root.parent), str(overlay_root))) != str(dataset_root.parent):
                raise PolicyWorkerError("policy_overlay_invalid", "policy overlay is outside the dataset parent")
            scorer = None
            resource_fingerprint = None
            if hello.quality.enabled:
                assert hello.resourceManifestRelativePath is not None and hello.resourceFingerprint is not None
                manifest, files = load_policy_resource(
                    install_root,
                    hello.resourceManifestRelativePath,
                    hello.resourceFingerprint,
                )
                if manifest.get("resourceId") != hello.quality.resourceId:
                    raise PolicyWorkerError("policy_model_invalid", "policy model id does not match hello")
                scorer = self.scorer_loader(files, hello.quality.device)
                if scorer.load_count != 1:
                    raise PolicyWorkerError("policy_model_invalid", "policy scorer did not load exactly once")
                resource_fingerprint = str(manifest["fingerprint"])
        except PolicyWorkerError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, PolicyResourceError, ValueError, RuntimeError) as exc:
            raise PolicyWorkerError("policy_initialization_failed", str(exc)) from exc
        self.hello = hello
        self.dataset_root = dataset_root
        self.overlay_root = overlay_root
        self.scorer = scorer
        self.resource_fingerprint = resource_fingerprint
        return {
            "schemaVersion": 1,
            "payloadType": "policy_hello_result",
            "ready": True,
            "qualityEnabled": hello.quality.enabled,
            "device": scorer.device_name if scorer is not None else None,
            "modelLoadCount": scorer.load_count if scorer is not None else 0,
            "resourceFingerprint": resource_fingerprint,
        }

    def _json_path(self, item: PolicyWorkItemV1) -> Path:
        assert self.dataset_root is not None and self.overlay_root is not None
        overlay = _within(self.overlay_root / "annotations", item.annotationKey + ".json")
        if overlay.is_file():
            return overlay
        return _within(self.dataset_root, item.annotationKey + ".json")

    def _read_json(self, item: PolicyWorkItemV1) -> dict[str, object]:
        path = self._json_path(item)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise PolicyWorkerError("policy_json_missing", "working JSON does not exist") from exc
        except OSError as exc:
            raise PolicyWorkerError("policy_json_read_failed", "working JSON cannot be read") from exc
        if not data or len(data) > MAX_JSON_BYTES:
            raise PolicyWorkerError("policy_json_invalid", "working JSON is blank or exceeds 1 MiB")
        try:
            value = json.loads(data.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PolicyWorkerError("policy_json_invalid", "working JSON is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise PolicyWorkerError("policy_json_invalid", "working JSON must be an object")
        return value

    def _open_image(self, item: PolicyWorkItemV1) -> Image.Image:
        assert self.dataset_root is not None
        path = _within(self.dataset_root, item.relativeImagePath)
        try:
            information = path.stat()
            if information.st_size != item.imageSize or information.st_mtime_ns != item.imageMtimeNs:
                raise PolicyWorkerError("policy_source_fingerprint_mismatch", "image size or timestamp changed")
            if item.imageFileId is not None and _image_file_id(information) != item.imageFileId:
                raise PolicyWorkerError("policy_source_fingerprint_mismatch", "image file identity changed")
            image = Image.open(path)
            if int(getattr(image, "n_frames", 1)) != 1:
                image.close()
                raise PolicyWorkerError("policy_image_decode_failed", "multi-frame images are unsupported")
            image.seek(0)
            prepared = ImageOps.exif_transpose(image)
            prepared.load()
            if prepared is not image:
                image.close()
            return prepared
        except PolicyWorkerError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise PolicyWorkerError("policy_image_decode_failed", "image decoding failed") from exc

    def _prepared_path(self, lease_id: str) -> tuple[Path, str]:
        assert self.overlay_root is not None
        if not SAFE_LEASE.fullmatch(lease_id):
            raise PolicyWorkerError("policy_protocol_violation", "lease id is unsafe")
        relative = f"prepared\\dropout\\{lease_id}.json"
        return _within(self.overlay_root, relative), relative

    @staticmethod
    def _issue(item: PolicyWorkItemV1, error: PolicyWorkerError) -> dict[str, object]:
        repair = "classify" if error.code in {"policy_json_missing", "policy_json_invalid", "policy_json_read_failed"} else "dropout"
        return {
            "schemaVersion": 1,
            "status": "issue",
            "sampleId": item.sampleId,
            "leaseId": item.leaseId,
            "relativeImagePath": item.relativeImagePath,
            "code": error.code,
            "message": str(error),
            "retriable": True,
            "repairStartModule": repair,
        }

    def process(self, items: Sequence[PolicyWorkItemV1]) -> dict[str, object]:
        if self.hello is None or self.dataset_root is None or self.overlay_root is None:
            raise PolicyWorkerError("policy_protocol_violation", "policy worker is not initialized")
        if not items or len(items) > MAX_BATCH_SIZE:
            raise PolicyWorkerError("policy_protocol_violation", "policy batch exceeds the protocol limit")

        payloads: dict[int, dict[str, object]] = {}
        images: dict[int, Image.Image] = {}
        outcomes: dict[int, dict[str, object]] = {}
        for item in items:
            try:
                payloads[item.sampleId] = self._read_json(item)
                if self.hello.quality.enabled:
                    images[item.sampleId] = self._open_image(item)
            except PolicyWorkerError as exc:
                outcomes[item.sampleId] = self._issue(item, exc)

        scores: dict[int, float] = {}
        score_items = [item for item in items if item.sampleId in images and item.sampleId not in outcomes]
        if score_items:
            if self.scorer is None:
                raise PolicyWorkerError("policy_protocol_violation", "quality scorer is unavailable")
            micro_batch_size = self.hello.quality.batchSize
            try:
                for start in range(0, len(score_items), micro_batch_size):
                    micro_items = score_items[start : start + micro_batch_size]
                    try:
                        values = self.scorer.score([images[item.sampleId] for item in micro_items])
                        if len(values) != len(micro_items):
                            raise RuntimeError("policy scorer returned the wrong result count")
                        # The offset is applied to the inferred score itself, not to the
                        # tier cut, so the reported aestheticScore and the frozen tiers
                        # both see the same raised value.
                        scores.update(
                            (item.sampleId, lift_aesthetic_score(score))
                            for item, score in zip(micro_items, values, strict=True)
                        )
                    except Exception as exc:
                        error = PolicyWorkerError("policy_inference_failed", f"aesthetic inference failed: {exc}")
                        for item in micro_items:
                            outcomes[item.sampleId] = self._issue(item, error)
            finally:
                for image in images.values():
                    image.close()

        for item in items:
            if item.sampleId in outcomes:
                continue
            try:
                result, decision = apply_policy(
                    payloads[item.sampleId],
                    annotation_key=item.annotationKey,
                    relative_image_path=item.relativeImagePath,
                    config=self.hello.policy,
                    aesthetic_score=scores.get(item.sampleId),
                    artist_root_name=self.hello.artistRootName,
                )
                data = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                if len(data) > MAX_JSON_BYTES:
                    raise PolicyWorkerError("policy_json_invalid", "policy output exceeds 1 MiB")
                prepared, relative = self._prepared_path(item.leaseId)
                digest = _atomic_write(prepared, data)
                outcomes[item.sampleId] = {
                    "schemaVersion": 1,
                    "status": "prepared",
                    "sampleId": item.sampleId,
                    "leaseId": item.leaseId,
                    "relativeImagePath": item.relativeImagePath,
                    "preparedRelativePath": relative,
                    "sha256": digest,
                    "aestheticScore": scores.get(item.sampleId),
                    "quality": result["quality"],
                    "decision": asdict(decision),
                }
            except PolicyWorkerError as exc:
                outcomes[item.sampleId] = self._issue(item, exc)
            except (PolicyError, OSError, ValueError) as exc:
                outcomes[item.sampleId] = self._issue(item, PolicyWorkerError("policy_json_invalid", str(exc)))

        return {
            "schemaVersion": 1,
            "payloadType": "policy_batch_result",
            "outcomes": [outcomes[item.sampleId] for item in items],
            "modelLoadCount": self.scorer.load_count if self.scorer is not None else 0,
        }
