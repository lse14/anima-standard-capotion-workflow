from __future__ import annotations

import math
import re
import re
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Literal

from .policy import CoupledProbabilities, POLICY_VERSION, PolicyConfig, PolicyError


MAX_BATCH_SIZE = 16
MAX_PATH_BYTES = 16_384
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolicyProtocolError(ValueError):
    pass


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PolicyProtocolError(f"{field} must be an object")
    return value


def _exact_keys(value: dict[str, object], required: set[str], field: str) -> None:
    if set(value) != required:
        raise PolicyProtocolError(f"{field} fields are invalid")


def _string(value: object, field: str, *, max_bytes: int, nonblank: bool = True) -> str:
    if not isinstance(value, str) or "\x00" in value or len(value.encode("utf-8")) > max_bytes:
        raise PolicyProtocolError(f"{field} is invalid")
    if nonblank and not value:
        raise PolicyProtocolError(f"{field} must not be empty")
    return value


def _identifier(value: object, field: str) -> str:
    result = _string(value, field, max_bytes=128)
    if not IDENTIFIER.fullmatch(result):
        raise PolicyProtocolError(f"{field} is not a valid identifier")
    return result


def _absolute_path(value: object, field: str) -> str:
    result = _string(value, field, max_bytes=MAX_PATH_BYTES)
    if not PureWindowsPath(result).is_absolute():
        raise PolicyProtocolError(f"{field} must be an absolute Windows path")
    return result


def _relative_path(value: object, field: str) -> str:
    result = _string(value, field, max_bytes=MAX_PATH_BYTES)
    path = PureWindowsPath(result.replace("/", "\\"))
    if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
        raise PolicyProtocolError(f"{field} must be a safe relative path")
    return str(path)


def _sha256(value: object, field: str) -> str:
    result = _string(value, field, max_bytes=64)
    if not SHA256.fullmatch(result):
        raise PolicyProtocolError(f"{field} must be a lowercase SHA-256")
    return result


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyProtocolError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise PolicyProtocolError(f"{field} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class QualityConfigV1:
    enabled: bool
    dropoutProbability: float
    device: Literal["auto", "cuda", "cpu"]
    batchSize: int
    resourceId: str | None


@dataclass(frozen=True)
class PolicyHelloV1:
    jobId: str
    configHash: str
    datasetRoot: str
    overlayRoot: str
    artistRootName: str
    resourceManifestRelativePath: str | None
    resourceFingerprint: str | None
    quality: QualityConfigV1
    policy: PolicyConfig


@dataclass(frozen=True)
class PolicyWorkItemV1:
    sampleId: int
    leaseId: str
    relativeImagePath: str
    annotationKey: str
    imageSize: int
    imageMtimeNs: int
    imageFileId: str | None


def _coupled(value: object, field: str) -> CoupledProbabilities:
    item = _object(value, field)
    _exact_keys(item, {"dropNl", "dropAppearance"}, field)
    try:
        return CoupledProbabilities(
            _probability(item["dropNl"], f"{field}.dropNl"),
            _probability(item["dropAppearance"], f"{field}.dropAppearance"),
        )
    except PolicyError as exc:
        raise PolicyProtocolError(str(exc)) from exc


def parse_hello(value: object) -> PolicyHelloV1:
    item = _object(value, "policy hello")
    _exact_keys(
        item,
        {
            "schemaVersion", "payloadType", "jobId", "configHash", "datasetRoot", "overlayRoot",
            "artistRootName", "resourceManifestRelativePath", "resourceFingerprint", "policy",
        },
        "policy hello",
    )
    if item["schemaVersion"] != 1 or item["payloadType"] != "policy_hello_request":
        raise PolicyProtocolError("policy hello identity is invalid")
    raw_policy = _object(item["policy"], "policy")
    _exact_keys(raw_policy, {"policyVersion", "seed", "artist", "quality", "appearanceNl"}, "policy")
    artist = _object(raw_policy["artist"], "artist")
    quality = _object(raw_policy["quality"], "quality")
    appearance_nl = _object(raw_policy["appearanceNl"], "appearanceNl")
    artist_required = {"enabled", "dropoutProbability"}
    if not artist_required.issubset(artist) or set(artist) - {*artist_required, "addAtPrefix"}:
        raise PolicyProtocolError("artist fields are invalid")
    quality_required = {"enabled", "dropoutProbability", "device", "batchSize"}
    if not quality_required.issubset(quality) or set(quality) - {*quality_required, "resourceId"}:
        raise PolicyProtocolError("quality fields are invalid")
    _exact_keys(appearance_nl, {"enabled", "solo", "nonSolo", "unknown"}, "appearanceNl")
    for field, raw in (("artist.enabled", artist["enabled"]), ("quality.enabled", quality["enabled"]), ("appearanceNl.enabled", appearance_nl["enabled"])):
        if type(raw) is not bool:
            raise PolicyProtocolError(f"{field} must be boolean")
    if "addAtPrefix" in artist and type(artist["addAtPrefix"]) is not bool:
        raise PolicyProtocolError("artist.addAtPrefix must be boolean")
    device = quality["device"]
    if device not in {"auto", "cuda", "cpu"}:
        raise PolicyProtocolError("quality.device is invalid")
    batch_size = quality["batchSize"]
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise PolicyProtocolError(f"quality.batchSize must be between 1 and {MAX_BATCH_SIZE}")
    resource_id = quality.get("resourceId")
    if resource_id is not None and (
        not isinstance(resource_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", resource_id) is None
    ):
        raise PolicyProtocolError("quality.resourceId is invalid")
    resource_path = item["resourceManifestRelativePath"]
    resource_fingerprint = item["resourceFingerprint"]
    if quality["enabled"]:
        if resource_id is None:
            raise PolicyProtocolError("enabled quality requires resourceId")
        resource_path = _relative_path(resource_path, "resourceManifestRelativePath")
        resource_fingerprint = _sha256(resource_fingerprint, "resourceFingerprint")
    elif resource_path is not None or resource_fingerprint is not None:
        raise PolicyProtocolError("disabled quality must not select model resources")
    if raw_policy["policyVersion"] != POLICY_VERSION:
        raise PolicyProtocolError("policyVersion is invalid")
    try:
        policy = PolicyConfig(
            seed=_string(raw_policy["seed"], "seed", max_bytes=256),
            artistEnabled=artist["enabled"],
            artistDropoutProbability=_probability(artist["dropoutProbability"], "artist.dropoutProbability"),
            qualityEnabled=quality["enabled"],
            qualityDropoutProbability=_probability(quality["dropoutProbability"], "quality.dropoutProbability"),
            appearanceNlEnabled=appearance_nl["enabled"],
            solo=_coupled(appearance_nl["solo"], "appearanceNl.solo"),
            nonSolo=_coupled(appearance_nl["nonSolo"], "appearanceNl.nonSolo"),
            unknown=_coupled(appearance_nl["unknown"], "appearanceNl.unknown"),
            artistAddAtPrefix=artist.get("addAtPrefix", True),
        )
    except PolicyError as exc:
        raise PolicyProtocolError(str(exc)) from exc
    return PolicyHelloV1(
        jobId=_identifier(item["jobId"], "jobId"),
        configHash=_sha256(item["configHash"], "configHash"),
        datasetRoot=_absolute_path(item["datasetRoot"], "datasetRoot"),
        overlayRoot=_absolute_path(item["overlayRoot"], "overlayRoot"),
        artistRootName=_string(item["artistRootName"], "artistRootName", max_bytes=512),
        resourceManifestRelativePath=resource_path,
        resourceFingerprint=resource_fingerprint,
        quality=QualityConfigV1(
            quality["enabled"], policy.qualityDropoutProbability, device, batch_size, resource_id
        ),
        policy=policy,
    )


def parse_process(value: object) -> tuple[PolicyWorkItemV1, ...]:
    item = _object(value, "policy process")
    _exact_keys(item, {"schemaVersion", "payloadType", "items"}, "policy process")
    if item["schemaVersion"] != 1 or item["payloadType"] != "policy_process_request":
        raise PolicyProtocolError("policy process identity is invalid")
    values = item["items"]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_BATCH_SIZE:
        raise PolicyProtocolError(f"policy batch must contain 1 to {MAX_BATCH_SIZE} items")
    results: list[PolicyWorkItemV1] = []
    sample_ids: set[int] = set()
    lease_ids: set[str] = set()
    for index, raw in enumerate(values):
        work = _object(raw, f"items[{index}]")
        _exact_keys(
            work,
            {"schemaVersion", "sampleId", "leaseId", "relativeImagePath", "annotationKey", "imageSize", "imageMtimeNs", "imageFileId"},
            f"items[{index}]",
        )
        sample_id = work["sampleId"]
        image_size = work["imageSize"]
        image_mtime = work["imageMtimeNs"]
        if (
            work["schemaVersion"] != 1
            or type(sample_id) is not int
            or sample_id < 1
            or type(image_size) is not int
            or image_size < 1
            or type(image_mtime) is not int
            or image_mtime < 0
        ):
            raise PolicyProtocolError("work item identity is invalid")
        image_file_id = work["imageFileId"]
        if image_file_id is not None:
            image_file_id = _string(image_file_id, "imageFileId", max_bytes=512)
        lease_id = _identifier(work["leaseId"], "leaseId")
        if sample_id in sample_ids or lease_id in lease_ids:
            raise PolicyProtocolError("policy batch contains duplicate identities")
        sample_ids.add(sample_id)
        lease_ids.add(lease_id)
        results.append(PolicyWorkItemV1(
            sample_id,
            lease_id,
            _relative_path(work["relativeImagePath"], "relativeImagePath"),
            _relative_path(work["annotationKey"], "annotationKey"),
            image_size,
            image_mtime,
            image_file_id,
        ))
    return tuple(results)
