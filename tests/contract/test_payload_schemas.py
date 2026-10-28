"""F45: contracts/schemas is validated against real payloads instead of being decoration.

Before this test the only readers of contracts/schemas asserted `$id` strings, so every
schema could drift away from the code that produces the payloads without any failure.
"""
from __future__ import annotations

import json
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_core.caption_protocol import (
    CaptionHelloResultV1,
    CaptionIssueResultV1,
    CaptionProcessRequestV1,
    CaptionProcessResultV1,
    CaptionResultV1,
    CaptionTagV1,
    CaptionWorkItemV1,
)
from anima_core.classify_protocol import (
    ClassifyCountDecisionV1,
    ClassifyHelloResultV1,
    ClassifyIssueResultV1,
    ClassifyProcessRequestV1,
    ClassifyProcessResultV1,
    ClassifyProjectionV1,
    ClassifyResultV1,
    ClassifyWorkItemV1,
)
from anima_core.contracts import (
    JobConfig,
    validate_job_config,
)
from anima_core.job_preflight import config_from_dict
from tests.worker_test_support import worker_hello_payload

SCHEMAS = ROOT / "contracts" / "schemas"
INSTALL_ROOT = ROOT / ".runtime-build"
_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float), "null": type(None),
}


def _equal(value: object, expected: object) -> bool:
    # True == 1 in Python; JSON Schema treats booleans and numbers as different types.
    if isinstance(value, bool) != isinstance(expected, bool):
        return False
    return value == expected


def _matches_type(value: object, name: str) -> bool:
    if name in ("integer", "number") and isinstance(value, bool):
        return False
    if name == "integer":
        return isinstance(value, int)
    if name == "number":
        return isinstance(value, (int, float)) and math.isfinite(float(value))
    return isinstance(value, _TYPES[name])


def validate(instance: object, schema: object, root: dict, path: str = "$") -> list[str]:
    """Validate against the JSON Schema 2020-12 subset the frozen contracts actually use."""
    if not isinstance(schema, dict):
        return [] if schema is True else [f"{path}: schema is not a boolean-true or object"]
    errors: list[str] = []
    if "$ref" in schema:
        reference = str(schema["$ref"])
        if not reference.startswith("#/"):
            raise ValueError(f"unsupported $ref: {reference}")
        target: object = root
        for part in reference[2:].split("/"):
            target = target[part]  # type: ignore[index]
        errors.extend(validate(instance, target, root, path))
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_matches_type(instance, str(name)) for name in names):
            errors.append(f"{path}: expected type {schema['type']}, got {type(instance).__name__}")
    if "const" in schema and not _equal(instance, schema["const"]):
        errors.append(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and not any(_equal(instance, option) for option in schema["enum"]):
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']!r}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in instance}) != len(instance):
            errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], root, f"{path}[{index}]"))
        if "prefixItems" in schema:
            for index, item_schema in enumerate(schema["prefixItems"]):
                if index < len(instance):
                    errors.extend(validate(instance[index], item_schema, root, f"{path}[{index}]"))
    if isinstance(instance, dict):
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(f"{path}: fewer than minProperties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            errors.append(f"{path}: more than maxProperties")
        if "propertyNames" in schema:
            for name in instance:
                errors.extend(validate(name, schema["propertyNames"], root, f"{path}.<propertyName>"))
        for name in schema.get("required", ()):
            if name not in instance:
                errors.append(f"{path}: missing required property {name!r}")
        properties = schema.get("properties", {})
        for name, value in instance.items():
            if name in properties:
                errors.extend(validate(value, properties[name], root, f"{path}.{name}"))
            else:
                additional = schema.get("additionalProperties")
                if additional is False:
                    errors.append(f"{path}: unexpected property {name!r}")
                elif isinstance(additional, dict):
                    errors.extend(validate(value, additional, root, f"{path}.{name}"))
    for option in schema.get("allOf", ()):
        errors.extend(validate(instance, option, root, path))
    for keyword in ("anyOf", "oneOf"):
        if keyword not in schema:
            continue
        results = [validate(instance, option, root, path) for option in schema[keyword]]
        matched = sum(1 for result in results if not result)
        if (keyword == "anyOf" and not matched) or (keyword == "oneOf" and matched != 1):
            closest = min(results, key=len) if results else []
            errors.append(f"{path}: matched {matched} {keyword} branches; closest branch says {closest}")
    if "not" in schema and not validate(instance, schema["not"], root, path):
        errors.append(f"{path}: must not match the 'not' schema")
    if "if" in schema and not validate(instance, schema["if"], root, path):
        errors.extend(validate(instance, schema.get("then", True), root, path))
    elif "if" in schema:
        errors.extend(validate(instance, schema.get("else", True), root, path))
    return errors


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


def _wire(payload: dict) -> dict:
    """Validate what crosses stdio: dataclass tuples become JSON arrays on the way out."""
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _hello(runtime_id: str) -> dict:
    with worker_hello_payload(runtime_id, INSTALL_ROOT) as (mode, payload):
        assert mode == "normal", runtime_id
        return dict(payload)


def _classify_projection() -> ClassifyProjectionV1:
    return ClassifyProjectionV1(
        quality=(), count="solo", character="nala_(the_lion_king)", series="", artist="",
        appearance=("mammal",), tags=("blush",), environment=("forest",), nl="",
    )


def _classify_decision() -> ClassifyCountDecisionV1:
    return ClassifyCountDecisionV1(
        value="solo", baseValue="solo", selectedSource="wiki_tags", originalRaw=None, originalNormalized=None,
        wikiValue="solo", matchedTags=("solo",), conflict=False, issueCodes=(), warnings=(), appliedLowerBounds=(),
    )


def _caption_item() -> CaptionWorkItemV1:
    return CaptionWorkItemV1(
        sampleId=7, leaseId="lease-7", relativeImagePath="nested\\image.png", annotationKey="nested\\image",
        imageFormat="png", imageSize=123, imageMtimeNs=456, imageFileId="1:2",
    )


# Literal payloads mirror the worker lines named in each comment; the schema is the contract
# and this table is what makes a drift in either direction fail.
POLICY_HELLO_RESULT = {  # workers/policy/src/anima_policy_worker/worker.py:113-121
    "schemaVersion": 1, "payloadType": "policy_hello_result", "ready": True, "qualityEnabled": True,
    "device": "CPUExecutionProvider", "modelLoadCount": 1, "resourceFingerprint": "c" * 64,
}
POLICY_HELLO_REQUEST = {  # core/src/anima_core/policy_runner.py:137-143
    "schemaVersion": 1, "payloadType": "policy_hello_request", "jobId": "job-1", "configHash": "a" * 64,
    "datasetRoot": "E:\\dataset", "overlayRoot": "E:\\.dataset.anima-overlay-job-1", "artistRootName": "10_artist",
    "resourceManifestRelativePath": "manifests\\resources\\policy-lse14.json", "resourceFingerprint": "c" * 64,
    "policy": {
        "policyVersion": "dataset-batch-policy-v1", "seed": "anima-policy-default-v1",
        "artist": {"enabled": True, "dropoutProbability": 0.0},
        "quality": {"enabled": True, "dropoutProbability": 0.0, "device": "auto", "batchSize": 4, "resourceId": "lse14-scorer-5k-v1"},
        "appearanceNl": {
            "enabled": True, "solo": {"dropNl": 0.7, "dropAppearance": 0.05},
            "nonSolo": {"dropNl": 0.05, "dropAppearance": 0.7}, "unknown": {"dropNl": 0.15, "dropAppearance": 0.15},
        },
    },
}
POLICY_PROCESS_REQUEST = {  # core/src/anima_core/policy_runner.py:226-231
    "schemaVersion": 1, "payloadType": "policy_process_request",
    "items": [{
        "schemaVersion": 1, "sampleId": 1, "leaseId": "lease-1", "relativeImagePath": "1_artist\\a.png",
        "annotationKey": "1_artist\\a", "imageSize": 1024, "imageMtimeNs": 1_000_000, "imageFileId": "volume:1",
    }],
}
POLICY_BATCH_RESULT = {  # workers/policy/src/anima_policy_worker/worker.py:255-269 and :182-192
    "schemaVersion": 1, "payloadType": "policy_batch_result", "modelLoadCount": 1,
    "outcomes": [
        {
            "schemaVersion": 1, "status": "prepared", "sampleId": 1, "leaseId": "lease-1",
            "relativeImagePath": "1_artist\\a.png", "preparedRelativePath": "prepared\\dropout\\lease-1.json",
            "sha256": "d" * 64, "aestheticScore": 6.5, "quality": ["masterpiece", "best quality"],
            "decision": {"artistDropped": False, "qualityDropped": False, "appearanceNlAction": "drop_nl"},
        },
        {
            "schemaVersion": 1, "status": "issue", "sampleId": 2, "leaseId": "lease-2",
            "relativeImagePath": "1_artist\\b.png", "code": "policy_json_missing",
            "message": "working JSON does not exist", "retriable": True, "repairStartModule": "classify",
        },
    ],
}
EXPORT_HELLO_RESULT = {"schemaVersion": 1, "payloadType": "export_hello_result", "ready": True}  # worker.py:39
EXPORT_PROCESS_REQUEST = {  # core/src/anima_core/export_runner.py process request
    "schemaVersion": 1, "payloadType": "export_process_request",
    "items": [{"schemaVersion": 1, "sampleId": 1, "leaseId": "lease-1", "relativeImagePath": "a.png", "annotationKey": "a"}],
}
EXPORT_BATCH_RESULT = {  # workers/export/src/anima_export_worker/worker.py:50-63
    "schemaVersion": 1, "payloadType": "export_batch_result",
    "outcomes": [
        {
            "schemaVersion": 1, "status": "prepared", "sampleId": 1, "leaseId": "lease-1", "relativeImagePath": "a.png",
            "artifacts": [
                {"kind": "json", "relativePath": "prepared\\export\\lease-1.json", "sha256": "e" * 64},
                {"kind": "txt", "relativePath": "prepared\\export\\lease-1.txt", "sha256": "f" * 64},
            ],
            "conversions": {"missing_field_defaulted": 2, "array_duplicate_removed": 1},
        },
        {
            "schemaVersion": 1, "status": "issue", "sampleId": 2, "leaseId": "lease-2", "relativeImagePath": "b.png",
            "code": "final_json_invalid",
            "fieldErrors": [{"field": "artist", "code": "tag_not_flat_txt_representable"}, {"code": "json_read_failed"}],
        },
    ],
}
REPLACE_RESULT = {  # workers/replace/src/anima_replace_worker/worker.py:62-66
    "schemaVersion": 1, "payloadType": "replace_result", "sampleId": 1, "leaseId": "lease-1", "source": "e621",
    "relativeImagePath": "a.png", "replaced": 3, "dropped": 1, "passthrough": 5, "keepRewritten": 2,
    "projection": {
        "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
        "appearance": ["fat"], "tags": ["blush"], "environment": [], "nl": "",
    },
}
REPLACE_PROCESS_REQUEST = {
    "schemaVersion": 1,
    "payloadType": "replace_process_request",
    "items": [
        {
            "schemaVersion": 1, "sampleId": 1, "leaseId": "lease-1", "source": "e621",
            "relativeImagePath": "a.png", "projection": REPLACE_RESULT["projection"],
        },
        {
            "schemaVersion": 1, "sampleId": 2, "leaseId": "lease-2", "source": "e621",
            "relativeImagePath": "b.png", "projection": REPLACE_RESULT["projection"],
        },
    ],
}
REPLACE_PROCESS_RESULT = {
    "schemaVersion": 1,
    "payloadType": "replace_process_result",
    "outcomes": [
        REPLACE_RESULT,
        {
            "schemaVersion": 1, "payloadType": "replace_issue", "sampleId": 2, "leaseId": "lease-2",
            "source": "e621", "relativeImagePath": "b.png", "code": "replace_json_invalid",
            "severity": "error", "blocking": True, "retriable": False, "message": "projection rejected",
        },
    ],
}
REPLACE_HELLO_RESULT = {  # workers/replace/src/anima_replace_worker/worker.py:39-45
    "schemaVersion": 1, "payloadType": "replace_hello_result", "ready": True, "indexLoads": 1,
    "ruleCount": 86922, "resourceFingerprint": "b" * 64, "keepNonCanonical": 269, "canonicalDirectionConflict": 0,
}
REPLACE_CUSTOM_HELLO_REQUEST = {  # core/src/anima_core/replace_runner.py:112
    "schemaVersion": 1, "payloadType": "replace_hello_request", "jobId": "job-1", "configHash": "a" * 64,
    "customIndexOverlayRoot": "E:\\.dataset.anima-overlay-job-1",
    "customIndexPath": "resources\\replace\\custom.csv", "customIndexSha256": "b" * 64, "customIndexRuleCount": 12,
}

OCR_INFERENCE = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": True,
    "textRecScoreThresh": 0,
    "textDetLimitSideLen": 1920,
    "textDetLimitType": "max",
}
OCR_HELLO_REQUEST = {
    "schemaVersion": 1,
    "payloadType": "ocr_hello_request",
    "jobId": "job-1",
    "configHash": "a" * 64,
    "resourceId": "ocr-ppocrv5-server-paddle-v1",
    "resourceManifestRelativePath": "ocr-models\\ocr-ppocrv5-server-paddle-v1\\resource.json",
    "resourceFingerprint": "b" * 64,
    "inference": OCR_INFERENCE,
}
OCR_HELLO_RESULT = {
    "schemaVersion": 1,
    "payloadType": "ocr_hello_result",
    "ready": True,
    "executable": "C:\\app\\runtimes\\ocr-paddle\\python.exe",
    "pythonVersion": "3.11.15",
    "modelSessionLoads": 1,
    "resourceFingerprint": "b" * 64,
}
OCR_PROCESS_REQUEST = {
    "schemaVersion": 1,
    "payloadType": "ocr_process_request",
    "items": [
        {
            "schemaVersion": 1,
            "sampleId": 1,
            "leaseId": "lease-1",
            "relativeImagePath": "cats\\poster.jpg",
            "imagePath": "C:\\dataset\\cats\\poster.jpg",
            "imageSize": 123456,
            "imageSha256": "c" * 64,
        },
        {
            "schemaVersion": 1,
            "sampleId": 2,
            "leaseId": "lease-2",
            "relativeImagePath": "cats\\empty.jpg",
            "imagePath": "C:\\dataset\\cats\\empty.jpg",
            "imageSize": 123457,
            "imageSha256": "d" * 64,
        },
        {
            "schemaVersion": 1,
            "sampleId": 3,
            "leaseId": "lease-3",
            "relativeImagePath": "cats\\failed.jpg",
            "imagePath": "C:\\dataset\\cats\\failed.jpg",
            "imageSize": 123458,
            "imageSha256": "e" * 64,
        },
    ],
}
OCR_PROCESS_RESULT = {
    "schemaVersion": 1,
    "payloadType": "ocr_process_result",
    "items": [
        {
            "schemaVersion": 1,
            "status": "success",
            "sampleId": 1,
            "leaseId": "lease-1",
            "relativeImagePath": "cats\\poster.jpg",
            "image": {"width": 100, "height": 80, "sizeBytes": 123456, "sha256": "c" * 64},
            "items": [
                {
                    "text": "Hello",
                    "confidence": 0.98,
                    "polygonPixels": [[0, 0], [30, 0], [30, 20], [0, 20]],
                    "bboxPixels": [0, 0, 30, 20],
                    "textlineOrientationDegrees": 0,
                },
                {
                    "text": "Hello",
                    "confidence": 0.49,
                    "polygonPixels": [[60, 60], [100, 60], [100, 80], [60, 80]],
                    "bboxPixels": [60, 60, 100, 80],
                    "textlineOrientationDegrees": 0,
                },
            ],
        },
        {
            "schemaVersion": 1,
            "status": "no_text",
            "sampleId": 2,
            "leaseId": "lease-2",
            "relativeImagePath": "cats\\empty.jpg",
            "image": {"width": 100, "height": 80, "sizeBytes": 123457, "sha256": "d" * 64},
            "items": [],
        },
        {
            "schemaVersion": 1,
            "status": "failed",
            "sampleId": 3,
            "leaseId": "lease-3",
            "relativeImagePath": "cats\\failed.jpg",
            "image": {"width": 100, "height": 80, "sizeBytes": 123458, "sha256": "e" * 64},
            "items": [],
            "error": {
                "code": "ocr_inference_failed",
                "message": "The OCR engine failed for this image.",
                "retriable": True,
            },
        },
        {
            "schemaVersion": 1,
            "status": "failed",
            "sampleId": 4,
            "leaseId": "lease-4",
            "relativeImagePath": "cats\\oversize.jpg",
            "image": {"width": 20_001, "height": 2_000, "sizeBytes": 123459, "sha256": "f" * 64},
            "items": [],
            "error": {
                "code": "ocr_image_too_large",
                "message": "OCR image dimensions exceed the first-release safety limit.",
                "retriable": False,
                "details": {
                    "actualPixels": 40_002_000,
                    "maxPixels": 40_000_000,
                    "maxSide": 16_384,
                },
            },
        },
    ],
}


CAPTION_HELLO_REQUEST = {  # tests/unit/test_caption_protocol.py mirrors the same frozen payload
    "schemaVersion": 1, "payloadType": "caption_hello_request", "jobId": "job-1", "configHash": "a" * 64,
    "profile": "e621", "datasetRoot": "E:\\dataset",
    "resourceManifestRelativePath": "manifests\\resources\\caption-e621.json", "resourceFingerprint": "b" * 64,
    "thresholdPolicy": {"mode": "model_default"},
    "captionFormat": {"replaceUnderscoresWithSpaces": True, "preserveEscapes": True, "triggersEnabled": True, "triggerTerms": ["anima_style"]},
    "imageDecode": {"extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp"], "rejectMultiFrame": True, "applyExifTranspose": True, "alphaBackground": "#FFFFFF"},
}


def _payloads() -> list[tuple[str, str, dict]]:
    caption_item = _caption_item()
    classify_item = ClassifyWorkItemV1(
        sampleId=1,
        leaseId="lease-1",
        relativeImagePath="a.png",
        annotationKey="a",
        txtText="solo, blush",
        txtProvenance="module1_written",
        originalCount=None,
    )
    classify_second_item = ClassifyWorkItemV1(
        sampleId=2,
        leaseId="lease-2",
        relativeImagePath="b.png",
        annotationKey="b",
        txtText="solo, blue eyes",
        txtProvenance="module1_written",
        originalCount=None,
    )
    return [
        ("caption-worker-v1", "caption_hello_request", CAPTION_HELLO_REQUEST),
        ("caption-worker-v1", "caption_danbooru_hello_request", {
            **CAPTION_HELLO_REQUEST,
            "profile": "danbooru",
            "resourceManifestRelativePath": "tagging-models\\caption-danbooru-cl-tagger-v2-00\\resource.json",
            "thresholdPolicy": {
                "mode": "per_category",
                "categoryThresholds": {"general": 0.55, "character": 0.55, "copyright": 0.55},
            },
        }),
        ("caption-worker-v1", "caption_hello_result", CaptionHelloResultV1(
            executable="C:\\app\\runtimes\\caption-e621\\python.exe", provider="CPUExecutionProvider",
            resourceFingerprint="b" * 64,
        ).to_dict()),
        ("caption-worker-v1", "caption_danbooru_hello_result", CaptionHelloResultV1(
            executable="C:\\app\\runtimes\\caption-e621\\python.exe", provider="CPUExecutionProvider",
            resourceFingerprint="b" * 64, tagCount=106_536,
        ).to_dict()),
        ("caption-worker-v1", "caption_process_request", CaptionProcessRequestV1((
            caption_item,
            CaptionWorkItemV1(
                sampleId=8, leaseId="lease-8", relativeImagePath="nested\\image-8.png",
                annotationKey="nested\\image-8", imageFormat="png", imageSize=124,
                imageMtimeNs=457, imageFileId="1:3",
            ),
        )).to_dict()),
        ("caption-worker-v1", "caption_result", CaptionResultV1(
            sampleId=caption_item.sampleId, leaseId=caption_item.leaseId,
            relativeImagePath=caption_item.relativeImagePath, tags=(CaptionTagV1("blush", 0.87, "general"),),
            formattedTxt="blush", provider="CPUExecutionProvider",
        ).to_dict()),
        ("caption-worker-v1", "caption_danbooru_result", CaptionResultV1(
            sampleId=caption_item.sampleId, leaseId=caption_item.leaseId,
            relativeImagePath=caption_item.relativeImagePath,
            tags=(CaptionTagV1("hatsune_miku", 0.91, "character"),),
            formattedTxt="hatsune_miku", provider="CPUExecutionProvider", source="danbooru",
        ).to_dict()),
        ("caption-worker-v1", "caption_issue", CaptionIssueResultV1(
            sampleId=caption_item.sampleId, leaseId=caption_item.leaseId,
            relativeImagePath=caption_item.relativeImagePath, code="caption_no_tags", retriable=False,
            message="No model tags matched the frozen thresholds.",
        ).to_dict()),
        ("caption-worker-v1", "caption_oversize_issue", CaptionIssueResultV1(
            sampleId=caption_item.sampleId, leaseId=caption_item.leaseId,
            relativeImagePath=caption_item.relativeImagePath, code="caption_result_too_large", retriable=False,
            message="Caption result exceeds the 1 MiB worker response limit.",
        ).to_dict()),
        ("caption-worker-v1", "caption_process_result", CaptionProcessResultV1((
            CaptionResultV1(
                sampleId=caption_item.sampleId, leaseId=caption_item.leaseId,
                relativeImagePath=caption_item.relativeImagePath,
                tags=(CaptionTagV1("blush", 0.87, "general"),), formattedTxt="blush",
                provider="CPUExecutionProvider",
            ),
            CaptionIssueResultV1(
                sampleId=8, leaseId="lease-8", relativeImagePath="nested\\image-8.png",
                code="caption_no_tags", retriable=False,
                message="No model tags matched the frozen thresholds.",
            ),
        )).to_dict()),
        ("classify-worker-v1", "classify_hello_request", _hello("classify-e621")),
        ("classify-worker-v1", "classify_danbooru_hello_request", {
            **_hello("classify-e621"),
            "profile": "danbooru",
            "resourceManifestRelativePath": "classification-indexes\\danbooru-classify-20260727-v1\\resource.json",
            "wikiDataSourceId": "danbooru-wiki-test-v1",
        }),
        ("classify-worker-v1", "classify_hello_result", ClassifyHelloResultV1(
            "C:\\app\\runtimes\\classify-e621\\python.exe", "b" * 64,
        ).to_dict()),
        ("classify-worker-v1", "classify_process_request", ClassifyProcessRequestV1((
            classify_item,
            classify_second_item,
        )).to_dict()),
        ("classify-worker-v1", "classify_result", ClassifyResultV1(
            sampleId=1, leaseId="lease-1", relativeImagePath="a.png", projection=_classify_projection(),
            countDecision=_classify_decision(), inputTagCount=4, outputTagCount=3, droppedTagCount=1,
        ).to_dict()),
        ("classify-worker-v1", "classify_danbooru_result", ClassifyResultV1(
            sampleId=1,
            leaseId="lease-1",
            relativeImagePath="a.png",
            projection=ClassifyProjectionV1(
                (), "duo", "hatsune_miku", "vocaloid", "", (), ("2girls",), (), "",
            ),
            countDecision=ClassifyCountDecisionV1(
                value="duo", baseValue="duo", selectedSource="wiki_tags", originalRaw=None,
                originalNormalized=None, wikiValue="duo", matchedTags=("2girls", "multiple_girls"),
                conflict=False, issueCodes=(), warnings=("count_lower_bound:danbooru:multiple_girls",),
                appliedLowerBounds=("danbooru_girl",),
            ),
            inputTagCount=4,
            outputTagCount=3,
            droppedTagCount=1,
            source="danbooru",
        ).to_dict()),
        ("classify-worker-v1", "classify_issue", ClassifyIssueResultV1(
            sampleId=1, leaseId="lease-1", relativeImagePath="a.png", code="classify_no_writable_tags",
            retriable=False, message="no writable tag survived the projection", repairStartModule=None,
        ).to_dict()),
        ("classify-worker-v1", "classify_process_result", ClassifyProcessResultV1((
            ClassifyResultV1(
                sampleId=classify_item.sampleId,
                leaseId=classify_item.leaseId,
                relativeImagePath=classify_item.relativeImagePath,
                projection=_classify_projection(),
                countDecision=_classify_decision(),
                inputTagCount=4,
                outputTagCount=3,
                droppedTagCount=1,
            ),
            ClassifyIssueResultV1(
                sampleId=classify_second_item.sampleId,
                leaseId=classify_second_item.leaseId,
                relativeImagePath=classify_second_item.relativeImagePath,
                code="classify_no_writable_tags",
                retriable=False,
                message="no writable tag survived the projection",
                repairStartModule=None,
            ),
        )).to_dict()),
        ("replace-worker-v1", "replace_hello_request", _hello("replace-e621")),
        ("replace-worker-v1", "replace_hello_request_custom", REPLACE_CUSTOM_HELLO_REQUEST),
        ("replace-worker-v1", "replace_hello_result", REPLACE_HELLO_RESULT),
        ("replace-worker-v1", "replace_process_request", REPLACE_PROCESS_REQUEST),
        ("replace-worker-v1", "replace_process_result", REPLACE_PROCESS_RESULT),
        ("ocr-worker-v1", "ocr_hello_request", OCR_HELLO_REQUEST),
        ("ocr-worker-v1", "ocr_hello_result", OCR_HELLO_RESULT),
        ("ocr-worker-v1", "ocr_process_request", OCR_PROCESS_REQUEST),
        ("ocr-worker-v1", "ocr_process_result", OCR_PROCESS_RESULT),
        ("token-budget-worker-v1", "token_budget_hello_request", {
            "schemaVersion": 1, "payloadType": "token_budget_hello_request", "jobId": "worker-test",
            "configHash": "a" * 64, "resourceId": "tokenizer-qwen3-0.6b-anima-v1",
            "resourceManifestRelativePath": "tokenizers\\tokenizer-qwen3-0.6b-anima-v1\\resource.json",
            "resourceFingerprint": "b" * 64, "contextLimit": 32768, "maxTokens": 512,
        }),
        ("token-budget-worker-v1", "token_budget_process_result", {
            "schemaVersion": 1, "payloadType": "token_budget_process_result", "outcomes": [{
                "schemaVersion": 1, "payloadType": "token_budget_outcome", "sampleId": 1, "leaseId": "lease-1",
                "status": "overflow", "originalTokens": 513, "finalTokens": 513,
                "removed": {"quality": [], "environment": [], "tags": [], "appearance": []},
            }],
        }),
        ("nl-worker-v1", "nl_hello_request", _hello("nl")),
        ("nl-worker-v1", "nl_moss_hello_request", {
            **_hello("nl"),
            "responseProtocol": "nl-moss-v1",
            "promptVersion": "nl-default-prompt-moss-v1",
        }),
        ("nl-worker-v1", "nl_moss_process_result", {
            "schemaVersion": 1,
            "payloadType": "nl_process_result",
            "items": [{
                "schemaVersion": 1, "payloadType": "nl_moss_result", "sampleId": 1,
                "leaseId": "lease-1", "relativeImagePath": "sample.png", "requestId": "request-1",
                "usage": {"total_tokens": 10}, "httpAttempts": 1,
                "moss": {
                    "caption": {"long": "A cat sits by a window.", "medium": "A cat by a window.", "short": "A cat."},
                    "watermark": [], "borders": ["none"],
                },
            }],
        }),
        ("export-worker-v1", "export_hello_request", _hello("export")),
        ("export-worker-v1", "export_hello_result", EXPORT_HELLO_RESULT),
        ("export-worker-v1", "export_process_request", EXPORT_PROCESS_REQUEST),
        ("export-worker-v1", "export_batch_result", EXPORT_BATCH_RESULT),
        ("policy-worker-v1", "policy_hello_request", POLICY_HELLO_REQUEST),
        ("policy-worker-v1", "policy_hello_result", POLICY_HELLO_RESULT),
        ("policy-worker-v1", "policy_process_request", POLICY_PROCESS_REQUEST),
        ("policy-worker-v1", "policy_batch_result", POLICY_BATCH_RESULT),
    ]


class PayloadSchemaTests(unittest.TestCase):
    def test_job_config_v10_requires_complete_batch_map_and_flat_txt_layout(self) -> None:
        schema_path = SCHEMAS / "job-config-v10.schema.json"
        self.assertTrue(schema_path.is_file(), "JobConfig v10 schema must exist before v10 is accepted")
        document = _schema("job-config-v10")
        payload = _wire(JobConfig(
            workMode="in_place", overwriteMode="incremental", sourceRoot="C:\\data",
        ).to_dict())
        self.assertEqual(10, payload["schemaVersion"])
        self.assertEqual(
            {
                "caption": 4, "classify": 128, "replace": 128, "ocr": 4, "nl": 3,
                "countReview": 100, "dropout": 4, "tokenBudget": 128, "export": 500,
            },
            payload["moduleBatchSize"],
        )
        self.assertEqual("nl_newline", payload["captionFormat"]["flatTxtLayout"])
        self.assertEqual([], validate(payload, document, document))

        for field in payload["moduleBatchSize"]:
            incomplete = _wire(payload)
            incomplete["moduleBatchSize"].pop(field)
            self.assertNotEqual([], validate(incomplete, document, document), field)
        invalid_layout = _wire(payload)
        invalid_layout["captionFormat"]["flatTxtLayout"] = "paragraphs"
        self.assertNotEqual([], validate(invalid_layout, document, document))

        legacy = _wire(payload)
        legacy["schemaVersion"] = 9
        with self.assertRaisesRegex(ValueError, "legacy JobConfig is incompatible"):
            config_from_dict(legacy)

    def test_job_config_v10_removes_task_profile_and_separates_classify_input_modes(self) -> None:
        schema_path = SCHEMAS / "job-config-v10.schema.json"
        self.assertTrue(schema_path.is_file(), "JobConfig v10 schema must exist before v10 is accepted")
        document = _schema("job-config-v10")
        payload = JobConfig(
            workMode="in_place", overwriteMode="incremental", sourceRoot="C:\\data",
        ).to_dict()
        self.assertNotIn("profile", payload)
        self.assertEqual(10, payload["schemaVersion"])
        self.assertEqual([], validate(_wire(payload), document, document))

        with_profile = json.loads(json.dumps(payload))
        with_profile["profile"] = "e621"
        self.assertNotEqual([], validate(with_profile, document, document))

        mixed = json.loads(json.dumps(payload))
        mixed["classify"]["indexMode"] = "custom"
        mixed["classify"]["customResourcePath"] = "C:\\resources\\resource.json"
        self.assertNotEqual([], validate(mixed, document, document))

    def test_every_worker_owns_a_schema_and_each_file_is_wellformed(self) -> None:
        names = {path.name[: -len(".schema.json")] for path in SCHEMAS.glob("*.schema.json")}
        self.assertEqual(
            {
                "caption-worker-v1", "classify-worker-v1", "replace-worker-v1", "ocr-worker-v1", "token-budget-worker-v1", "nl-worker-v1",
                "policy-worker-v1", "export-worker-v1", "job-config-v2", "job-config-v3", "job-config-v4",
                "job-config-v5", "job-config-v6", "job-config-v7", "job-config-v8", "job-config-v9", "job-config-v10",
                "sample-manifest-v1",
                "worker-envelope-v1",
            },
            names,
            "every module's stdio protocol must have a frozen schema",
        )
        for name in sorted(names):
            with self.subTest(schema=name):
                document = _schema(name)
                self.assertEqual("https://json-schema.org/draft/2020-12/schema", document["$schema"])
                self.assertTrue(str(document["$id"]).startswith("anima://contracts/"))

    def test_real_worker_payloads_validate_against_their_schema(self) -> None:
        for name, label, payload in _payloads():
            with self.subTest(schema=name, payload=label):
                document = _schema(name)
                self.assertEqual([], validate(_wire(payload), document, document), f"{label} does not match {name}")

    def test_schemas_reject_an_unknown_property_and_a_wrong_constant(self) -> None:
        # Without this the validator could accept everything and the suite would prove nothing.
        for name, label, payload in _payloads():
            with self.subTest(schema=name, payload=label):
                document = _schema(name)
                self.assertNotEqual([], validate(_wire({**payload, "unexpected": 1}), document, document))
                self.assertNotEqual([], validate(_wire({**payload, "schemaVersion": 99}), document, document))

    def test_ocr_schema_freezes_six_settings_and_strict_failure_union(self) -> None:
        document = _schema("ocr-worker-v1")
        self.assertEqual([], validate(_wire(OCR_HELLO_REQUEST), document, document))
        self.assertEqual([], validate(_wire(OCR_PROCESS_RESULT), document, document))

        extended_request = {
            **OCR_HELLO_REQUEST,
            "requestedDevice": "auto",
            "expectedRuntimeId": "ocr-paddle",
            "expectedRuntimeFingerprint": "c" * 64,
        }
        extended_result = {
            **OCR_HELLO_RESULT,
            "requestedDevice": "auto",
            "observedDevice": "cpu",
            "runtimeId": "ocr-paddle",
            "runtimeFingerprint": "c" * 64,
            "paddleVersion": "3.2.2",
            "compiledWithCuda": False,
            "cudaVersion": None,
            "gpuName": None,
            "totalVramBytes": None,
        }
        self.assertEqual([], validate(_wire(extended_request), document, document))
        self.assertEqual([], validate(_wire(extended_result), document, document))
        for payload, field in ((extended_request, "expectedRuntimeId"), (extended_result, "gpuName"), (extended_result, "totalVramBytes")):
            with self.subTest(partial=field):
                malformed = _wire(payload)
                malformed.pop(field)
                self.assertNotEqual([], validate(malformed, document, document))
        for requested_device, expected_runtime_id in (
            ("cuda", "ocr-paddle"),
            ("cpu", "ocr-paddle-gpu"),
        ):
            with self.subTest(requested_device=requested_device, expected_runtime_id=expected_runtime_id):
                malformed = {
                    **extended_request,
                    "requestedDevice": requested_device,
                    "expectedRuntimeId": expected_runtime_id,
                }
                self.assertNotEqual([], validate(_wire(malformed), document, document))

        missing = dict(OCR_INFERENCE)
        missing.pop("textDetLimitType")
        for inference in (
            {**OCR_INFERENCE, "textDetLimitSideLen": "1920"},
            {**OCR_INFERENCE, "textDetLimitSideLen": True},
            {**OCR_INFERENCE, "textDetLimitSideLen": 1920.0},
            {**OCR_INFERENCE, "textDetLimitSideLen": 960},
            {**OCR_INFERENCE, "textDetLimitType": "min"},
            missing,
            {**OCR_INFERENCE, "unexpected": 1},
        ):
            with self.subTest(inference=inference):
                payload = _wire(OCR_HELLO_REQUEST)
                payload["inference"] = inference
                self.assertNotEqual([], validate(payload, document, document))

        normal_with_details = _wire(OCR_PROCESS_RESULT)
        normal_with_details["items"][2]["error"]["details"] = {
            "actualPixels": 8_000,
            "maxPixels": 40_000_000,
            "maxSide": 16_384,
        }
        self.assertNotEqual([], validate(normal_with_details, document, document))

        for item_index in (2, 3):
            with self.subTest(error_branch=item_index, case="absolute-path-message"):
                payload = _wire(OCR_PROCESS_RESULT)
                payload["items"][item_index]["error"]["message"] = "OCR failed for C:\\dataset\\image.png"
                self.assertNotEqual([], validate(payload, document, document))

        for label, mutate in (
            ("missing-details", lambda error: error.pop("details")),
            ("wrong-retriable", lambda error: error.__setitem__("retriable", True)),
            ("boolean-actual", lambda error: error["details"].__setitem__("actualPixels", True)),
            ("wrong-max-pixels", lambda error: error["details"].__setitem__("maxPixels", 40_000_001)),
            ("wrong-max-side", lambda error: error["details"].__setitem__("maxSide", 16_383)),
            ("details-extra", lambda error: error["details"].__setitem__("extra", 1)),
            ("error-extra", lambda error: error.__setitem__("extra", 1)),
        ):
            with self.subTest(case=label):
                payload = _wire(OCR_PROCESS_RESULT)
                mutate(payload["items"][3]["error"])
                self.assertNotEqual([], validate(payload, document, document))

    def test_current_job_config_matches_the_v10_schema(self) -> None:
        document = _schema("job-config-v10")
        configs = (
            JobConfig(workMode="in_place", overwriteMode="incremental", sourceRoot="C:\\data"),
            JobConfig(
                workMode="full_copy", overwriteMode="rebuild",
                sourceRoot="C:\\data", outputRoot="D:\\out",
            ),
        )
        for config in configs:
            with self.subTest(workMode=config.workMode):
                payload = _wire(config.to_dict())
                self.assertEqual(10, payload["schemaVersion"])
                self.assertNotIn("profile", payload)
                self.assertEqual([], validate(payload, document, document))
                validate_job_config(config)

        invalid = _wire(configs[0].to_dict())
        invalid["imageDecode"]["invalidImageAction"] = "ignore"
        self.assertNotEqual([], validate(invalid, document, document))

    def test_legacy_job_config_versions_are_rejected_explicitly(self) -> None:
        current = _wire(JobConfig(
            workMode="in_place", overwriteMode="incremental", sourceRoot="C:\\data",
        ).to_dict())
        for schema_version in range(2, 10):
            with self.subTest(schema_version=schema_version):
                legacy = _wire(current)
                legacy["schemaVersion"] = schema_version
                legacy["profile"] = "e621"
                legacy_document = _schema(f"job-config-v{schema_version}")
                self.assertNotEqual([], validate(legacy, legacy_document, legacy_document))
                with self.assertRaisesRegex(
                    ValueError,
                    "legacy JobConfig is incompatible",
                ):
                    config_from_dict(legacy)

    def test_token_budget_worker_schema_rejects_untrusted_outcomes(self) -> None:
        schema_path = SCHEMAS / "token-budget-worker-v1.schema.json"
        self.assertTrue(schema_path.is_file(), "Token Budget worker schema must exist before worker output is accepted")
        if not schema_path.is_file():
            return
        document = _schema("token-budget-worker-v1")
        valid = {
            "schemaVersion": 1,
            "payloadType": "token_budget_outcome",
            "sampleId": 1,
            "leaseId": "lease-1",
            "status": "trimmed",
            "originalTokens": 8,
            "finalTokens": 4,
            "removed": {"quality": [], "environment": [], "tags": ["tail"], "appearance": []},
            "annotation": {
                "quality": [], "count": "solo", "character": "", "series": "", "artist": "",
                "appearance": [], "tags": ["head"], "environment": [], "nl": "caption.",
            },
            "flatTextSha256": "a" * 64,
        }
        self.assertEqual([], validate(valid, document, document))
        for changed in (
            {"status": "prepared"},
            {"finalTokens": True},
            {"originalTokens": math.nan},
            {"removed": {"quality": [], "environment": [], "tags": [], "appearance": [], "extra": []}},
            {"flatTextSha256": "A" * 64},
        ):
            with self.subTest(changed=changed):
                candidate = {**valid, **changed}
                self.assertNotEqual([], validate(candidate, document, document))
        overflow = {key: value for key, value in valid.items() if key not in {"annotation", "flatTextSha256"}}
        overflow.update({"status": "overflow", "finalTokens": 9, "removed": {"quality": [], "environment": [], "tags": ["head", "tail"], "appearance": []}})
        self.assertEqual([], validate(overflow, document, document))

    def test_nl_worker_schema_keeps_legacy_items_and_allows_only_strict_v5_v6_context_shapes(self) -> None:
        document = _schema("nl-worker-v1")
        legacy = {
            "schemaVersion": 1,
            "payloadType": "nl_process_request",
            "httpAttemptAllowance": 1,
            "items": [{
                "schemaVersion": 1,
                "sampleId": 1,
                "leaseId": "lease-1",
                "relativeImagePath": "sample.png",
                "imagePath": None,
                "jsonContext": "{\"nl\":\"\"}",
            }],
        }
        v5 = _wire(legacy)
        v5["items"][0]["ocrContext"] = {"items": [["top-left", "Hello"], ["bottom-right", "Hello"]]}
        self.assertEqual([], validate(legacy, document, document))
        self.assertEqual([], validate(v5, document, document))
        v6 = _wire(v5)
        v6["items"][0].update({
            "captionPreset": "character",
            "lengthTier": "medium",
            "primaryCharacterName": "主角",
            "userSupplement": "bounded supplement",
        })
        self.assertEqual([], validate(v6, document, document))
        for label, mutate in (
            ("metadata", lambda value: value["items"][0]["ocrContext"].__setitem__("confidence", 0.8)),
            ("bad-position", lambda value: value["items"][0]["ocrContext"]["items"][0].__setitem__(0, "corner")),
            ("extra-item-field", lambda value: value["items"][0]["ocrContext"]["items"][0].append("extra")),
        ):
            with self.subTest(case=label):
                malformed = _wire(v5)
                mutate(malformed)
                self.assertNotEqual([], validate(malformed, document, document))
        for label, mutate in (
            ("missing-tier", lambda value: value["items"][0].pop("lengthTier")),
            ("character-name-on-general", lambda value: value["items"][0].update({"captionPreset": "general", "primaryCharacterName": "主角"})),
            ("unknown-v6-item-field", lambda value: value["items"][0].__setitem__("extra", "no")),
        ):
            with self.subTest(case=label):
                malformed = _wire(v6)
                mutate(malformed)
                self.assertNotEqual([], validate(malformed, document, document))

    def test_nl_and_dropout_sections_are_no_longer_unconstrained(self) -> None:
        # F45: both used to be bare {"type": "object"}, so an illegal policy only failed in
        # nl_runner.py after Caption, Classify and Replace had already run.
        document = _schema("job-config-v9")
        base = JobConfig(workMode="in_place", overwriteMode="incremental", sourceRoot="C:\\data").to_dict()
        invalid = (
            ("nl-unknown-key", {**base, "nl": {**base["nl"], "typo": True}}),
            ("nl-api-without-context", {**base, "nl": {**base["nl"], "apiEnabled": True, "useImage": False, "useFullJson": False}}),
            ("nl-concurrency-out-of-range", {**base, "nl": {**base["nl"], "apiPolicy": {"concurrency": 64}}}),
            ("nl-unknown-policy-key", {**base, "nl": {**base["nl"], "apiPolicy": {"retries": 3}}}),
            ("nl-frozen-policy-value", {**base, "nl": {**base["nl"], "apiPolicy": {"maxNlBytes": 1}}}),
            ("nl-rpm-not-a-number", {**base, "nl": {**base["nl"], "apiPolicy": {"maxRequestsPerMinute": "fast"}}}),
            ("dropout-unknown-key", {**base, "dropout": {**base["dropout"], "typo": 1}}),
            ("dropout-policy-version", {**base, "dropout": {**base["dropout"], "policyVersion": "v2"}}),
            ("dropout-probability-out-of-range", {
                **base,
                "dropout": {**base["dropout"], "artist": {"enabled": True, "dropoutProbability": 1.5}},
            }),
            ("dropout-batch-size", {
                **base,
                "dropout": {**base["dropout"], "quality": {**base["dropout"]["quality"], "batchSize": 0}},
            }),
            ("dropout-coupled-shape", {
                **base,
                "dropout": {**base["dropout"], "appearanceNl": {**base["dropout"]["appearanceNl"], "solo": {"dropNl": 0.5}}},
            }),
            ("count-review-unknown-key", {**base, "countReview": {**base["countReview"], "typo": True}}),
            ("count-review-protocol-version", {
                **base, "countReview": {**base["countReview"], "protocolVersion": "count-review-v2"},
            }),
            ("count-review-without-classify", {
                **base, "classify": {**base["classify"], "enabled": False},
            }),
            ("export-format", {**base, "export": {"format": "yaml"}}),
        )
        for label, payload in invalid:
            with self.subTest(case=label):
                self.assertNotEqual([], validate(_wire(payload), document, document), f"{label} should be rejected")


if __name__ == "__main__":
    unittest.main()
