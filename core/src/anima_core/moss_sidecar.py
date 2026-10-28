"""Strict MOSS image-caption JSON sidecar format."""
from __future__ import annotations

import json


MAX_MOSS_BYTES = 16_384
MOSS_BORDER_DIRECTIONS = frozenset({"left", "right", "up", "down"})


class MossSidecarError(ValueError):
    pass


def _validate_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MossSidecarError(f"MOSS {field} is invalid")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise MossSidecarError(f"MOSS {field} is invalid") from exc
    if size > MAX_MOSS_BYTES:
        raise MossSidecarError(f"MOSS {field} is invalid")
    return value


def validate_moss_json(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"caption", "watermark", "borders"}:
        raise MossSidecarError("MOSS JSON fields are invalid")
    caption = value["caption"]
    if not isinstance(caption, dict) or set(caption) != {"long", "medium", "short"}:
        raise MossSidecarError("MOSS caption fields are invalid")
    normalized_caption: dict[str, str] = {}
    for field in ("long", "medium", "short"):
        normalized_caption[field] = _validate_text(caption[field], f"caption.{field}")
    raw_watermarks = value["watermark"]
    if not isinstance(raw_watermarks, list) or len(raw_watermarks) > 1024:
        raise MossSidecarError("MOSS watermark is invalid")
    normalized_watermarks: list[dict[str, object]] = []
    for entry in raw_watermarks:
        if not isinstance(entry, dict) or set(entry) != {"box_2d", "label"}:
            raise MossSidecarError("MOSS watermark entry is invalid")
        box = entry["box_2d"]
        label = entry["label"]
        valid_box = (
            not isinstance(box, list) or len(box) != 4
            or any(type(coordinate) is not int or not 0 <= coordinate <= 1000 for coordinate in box)
            or box[0] > box[2] or box[1] > box[3]
        )
        if valid_box:
            raise MossSidecarError("MOSS watermark entry is invalid")
        normalized_watermarks.append({"box_2d": list(box), "label": _validate_text(label, "watermark label")})
    raw_borders = value["borders"]
    if not isinstance(raw_borders, list) or not 1 <= len(raw_borders) <= 4 or any(not isinstance(item, str) for item in raw_borders):
        raise MossSidecarError("MOSS borders are invalid")
    if raw_borders != ["none"] and (
        len(set(raw_borders)) != len(raw_borders)
        or any(direction not in MOSS_BORDER_DIRECTIONS for direction in raw_borders)
    ):
        raise MossSidecarError("MOSS borders are invalid")
    normalized = {"caption": normalized_caption, "watermark": normalized_watermarks, "borders": list(raw_borders)}
    try:
        serialized_size = len(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise MossSidecarError("MOSS JSON contains invalid Unicode") from exc
    if serialized_size > MAX_MOSS_BYTES:
        raise MossSidecarError("MOSS JSON exceeds its limit")
    return normalized


def serialize_moss_json(value: object) -> bytes:
    normalized = validate_moss_json(value)
    return (json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def parse_moss_json(data: bytes) -> dict[str, object]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_MOSS_BYTES + 1:
        raise MossSidecarError("MOSS JSON bytes are invalid")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise MossSidecarError("MOSS JSON contains a duplicate key")
            result[key] = item
        return result

    def reject_constant(_: str) -> object:
        raise MossSidecarError("MOSS JSON contains a non-finite number")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, MossSidecarError) as exc:
        raise MossSidecarError("MOSS JSON is invalid") from exc
    return validate_moss_json(value)
