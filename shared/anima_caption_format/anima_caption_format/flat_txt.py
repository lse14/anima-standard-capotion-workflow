"""Deterministic module-6 flat TXT serialization for an already valid payload."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

from .normalizer import ARRAY_FIELDS, CaptionDisplayPolicy, FIELDS, display_tag, flat_txt_representable


_TAG_FIELDS = FIELDS[:-1]


class FlatTextSerializationError(ValueError):
    pass


def _display_tag(value: str, policy: CaptionDisplayPolicy) -> str:
    result = display_tag(value, policy)
    if not flat_txt_representable(result):
        raise FlatTextSerializationError("normalized tag is not representable in flat TXT")
    return result


def _field_values(payload: Mapping[str, object], field: str) -> list[str]:
    value = payload[field]
    if field in ARRAY_FIELDS:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise FlatTextSerializationError(f"{field} is not a normalized string array")
        return list(value)
    if field == "character":
        if not isinstance(value, str):
            raise FlatTextSerializationError("character is not a normalized string")
        return [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, str):
        raise FlatTextSerializationError(f"{field} is not a normalized string")
    return [value] if value else []


def serialize_flat_txt(payload: Mapping[str, object], caption_policy: CaptionDisplayPolicy) -> bytes:
    """Serialize a normalized payload without changing JSON field semantics."""
    if set(payload) != set(FIELDS):
        raise FlatTextSerializationError("flat TXT requires exactly the normalized nine fields")
    # Trigger terms come first, then the eight tag-bearing fields in their
    # frozen order. `count` legitimately repeats a bucket label that also
    # appears in `tags`, so the flat TXT prints the first occurrence only;
    # the JSON payload stays untouched.
    values: list[str] = []
    if caption_policy.triggers_enabled:
        values.extend(caption_policy.trigger_terms)
    for field in _TAG_FIELDS:
        values.extend(_field_values(payload, field))
    seen: set[str] = set()
    tags: list[str] = []
    for value in values:
        displayed = _display_tag(value, caption_policy)
        key = displayed.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(displayed)
    nl = payload["nl"]
    if not isinstance(nl, str):
        raise FlatTextSerializationError("nl is not a normalized string")
    if "\r" in nl or "\n" in nl:
        raise FlatTextSerializationError("nl must not contain a line separator")
    if not tags and not nl:
        raise FlatTextSerializationError("flat TXT cannot serialize an empty payload")
    tag_text = ", ".join(tags)
    if caption_policy.flat_txt_layout == "single_line":
        result = ", ".join(section for section in (tag_text, nl) if section)
    elif caption_policy.flat_txt_layout == "nl_newline":
        result = f"{tag_text}\n{nl}" if tag_text and nl else tag_text or nl
    else:
        raise FlatTextSerializationError("flat TXT layout is invalid")
    if not result.endswith("."):
        result += "."
    return result.encode("utf-8")


def flat_txt_sha256(payload: Mapping[str, object], caption_policy: CaptionDisplayPolicy) -> str:
    return hashlib.sha256(serialize_flat_txt(payload, caption_policy)).hexdigest()
