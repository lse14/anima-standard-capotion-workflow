from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


FRAGMENT_NAMES = ("base", "general", "style", "character", "short", "medium", "long")
MOSS_FRAGMENT_NAMES = ("base", "general", "style", "character")
MAX_FRAGMENT_BYTES = 65_536
MAX_SUPPLEMENT_BYTES = 16_384
MAX_CHARACTER_NAME_BYTES = 512
PRESETS = frozenset({"general", "style", "character"})
MOSS_PRESETS = frozenset({"moss-general", "moss-style", "moss-character"})
TIERS = frozenset({"short", "medium", "long"})


class PromptResourceError(ValueError):
    pass


def _validate_text(data: bytes, name: str) -> str:
    if data.startswith(b"\xef\xbb\xbf") or not data or len(data) > MAX_FRAGMENT_BYTES or b"\x00" in data:
        raise PromptResourceError(f"NL v4 fragment {name} is invalid")
    try:
        value = data.decode("utf-8").replace("\r\n", "\n").strip()
    except UnicodeDecodeError as exc:
        raise PromptResourceError(f"NL v4 fragment {name} is not UTF-8") from exc
    if not value:
        raise PromptResourceError(f"NL v4 fragment {name} is empty")
    return value


def _verify_manifest(manifest_path: Path, data: bytes, resource_id: str) -> None:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromptResourceError("NL v4 prompt manifest is invalid") from exc
    expected = {"schemaVersion", "resourceId", "owner", "relativePath", "sizeBytes", "sha256"}
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schemaVersion") != 1
        or value.get("resourceId") != resource_id
        or value.get("owner") != "nl"
        or value.get("relativePath") != f"resources\\{resource_id}.txt"
        or type(value.get("sizeBytes")) is not int
        or value.get("sizeBytes") != len(data)
        or value.get("sha256") != hashlib.sha256(data).hexdigest()
    ):
        raise PromptResourceError("NL v4 prompt manifest identity or digest is invalid")


def _roots() -> tuple[tuple[Path, bool], ...]:
    candidates: list[tuple[Path, bool]] = []
    configured = os.environ.get("ANIMA_INSTALL_ROOT")
    if configured:
        candidates.append((Path(configured), True))
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidates.append((parent, True))
        candidates.append((parent, False))
        candidates.append((parent / "packaging" / "resources", False))
    unique: list[tuple[Path, bool]] = []
    seen: set[str] = set()
    for root, installed in candidates:
        key = str(root).casefold()
        if key not in seen:
            seen.add(key)
            unique.append((root, installed))
    return tuple(unique)


def load_v4_fragments() -> dict[str, str]:
    for root, installed in _roots():
        resource_root = root / "resources" if installed else root
        first = resource_root / "nl-default-prompt-v4-base.txt"
        if not first.is_file():
            continue
        fragments: dict[str, str] = {}
        for name in FRAGMENT_NAMES:
            resource_id = f"nl-default-prompt-v4-{name}"
            path = resource_root / f"{resource_id}.txt"
            if not path.is_file():
                raise PromptResourceError("NL v4 prompt fragment set is incomplete")
            data = path.read_bytes()
            fragments[name] = _validate_text(data, name)
            if installed:
                _verify_manifest(root / "manifests" / "resources" / f"{resource_id}.json", data, resource_id)
        return fragments
    raise PromptResourceError("NL v4 prompt resources are unavailable")


def load_moss_fragments() -> dict[str, str]:
    for root, installed in _roots():
        resource_root = root / "resources" if installed else root
        first = resource_root / "nl-default-prompt-moss-v1-base.txt"
        if not first.is_file():
            continue
        fragments: dict[str, str] = {}
        for name in MOSS_FRAGMENT_NAMES:
            resource_id = f"nl-default-prompt-moss-v1-{name}"
            path = resource_root / f"{resource_id}.txt"
            if not path.is_file():
                raise PromptResourceError("MOSS NL prompt fragment set is incomplete")
            data = path.read_bytes()
            fragments[name] = _validate_text(data, f"moss-{name}")
            if installed:
                _verify_manifest(root / "manifests" / "resources" / f"{resource_id}.json", data, resource_id)
        return fragments
    raise PromptResourceError("MOSS NL prompt resources are unavailable")


def compose_v4_prompt(
    *, fragments: dict[str, str], caption_preset: str, length_tier: str,
    primary_character_name: str | None, user_supplement: str,
) -> str:
    if caption_preset not in PRESETS or length_tier not in TIERS:
        raise PromptResourceError("NL v4 preset or length tier is invalid")
    if caption_preset in {"character", "moss-character"}:
        if not isinstance(primary_character_name, str) or not primary_character_name.strip():
            raise PromptResourceError("character preset requires a primary character name")
        if len(primary_character_name.encode("utf-8")) > MAX_CHARACTER_NAME_BYTES:
            raise PromptResourceError("primary character name exceeds its bound")
    elif primary_character_name is not None:
        raise PromptResourceError("non-character preset must not include a primary character name")
    if not isinstance(user_supplement, str) or "\x00" in user_supplement or len(user_supplement.encode("utf-8")) > MAX_SUPPLEMENT_BYTES:
        raise PromptResourceError("NL user supplement exceeds its bound")
    if set(fragments) != set(FRAGMENT_NAMES):
        raise PromptResourceError("NL v4 prompt fragment set is incomplete")
    preset_data = json.dumps(
        {"captionPreset": caption_preset, "primaryCharacterName": primary_character_name},
        ensure_ascii=False, separators=(",", ":"),
    )
    length_data = json.dumps({"lengthTier": length_tier}, separators=(",", ":"))
    supplement_data = json.dumps({"userSupplement": user_supplement}, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((
        "NL_PROMPT_BASE:\n" + fragments["base"],
        "NL_PROMPT_PRESET:\n" + fragments[caption_preset],
        "NL_PROMPT_PRESET_DATA_JSON:\n" + preset_data,
        "NL_PROMPT_LENGTH:\n" + fragments[length_tier],
        "NL_PROMPT_LENGTH_DATA_JSON:\n" + length_data,
        "NL_PROMPT_SUPPLEMENT:\n" + supplement_data,
    ))


def compose_moss_prompt(
    *, fragments: dict[str, str], caption_preset: str,
    primary_character_name: str | None, user_supplement: str,
) -> str:
    if caption_preset not in MOSS_PRESETS:
        raise PromptResourceError("MOSS NL preset is invalid")
    if caption_preset == "moss-character":
        if not isinstance(primary_character_name, str) or not primary_character_name.strip():
            raise PromptResourceError("MOSS character preset requires a primary character name")
        if len(primary_character_name.encode("utf-8")) > MAX_CHARACTER_NAME_BYTES:
            raise PromptResourceError("primary character name exceeds its bound")
    elif primary_character_name is not None:
        raise PromptResourceError("non-character MOSS preset must not include a primary character name")
    if not isinstance(user_supplement, str) or "\x00" in user_supplement or len(user_supplement.encode("utf-8")) > MAX_SUPPLEMENT_BYTES:
        raise PromptResourceError("NL user supplement exceeds its bound")
    if set(fragments) != set(MOSS_FRAGMENT_NAMES):
        raise PromptResourceError("MOSS NL prompt fragment set is incomplete")
    preset_data = json.dumps(
        {"captionPreset": caption_preset, "primaryCharacterName": primary_character_name},
        ensure_ascii=False, separators=(",", ":"),
    )
    supplement_data = json.dumps({"userSupplement": user_supplement}, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((
        "MOSS_NL_PROMPT_BASE:\n" + fragments["base"],
        "MOSS_NL_PROMPT_PRESET:\n" + fragments[caption_preset.removeprefix("moss-")],
        "MOSS_NL_PROMPT_PRESET_DATA_JSON:\n" + preset_data,
        "MOSS_NL_PROMPT_SUPPLEMENT:\n" + supplement_data,
    ))
