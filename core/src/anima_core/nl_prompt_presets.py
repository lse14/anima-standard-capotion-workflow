from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .nl_profiles import (
    MOSS_CHARACTER_PROMPT_VERSION,
    MOSS_GENERAL_PROMPT_VERSION,
    MOSS_STYLE_PROMPT_VERSION,
    V4_BASE_PROMPT_VERSION,
    V4_CHARACTER_PROMPT_VERSION,
    V4_GENERAL_PROMPT_VERSION,
    V4_STYLE_PROMPT_VERSION,
    load_default_system_prompt,
)


NlPresetType = Literal["general", "style", "character", "moss-general", "moss-style", "moss-character"]
PRESET_TYPES: tuple[NlPresetType, ...] = ("general", "style", "character")
BUILTIN_PRESET_IDS: tuple[str, ...] = (
    "builtin:nl-preset-v1-general",
    "builtin:nl-preset-v1-style",
    "builtin:nl-preset-v1-character",
)
BUILTIN_V4_BASE_PRESET_ID = "builtin:nl-default-prompt-v4-base"  # v1 diagnostic compatibility alias
MOSS_BUILTIN_PRESET_IDS: tuple[str, ...] = (
    "builtin:nl-preset-moss-v1-general",
    "builtin:nl-preset-moss-v1-style",
    "builtin:nl-preset-moss-v1-character",
)
MOSS_PRESET_TYPES: tuple[NlPresetType, ...] = ("moss-general", "moss-style", "moss-character")
MAX_CUSTOM_PRESETS = 100
MAX_STORE_BYTES = 8 * 1024 * 1024
MAX_NAME_BYTES = 256
MAX_PROMPT_BYTES = 65_536
_CUSTOM_ID = re.compile(r"^custom:[0-9a-f]{32}$")

_BUILTINS: tuple[tuple[str, str, NlPresetType, str], ...] = (
    (BUILTIN_PRESET_IDS[0], "General", "general", V4_GENERAL_PROMPT_VERSION),
    (BUILTIN_PRESET_IDS[1], "Style", "style", V4_STYLE_PROMPT_VERSION),
    (BUILTIN_PRESET_IDS[2], "Character", "character", V4_CHARACTER_PROMPT_VERSION),
)
_MOSS_BUILTINS: tuple[tuple[str, str, NlPresetType, str], ...] = (
    (MOSS_BUILTIN_PRESET_IDS[0], "MOSS General", "moss-general", MOSS_GENERAL_PROMPT_VERSION),
    (MOSS_BUILTIN_PRESET_IDS[1], "MOSS Style", "moss-style", MOSS_STYLE_PROMPT_VERSION),
    (MOSS_BUILTIN_PRESET_IDS[2], "MOSS Character", "moss-character", MOSS_CHARACTER_PROMPT_VERSION),
)
_BUILTIN_BY_ID = {item[0]: item for item in (*_BUILTINS, *_MOSS_BUILTINS)}


class PromptPresetError(ValueError):
    pass


class PromptPresetValidationError(PromptPresetError):
    pass


class PromptPresetConflictError(PromptPresetError):
    pass


class PromptPresetNotFoundError(PromptPresetError):
    pass


@dataclass(frozen=True)
class NlPromptPreset:
    presetId: str
    name: str
    builtIn: bool
    type: NlPresetType
    promptText: str
    sha256: str
    sizeBytes: int

    @property
    def basePrompt(self) -> str:
        """Compatibility spelling used by the v1 diagnostics client."""
        return self.promptText


def _reject_duplicate_keys(pairs: list[tuple[object, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if not isinstance(key, str) or key in value:
            raise PromptPresetValidationError("prompt preset store contains duplicate JSON keys")
        value[key] = item
    return value


def _validate_text(value: object, *, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PromptPresetValidationError(f"prompt preset {field} is invalid")
    if len(value.encode("utf-8")) > limit:
        raise PromptPresetValidationError(f"prompt preset {field} exceeds its limit")
    return value


def _validate_type(value: object) -> NlPresetType:
    if value not in (*PRESET_TYPES, *MOSS_PRESET_TYPES):
        raise PromptPresetValidationError("prompt preset type is invalid")
    return value  # type: ignore[return-value]


def _prompt_value(prompt_text: str | None, base_prompt: str | None) -> str:
    if prompt_text is not None and base_prompt is not None and prompt_text != base_prompt:
        raise PromptPresetValidationError("prompt preset promptText and basePrompt disagree")
    value = prompt_text if prompt_text is not None else base_prompt
    return _validate_text(value, field="prompt text", limit=MAX_PROMPT_BYTES)


def _preset(*, preset_id: str, name: str, built_in: bool, preset_type: NlPresetType, prompt_text: str) -> NlPromptPreset:
    data = prompt_text.encode("utf-8")
    return NlPromptPreset(
        presetId=preset_id,
        name=name,
        builtIn=built_in,
        type=preset_type,
        promptText=prompt_text,
        sha256=hashlib.sha256(data).hexdigest(),
        sizeBytes=len(data),
    )


class NlPromptPresetStore:
    """Persist one visible, typed NL prompt library for tasks and diagnostics."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path)
            if path is not None
            else Path(os.environ["LOCALAPPDATA"]) / "AnimaDatasetTool" / "nl-prompt-presets.json"
        )

    def _builtin(self, preset_id: str) -> NlPromptPreset:
        definition = _BUILTIN_BY_ID.get(preset_id)
        if definition is None:
            raise PromptPresetNotFoundError("prompt preset was not found")
        _, name, preset_type, prompt_version = definition
        prompt = load_default_system_prompt(prompt_version=prompt_version)
        override = self._load_records()[0].get(preset_id)
        return _preset(
            preset_id=preset_id,
            name=name,
            built_in=True,
            preset_type=preset_type,
            prompt_text=override or prompt,
        )

    def _read_records(self) -> tuple[dict[str, str], dict[str, tuple[str, NlPresetType, str]], bool]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {}, {}, False
        except OSError as exc:
            raise PromptPresetValidationError("prompt preset store is unreadable") from exc
        if len(raw) > MAX_STORE_BYTES:
            raise PromptPresetValidationError("prompt preset store exceeds its limit")
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromptPresetValidationError("prompt preset store is invalid JSON") from exc
        if not isinstance(value, dict):
            raise PromptPresetValidationError("prompt preset store shape is invalid")

        if value.get("schemaVersion") == 1:
            if set(value) != {"schemaVersion", "presets"} or not isinstance(value.get("presets"), list):
                raise PromptPresetValidationError("prompt preset store shape is invalid")
            custom: dict[str, tuple[str, NlPresetType, str]] = {}
            for record in value["presets"]:
                if not isinstance(record, dict) or set(record) != {"presetId", "name", "basePrompt"}:
                    raise PromptPresetValidationError("prompt preset store record fields are invalid")
                preset_id = record["presetId"]
                if preset_id == BUILTIN_V4_BASE_PRESET_ID:
                    # The legacy diagnostic built-in is replaced by the three packaged entries.
                    continue
                if not isinstance(preset_id, str) or not _CUSTOM_ID.fullmatch(preset_id) or preset_id in custom:
                    raise PromptPresetValidationError("prompt preset id is invalid")
                custom[preset_id] = (
                    _validate_text(record["name"], field="name", limit=MAX_NAME_BYTES),
                    "general",
                    _validate_text(record["basePrompt"], field="prompt text", limit=MAX_PROMPT_BYTES),
                )
            return {}, custom, True

        if (
            set(value) != {"schemaVersion", "builtInOverrides", "customPresets"}
            or value.get("schemaVersion") != 2
            or not isinstance(value.get("builtInOverrides"), list)
            or not isinstance(value.get("customPresets"), list)
            or len(value["customPresets"]) > MAX_CUSTOM_PRESETS
        ):
            raise PromptPresetValidationError("prompt preset store shape is invalid")
        overrides: dict[str, str] = {}
        for record in value["builtInOverrides"]:
            if not isinstance(record, dict) or set(record) != {"presetId", "promptText"}:
                raise PromptPresetValidationError("prompt preset built-in override fields are invalid")
            preset_id = record["presetId"]
            if not isinstance(preset_id, str) or preset_id not in _BUILTIN_BY_ID or preset_id in overrides:
                raise PromptPresetValidationError("prompt preset built-in id is invalid")
            overrides[preset_id] = _validate_text(record["promptText"], field="prompt text", limit=MAX_PROMPT_BYTES)
        custom = {}
        for record in value["customPresets"]:
            if not isinstance(record, dict) or set(record) != {"presetId", "name", "type", "promptText"}:
                raise PromptPresetValidationError("prompt preset store record fields are invalid")
            preset_id = record["presetId"]
            if not isinstance(preset_id, str) or not _CUSTOM_ID.fullmatch(preset_id) or preset_id in custom:
                raise PromptPresetValidationError("prompt preset id is invalid")
            custom[preset_id] = (
                _validate_text(record["name"], field="name", limit=MAX_NAME_BYTES),
                _validate_type(record["type"]),
                _validate_text(record["promptText"], field="prompt text", limit=MAX_PROMPT_BYTES),
            )
        return overrides, custom, False

    def _write_records(self, overrides: dict[str, str], custom: dict[str, tuple[str, NlPresetType, str]]) -> None:
        if len(custom) > MAX_CUSTOM_PRESETS:
            raise PromptPresetValidationError("prompt preset limit is exceeded")
        target = {
            "schemaVersion": 2,
            "builtInOverrides": [
                {"presetId": preset_id, "promptText": overrides[preset_id]}
                for preset_id in sorted(overrides)
            ],
            "customPresets": [
                {
                    "presetId": preset_id,
                    "name": custom[preset_id][0],
                    "type": custom[preset_id][1],
                    "promptText": custom[preset_id][2],
                }
                for preset_id in sorted(custom)
            ],
        }
        encoded = (json.dumps(target, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if len(encoded) > MAX_STORE_BYTES:
            raise PromptPresetValidationError("prompt preset store exceeds its limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _load_records(self) -> tuple[dict[str, str], dict[str, tuple[str, NlPresetType, str]]]:
        overrides, custom, migrated = self._read_records()
        if migrated:
            self._write_records(overrides, custom)
        return overrides, custom

    @staticmethod
    def _summary(preset: NlPromptPreset) -> dict[str, object]:
        return {
            "presetId": preset.presetId,
            "name": preset.name,
            "type": preset.type,
            "builtIn": preset.builtIn,
            "sha256": preset.sha256,
            "sizeBytes": preset.sizeBytes,
        }

    def list_summaries(self, *, include_moss: bool = False) -> tuple[dict[str, object], ...]:
        overrides, custom = self._load_records()
        result: list[dict[str, object]] = []
        builtins = (*_BUILTINS, *_MOSS_BUILTINS) if include_moss else _BUILTINS
        for preset_id, name, preset_type, prompt_version in builtins:
            prompt = overrides.get(preset_id) or load_default_system_prompt(prompt_version=prompt_version)
            result.append(self._summary(_preset(
                preset_id=preset_id, name=name, built_in=True, preset_type=preset_type, prompt_text=prompt,
            )))
        result.extend(self._summary(_preset(
            preset_id=preset_id, name=name, built_in=False, preset_type=preset_type, prompt_text=prompt,
        )) for preset_id, (name, preset_type, prompt) in sorted(custom.items()))
        return tuple(result)

    def get(self, preset_id: str) -> NlPromptPreset:
        if preset_id == BUILTIN_V4_BASE_PRESET_ID:
            return _preset(
                preset_id=BUILTIN_V4_BASE_PRESET_ID,
                name=V4_BASE_PROMPT_VERSION,
                built_in=True,
                preset_type="general",
                prompt_text=load_default_system_prompt(prompt_version=V4_BASE_PROMPT_VERSION),
            )
        overrides, custom = self._load_records()
        definition = _BUILTIN_BY_ID.get(preset_id)
        if definition is not None:
            _, name, preset_type, prompt_version = definition
            return _preset(
                preset_id=preset_id,
                name=name,
                built_in=True,
                preset_type=preset_type,
                prompt_text=overrides.get(preset_id) or load_default_system_prompt(prompt_version=prompt_version),
            )
        record = custom.get(preset_id)
        if record is None:
            raise PromptPresetNotFoundError("prompt preset was not found")
        return _preset(preset_id=preset_id, name=record[0], built_in=False, preset_type=record[1], prompt_text=record[2])

    def create(
        self,
        *,
        name: str,
        preset_type: NlPresetType = "general",
        prompt_text: str | None = None,
        base_prompt: str | None = None,
    ) -> NlPromptPreset:
        valid_name = _validate_text(name, field="name", limit=MAX_NAME_BYTES)
        valid_type = _validate_type(preset_type)
        valid_prompt = _prompt_value(prompt_text, base_prompt)
        overrides, custom = self._load_records()
        if len(custom) >= MAX_CUSTOM_PRESETS:
            raise PromptPresetValidationError("prompt preset limit is exceeded")
        preset_id = f"custom:{uuid.uuid4().hex}"
        while preset_id in custom:
            preset_id = f"custom:{uuid.uuid4().hex}"
        custom[preset_id] = (valid_name, valid_type, valid_prompt)
        self._write_records(overrides, custom)
        return _preset(preset_id=preset_id, name=valid_name, built_in=False, preset_type=valid_type, prompt_text=valid_prompt)

    def update(
        self,
        preset_id: str,
        *,
        name: str,
        preset_type: NlPresetType = "general",
        prompt_text: str | None = None,
        base_prompt: str | None = None,
    ) -> NlPromptPreset:
        valid_name = _validate_text(name, field="name", limit=MAX_NAME_BYTES)
        valid_type = _validate_type(preset_type)
        valid_prompt = _prompt_value(prompt_text, base_prompt)
        overrides, custom = self._load_records()
        definition = _BUILTIN_BY_ID.get(preset_id)
        if definition is not None:
            _, stable_name, stable_type, _ = definition
            if valid_name != stable_name or valid_type != stable_type:
                raise PromptPresetConflictError("built-in preset name and type are fixed")
            overrides[preset_id] = valid_prompt
            self._write_records(overrides, custom)
            return self.get(preset_id)
        if preset_id == BUILTIN_V4_BASE_PRESET_ID:
            raise PromptPresetConflictError("legacy built-in prompt preset is immutable")
        if preset_id not in custom:
            raise PromptPresetNotFoundError("prompt preset was not found")
        custom[preset_id] = (valid_name, valid_type, valid_prompt)
        self._write_records(overrides, custom)
        return self.get(preset_id)

    def reset(self, preset_id: str) -> NlPromptPreset:
        if preset_id not in _BUILTIN_BY_ID:
            if preset_id == BUILTIN_V4_BASE_PRESET_ID:
                raise PromptPresetConflictError("legacy built-in prompt preset is immutable")
            raise PromptPresetNotFoundError("prompt preset was not found")
        overrides, custom = self._load_records()
        overrides.pop(preset_id, None)
        self._write_records(overrides, custom)
        return self.get(preset_id)

    def delete(self, preset_id: str) -> None:
        if preset_id in _BUILTIN_BY_ID or preset_id == BUILTIN_V4_BASE_PRESET_ID:
            raise PromptPresetConflictError("built-in prompt preset is immutable")
        overrides, custom = self._load_records()
        if preset_id not in custom:
            raise PromptPresetNotFoundError("prompt preset was not found")
        del custom[preset_id]
        self._write_records(overrides, custom)
