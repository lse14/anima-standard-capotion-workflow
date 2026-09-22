from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workers" / "export" / "src"))
sys.path.insert(0, str(ROOT / "shared" / "anima_caption_format"))

from anima_export_worker.normalizer import CaptionDisplayPolicy, MAX_JSON_BYTES, MAX_NL_BYTES, normalize_json_bytes
from anima_export_worker.flat_txt import FlatTextSerializationError, serialize_flat_txt


POLICY = CaptionDisplayPolicy(True, True, True, ("anima_style",), "nl_newline")


def _payload(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "quality": [], "count": "solo", "character": "amy_rose", "series": "sonic", "artist": "",
        "appearance": ["blue_eyes"], "tags": ["smile"], "environment": ["outdoors"], "nl": "Amy smiles, happily.",
    }
    value.update(changes)
    return value


def _normalise(value: object, policy: CaptionDisplayPolicy = POLICY):
    return normalize_json_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), policy)


class ExportNormalizerTests(unittest.TestCase):
    def test_canonical_payload_preserves_every_business_value(self) -> None:
        payload = _payload(
            quality=["high_quality"], count="duo", character="amy_rose, blaze", series="sonic",
            artist="artist_(name)", appearance=["blue_eyes"], tags=["smile", "looking_at_viewer"],
            environment=["outdoors"], nl="Amy smiles, happily.",
        )
        result = _normalise(payload)
        self.assertTrue(result.valid)
        self.assertEqual(payload, result.payload)
        self.assertEqual({}, result.conversions)

    def test_canonical_bytes_missing_defaults_and_safe_conversions_are_idempotent(self) -> None:
        raw = _payload(quality=" high_quality, high_quality, ", character="amy_rose, Amy_Rose", tags=["smile, grin", "GRIN"], nl=["  A\nsmile, here.  "])
        raw.pop("artist")
        result = _normalise(raw)
        self.assertTrue(result.valid)
        self.assertEqual("", result.payload["artist"])
        self.assertEqual(["high_quality"], result.payload["quality"])
        self.assertEqual("amy_rose", result.payload["character"])
        self.assertEqual(["smile", "grin"], result.payload["tags"])
        self.assertEqual("A smile, here.", result.payload["nl"])
        self.assertFalse(result.json_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertTrue(result.json_bytes.endswith(b"\n"))
        self.assertEqual(
            ["quality", "count", "character", "series", "artist", "appearance", "tags", "environment", "nl"],
            list(json.loads(result.json_bytes)),
        )
        second = normalize_json_bytes(result.json_bytes, POLICY)
        self.assertEqual(result.json_bytes, second.json_bytes)

    def test_utf8_bom_and_all_strict_json_failures_are_distinctly_blocking(self) -> None:
        valid = b"\xef\xbb\xbf" + json.dumps(_payload()).encode("utf-8")
        self.assertTrue(normalize_json_bytes(valid, POLICY).valid)
        cases = (b"{\"tags\":NaN}", b"{\"tags\":[],\"tags\":[]}", b"{/*x*/}", b'{"tags":[],}', b"\xff")
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertFalse(normalize_json_bytes(raw, POLICY).valid)

    def test_size_type_extra_nested_and_count_errors_do_not_guess(self) -> None:
        cases = (
            (b"", "json_missing_or_blank"),
            (b" " * (MAX_JSON_BYTES + 1), "json_too_large"),
            (json.dumps([1]).encode(), "json_root_not_object"),
            (json.dumps(_payload(extra="x")).encode(), "extra_field"),
            (json.dumps(_payload(tags=["ok", ["nested"]])).encode(), "array_element_type_invalid"),
            (json.dumps(_payload(series=3)).encode(), "field_type_invalid"),
            (json.dumps(_payload(artist=True)).encode(), "field_type_invalid"),
            (json.dumps(_payload(count={"value": "solo"})).encode(), "field_type_invalid"),
            (json.dumps(_payload(nl=["one", "two"])).encode(), "field_type_invalid"),
            (json.dumps(_payload(count="2 characters")).encode(), "count_invalid"),
        )
        for raw, code in cases:
            with self.subTest(code=code):
                self.assertIn(code, {error.code for error in normalize_json_bytes(raw, POLICY).field_errors})

    def test_control_nl_limit_format_and_trigger_collisions_are_blocking(self) -> None:
        cases = (
            _payload(tags=["line\nbreak"]),
            _payload(nl="bad\x00control"),
            _payload(nl="a" * (MAX_NL_BYTES + 1)),
            _payload(series="Smile", tags=["smile"]),
            _payload(series="blue eyes", appearance=["blue_eyes"]),
            _payload(tags=["anima_style"]),
        )
        for value in cases:
            with self.subTest(value=value):
                self.assertFalse(_normalise(value).valid)

    def test_unicode_and_empty_nl_are_valid_but_wholly_empty_payload_is_not(self) -> None:
        valid = _normalise(_payload(character="初音ミク", tags=["笑顔"], nl=""))
        self.assertTrue(valid.valid)
        self.assertIn("初音ミク".encode("utf-8"), valid.json_bytes)
        empty = _normalise({field: [] if field in {"quality", "appearance", "tags", "environment"} else "" for field in (
            "quality", "count", "character", "series", "artist", "appearance", "tags", "environment", "nl",
        )})
        self.assertIn("payload_all_empty", {error.code for error in empty.field_errors})

    def test_flat_txt_has_frozen_field_order_separators_and_no_normalizer_dependency(self) -> None:
        result = _normalise(_payload(
            quality=["high_quality"], count="solo", character="amy_rose, blaze", series="sonic",
            artist="artist_(name)", appearance=["blue_eyes"], tags=["smile"], environment=["outdoors"],
            nl="A smile, outside",
        ))
        self.assertTrue(result.valid)
        output = serialize_flat_txt(result.payload, POLICY).decode("utf-8")
        self.assertEqual(
            "anima style, high quality, solo, amy rose, blaze, sonic, artist \\(name\\), blue eyes, smile, outdoors\nA smile, outside.",
            output,
        )
        self.assertEqual(output, serialize_flat_txt(result.payload, POLICY).decode("utf-8"))
        only_tags = _normalise(_payload(quality=[], count="", character="", series="", artist="", appearance=[], tags=["smile"], environment=[], nl=""))
        self.assertEqual("anima style, smile.", serialize_flat_txt(only_tags.payload, POLICY).decode("utf-8"))
        with self.assertRaises(FlatTextSerializationError):
            serialize_flat_txt({"tags": []}, POLICY)

    def test_tag_field_and_nl_punctuation_are_isolated(self) -> None:
        result = _normalise(_payload(
            quality=["high_quality", "best_quality"],
            character="amy_rose, blaze",
            appearance=["green_eyes"],
            tags=["blue_eyes", "long_hair"],
            nl="Amy smiles. She rests, quietly.",
        ))
        self.assertTrue(result.valid)
        output = serialize_flat_txt(result.payload, POLICY).decode("utf-8")
        self.assertIn("high quality, best quality", output)
        self.assertIn("amy rose, blaze", output)
        self.assertIn("best quality, solo, amy rose, blaze", output)
        self.assertTrue(output.endswith("Amy smiles. She rests, quietly."))
        self.assertEqual(1, output.count("\n"))

    def test_flat_txt_serializer_does_not_invoke_semantic_normalization(self) -> None:
        payload = _payload(count="2 characters", tags=["tag_two"])
        output = serialize_flat_txt(payload, POLICY).decode("utf-8")
        self.assertIn("2 characters", output)
        self.assertIn("tag two", output)

    def test_count_bucket_label_may_repeat_inside_other_tag_fields(self) -> None:
        # F01: Classify writes the count bucket into both `count` and `tags`.
        result = _normalise(_payload(count="solo", tags=["solo", "smile"]))
        self.assertTrue(result.valid, {error.code for error in result.field_errors})
        self.assertEqual("solo", result.payload["count"])
        self.assertEqual(["solo", "smile"], result.payload["tags"])
        # Exact tags/non-tags duplicates are removed from tags; non-tags pairs
        # are intentionally retained without a field-priority collision.
        duplicate = _normalise(_payload(series="smile", tags=["smile", "other"]))
        self.assertTrue(duplicate.valid)
        self.assertEqual(["other"], duplicate.payload["tags"])
        self.assertEqual(1, duplicate.conversions["cross_field_duplicate_removed"])
        self.assertTrue(_normalise(_payload(series="smile", artist="smile", tags=[])).valid)

    def test_existing_escapes_are_not_escaped_a_second_time(self) -> None:
        # F07: ROADMAP.md:978 allows an input tag to carry its own `\(` / `\)`.
        result = _normalise(_payload(series="hazbin_hotel_\\(series\\)", artist="artist_(name)"))
        self.assertTrue(result.valid, {error.code for error in result.field_errors})
        output = serialize_flat_txt(result.payload, POLICY).decode("utf-8")
        self.assertIn("hazbin hotel \\(series\\)", output)
        self.assertNotIn("\\\\", output)
        self.assertIn("artist \\(name\\)", output)
        self.assertEqual(output, serialize_flat_txt(_normalise(result.payload).payload, POLICY).decode("utf-8"))

    def test_display_layer_checks_only_apply_when_flat_txt_is_produced(self) -> None:
        # F08: ROADMAP.md:980/1023 limit formatted and trigger collisions to flat TXT.
        formatted = _payload(series="blue eyes", appearance=["blue_eyes"])
        trigger = _payload(tags=["anima_style"])
        for value in (formatted, trigger):
            with self.subTest(value=value):
                self.assertFalse(_normalise(value).valid)
                self.assertTrue(normalize_json_bytes(json.dumps(value).encode("utf-8"), POLICY, export_format="json").valid)
        # Structural checks stay on for pure JSON exports, while exact
        # cross-field duplicate removal is format-independent.
        json_duplicate = normalize_json_bytes(json.dumps(_payload(series="smile", tags=["smile"])).encode("utf-8"), POLICY, export_format="json")
        self.assertTrue(json_duplicate.valid)
        self.assertEqual([], json_duplicate.payload["tags"])
        with self.assertRaises(ValueError):
            normalize_json_bytes(b"{}", POLICY, export_format="csv")

    def test_multi_artist_string_is_a_field_error_not_a_serializer_exception(self) -> None:
        # F22: `artist_a, artist_b` is normal on e621 and used to escape as an exception.
        result = _normalise(_payload(artist="artist_a, artist_b"))
        self.assertFalse(result.valid)
        self.assertEqual(
            {("artist", "tag_not_flat_txt_representable")},
            {(error.field, error.code) for error in result.field_errors},
        )
        json_only = normalize_json_bytes(json.dumps(_payload(artist="artist_a, artist_b")).encode("utf-8"), POLICY, export_format="json")
        self.assertTrue(json_only.valid)
        self.assertEqual("artist_a, artist_b", json_only.payload["artist"])


if __name__ == "__main__":
    unittest.main()
