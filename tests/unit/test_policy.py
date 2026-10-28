from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workers" / "policy" / "src"))
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_policy_worker.policy import (
    CoupledProbabilities,
    PolicyConfig,
    PolicyError,
    apply_policy,
    artist_from_image_path,
    quality_for_score,
    stable_random,
)
from anima_core.contracts import JobConfig, validate_job_config


def _payload(*, count: str = "solo", appearance: list[str] | None = None, nl: str = "A person smiles.") -> dict[str, object]:
    return {
        "quality": [],
        "count": count,
        "character": "amy_rose",
        "series": "",
        "artist": "",
        "appearance": ["white hair"] if appearance is None else appearance,
        "tags": ["smile"],
        "environment": ["outdoors"],
        "nl": nl,
    }


def _config() -> PolicyConfig:
    return PolicyConfig(
        seed="fixed-seed",
        artistEnabled=True,
        artistDropoutProbability=0.0,
        qualityEnabled=True,
        qualityDropoutProbability=0.0,
        appearanceNlEnabled=True,
        solo=CoupledProbabilities(0.70, 0.05),
        nonSolo=CoupledProbabilities(0.05, 0.70),
        unknown=CoupledProbabilities(0.15, 0.15),
    )


class PolicyTests(unittest.TestCase):
    def test_artist_uses_first_directory_and_preserves_suffix(self) -> None:
        self.assertEqual("@Crow (Siranui)", artist_from_image_path(r"1_Crow (Siranui)\image.png"))
        self.assertEqual("@6suan", artist_from_image_path(r"10_6suan\nested\image.webp"))
        self.assertEqual("@artist", artist_from_image_path(r"1_artist\image.jpg"))
        for invalid in ("image.png", r"artist\image.png", r"1_\image.png", r"..\1_artist\image.png"):
            with self.subTest(path=invalid), self.assertRaises(PolicyError):
                artist_from_image_path(invalid)

    def test_artist_setting_appends_folder_artist_to_existing_artist(self) -> None:
        payload = _payload()
        payload["artist"] = "kannos"
        result, _ = apply_policy(
            payload,
            annotation_key=r"1_Crow\image",
            relative_image_path=r"1_Crow\image.png",
            config=_config(),
            aesthetic_score=4.0,
        )
        self.assertEqual("kannos, @Crow", result["artist"])

    def test_artist_prefix_can_be_disabled(self) -> None:
        payload = _payload()
        result, _ = apply_policy(
            payload,
            annotation_key=r"1_Crow\image",
            relative_image_path=r"1_Crow\image.png",
            config=replace(_config(), artistAddAtPrefix=False),
            aesthetic_score=4.0,
        )
        self.assertEqual("Crow", result["artist"])

    def test_artist_setting_deduplicates_case_insensitively(self) -> None:
        payload = _payload()
        payload["artist"] = "kannos, @crow"
        result, _ = apply_policy(
            payload,
            annotation_key=r"1_Crow\image",
            relative_image_path=r"1_Crow\image.png",
            config=_config(),
            aesthetic_score=4.0,
        )
        self.assertEqual("kannos, @crow", result["artist"])

    def test_disabled_artist_setting_preserves_existing_artist(self) -> None:
        payload = _payload()
        payload["artist"] = "kannos"
        result, _ = apply_policy(
            payload,
            annotation_key=r"1_Crow\image",
            relative_image_path=r"1_Crow\image.png",
            config=replace(_config(), artistEnabled=False),
            aesthetic_score=4.0,
        )
        self.assertEqual("kannos", result["artist"])

    def test_artist_dropout_clears_original_and_appended_artists(self) -> None:
        payload = _payload()
        payload["artist"] = "kannos"
        result, decision = apply_policy(
            payload,
            annotation_key=r"1_Crow\image",
            relative_image_path=r"1_Crow\image.png",
            config=replace(_config(), artistDropoutProbability=1.0),
            aesthetic_score=4.0,
        )
        self.assertTrue(decision.artistDropped)
        self.assertEqual("", result["artist"])

    def test_quality_boundaries_cover_the_frozen_one_to_five_range(self) -> None:
        cases = (
            (1.0, ["low quality"]),
            (2.0, ["low quality"]),
            (2.01, ["normal quality"]),
            (3.0, ["normal quality"]),
            (3.01, ["good quality"]),
            (4.0, ["good quality"]),
            (4.01, ["masterpiece", "best quality"]),
            (5.0, ["masterpiece", "best quality"]),
        )
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(expected, quality_for_score(score))
        for invalid in (float("nan"), float("inf"), 0.99, 5.01):
            with self.subTest(score=invalid), self.assertRaises(PolicyError):
                quality_for_score(invalid)

    def test_coupled_dropout_can_never_remove_appearance_and_nl_together(self) -> None:
        config = _config()
        original = _payload()
        with patch("anima_policy_worker.policy.stable_random", return_value=0.0):
            dropped_nl, decision = apply_policy(
                original,
                annotation_key=r"1_Crow (Siranui)\image",
                relative_image_path=r"1_Crow (Siranui)\image.png",
                config=config,
                aesthetic_score=4.5,
            )
        self.assertEqual("drop_nl", decision.appearanceNlAction)
        self.assertEqual("", dropped_nl["nl"])
        self.assertEqual(["white hair"], dropped_nl["appearance"])

        with patch("anima_policy_worker.policy.stable_random", return_value=0.71):
            dropped_appearance, decision = apply_policy(
                original,
                annotation_key=r"1_Crow (Siranui)\image",
                relative_image_path=r"1_Crow (Siranui)\image.png",
                config=config,
                aesthetic_score=4.5,
            )
        self.assertEqual("drop_appearance", decision.appearanceNlAction)
        self.assertEqual([], dropped_appearance["appearance"])
        self.assertEqual("A person smiles.", dropped_appearance["nl"])

    def test_count_selects_solo_or_non_solo_dropout_policy(self) -> None:
        cases = (
            ("solo", "drop_nl"),
            ("duo", "drop_appearance"),
            ("trio", "drop_appearance"),
            ("group", "drop_appearance"),
        )
        for count, expected_action in cases:
            with self.subTest(count=count), patch(
                "anima_policy_worker.policy.stable_random", return_value=0.10
            ):
                result, decision = apply_policy(
                    _payload(count=count),
                    annotation_key=rf"1_Crow (Siranui)\{count}",
                    relative_image_path=rf"1_Crow (Siranui)\{count}.png",
                    config=_config(),
                    aesthetic_score=4.5,
                )
                self.assertEqual(expected_action, decision.appearanceNlAction)
                self.assertEqual(count, result["count"])
                self.assertEqual(expected_action != "drop_nl", bool(result["nl"]))
                self.assertEqual(expected_action != "drop_appearance", bool(result["appearance"]))

    def test_an_existing_empty_side_protects_the_other_side(self) -> None:
        config = _config()
        inputs = (
            (_payload(appearance=[]), [], "A person smiles."),
            (_payload(nl=""), ["white hair"], ""),
            (_payload(appearance=[], nl=""), [], ""),
        )
        for original, expected_appearance, expected_nl in inputs:
            with self.subTest(original=original), patch(
                "anima_policy_worker.policy.stable_random", return_value=0.0
            ):
                result, decision = apply_policy(
                    original,
                    annotation_key=r"1_Crow (Siranui)\image",
                    relative_image_path=r"1_Crow (Siranui)\image.png",
                    config=config,
                    aesthetic_score=4.5,
                )
                self.assertEqual("unchanged", decision.appearanceNlAction)
                self.assertEqual(expected_appearance, result["appearance"])
                self.assertEqual(expected_nl, result["nl"])

    def test_protected_fields_remain_byte_for_byte_equivalent(self) -> None:
        original = _payload(count="duo")
        with patch("anima_policy_worker.policy.stable_random", return_value=0.1):
            result, _ = apply_policy(
                original,
                annotation_key=r"10_6suan\image",
                relative_image_path=r"10_6suan\image.png",
                config=_config(),
                aesthetic_score=3.5,
            )
        for field in ("count", "character", "series", "tags", "environment"):
            self.assertEqual(original[field], result[field])

    def test_stable_random_is_repeatable_and_decision_specific(self) -> None:
        config = _config()
        first = stable_random(config, r"1_artist\image", "artist")
        self.assertEqual(first, stable_random(config, r"1_artist\image", "artist"))
        self.assertEqual(first, stable_random(config, "1_ARTIST/image", "artist"))
        self.assertNotEqual(first, stable_random(config, r"1_artist\image", "quality"))

    def test_invalid_coupled_probability_sum_is_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            CoupledProbabilities(0.7, 0.31)

    def test_job_config_freezes_valid_policy_defaults_and_rejects_double_dropout_probability(self) -> None:
        config = JobConfig(
            workMode="in_place",
            overwriteMode="incremental",
            sourceRoot=r"E:\dataset",
        )
        validate_job_config(config)
        self.assertEqual(0.70, config.dropout["appearanceNl"]["solo"]["dropNl"])
        self.assertEqual(0.05, config.dropout["appearanceNl"]["solo"]["dropAppearance"])
        config.dropout["appearanceNl"]["solo"] = {"dropNl": 0.70, "dropAppearance": 0.31}
        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            validate_job_config(config)


if __name__ == "__main__":
    unittest.main()
