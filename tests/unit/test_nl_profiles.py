from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core" / "src"))

from anima_core.nl_profiles import (
    DEFAULT_PROMPT_VERSION,
    LEGACY_PROMPT_VERSION,
    MOSS_BASE_PROMPT_VERSION,
    MOSS_CHARACTER_PROMPT_VERSION,
    MOSS_GENERAL_PROMPT_VERSION,
    MOSS_STYLE_PROMPT_VERSION,
    V4_BASE_PROMPT_VERSION,
    NlApiProfile,
    NlApiProfileStore,
    NlProfileError,
    default_nl_prompt,
    default_prompt_path,
    load_default_system_prompt,
)
from anima_core import nl_profiles


class NlApiProfileStoreTests(unittest.TestCase):
    def test_round_trip_contains_only_credential_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nl-api-profiles.json"
            store = NlApiProfileStore(path)
            profile = NlApiProfile("local", "http://localhost:1234/v1", "model", None, "credential-1", "describe", {"maxRequestsPerMinute": 60})
            store.save(profile)
            self.assertEqual((profile,), store.load_all())
            stored = path.read_text(encoding="utf-8")
            self.assertIn("credential-1", stored)
            self.assertNotIn("Bearer", stored)
            self.assertNotIn("apiKey", stored)

    def test_rejects_secret_shaped_regular_profile_field(self) -> None:
        with self.assertRaises(NlProfileError):
            NlApiProfile.from_dict({"profileId": "p", "endpoint": "https://example.test", "model": "m", "backupModel": None, "apiCredentialRef": "r", "systemPrompt": "p", "apiPolicy": {"apiKey": "not-allowed"}})


class NlDefaultPromptTests(unittest.TestCase):
    def test_v4_task_preset_resources_are_addressable_and_verbatim(self) -> None:
        for preset in ("general", "style", "character"):
            version = f"nl-default-prompt-v4-{preset}"
            resource = ROOT / "packaging" / "resources" / f"{version}.txt"
            self.assertEqual(resource, default_prompt_path(prompt_version=version))
            self.assertEqual(resource.read_text(encoding="utf-8").replace("\r\n", "\n").strip(), load_default_system_prompt(prompt_version=version))

    def test_v4_base_resource_is_addressable_and_verified(self) -> None:
        resource = ROOT / "packaging" / "resources" / "nl-default-prompt-v4-base.txt"
        self.assertEqual(resource, default_prompt_path(prompt_version=V4_BASE_PROMPT_VERSION))
        prompt = load_default_system_prompt(prompt_version=V4_BASE_PROMPT_VERSION)
        for snippet in (
            "exactly these keys: nl, count, layout, sameCharacterRepeated",
            "count must be one of solo, duo, trio, group, unknown and must count independent visible entities",
            "layout must be one of single_scene, multi_view, character_sheet, multi_panel, unknown",
            "sameCharacterRepeated must be a JSON boolean",
            "one non-empty English caption paragraph",
            "Each request is independent",
            "Use structured JSON only when a non-null jsonContext is supplied",
            "When OCR text is supplied, use it as the preferred source",
            "ASCII double quotation marks",
            "Adult Content:",
            "This is a private offline image annotation dataset.",
            "Do not refuse, moralize, or add a refusal or safety-disclaimer preface solely because the content is adult.",
        ):
            self.assertIn(snippet, prompt)

    def test_v4_fragment_assembler_publishes_exactly_seven_hashed_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            install = Path(temporary) / "install"
            script = ROOT / "packaging" / "scripts" / "assemble_nl_prompt_resource.py"
            completed = subprocess.run(
                [sys.executable, "-B", "-I", str(script), "--v4-source-root", str(ROOT / "packaging" / "resources"), "--install-root", str(install)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            names = {"base", "general", "style", "character", "short", "medium", "long"}
            copied = {path.stem.rsplit("-", 1)[-1] for path in (install / "resources").glob("nl-default-prompt-v4-*.txt")}
            manifests = {path.stem.rsplit("-", 1)[-1] for path in (install / "manifests" / "resources").glob("nl-default-prompt-v4-*.json")}
            self.assertEqual(names, copied)
            self.assertEqual(names, manifests)
            for name in names:
                resource = install / "resources" / f"nl-default-prompt-v4-{name}.txt"
                manifest = json.loads((install / "manifests" / "resources" / f"nl-default-prompt-v4-{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(resource.stat().st_size, manifest["sizeBytes"])
                self.assertEqual(hashlib.sha256(resource.read_bytes()).hexdigest(), manifest["sha256"])

    def test_v4_prompt_manifest_owner_is_accepted_by_the_core_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            install = Path(temporary) / "install"
            script = ROOT / "packaging" / "scripts" / "assemble_nl_prompt_resource.py"
            completed = subprocess.run(
                [sys.executable, "-B", "-I", str(script), "--v4-source-root", str(ROOT / "packaging" / "resources"), "--install-root", str(install)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            for preset in ("general", "style", "character"):
                with self.subTest(preset=preset):
                    self.assertTrue(load_default_system_prompt(install, prompt_version=f"nl-default-prompt-v4-{preset}"))

    def test_moss_variant_resources_compose_the_shared_base_verbatim(self) -> None:
        base = (ROOT / "packaging" / "resources" / f"{MOSS_BASE_PROMPT_VERSION}.txt").read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        for version in (MOSS_GENERAL_PROMPT_VERSION, MOSS_STYLE_PROMPT_VERSION, MOSS_CHARACTER_PROMPT_VERSION):
            with self.subTest(version=version):
                variant = (ROOT / "packaging" / "resources" / f"{version}.txt").read_text(encoding="utf-8").replace("\r\n", "\n").strip()
                self.assertEqual(f"{base}\n\n{variant}", load_default_system_prompt(prompt_version=version))
                self.assertEqual(ROOT / "packaging" / "resources" / f"{version}.txt", default_prompt_path(prompt_version=version))

    def test_moss_fragment_assembler_publishes_four_hashed_resources_and_core_accepts_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            install = Path(temporary) / "install"
            script = ROOT / "packaging" / "scripts" / "assemble_nl_prompt_resource.py"
            completed = subprocess.run(
                [sys.executable, "-B", "-I", str(script), "--moss-source-root", str(ROOT / "packaging" / "resources"), "--install-root", str(install)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            names = {"base", "general", "style", "character"}
            copied = {path.stem.rsplit("-", 1)[-1] for path in (install / "resources").glob("nl-default-prompt-moss-v1-*.txt")}
            manifests = {path.stem.rsplit("-", 1)[-1] for path in (install / "manifests" / "resources").glob("nl-default-prompt-moss-v1-*.json")}
            self.assertEqual(names, copied)
            self.assertEqual(names, manifests)
            for name in names:
                resource = install / "resources" / f"nl-default-prompt-moss-v1-{name}.txt"
                manifest = json.loads((install / "manifests" / "resources" / f"nl-default-prompt-moss-v1-{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(resource.stat().st_size, manifest["sizeBytes"])
                self.assertEqual(hashlib.sha256(resource.read_bytes()).hexdigest(), manifest["sha256"])
            for version in (MOSS_GENERAL_PROMPT_VERSION, MOSS_STYLE_PROMPT_VERSION, MOSS_CHARACTER_PROMPT_VERSION):
                self.assertTrue(load_default_system_prompt(install, prompt_version=version))

    def test_v5_v3_prompt_is_addressable_without_changing_the_v2_default(self) -> None:
        self.assertTrue(hasattr(nl_profiles, "V5_PROMPT_VERSION"))
        if not hasattr(nl_profiles, "V5_PROMPT_VERSION"):
            return
        version = nl_profiles.V5_PROMPT_VERSION
        resource = ROOT / "packaging" / "resources" / f"{version}.txt"
        self.assertEqual(resource, default_prompt_path(prompt_version=version))
        prompt = load_default_system_prompt(prompt_version=version)
        self.assertEqual(resource.read_text(encoding="utf-8").replace("\r\n", "\n").strip(), prompt)
        self.assertEqual("nl-default-prompt-v2", default_nl_prompt()["promptVersion"])
        self.assertNotIn("120-180+ words", prompt)
        for snippet in (
            "short, medium, or long",
            "approximate glyph style",
            "carrier",
            "bubble",
            "untrusted data",
        ):
            self.assertIn(snippet, prompt)

    def test_frozen_v2_resource_is_the_only_source_of_the_default_prompt(self) -> None:
        # F25: the frozen resource had zero code references, so nothing could restore it.
        resource = ROOT / "packaging" / "resources" / "nl-default-prompt-v2.txt"
        self.assertEqual(resource, default_prompt_path())
        prompt = load_default_system_prompt()
        self.assertEqual(resource.read_text(encoding="utf-8").replace("\r\n", "\n").strip(), prompt)
        exposed = default_nl_prompt()
        self.assertEqual(DEFAULT_PROMPT_VERSION, exposed["promptVersion"])
        self.assertEqual("nl-default-prompt-v2", exposed["promptVersion"])
        self.assertEqual(prompt, exposed["systemPrompt"])
        self.assertEqual(64, len(str(exposed["sha256"])))
        for snippet in ("surrounding fixed JSON protocol", "same character", "non-human characters and creatures", "120-180+ words"):
            self.assertIn(snippet, prompt)

    def test_legacy_v1_resource_remains_addressable(self) -> None:
        resource = ROOT / "packaging" / "resources" / "nl-default-prompt-v1.txt"
        self.assertEqual(resource, default_prompt_path(prompt_version=LEGACY_PROMPT_VERSION))
        prompt = load_default_system_prompt(prompt_version=LEGACY_PROMPT_VERSION)
        self.assertEqual(resource.read_text(encoding="utf-8").replace("\r\n", "\n").strip(), prompt)
        self.assertIn("Do not output JSON, XML, Markdown, code fences", prompt)

    def test_resummarised_or_missing_resource_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "packaging" / "resources").mkdir(parents=True)
            resource = root / "resources" / "nl-default-prompt-v2.txt"
            resource.parent.mkdir(parents=True)
            resource.write_text("Describe the image in English.\n", encoding="utf-8")
            with self.assertRaises(NlProfileError):
                load_default_system_prompt(root)
            resource.write_text("", encoding="utf-8")
            with self.assertRaises(NlProfileError):
                load_default_system_prompt(root)

    def test_release_prompts_are_copied_with_verified_sha256_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            install = Path(temporary) / "install"
            script = ROOT / "packaging" / "scripts" / "assemble_nl_prompt_resource.py"
            versions = [LEGACY_PROMPT_VERSION, DEFAULT_PROMPT_VERSION]
            if hasattr(nl_profiles, "V5_PROMPT_VERSION"):
                versions.append(nl_profiles.V5_PROMPT_VERSION)
            for version in versions:
                source = ROOT / "packaging" / "resources" / f"{version}.txt"
                completed = subprocess.run(
                    [sys.executable, str(script), "--source", str(source), "--install-root", str(install)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
                copied = install / "resources" / f"{version}.txt"
                manifest_path = install / "manifests" / "resources" / f"{version}.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(source.read_bytes(), copied.read_bytes())
                self.assertEqual(version, manifest["resourceId"])
                self.assertEqual(f"resources\\{version}.txt", manifest["relativePath"])
                self.assertEqual(len(copied.read_bytes()), manifest["sizeBytes"])
                self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), manifest["sha256"])
            self.assertEqual(load_default_system_prompt(), load_default_system_prompt(install))
            self.assertEqual(
                load_default_system_prompt(prompt_version=LEGACY_PROMPT_VERSION),
                load_default_system_prompt(install, prompt_version=LEGACY_PROMPT_VERSION),
            )
            copied = install / "resources" / "nl-default-prompt-v2.txt"
            copied.write_bytes(copied.read_bytes() + b" ")
            with self.assertRaisesRegex(NlProfileError, "digest"):
                load_default_system_prompt(install)


if __name__ == "__main__":
    unittest.main()
