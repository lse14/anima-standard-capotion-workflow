"""Validate the project-local resource library before a release is started or assembled."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPOSITORY_ROOT / "core" / "src"))

from anima_core.resource_catalog import ResourceCatalog, verify_tagger_dictionary_compatibility
from copy_resource_library import assert_no_local_only_leaks


def validate(
    root: Path,
    *,
    release: bool = False,
    require_quality_stack: bool = False,
) -> dict[str, object]:
    root = root.resolve()
    snapshot = ResourceCatalog(root).scan(require_quality_stack=require_quality_stack)
    if snapshot.invalid:
        details = "; ".join(f"{item.relative_path}: {item.reason}" for item in snapshot.invalid)
        raise ValueError(f"resource library contains invalid packages: {details}")
    for package in snapshot.packages:
        package.verify_files(verify_hashes=True)
    defaults = snapshot.defaults_for()
    tagging = snapshot.package(
        "tagging-model", defaults["taggingModel"], verify_hashes=False,
    )
    classification = snapshot.package(
        "classification-index", defaults["classificationIndex"], verify_hashes=False,
    )
    verify_tagger_dictionary_compatibility(tagging, classification)
    compatible_profiles = sorted({
        package.profile for package in (tagging, classification)
        if package.profile != "shared"
    })
    if release:
        assert_no_local_only_leaks(root)
    return {
        "schemaVersion": 2,
        "root": str(root.resolve()),
        "compatibleProfiles": compatible_profiles,
        "resources": {
            package.resource_id: package.fingerprint
            for package in sorted(snapshot.packages, key=lambda item: item.resource_id)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--require-quality-stack", action="store_true")
    arguments = parser.parse_args()
    result = validate(
        arguments.root,
        release=arguments.release,
        require_quality_stack=arguments.require_quality_stack,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
