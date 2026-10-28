from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from .count import (
    DanbooruCountRules,
    DanbooruCountRulesError,
    WikiCountError,
    WikiCountResolver,
    decide_count,
    decide_danbooru_count,
)
from .dictionary import ClassifyDictionaryError, DanbooruDictionary, E621Dictionary
from .parsing import ClassifyTextError, parse_tag_text
from .protocol import ClassifyPayloadError, validate_hello_payload, validate_work_item
from .resource import ClassifyResourceError, WorkerClassifyResource, load_classify_resource


def bounded_message(message: str, limit: int = 1_024) -> str:
    """Truncate an error message to the parent's UTF-8 byte limit.

    The core validates `message` by encoded byte length, while slicing a Python
    string counts characters. Text with three-byte characters would therefore
    pass the slice and still be rejected, which fails a whole worker batch.
    """
    raw = message.encode("utf-8")
    if len(raw) <= limit:
        return message
    return raw[:limit].decode("utf-8", errors="ignore")


class ClassifyWorkerInitializationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ClassifyWorker:
    def __init__(self) -> None:
        self.hello: dict[str, object] | None = None
        self.resource: WorkerClassifyResource | None = None
        self.dictionary: E621Dictionary | DanbooruDictionary | None = None
        self.count_rules: DanbooruCountRules | None = None
        self.resolver: WikiCountResolver | None = None

    def initialize(self, payload: object, *, install_root: Path) -> dict[str, object]:
        if self.hello is not None:
            raise ClassifyWorkerInitializationError("classify_protocol_violation", "classify worker is already initialized")
        try:
            hello = validate_hello_payload(payload)
        except ClassifyPayloadError as exc:
            raise ClassifyWorkerInitializationError("classify_protocol_violation", str(exc)) from exc
        resource: WorkerClassifyResource | None = None
        try:
            resource = load_classify_resource(
                install_root,
                str(hello["resourceManifestRelativePath"]),
                str(hello["resourceFingerprint"]),
            )
            if resource.wiki_data_source_id != hello["wikiDataSourceId"]:
                raise ClassifyResourceError("classification resource Wiki data source does not match hello")
            temporary_e621_fallback = (
                hello["profile"] == "danbooru"
                and resource.profile == "e621"
                and resource.resource_id == "classify-e621-20260724-v1"
            )
            if resource.profile != hello["profile"] and not temporary_e621_fallback:
                raise ClassifyResourceError("classification resource profile does not match hello")
            if resource.profile == "e621":
                dictionary: E621Dictionary | DanbooruDictionary = E621Dictionary(resource.dictionary)
                count_rules = None
            else:
                dictionary = DanbooruDictionary(resource.dictionary)
                if resource.count_rules is None:
                    raise ClassifyResourceError("Danbooru classification resource has no count rules")
                count_rules = DanbooruCountRules.from_payload(resource.count_rules)
                if count_rules.wiki_titles != resource.wiki_page_titles:
                    raise ClassifyResourceError("Danbooru count rules and Wiki page set do not match")
        except (
            ClassifyResourceError,
            ClassifyDictionaryError,
            DanbooruCountRulesError,
            OSError,
            ValueError,
        ) as exc:
            if resource is not None:
                resource.close()
            raise ClassifyWorkerInitializationError("classify_resource_invalid", str(exc)) from exc
        self.hello = hello
        self.resource = resource
        self.dictionary = dictionary
        self.count_rules = count_rules
        self.resolver = WikiCountResolver(resource.wiki_connection)
        return {
            "schemaVersion": 1,
            "payloadType": "classify_hello_result",
            "executable": sys.executable,
            "pythonVersion": ".".join(map(str, sys.version_info[:3])),
            "ready": True,
            "dictionaryLoads": 1,
            "wikiConnectionLoads": 1,
            "entryCount": len(dictionary.entries),
            "wikiSchemaVersion": resource.wiki_schema_version,
            "wikiDataSourceId": resource.wiki_data_source_id,
            "resourceFingerprint": resource.fingerprint,
        }

    def close(self) -> None:
        if self.resource is not None:
            self.resource.close()
            self.resource = None

    @staticmethod
    def _issue(item: dict[str, object], code: str, message: str, *, retriable: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "schemaVersion": 1,
            "payloadType": "classify_issue",
            "sampleId": item["sampleId"],
            "leaseId": item["leaseId"],
            "source": item["source"],
            "relativeImagePath": item["relativeImagePath"],
            "code": code,
            "severity": "error",
            "blocking": True,
            "retriable": retriable,
            "message": bounded_message(message),
        }
        if retriable:
            result["repairStartModule"] = "classify"
        return result

    def process(self, value: object) -> dict[str, object]:
        if self.hello is None or self.dictionary is None or self.resolver is None:
            raise ClassifyWorkerInitializationError("classify_protocol_violation", "classify worker is not initialized")
        item = validate_work_item(value)
        if item["source"] != self.hello["profile"]:
            raise ClassifyWorkerInitializationError(
                "classify_protocol_violation", "classify work item profile does not match hello"
            )
        try:
            tags = parse_tag_text(
                str(item["txtText"]),
                self.hello["captionFormat"],  # type: ignore[arg-type]
                str(item["txtProvenance"]),
            )
            projection = self.dictionary.classify(tags)
            uses_e621_count_rules = (
                self.resource.profile == "e621"
                if self.resource is not None
                else self.hello["profile"] == "e621"
            )
            if uses_e621_count_rules:
                decision = decide_count(
                    item["originalCount"],  # type: ignore[arg-type]
                    projection.evidence_tags,
                    projection.canonical_character_ids,
                    projection.evidence_tags,
                    self.resolver,
                    bool(self.hello["overwriteCount"]),
                )
            else:
                if self.count_rules is None:
                    raise ClassifyDictionaryError("Danbooru count rules are unavailable")
                decision = decide_danbooru_count(
                    item["originalCount"],  # type: ignore[arg-type]
                    projection.evidence_tags,
                    self.resolver,
                    self.count_rules,
                    bool(self.hello["overwriteCount"]),
                )
        except WikiCountError as exc:
            return self._issue(item, "classify_wiki_io_failed", f"Wiki count query failed: {exc}", retriable=True)
        except ClassifyTextError as exc:
            # Caption defects are deterministic: retrying the same TXT can never converge.
            return self._issue(item, "classify_text_invalid", f"Caption text is invalid: {exc}", retriable=False)
        except (ClassifyDictionaryError, sqlite3.Error) as exc:
            return self._issue(item, "classify_processing_failed", f"Classification failed: {exc}", retriable=True)
        if not projection.output_tag_count:
            return self._issue(
                item, "classify_no_writable_tags", "caption produced no writable classification tag", retriable=False
            )
        if decision.blocking_code is not None:
            return self._issue(item, decision.blocking_code, decision.blocking_code.replace("_", " "), retriable=False)
        return {
            "schemaVersion": 1,
            "payloadType": "classify_result",
            "sampleId": item["sampleId"],
            "leaseId": item["leaseId"],
            "source": item["source"],
            "relativeImagePath": item["relativeImagePath"],
            "projection": projection.to_json_projection(decision.value),
            "countDecision": {
                "value": decision.value,
                "baseValue": decision.base_value,
                "selectedSource": decision.selected_source,
                "originalRaw": decision.original_raw,
                "originalNormalized": decision.original_normalized,
                "wikiValue": decision.wiki_value,
                "matchedTags": list(decision.matched_tags),
                "conflict": decision.conflict,
                "issueCodes": list(decision.issue_codes),
                "warnings": list(decision.warnings),
                "appliedLowerBounds": list(decision.applied_lower_bounds),
            },
            "inputTagCount": projection.input_tag_count,
            "outputTagCount": projection.output_tag_count,
            "droppedTagCount": projection.dropped_tag_count,
        }

    def process_batch(self, items: list[dict[str, object]]) -> list[dict[str, object]]:
        """Reuse initialized resources while preserving item-local outcomes."""
        return [self.process(item) for item in items]
