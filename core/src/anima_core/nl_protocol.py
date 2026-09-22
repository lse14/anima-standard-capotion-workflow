from __future__ import annotations

import re
from dataclasses import dataclass

from .count_review_protocol import CountObservationV1, CountReviewProtocolError
from .path_safety import PathSafetyError, safe_relative_path


MAX_NL_BYTES = 16_384
REFUSAL_SNIPPETS = (
    "no image was provided", "i cannot analyze images", "i can't analyze images",
    "i cannot assist with that", "i can't assist with that", "i cannot help with that",
    "i can't help with that", "i cannot generate", "i can't generate",
    "i am unable to help", "i'm unable to help", "as an ai language model",
    "content policy", "content_policy", "request was blocked", "request has been blocked",
    "request was rejected", "moderation", "policy violation",
)
# F29: the last gate before overlay must also reject localized or proxy-custom moderation text.
REFUSAL_PATTERN = re.compile("|".join((
    r"无法(?:分析|识别|处理|查看|描述|生成|提供|回答|完成)",
    r"不能(?:分析|识别|处理|查看|描述|生成|提供|回答|协助|帮助)",
    r"(?:抱歉|对不起|很遗憾)[^\n]{0,8}?(?:无法|不能|不便|不予)",
    r"(?:内容|安全|合规|风控)?(?:审核|审查)(?:未通过|不通过|失败|拦截)",
    r"(?:命中|触发)(?:了)?(?:敏感|违规|风控|安全)",
    r"违反(?:了)?[^\n]{0,8}?(?:政策|规定|规范|条款|法律法规)",
    r"(?:已)?被(?:拦截|屏蔽|拒绝)",
    r"敏感(?:词|内容|信息)",
    r"请求(?:被)?(?:拒绝|拦截|屏蔽)",
)))


class NlProtocolError(ValueError):
    pass


def validate_nl(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value.encode("utf-8")) > MAX_NL_BYTES:
        raise NlProtocolError("NL result is invalid")
    nl = value.strip()
    if any(text in nl.casefold() for text in REFUSAL_SNIPPETS) or REFUSAL_PATTERN.search(nl) is not None or nl[-1] in ",:;-/":
        raise NlProtocolError("NL result is refused or truncated")
    return nl


@dataclass(frozen=True)
class NlOutcomeV1:
    sampleId: int
    leaseId: str
    relativeImagePath: str
    nl: str | None
    code: str | None
    retriable: bool
    httpAttempts: int
    observation: CountObservationV1 | None = None
    message: str | None = None


def _valid_usage(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) <= {"prompt_tokens", "completion_tokens", "total_tokens"}
        and all(type(item) is int and 0 <= item <= 1_000_000 for item in value.values())
    )


def parse_outcomes(
    payload: object,
    expected: dict[tuple[int, str], str],
    *,
    response_protocol: str = "nl-v1",
) -> tuple[NlOutcomeV1, ...]:
    if response_protocol not in {"nl-v1", "nl-count-v2"}:
        raise NlProtocolError("NL response protocol is invalid")
    if not isinstance(payload, dict) or set(payload) != {"schemaVersion", "payloadType", "items"} or payload.get("schemaVersion") != 1 or payload.get("payloadType") != "nl_process_result" or not isinstance(payload.get("items"), list) or len(payload["items"]) != len(expected):
        raise NlProtocolError("NL process result is invalid")
    outcomes: list[NlOutcomeV1] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise NlProtocolError("NL result item is invalid")
        common = {"schemaVersion", "payloadType", "sampleId", "leaseId", "relativeImagePath"}
        if not common.issubset(item) or item["schemaVersion"] != 1 or type(item["sampleId"]) is not int or not isinstance(item["leaseId"], str) or not isinstance(item["relativeImagePath"], str):
            raise NlProtocolError("NL result identity is invalid")
        key = (item["sampleId"], item["leaseId"])
        if key not in expected or item["relativeImagePath"] != expected[key]:
            raise NlProtocolError("NL result does not match a leased sample")
        try:
            safe_relative_path(item["relativeImagePath"])
        except PathSafetyError as exc:
            raise NlProtocolError("NL result relative path is unsafe") from exc
        expected_result_type = "nl_result" if response_protocol == "nl-v1" else "nl_result_v2"
        if item["payloadType"] == expected_result_type:
            result_fields = {"nl", "requestId", "usage", "httpAttempts"}
            if response_protocol == "nl-count-v2":
                result_fields.add("observation")
            if (
                set(item) != common | result_fields
                or (item["requestId"] is not None and (
                    not isinstance(item["requestId"], str) or len(item["requestId"]) > 512
                ))
                or not _valid_usage(item["usage"])
                or type(item["httpAttempts"]) is not int
                or not 1 <= item["httpAttempts"] <= 5
            ):
                raise NlProtocolError("NL result fields are invalid")
            observation = None
            if response_protocol == "nl-count-v2":
                try:
                    observation = CountObservationV1.from_dict(item["observation"])
                except CountReviewProtocolError as exc:
                    raise NlProtocolError(str(exc)) from exc
                if observation.status not in {"observed", "invalid"}:
                    raise NlProtocolError("requested NL observation status is invalid")
            outcomes.append(NlOutcomeV1(
                item["sampleId"], item["leaseId"], item["relativeImagePath"],
                validate_nl(item["nl"]), None, False, item["httpAttempts"], observation, None,
            ))
        elif item["payloadType"] == "nl_issue":
            if set(item) != common | {"code", "severity", "blocking", "retriable", "message", "httpAttempts"} or not isinstance(item["code"], str) or item["severity"] != "error" or item["blocking"] is not True or type(item["retriable"]) is not bool or not isinstance(item["message"], str) or type(item["httpAttempts"]) is not int or not 0 <= item["httpAttempts"] <= 5:
                raise NlProtocolError("NL issue fields are invalid")
            outcomes.append(NlOutcomeV1(
                item["sampleId"], item["leaseId"], item["relativeImagePath"],
                None, item["code"], item["retriable"], item["httpAttempts"], None, item["message"],
            ))
        else:
            raise NlProtocolError("NL outcome type is invalid")
    if len({(outcome.sampleId, outcome.leaseId) for outcome in outcomes}) != len(outcomes):
        raise NlProtocolError("NL response contains duplicate outcomes")
    return tuple(outcomes)
