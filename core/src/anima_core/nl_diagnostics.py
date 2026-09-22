from __future__ import annotations

import json
import ipaddress
import re
import socket
import ssl
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


TEST_MESSAGE = "This is an Anima API connectivity test. Reply with one short confirmation sentence."
MAX_RESPONSE_BYTES = 1_048_576
_MAX_ENDPOINT_BYTES = 2_048
_MAX_MODEL_BYTES = 512
_MAX_BASE_PROMPT_BYTES = 65_536
_MAX_REASON_CHARS = 512
_REQUEST_TIMEOUT_SECONDS = 15.0
_USER_AGENT = "Anima-Dataset-Tool/1.0"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
_WHITESPACE = re.compile(r"\s+")
_AUTHORIZATION = re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+")
_WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:\\|/)(?:[^\s]+)")


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _trust_environment_proxy(endpoint: str) -> bool:
    hostname = urlsplit(endpoint).hostname
    if hostname is None:
        return False
    if hostname.casefold() == "localhost" or hostname.casefold().endswith(".localhost"):
        return False
    try:
        return not ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return True


def _default_opener(endpoint: str) -> object:
    handlers: list[object] = [_NoRedirectHandler()]
    if not _trust_environment_proxy(endpoint):
        handlers.insert(0, ProxyHandler({}))
    return build_opener(*handlers)


def _strict_json(data: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[object, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if not isinstance(key, str) or key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)


def _normalize_chat_endpoint(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_ENDPOINT_BYTES or any(char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or parsed.hostname is None
    ):
        return None
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _models_endpoint(chat_endpoint: str) -> str:
    parsed = urlsplit(chat_endpoint)
    path = parsed.path[: -len("/chat/completions")] + "/models"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _valid_text(value: object, *, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value and len(value.encode("utf-8")) <= limit


def _latency_ms(start: float, end: float) -> int:
    return max(0, int((end - start) * 1_000))


def _sanitize_reason(value: object, *, api_key: str, endpoint: str) -> str | None:
    if not isinstance(value, str):
        return None
    text = _AUTHORIZATION.sub("[redacted]", value)
    if api_key:
        text = text.replace(f"Bearer {api_key}", "[redacted]").replace(api_key, "[redacted]")
    text = text.replace(endpoint, "[redacted]")
    text = _WINDOWS_PATH.sub("[redacted-path]", text)
    text = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", text)).strip()
    return text[:_MAX_REASON_CHARS] or None


def _provider_message(error: HTTPError, *, api_key: str, endpoint: str) -> str | None:
    message: object = error.reason
    try:
        body = error.read(MAX_RESPONSE_BYTES + 1)
        if len(body) <= MAX_RESPONSE_BYTES:
            parsed = _strict_json(body)
            if isinstance(parsed, dict):
                nested = parsed.get("error")
                if isinstance(nested, dict):
                    message = nested.get("message", message)
                else:
                    message = parsed.get("message", message)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        pass
    return _sanitize_reason(message, api_key=api_key, endpoint=endpoint)


def _http_error_code(status: int) -> str:
    if 300 <= status < 400:
        return "redirect_rejected"
    if status in {401, 403}:
        return "authentication_failed"
    if status == 404:
        return "endpoint_or_model_not_found"
    if status == 429:
        return "rate_limited"
    if 500 <= status < 600:
        return "provider_unavailable"
    return "provider_rejected"


def _transport_error_code(error: BaseException) -> str:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(error, URLError) and isinstance(error.reason, (TimeoutError, socket.timeout)):
        return "timeout"
    return "connection_failed"


class NlDiagnosticClient:
    """Single-request diagnostic traffic kept outside normal NL jobs and worker protocols."""

    def __init__(self, *, opener: object | None = None, clock: Callable[[], float] = time.perf_counter) -> None:
        self._opener = opener
        self._clock = clock

    def _request(self, request: Request, *, api_key: str, endpoint: str) -> tuple[bytes | None, str | None, str | None]:
        try:
            opener = self._opener if self._opener is not None else _default_opener(endpoint)
            response = opener.open(request, timeout=_REQUEST_TIMEOUT_SECONDS)
            with response:
                status = response.getcode()
                if not isinstance(status, int) or status < 200 or status >= 300:
                    return None, _http_error_code(status if isinstance(status, int) else 500), None
                body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                return None, "response_too_large", None
            return body, None, None
        except HTTPError as exc:
            return None, _http_error_code(exc.code), _provider_message(exc, api_key=api_key, endpoint=endpoint)
        except (TimeoutError, socket.timeout, ssl.SSLError, URLError, OSError) as exc:
            return None, _transport_error_code(exc), None

    @staticmethod
    def _discovery_failure(*, latency_ms: int, code: str, reason: str | None = None) -> dict[str, object]:
        return {
            "ok": False,
            "latencyMs": latency_ms,
            "models": [],
            "errorCode": code,
            "errorReason": reason or code.replace("_", " "),
        }

    @staticmethod
    def _message_failure(*, latency_ms: int, code: str, reason: str | None = None) -> dict[str, object]:
        return {
            "ok": False,
            "latencyMs": latency_ms,
            "actualModel": None,
            "replyText": None,
            "usage": None,
            "errorCode": code,
            "errorReason": reason or code.replace("_", " "),
        }

    @staticmethod
    def _parse_models(body: bytes) -> list[str]:
        try:
            value = _strict_json(body)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("model list is invalid") from exc
        if not isinstance(value, dict) or not isinstance(value.get("data"), list):
            raise ValueError("model list is invalid")
        unique: dict[str, str] = {}
        for item in value["data"]:
            if not isinstance(item, dict) or not _valid_text(item.get("id"), limit=_MAX_MODEL_BYTES):
                raise ValueError("model entry is invalid")
            model_id = item["id"]
            unique.setdefault(model_id.casefold(), model_id)
        return sorted(unique.values(), key=lambda item: (item.casefold(), item))[:1_000]

    @staticmethod
    def _parse_completion(body: bytes) -> tuple[str, str, dict[str, int | None]]:
        try:
            value = _strict_json(body)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("completion is invalid") from exc
        if not isinstance(value, dict) or not _valid_text(value.get("model"), limit=_MAX_MODEL_BYTES):
            raise ValueError("completion model is invalid")
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("completion choices are invalid")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("completion message is invalid")
        content = message.get("content")
        if isinstance(content, list):
            pieces: list[str] = []
            for part in content:
                if not isinstance(part, dict) or set(part) != {"type", "text"} or part.get("type") != "text" or not isinstance(part.get("text"), str):
                    raise ValueError("completion content parts are invalid")
                pieces.append(part["text"])
            content = "".join(pieces)
        if not _valid_text(content, limit=MAX_RESPONSE_BYTES):
            raise ValueError("completion content is invalid")
        usage = value.get("usage")
        usage_summary: dict[str, int | None] = {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
        }
        if isinstance(usage, dict):
            for source, target in (
                ("prompt_tokens", "promptTokens"),
                ("completion_tokens", "completionTokens"),
                ("total_tokens", "totalTokens"),
            ):
                candidate = usage.get(source)
                if type(candidate) is int and candidate >= 0:
                    usage_summary[target] = candidate
        return value["model"], content, usage_summary

    def discover_models(self, *, endpoint: str, api_key: str) -> dict[str, object]:
        start = self._clock()
        chat_endpoint = _normalize_chat_endpoint(endpoint)
        if chat_endpoint is None:
            return self._discovery_failure(latency_ms=_latency_ms(start, self._clock()), code="invalid_endpoint")
        if not _valid_text(api_key, limit=MAX_RESPONSE_BYTES):
            return self._discovery_failure(latency_ms=_latency_ms(start, self._clock()), code="credential_unavailable")
        request = Request(_models_endpoint(chat_endpoint), method="GET", headers={"Authorization": f"Bearer {api_key}", "User-Agent": _USER_AGENT})
        body, error_code, error_reason = self._request(request, api_key=api_key, endpoint=chat_endpoint)
        latency_ms = _latency_ms(start, self._clock())
        if error_code is not None:
            return self._discovery_failure(latency_ms=latency_ms, code=error_code, reason=error_reason)
        assert body is not None
        try:
            models = self._parse_models(body)
        except ValueError:
            return self._discovery_failure(latency_ms=latency_ms, code="response_invalid")
        return {"ok": True, "latencyMs": latency_ms, "models": models, "errorCode": None, "errorReason": None}

    def test_message(self, *, endpoint: str, model: str, api_key: str, base_prompt: str) -> dict[str, object]:
        start = self._clock()
        chat_endpoint = _normalize_chat_endpoint(endpoint)
        if chat_endpoint is None:
            return self._message_failure(latency_ms=_latency_ms(start, self._clock()), code="invalid_endpoint")
        if not _valid_text(api_key, limit=MAX_RESPONSE_BYTES):
            return self._message_failure(latency_ms=_latency_ms(start, self._clock()), code="credential_unavailable")
        if not _valid_text(model, limit=_MAX_MODEL_BYTES) or not _valid_text(base_prompt, limit=_MAX_BASE_PROMPT_BYTES):
            return self._message_failure(latency_ms=_latency_ms(start, self._clock()), code="invalid_request")
        payload = json.dumps(
            {
                "model": model,
                "temperature": 0,
                "max_tokens": 64,
                "messages": [
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": TEST_MESSAGE},
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            chat_endpoint,
            data=payload,
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json; charset=utf-8", "User-Agent": _USER_AGENT},
        )
        body, error_code, error_reason = self._request(request, api_key=api_key, endpoint=chat_endpoint)
        latency_ms = _latency_ms(start, self._clock())
        if error_code is not None:
            return self._message_failure(latency_ms=latency_ms, code=error_code, reason=error_reason)
        assert body is not None
        try:
            actual_model, reply_text, usage = self._parse_completion(body)
        except ValueError:
            return self._message_failure(latency_ms=latency_ms, code="response_invalid")
        return {
            "ok": True,
            "latencyMs": latency_ms,
            "actualModel": actual_model,
            "replyText": reply_text,
            "usage": usage,
            "errorCode": None,
            "errorReason": None,
        }
