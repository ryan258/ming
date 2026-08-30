from __future__ import annotations

import asyncio
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .contracts import ModelResponse, RouteAttempt, usage_from_provider

NETWORK_ERRORS = (OSError, TimeoutError, http.client.HTTPException)


class GatewayError(RuntimeError):
    """A safe, credential-free gateway failure."""


class MissingAPIKeyError(GatewayError):
    pass


class AllModelsFailedError(GatewayError):
    def __init__(
        self, message: str, *, attempt_trace: list[RouteAttempt] | None = None
    ) -> None:
        super().__init__(message)
        self.attempt_trace: list[RouteAttempt] = list(attempt_trace) if attempt_trace else []


@dataclass(slots=True)
class TransportResponse:
    status: int
    body: dict[str, Any]


Transport = Callable[[str, Mapping[str, str], bytes, float], TransportResponse]


@dataclass(slots=True)
class TransportOutcome:
    response: TransportResponse
    cancelled: bool = False


async def dispatch_transport(
    transport: Transport,
    url: str,
    headers: Mapping[str, str],
    payload: bytes,
    timeout: float,
) -> TransportOutcome:
    """Run transport in an executor so cancellation cannot drop a completed HTTP body."""
    loop = asyncio.get_running_loop()
    cf_future = loop.run_in_executor(None, transport, url, headers, payload, timeout)
    inner = asyncio.wrap_future(cf_future)
    try:
        return TransportOutcome(await asyncio.shield(inner), cancelled=False)
    except asyncio.CancelledError:
        try:
            response = await inner
        except Exception:
            raise asyncio.CancelledError from None
        return TransportOutcome(response, cancelled=True)

# Match the image-encoded cap so a hostile 200 cannot unbounded-buffer the process.
MAX_TRANSPORT_BYTES = 48 * 1024 * 1024
_NON_JSON_BODY = {"error": {"message": "OpenRouter returned a non-JSON response"}}
_OVERSIZE_BODY = {"error": {"message": "OpenRouter response exceeded the safe size limit"}}


def _read_limited(source: object) -> bytes | None:
    raw = source.read(MAX_TRANSPORT_BYTES + 1)  # type: ignore[attr-defined]
    if not isinstance(raw, (bytes, bytearray)):
        return None
    if len(raw) > MAX_TRANSPORT_BYTES:
        return None
    return bytes(raw)


def _json_object(raw: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _body_from_http(source: object) -> dict[str, Any]:
    raw = _read_limited(source)
    if raw is None:
        return dict(_OVERSIZE_BODY)
    return _json_object(raw) or dict(_NON_JSON_BODY)


class _StripAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        old_host = urllib.parse.urlparse(req.full_url).netloc.casefold()
        new_host = urllib.parse.urlparse(new_req.full_url).netloc.casefold()
        if old_host != new_host:
            for mapping in (new_req.headers, new_req.unredirected_hdrs):
                for key in [name for name in mapping if name.lower() == "authorization"]:
                    del mapping[key]
        return new_req


_OPENER = urllib.request.build_opener(_StripAuthOnCrossHostRedirect)


def _open_http(request: urllib.request.Request, timeout: float) -> object:
    return _OPENER.open(request, timeout=timeout)


def _urllib_transport(
    url: str, headers: Mapping[str, str], payload: bytes, timeout: float
) -> TransportResponse:
    request = urllib.request.Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        with _open_http(request, timeout) as response:  # type: ignore[union-attr]
            return TransportResponse(status=response.status, body=_body_from_http(response))
    except urllib.error.HTTPError as exc:
        return TransportResponse(status=exc.code, body=_body_from_http(exc))


class SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.limit:
                    self._timestamps.append(now)
                    return
                delay = max(0.01, 60 - (now - self._timestamps[0]))
            await asyncio.sleep(delay)


class OpenRouterGateway:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: Transport = _urllib_transport,
        timeout: float = 60.0,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.timeout = timeout
        self.limiter = SlidingWindowLimiter(settings.requests_per_minute)

    async def complete(
        self,
        role: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        response_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if not self.settings.api_key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY is not configured; use --offline or add it to .env"
            )

        failures: list[str] = []
        attempts = 0
        attempt_trace: list[RouteAttempt] = []
        for model in self.settings.models_for(role):
            drop_require_parameters = False
            parameter_modes = (True, False) if response_schema is not None else (False,)
            for require_parameters in parameter_modes:
                if (
                    response_schema is not None
                    and require_parameters is False
                    and not drop_require_parameters
                ):
                    continue
                model_attempts = self.settings.max_retries + 1
                for retry in range(model_attempts):
                    attempts += 1
                    await self.limiter.acquire()
                    started = time.perf_counter()
                    payload: dict[str, Any] = {
                        "model": model,
                        "messages": list(messages),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if response_schema is not None:
                        payload["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ming_response",
                                "strict": True,
                                "schema": response_schema,
                            },
                        }
                        if require_parameters:
                            payload["provider"] = {"require_parameters": True}
                    headers = {
                        "Authorization": f"Bearer {self.settings.api_key}",
                        "Content-Type": "application/json",
                        "X-Title": "Ming Local Agent Harness",
                        "X-OpenRouter-Metadata": "enabled",
                    }
                    try:
                        outcome = await dispatch_transport(
                            self.transport,
                            f"{self.settings.base_url}/chat/completions",
                            headers,
                            json.dumps(payload).encode("utf-8"),
                            self.timeout,
                        )
                        response = outcome.response
                    except NETWORK_ERRORS as exc:
                        err_cls = f"network {type(exc).__name__}"
                        failures.append(f"{model}: {err_cls}")
                        attempt_trace.append(
                            RouteAttempt(
                                route=model,
                                status_or_error=err_cls,
                                retry=retry,
                                outcome="retry" if retry < self.settings.max_retries else "failed",
                            )
                        )
                        if retry < self.settings.max_retries:
                            await asyncio.sleep(min(2**retry, 4))
                            continue
                        break

                    latency_ms = (time.perf_counter() - started) * 1000
                    if outcome.cancelled:
                        raise asyncio.CancelledError
                    if response.status == 200:
                        try:
                            parsed = self._parse_response(
                                response.body, model, latency_ms, attempts
                            )
                            attempt_trace.append(
                                RouteAttempt(
                                    route=model,
                                    status_or_error="HTTP 200",
                                    retry=retry,
                                    outcome="success",
                                )
                            )
                            parsed.attempt_trace = attempt_trace
                            return parsed
                        except GatewayError as exc:
                            # A named model returning malformed content is a dead route.
                            # The dynamic router can select a different free model on its
                            # next request, so give it the configured retry budget.
                            retry_dynamic = (
                                model == "openrouter/free"
                                and retry < self.settings.max_retries
                            )
                            failures.append(f"{model}: {exc}")
                            attempt_trace.append(
                                RouteAttempt(
                                    route=model,
                                    status_or_error=type(exc).__name__,
                                    retry=retry,
                                    outcome="retry" if retry_dynamic else "failed",
                                )
                            )
                            if retry_dynamic:
                                continue
                            break

                    message = self._safe_error_message(response.body)
                    if self.settings.api_key:
                        message = message.replace(self.settings.api_key, "[REDACTED]")
                    failures.append(f"{model}: HTTP {response.status} {message}")
                    is_retryable = (
                        response.status == 429 or response.status >= 500
                    ) and retry < self.settings.max_retries
                    attempt_trace.append(
                        RouteAttempt(
                            route=model,
                            status_or_error=f"HTTP {response.status} {message}",
                            retry=retry,
                            outcome="retry" if is_retryable else "failed",
                        )
                    )
                    if is_retryable:
                        await asyncio.sleep(min(2**retry, 4))
                        continue
                    if (
                        response_schema is not None
                        and require_parameters
                        and response.status in {400, 422}
                    ):
                        drop_require_parameters = True
                    break
                if drop_require_parameters and require_parameters:
                    continue
                break

        summary = "; ".join(failures[-6:]) or "no response"
        raise AllModelsFailedError(
            f"all configured free routes failed: {summary}",
            attempt_trace=attempt_trace,
        )

    @staticmethod
    def _safe_error_message(body: dict[str, Any]) -> str:
        error = body.get("error", {})
        if isinstance(error, dict):
            message = str(error.get("message", "request failed"))
        else:
            message = "request failed"
        return message.replace("\n", " ")[:240]

    @staticmethod
    def _parse_response(
        body: dict[str, Any], requested_model: str, latency_ms: float, attempts: int
    ) -> ModelResponse:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayError("OpenRouter response did not contain assistant content") from exc
        if not isinstance(content, str):
            raise GatewayError("OpenRouter assistant content was not text")
        usage = usage_from_provider(body.get("usage"))
        return ModelResponse(
            content=content,
            requested_model=requested_model,
            actual_model=str(body.get("model") or requested_model),
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempts,
        )
