from __future__ import annotations

import http.client
import io
import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from engine.config import Settings
from engine.llm import (
    AllModelsFailedError,
    MissingAPIKeyError,
    OpenRouterGateway,
    TransportResponse,
    _StripAuthOnCrossHostRedirect,
    _urllib_transport,
)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_stops_before_transport(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            raise AssertionError("must not be called")

        gateway = OpenRouterGateway(Settings(api_key=None), transport=transport)
        with self.assertRaises(MissingAPIKeyError):
            await gateway.complete("fast", [{"role": "user", "content": "hi"}])
        self.assertFalse(called)

    async def test_request_uses_bearer_auth_and_parses_actual_model(self) -> None:
        captured: dict[str, object] = {}

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            captured.update(url=url, headers=headers, payload=json.loads(payload), timeout=timeout)
            return TransportResponse(
                200,
                {
                    "model": "actual/free-model",
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                },
            )

        settings = Settings(
            api_key="secret",
            fast_models=("example/fast:free", "openrouter/free"),
            max_retries=0,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "fast", [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(response.actual_model, "actual/free-model")
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(
            captured["headers"]["Authorization"], "Bearer secret"  # type: ignore[index]
        )
        self.assertEqual(captured["payload"]["model"], "example/fast:free")  # type: ignore[index]

    async def test_http_exception_falls_back_to_next_model(self) -> None:
        models: list[str] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            model = json.loads(payload)["model"]
            models.append(model)
            if model == "first/model:free":
                raise http.client.RemoteDisconnected("Remote end closed connection")
            return TransportResponse(
                200,
                {"model": "second/routed", "choices": [{"message": {"content": "ok"}}]},
            )

        settings = Settings(
            api_key="secret",
            fast_models=("first/model:free", "second/model:free"),
            max_retries=0,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "fast", [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(models, ["first/model:free", "second/model:free"])
        self.assertEqual(response.content, "ok")
        self.assertEqual(response.attempt_trace[0].outcome, "failed")
        self.assertIn("RemoteDisconnected", response.attempt_trace[0].status_or_error)

    async def test_model_failure_falls_back_to_dynamic_free_router(self) -> None:
        models: list[str] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            model = json.loads(payload)["model"]
            models.append(model)
            if model != "openrouter/free":
                return TransportResponse(404, {"error": {"message": "model unavailable"}})
            return TransportResponse(
                200,
                {"model": "routed/free", "choices": [{"message": {"content": "ok"}}]},
            )

        settings = Settings(
            api_key="secret",
            fast_models=("rotated/model:free", "openrouter/free"),
            max_retries=0,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "fast", [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(models, ["rotated/model:free", "openrouter/free"])
        self.assertEqual(response.actual_model, "routed/free")
        self.assertEqual(len(response.attempt_trace), 2)
        self.assertEqual(response.attempt_trace[0].route, "rotated/model:free")
        self.assertEqual(response.attempt_trace[0].outcome, "failed")
        self.assertIn("HTTP 404", response.attempt_trace[0].status_or_error)
        self.assertEqual(response.attempt_trace[1].route, "openrouter/free")
        self.assertEqual(response.attempt_trace[1].outcome, "success")
        self.assertEqual(response.attempt_trace[1].status_or_error, "HTTP 200")

    async def test_malformed_200_falls_through_to_next_model(self) -> None:
        models: list[str] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            model = json.loads(payload)["model"]
            models.append(model)
            if model == "first/model:free":
                return TransportResponse(200, {"choices": []})  # HTTP 200, no content
            return TransportResponse(
                200,
                {"model": "second/routed", "choices": [{"message": {"content": "ok"}}]},
            )

        settings = Settings(
            api_key="secret",
            fast_models=("first/model:free", "second/model:free"),
            max_retries=0,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "fast", [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(models, ["first/model:free", "second/model:free"])
        self.assertEqual(response.content, "ok")
        self.assertEqual(len(response.attempt_trace), 2)
        self.assertEqual(response.attempt_trace[0].route, "first/model:free")
        self.assertEqual(response.attempt_trace[0].outcome, "failed")
        self.assertEqual(response.attempt_trace[0].status_or_error, "GatewayError")
        self.assertEqual(response.attempt_trace[1].route, "second/model:free")
        self.assertEqual(response.attempt_trace[1].outcome, "success")

    async def test_dynamic_router_retries_malformed_200(self) -> None:
        calls = 0

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            nonlocal calls
            calls += 1
            self.assertEqual(json.loads(payload)["model"], "openrouter/free")
            if calls == 1:
                return TransportResponse(
                    200,
                    {
                        "model": "routed/non-text:free",
                        "choices": [{"message": {"content": None}}],
                    },
                )
            return TransportResponse(
                200,
                {
                    "model": "routed/text:free",
                    "choices": [{"message": {"content": "usable answer"}}],
                },
            )

        settings = Settings(
            api_key="secret",
            fast_models=("openrouter/free",),
            max_retries=1,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "fast", [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(calls, 2)
        self.assertEqual(response.content, "usable answer")
        self.assertEqual(response.actual_model, "routed/text:free")
        self.assertEqual(len(response.attempt_trace), 2)
        self.assertEqual(response.attempt_trace[0].route, "openrouter/free")
        self.assertEqual(response.attempt_trace[0].status_or_error, "GatewayError")
        self.assertEqual(response.attempt_trace[0].outcome, "retry")
        self.assertEqual(response.attempt_trace[1].outcome, "success")

    async def test_malformed_usage_keeps_valid_content(self) -> None:
        models: list[str] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            model = json.loads(payload)["model"]
            models.append(model)
            if model == "first/model:free":
                return TransportResponse(
                    200,
                    {
                        "choices": [{"message": {"content": "usable answer"}}],
                        "usage": {"total_tokens": "unknown"},
                    },
                )
            return TransportResponse(
                200,
                {"choices": [{"message": {"content": "fallback answer"}}]},
            )

        settings = Settings(
            api_key="secret",
            fast_models=("first/model:free", "second/model:free"),
            max_retries=0,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "fast", [{"role": "user", "content": "hi"}]
        )
        self.assertEqual(models, ["first/model:free"])
        self.assertEqual(response.content, "usable answer")
        self.assertFalse(response.usage.available)
        self.assertEqual(response.usage.total_tokens, 0)

    async def test_error_redacts_key_echoed_by_provider(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return TransportResponse(400, {"error": {"message": "bad key secret-value"}})

        settings = Settings(
            api_key="secret-value",
            fast_models=("only/model:free",),
            max_retries=0,
        )
        with self.assertRaises(AllModelsFailedError) as context:
            await OpenRouterGateway(settings, transport=transport).complete(
                "fast", [{"role": "user", "content": "hi"}]
            )
        self.assertNotIn("secret-value", str(context.exception))
        self.assertIn("[REDACTED]", str(context.exception))

    async def test_schema_require_parameters_failure_retries_without_it(self) -> None:
        payloads: list[dict[str, object]] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            data = json.loads(payload)
            payloads.append(data)
            if data.get("provider", {}).get("require_parameters"):
                return TransportResponse(
                    400, {"error": {"message": "provider does not support parameters"}}
                )
            return TransportResponse(
                200,
                {"choices": [{"message": {"content": '{"ok": true}'}}]},
            )

        settings = Settings(
            api_key="secret",
            orchestrator_models=("example/fast:free",),
            max_retries=0,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "orchestrator",
            [{"role": "user", "content": "plan"}],
            response_schema={"type": "object"},
        )
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0].get("provider"), {"require_parameters": True})
        self.assertNotIn("provider", payloads[1])
        self.assertEqual(response.content, '{"ok": true}')

    async def test_schema_404_does_not_retry_same_model_without_require_parameters(
        self,
    ) -> None:
        models: list[str] = []

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            data = json.loads(payload)
            models.append(data["model"])
            if data["model"] == "first/model:free":
                return TransportResponse(404, {"error": {"message": "model unavailable"}})
            return TransportResponse(
                200,
                {"choices": [{"message": {"content": "ok"}}]},
            )

        settings = Settings(
            api_key="secret",
            orchestrator_models=("first/model:free", "second/model:free"),
            max_retries=0,
        )
        response = await OpenRouterGateway(settings, transport=transport).complete(
            "orchestrator",
            [{"role": "user", "content": "plan"}],
            response_schema={"type": "object"},
        )
        self.assertEqual(models, ["first/model:free", "second/model:free"])
        self.assertEqual(response.content, "ok")


class FakeHTTPResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return self._body
        return self._body[:n]

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class UrllibTransportTests(unittest.TestCase):
    def test_http_200_non_json_returns_safe_error_body(self) -> None:
        fake = FakeHTTPResponse(200, b"<html>nope</html>")
        with patch("engine.llm._open_http", return_value=fake):
            result = _urllib_transport(
                "https://openrouter.ai/api/v1/chat/completions", {}, b"{}", 1.0
            )
        self.assertEqual(result.status, 200)
        self.assertIn("non-JSON", result.body["error"]["message"])

    def test_http_200_oversize_returns_safe_error_body(self) -> None:
        fake = FakeHTTPResponse(200, b"x" * 32)
        with patch("engine.llm.MAX_TRANSPORT_BYTES", 8), patch(
            "engine.llm._open_http", return_value=fake
        ):
            result = _urllib_transport(
                "https://openrouter.ai/api/v1/chat/completions", {}, b"{}", 1.0
            )
        self.assertEqual(result.status, 200)
        self.assertIn("safe size limit", result.body["error"]["message"])

    def test_http_error_non_json_returns_safe_error_body(self) -> None:
        error = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            502,
            "Bad Gateway",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b"<html>upstream</html>"),
        )
        with patch("engine.llm._open_http", side_effect=error):
            result = _urllib_transport(
                "https://openrouter.ai/api/v1/chat/completions", {}, b"{}", 1.0
            )
        self.assertEqual(result.status, 502)
        self.assertIn("non-JSON", result.body["error"]["message"])

    def test_cross_host_redirect_strips_authorization(self) -> None:
        handler = _StripAuthOnCrossHostRedirect()
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/images",
            data=b"{}",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        redirected = handler.redirect_request(
            request,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="https://evil.example/collect",
        )
        self.assertIsNotNone(redirected)
        auth = [
            value
            for key, value in redirected.header_items()  # type: ignore[union-attr]
            if key.lower() == "authorization"
        ]
        self.assertEqual(auth, [])

    def test_same_host_redirect_keeps_authorization(self) -> None:
        handler = _StripAuthOnCrossHostRedirect()
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/images",
            data=b"{}",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        redirected = handler.redirect_request(
            request,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="https://openrouter.ai/api/v1/images/new",
        )
        self.assertIsNotNone(redirected)
        auth = [
            value
            for key, value in redirected.header_items()  # type: ignore[union-attr]
            if key.lower() == "authorization"
        ]
        self.assertEqual(auth, ["Bearer secret"])


if __name__ == "__main__":
    unittest.main()
