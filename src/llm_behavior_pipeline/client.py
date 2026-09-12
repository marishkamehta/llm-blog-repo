from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import requests

from .types import LLMResponse, Trajectory


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None

    def payload(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _messages(turn: str | Trajectory, system: str | None) -> Trajectory:
    messages: Trajectory = (
        [{"role": "user", "content": turn}] if isinstance(turn, str) else list(turn)
    )
    return ([{"role": "system", "content": system}] + messages) if system else messages


def _safe_response_headers(headers) -> dict[str, str]:
    sensitive = {"authorization", "proxy-authorization", "set-cookie"}
    return {
        str(name): "[REDACTED]" if str(name).lower() in sensitive else str(value)
        for name, value in headers.items()
    }


def _rate_limit_headers(headers) -> dict[str, str]:
    return {
        str(name): str(value)
        for name, value in headers.items()
        if "ratelimit" in str(name).lower().replace("-", "")
    }


class ModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        backend: str = "openai-compatible",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.backend = backend
        self.timeout = timeout

    def respond_with_metadata(
        self,
        turn: str | Trajectory,
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        config = config or GenerationConfig()
        messages = _messages(turn, system)
        payload = {"model": self.model, "messages": messages, **config.payload()}
        endpoint = f"{self.base_url}/chat/completions"
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        reply = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        elapsed = time.monotonic() - started
        reply.raise_for_status()
        body = reply.json()
        response_headers = _safe_response_headers(reply.headers)
        return LLMResponse(
            text=body["choices"][0]["message"].get("content") or "",
            request={
                "backend": self.backend,
                "endpoint": endpoint,
                "payload": payload,
                "headers": {"Authorization": "[REDACTED]"},
                "started_at_utc": started_at,
            },
            response={
                "http_status": reply.status_code,
                "model": body.get("model"),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"),
                "request_id": reply.headers.get("x-request-id"),
                "rate_limit_headers": _rate_limit_headers(reply.headers),
                "headers": response_headers,
                "latency_seconds": elapsed,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def respond(self, turn: str | Trajectory, **kwargs) -> str:
        return self.respond_with_metadata(turn, **kwargs).text


class MockModel:
    def respond_with_metadata(
        self,
        turn: str | Trajectory,
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        messages = _messages(turn, system)
        return LLMResponse(
            text="MOCK_RESPONSE",
            request={"backend": "mock", "messages": messages},
            response={"model": "mock", "usage": None},
        )

    def respond(self, turn: str | Trajectory, **kwargs) -> str:
        return self.respond_with_metadata(turn, **kwargs).text
