"""LLM objects: send text to a model and get text back.

`LLM` talks to an OpenAI-compatible chat endpoint, for example a local vLLM
container. `MockLLM` returns text with no server, so an experiment can run and
be tested offline.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import requests

from .contracts import (
    ExecutionConfig,
    LLMBase,
    LLMResponse,
    ModelConfig,
    Trajectory,
    Turn,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatModelConfig(ModelConfig):
    """Settings for a typical OpenAI-compatible chat-completions model.

    Every field defaults to `None` and is left out of the request unless you
    set it, so choosing a config is choosing which of these to set, not
    fighting a fixed universal default.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, str] | None = None


def _last_user_content(messages: Trajectory) -> str:
    """Return the last user message's content, or "" if there is none."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _as_messages(turn: str | Trajectory) -> Trajectory:
    """A plain string becomes one user turn; a list is a trajectory already."""
    if isinstance(turn, str):
        return [{"role": "user", "content": turn}]
    return list(turn)


def _error_message(response: requests.Response) -> str:
    """The provider's own error message from a non-2xx response, or "".

    Providers that return an OpenAI-style error body (`{"error": {"message":
    ...}}`) put the actual reason here, for example which specific quota was
    exceeded (a token-rate limit versus a request-rate limit) and which
    pricing tier it is measured against. Falls back to "" (not raised) if the
    body is not JSON or has no such field, since this is only ever used to
    enrich a log message, never to control behavior.
    """
    try:
        body = response.json()
    except ValueError:
        return ""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return str(body["error"].get("message") or "")
    return ""


def _rate_limit_headers(response: requests.Response) -> dict[str, str]:
    """Any response header whose name mentions "rate limit", verbatim.

    OpenAI-compatible endpoints commonly report the caller's own remaining
    quota on every response, not just a 429 (for example
    `x-ratelimit-remaining-tokens`, `x-ratelimit-remaining-requests`), so a
    caller can back off before hitting the limit instead of only after. This
    does not assume the exact header names, on purpose: which ones a given
    deployment sends is exactly the thing to find out by looking at this, not
    something to hardcode a guess for.
    """
    return {
        name: value
        for name, value in response.headers.items()
        if "ratelimit" in name.lower().replace("-", "")
    }


def _safe_response_headers(response: requests.Response) -> dict[str, str]:
    """Preserve provider headers while excluding credential-bearing values."""
    sensitive = {"authorization", "proxy-authorization", "set-cookie"}
    return {
        str(name): "[REDACTED]" if name.lower() in sensitive else str(value)
        for name, value in response.headers.items()
    }


def _raw_response_body(response: requests.Response):
    """Return the exact decoded JSON value, or raw response text otherwise."""
    try:
        return response.json()
    except ValueError:
        return response.text


def _attempt_record(response, started_at_utc, elapsed_seconds):
    return {
        "started_at_utc": started_at_utc,
        "elapsed_seconds": elapsed_seconds,
        "http_status": response.status_code,
        "headers": _safe_response_headers(response),
        "body": _raw_response_body(response),
        "scheduled_backoff_seconds": 0.0,
    }


def _responses_text(body: dict) -> str:
    """Extract text, preserving a successful textless model response as empty."""
    parts = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def _normalized_responses_usage(usage: dict | None) -> dict | None:
    """Map Responses token names onto the study's chat-compatible ledger."""
    if not isinstance(usage, dict):
        return None
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0) or 0,
        "completion_tokens": usage.get("output_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "prompt_tokens_details": {
            "cached_tokens": input_details.get("cached_tokens", 0) or 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": output_details.get("reasoning_tokens", 0) or 0,
        },
    }


def _check_config_type(config: ModelConfig, default: ModelConfig) -> None:
    """Stop early if a per-call config isn't the shape this LLM expects.

    Each `LLM`/`MockLLM` instance is built with one `ModelConfig` subclass as
    its default (see `registry.py`, which picks that subclass for a
    registered model). A per-call override must be that same subclass, or a
    further subclass of it: passing a different `ModelConfig` subclass would
    silently send the wrong fields to this model, which is exactly what
    per-model subclasses exist to prevent. Build a new `LLM` instead if a
    call genuinely needs a different config shape.
    """
    if not isinstance(config, type(default)):
        raise TypeError(
            f"this LLM expects a {type(default).__name__} (or a subclass of "
            f"it), got a {type(config).__name__}. Build a new LLM with "
            f"config={type(config).__name__}(...) if this model genuinely "
            f"needs a different config shape."
        )


@dataclass
class LLM(LLMBase):
    """Send text to a chat endpoint and return the model's text reply.

    base_url:
        The OpenAI-compatible root, for example http://localhost:8000/v1 for a
        local vLLM container (see the docker folder).
    model:
        The served model name, for example Qwen/Qwen2.5-7B-Instruct.
    api_key:
        A local vLLM server ignores this. A hosted provider needs a real key.
    system:
        An optional default system message (a persona, or any fixed
        instruction), sent on every call unless a call overrides it. Leave it
        unset to send no system message by default.
    config:
        The default `ModelConfig` for every call, unless a call passes its
        own. Which fields to set depends on the model, not just on it being
        OpenAI-compatible (see `ModelConfig`): a study that wants a fixed,
        reproducible temperature sets it here once; a model that needs
        different settings gets its own `ModelConfig` subclass instead. A
        call's `config` override must be this same subclass (or a further
        subclass of it): see `respond_with_metadata`.
    execution:
        The default `ExecutionConfig` (our own timeout and retry policy) for
        every call, unless a call passes its own.
    """

    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str = "EMPTY"
    backend: str = "openai_compatible"
    system: str | None = None
    config: ModelConfig = field(default_factory=ChatModelConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def respond_with_metadata(
        self,
        turn: str | Trajectory,
        system: str | None = None,
        *,
        config: ModelConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> LLMResponse:
        """Send `turn` (a string, or a trajectory list) and return the reply.

        A string becomes one self-contained user turn. A list is sent exactly
        as given: pass the same growing list across calls, only ever
        appending to it, so the request body's prefix stays byte-identical
        turn to turn. That is what lets a cache-aware backend reuse the
        unchanged prefix instead of paying for it again on every call.
        `system` overrides this object's default `system` for this one call;
        `config` and `execution` likewise override this object's defaults.
        `config`, if given, must be the same `ModelConfig` subclass as this
        object's default (or a further subclass of it): see
        `_check_config_type`.
        """
        if config is not None:
            _check_config_type(config, self.config)
        config = config or self.config
        execution = execution or self.execution
        system = system if system is not None else self.system
        messages = _as_messages(turn)
        if system:
            system_turn: Turn = {"role": "system", "content": system}
            messages = [system_turn] + messages

        payload = {
            "model": self.model,
            "messages": messages,
        }
        payload.update(config.to_payload())

        started = time.monotonic()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        attempts = 0
        attempt_log = []
        query_seconds = 0.0
        backoff_seconds = 0.0
        while True:
            attempts += 1
            attempt_started_at_utc = datetime.now(timezone.utc).isoformat()
            request_started = time.monotonic()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=execution.timeout,
            )
            attempt_elapsed = time.monotonic() - request_started
            query_seconds += attempt_elapsed
            attempt_record = _attempt_record(
                response, attempt_started_at_utc, attempt_elapsed
            )
            attempt_log.append(attempt_record)
            if response.status_code != 429 or attempts > execution.max_retries:
                if response.status_code == 429:
                    logger.warning(
                        "%s: rate limited (HTTP 429) and out of retries after "
                        "%d attempt(s); giving up. %s",
                        self.model,
                        attempts,
                        _error_message(response),
                    )
                elif response.status_code >= 400:
                    logger.warning(
                        "%s: request failed with HTTP %d on attempt %d. %s",
                        self.model,
                        response.status_code,
                        attempts,
                        _error_message(response),
                    )
                break
            retry_after = response.headers.get("Retry-After")
            retry_after_ms = response.headers.get("retry-after-ms")
            try:
                delay = float(retry_after) if retry_after is not None else None
            except ValueError:
                delay = None
            if delay is None and retry_after_ms is not None:
                try:
                    delay = float(retry_after_ms) / 1000.0
                except ValueError:
                    delay = None
            if delay is None:
                delay = execution.retry_base_seconds * (2 ** (attempts - 1))
            delay = min(max(delay, 0.0), execution.retry_max_seconds)
            attempt_record["scheduled_backoff_seconds"] = delay
            logger.warning(
                "%s: rate limited (HTTP 429) on attempt %d/%d; retrying in %.1fs. %s",
                self.model,
                attempts,
                execution.max_retries,
                delay,
                _error_message(response),
            )
            time.sleep(delay)
            backoff_seconds += delay
        latency = time.monotonic() - started
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        return LLMResponse(
            text=text,
            request={
                "model": self.model,
                "transport": "chat_full",
                "endpoint": f"{self.base_url}/chat/completions",
                "messages": messages,
                "generation": asdict(config),
                "execution": asdict(execution),
                "payload": payload,
                "request_headers": {"Authorization": "[REDACTED]"},
            },
            response={
                "id": body.get("id"),
                "model": body.get("model"),
                "created": body.get("created"),
                "system_fingerprint": body.get("system_fingerprint"),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"),
                "rate_limit_headers": _rate_limit_headers(response),
                "latency_seconds": latency,
                "query_seconds": query_seconds,
                "backoff_seconds": backoff_seconds,
                "attempts": attempts,
                "attempt_log": attempt_log,
                "started_at_utc": started_at_utc,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code,
                "headers": _safe_response_headers(response),
                "raw_body": body,
            },
        )

    def respond_statefully_with_metadata(
        self,
        input_text: str,
        *,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        config: ModelConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> LLMResponse:
        """Send one delta through Responses and optionally continue a chain."""
        if config is not None:
            _check_config_type(config, self.config)
        config = config or self.config
        execution = execution or self.execution
        generation = config.to_payload()
        if generation.get("seed") is not None and self.backend != "vllm":
            raise ValueError(
                "seeded Responses transport is currently enabled only for vLLM"
            )
        if "max_tokens" in generation:
            generation["max_output_tokens"] = generation.pop("max_tokens")
        reasoning_effort = generation.pop("reasoning_effort", None)

        payload = {"model": self.model, "input": input_text, "store": True}
        if instructions:
            payload["instructions"] = instructions
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        payload.update(generation)

        started = time.monotonic()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        attempts = 0
        attempt_log = []
        query_seconds = 0.0
        backoff_seconds = 0.0
        while True:
            attempts += 1
            attempt_started_at_utc = datetime.now(timezone.utc).isoformat()
            request_started = time.monotonic()
            response = requests.post(
                f"{self.base_url}/responses",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=execution.timeout,
            )
            attempt_elapsed = time.monotonic() - request_started
            query_seconds += attempt_elapsed
            attempt_record = _attempt_record(
                response, attempt_started_at_utc, attempt_elapsed
            )
            attempt_log.append(attempt_record)
            if response.status_code != 429 or attempts > execution.max_retries:
                break
            retry_after = response.headers.get("Retry-After")
            retry_after_ms = response.headers.get("retry-after-ms")
            try:
                delay = float(retry_after) if retry_after is not None else None
            except ValueError:
                delay = None
            if delay is None and retry_after_ms is not None:
                try:
                    delay = float(retry_after_ms) / 1000.0
                except ValueError:
                    delay = None
            if delay is None:
                delay = execution.retry_base_seconds * (2 ** (attempts - 1))
            delay = min(max(delay, 0.0), execution.retry_max_seconds)
            attempt_record["scheduled_backoff_seconds"] = delay
            logger.warning(
                "%s: Responses rate limited (HTTP 429) on attempt %d/%d; "
                "retrying in %.1fs. %s",
                self.model,
                attempts,
                execution.max_retries,
                delay,
                _error_message(response),
            )
            time.sleep(delay)
            backoff_seconds += delay
        latency = time.monotonic() - started
        response.raise_for_status()
        body = response.json()
        return LLMResponse(
            text=_responses_text(body),
            request={
                "model": self.model,
                "transport": "responses_stateful",
                "endpoint": f"{self.base_url}/responses",
                "input": input_text,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
                "generation": asdict(config),
                "execution": asdict(execution),
                "payload": payload,
                "request_headers": {"Authorization": "[REDACTED]"},
            },
            response={
                "id": body.get("id"),
                "model": body.get("model"),
                "created": body.get("created_at"),
                "status": body.get("status"),
                "usage": _normalized_responses_usage(body.get("usage")),
                "provider_usage": body.get("usage"),
                "rate_limit_headers": _rate_limit_headers(response),
                "latency_seconds": latency,
                "query_seconds": query_seconds,
                "backoff_seconds": backoff_seconds,
                "attempts": attempts,
                "attempt_log": attempt_log,
                "started_at_utc": started_at_utc,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code,
                "headers": _safe_response_headers(response),
                "raw_body": body,
            },
        )

    def respond_with_tools_metadata(
        self,
        turn: str | Trajectory,
        tools: list[dict],
        system: str | None = None,
        *,
        tool_choice: str | dict = "required",
        parallel_tool_calls: bool = False,
        config: ModelConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> LLMResponse:
        """Send a chat request with provider-native function tools."""
        if config is not None:
            _check_config_type(config, self.config)
        config = config or self.config
        execution = execution or self.execution
        system = system if system is not None else self.system
        messages = _as_messages(turn)
        if system:
            messages = [{"role": "system", "content": system}] + messages
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        payload.update(config.to_payload())
        started = time.monotonic()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=execution.timeout,
        )
        latency = time.monotonic() - started
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        return LLMResponse(
            text=message.get("content") or "",
            request={
                "model": self.model,
                "transport": "chat_native_tools",
                "endpoint": f"{self.base_url}/chat/completions",
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "generation": asdict(config),
                "execution": asdict(execution),
                "payload": payload,
                "request_headers": {"Authorization": "[REDACTED]"},
            },
            response={
                "id": body.get("id"),
                "model": body.get("model"),
                "created": body.get("created"),
                "system_fingerprint": body.get("system_fingerprint"),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"),
                "tool_calls": tool_calls,
                "assistant_message": message,
                "rate_limit_headers": _rate_limit_headers(response),
                "latency_seconds": latency,
                "query_seconds": latency,
                "backoff_seconds": 0.0,
                "attempts": 1,
                "started_at_utc": started_at_utc,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code,
                "headers": _safe_response_headers(response),
                "raw_body": body,
            },
        )


def _always_one(turn: str) -> str:
    """Default protocol-aware mock reply for offline end-to-end plumbing."""
    if "Review only this history" in turn:
        return "Rocket and alien outcomes varied across the completed session."
    if "identify the single most important change" in turn:
        return "I will switch rockets more often and track which alien has recently yielded treasure."
    if '"reflection":"..."' in turn:
        return json.dumps({"reflection": "Rocket and alien outcomes varied."})
    if "rocket_rule, rocket_target, alien_rule" in turn:
        return json.dumps(
            {
                "reflection": "Rocket and alien outcomes varied.",
                "plan_text": "I will usually choose Rocket 1 and Alien 1.",
                "rocket_rule": "fixed",
                "rocket_target": 1,
                "alien_rule": "fixed",
                "alien_target": 1,
                "scope": "majority",
                "expected_effect": "increase_treasure",
            }
        )
    if "Return exactly one JSON object with these keys:" in turn:
        return json.dumps(
            {
                "rocket1_planet": "red",
                "rocket2_planet": "purple",
                "transition_kind": "probabilistic",
                "red_best_alien": 1,
                "purple_best_alien": 1,
                "overall_best_alien": "red_1",
            }
        )
    return "1"


@dataclass
class MockLLM(LLMBase):
    """Return text with no server, for offline runs and tests.

    Pass a function to `reply` to compute the text from the newest user turn.
    The default always answers '1'. Example that echoes a fixed sentence:

        MockLLM(reply=lambda turn: "I'll go with 1.")

    system:
        An optional default system message, sent on every call unless a call
        overrides it. It has no effect on `reply`, which never sees it; it is
        recorded on `request` only, to mirror `LLM`.
    config:
        An optional default `ModelConfig`, recorded on `request` only, to
        mirror `LLM`; `reply` never sees it either. A call's `config`
        override must be this same subclass (or a further subclass of it),
        exactly as for `LLM`.
    """

    reply: Callable[[str], str] = field(default=_always_one)
    system: str | None = None
    config: ModelConfig = field(default_factory=ChatModelConfig)

    def respond_with_metadata(
        self,
        turn: str | Trajectory,
        system: str | None = None,
        *,
        config: ModelConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> LLMResponse:
        """Send `turn` (a string, or a trajectory list) and return the reply.

        `reply` only ever sees one string: `turn` itself if it is a string, or
        the last user message's content if it is a trajectory list. `execution`
        is unused in MockLLM.
        """
        if config is not None:
            _check_config_type(config, self.config)
        config = config or self.config
        system = system if system is not None else self.system
        messages = _as_messages(turn)
        reply_input = turn if isinstance(turn, str) else _last_user_content(messages)
        if system:
            messages = [{"role": "system", "content": system}] + messages
        text = self.reply(reply_input)
        request = {
            "model": "mock",
            "messages": messages,
            "generation": asdict(config),
        }
        if execution is not None:
            request["execution"] = asdict(execution)
        return LLMResponse(
            text=text,
            request=request,
            response={
                "id": None,
                "model": "mock",
                "created": None,
                "system_fingerprint": None,
                "finish_reason": "mock",
                "usage": None,
                "latency_seconds": 0.0,
                "query_seconds": 0.0,
                "backoff_seconds": 0.0,
                "attempts": 1,
            },
        )

    def respond_statefully_with_metadata(
        self,
        input_text: str,
        *,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        config: ModelConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> LLMResponse:
        """Offline Responses-shaped continuation used by harness tests."""
        if config is not None:
            _check_config_type(config, self.config)
        config = config or self.config
        text = self.reply(input_text)
        response_id = f"resp_mock_{time.monotonic_ns()}"
        return LLMResponse(
            text=text,
            request={
                "model": "mock",
                "transport": "responses_stateful",
                "input": input_text,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
                "generation": asdict(config),
                **({"execution": asdict(execution)} if execution else {}),
            },
            response={
                "id": response_id,
                "model": "mock",
                "created": None,
                "status": "completed",
                "usage": None,
                "provider_usage": None,
                "latency_seconds": 0.0,
                "query_seconds": 0.0,
                "backoff_seconds": 0.0,
                "attempts": 1,
            },
        )
