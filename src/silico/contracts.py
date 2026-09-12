"""Contracts followed by this library.

A `Turn` is one message in a conversation. A `Trajectory` is a sequence of
turns: a whole conversation, built up so far.

`LLMBase` is what every LLM object must implement: `respond_with_metadata`,
given a turn (or a whole trajectory), returns the model's reply as an
`LLMResponse` (the reply plus the exact request sent and the provider's
response). `LLM` and `MockLLM` (see llm.py) are the
two implementations. You only need to inherit from `LLMBase` yourself if you
are writing your own backend to drop in alongside them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TypedDict


class Turn(TypedDict):
    """One message in a trajectory: who sent it, and the text they sent.

    `role` is "user", "assistant", or "system". `content` is the text. For
    example: `{"role": "user", "content": "Pick a rocket: 1 or 2."}`.
    """

    role: str
    content: str


Trajectory = list[Turn]
"""A whole conversation so far, as a plain list of turns, in order."""


@dataclass(frozen=True)
class ExecutionConfig:
    """Our own HTTP client's behavior: timeout and retry policy.

    Never sent to the model. The same values work for every model served
    over an OpenAI-compatible endpoint, because this describes how OUR client
    behaves, not what the model itself accepts.
    """

    timeout: float = 120.0
    max_retries: int = 6
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 60.0


@dataclass(frozen=True)
class ModelConfig(ABC):
    """Base class for one model (or model family)'s generation settings.

    Which settings a model accepts depends on the model, not just on it being
    served over an OpenAI-compatible endpoint. For example, reasoning models
    such as o1/o3/gpt-5 reject a non-default temperature and top_p, and only
    they understand `reasoning_effort`; a vLLM-served open model might add its
    own knobs such as `top_k` or `repetition_penalty`. Subclass this per model
    family and declare exactly the fields that family accepts.
    `ChatModelConfig` (see llm.py) is the default, general-purpose subclass;
    add another subclass instead of piling more fields onto one shared
    dataclass.

    `to_payload()` works the same for every subclass, because this is a plain
    dataclass either way: `dataclasses.asdict(self)` turns its fields into a
    dict. `asdict` is the standard tool for that (rather than a hand-written
    `__dict__` method, or reading `vars(self)` directly): both of those also
    happen to work for a flat dataclass like the ones here, but `asdict`
    additionally converts any dataclass nested inside a field, so it stays
    correct if a future subclass ever adds one. `to_payload()` then drops any
    field left at `None`, so only the fields you actually set reach the
    request, and a subclass never needs to reimplement this method.
    """

    def to_payload(self) -> dict[str, object]:
        """This config's set fields, ready to merge into a request body."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class LLMResponse:
    """Text plus the exact request settings and available provider metadata."""

    text: str
    request: dict
    response: dict


class LLMBase(ABC):
    """Send one turn, or a whole trajectory, to a model and get text back.

    turn:
        Either a plain string (one self-contained turn, the common case), or a
        `Trajectory`, a list of turns built up so far by the caller. Pass a
        growing trajectory, only ever appending to it between calls, to run a
        multi-turn conversation: keeping that prefix unchanged is what lets a
        cache-aware backend (vLLM prefix caching, hosted prompt caching such
        as OpenAI/Azure, or Anthropic `cache_control`) reuse it instead of
        paying for it again on every call.
    system:
        An optional system message (a persona, or any fixed instruction).
        `LLM` and `MockLLM` let you set a default once at construction;
        passing it here overrides that default for one call.
    config, execution:
        Per-call overrides of this object's default `ModelConfig` and
        `ExecutionConfig` (see those classes). Every real implementation
        needs some notion of both, so they are part of this contract rather
        than bolted on separately by each backend.
    """

    @abstractmethod
    def respond_with_metadata(
        self,
        turn: str | Trajectory,
        system: str | None = None,
        *,
        config: ModelConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> LLMResponse:
        """Return the model's reply to `turn`, plus the request and response."""
        ...

    def respond(self, turn: str | Trajectory, system: str | None = None) -> str:
        """Return just the model's text reply to `turn`."""
        return self.respond_with_metadata(turn, system=system).text

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
        """Return a response that may contain provider-native tool calls."""
        raise NotImplementedError("this LLM backend does not support native tools")
