"""silico: put a language model in the agent's seat of a behavioral experiment.

A behavioral experiment runs as a loop of turns between two sides: an
environment presents a turn (a question, a game state, an instruction), and
an agent responds to it. Usually a human sits in the agent's seat. silico
lets a language model sit there instead, so your experiment loop keeps doing
the same thing (present a turn, get a reply) whether the reply comes from a
hosted model, a local server, or an offline stand-in for testing.

silico gives you three things:

    1. LLM: connects to a real model over the network (a local server, or a
       hosted API such as Azure).
    2. MockLLM: answers with no network call at all, for offline development
       and tests.
    3. make_llm(name): builds either one from a short, registered model name,
       so your experiment code can refer to a full model setup by just a name.

`LLM` and `MockLLM` look identical from your experiment's side (both have one
method, `respond`), so you can swap one for the other with no other change:
develop and test against `MockLLM`, then switch to `LLM` (or `make_llm`) to
run for real.

1. LLM: talk to a real model.

Why: use this to run your experiment against an actual model, whether hosted
(for example Azure) or served locally (for example vLLM).

Input: `respond(turn, system=None)`.

    turn:
        Either a plain string, one self-contained turn (the common case), or
        a `Trajectory` (a list of `Turn`s: `{"role", "content"}` dicts),
        built by your code as a whole conversation so far. Passing a
        trajectory is how you run a multi-turn conversation: keep the same
        list between calls, only ever appending to it (the user's new turn,
        then the reply you got back, as an assistant entry), and send the
        growing list each time. A cache-aware backend (vLLM prefix caching,
        hosted prompt caching such as OpenAI/Azure, or Anthropic
        `cache_control`) can then reuse the part that has not changed,
        instead of paying for it again on every call.
    system:
        An optional system message: a persona, or any fixed instruction
        ("You are anxious.", "Reply with only a number."). Set it once at
        construction to fix it for every call, or pass it here to override
        that default for one call.

What it does: sends `turn` to the endpoint's chat API over HTTP, retrying on
rate limits, and reads back the model's reply. (`respond_with_metadata`
returns the exact request sent and the provider's full response too, for
logging.)

Output: the model's text reply.

Example, a single turn in, text back:

    from silico import LLM

    llm = LLM(base_url="http://localhost:8000/v1",
             model="Qwen/Qwen2.5-7B-Instruct",
             system="You are playing a game.")
    text = llm.respond("Pick a rocket: 1 or 2.")
    # text -> "2"

Example, a growing trajectory, your code carries the history:

    trajectory = [{"role": "user", "content": "Turn 1: pick a rocket: 1 or 2."}]
    text = llm.respond(trajectory)
    # text -> "1"
    trajectory.append({"role": "assistant", "content": text})
    trajectory.append({"role": "user", "content": "Turn 2: pick a rocket: 1 or 2."})
    text = llm.respond(trajectory)
    # text -> "2"  (the model's reply can depend on turn 1, sent again above)

Tuning generation settings. Two different things can vary between models, so
they are two different config objects, both optional constructor fields on
`LLM` (also accepted as respond_with_metadata's keyword overrides for one
call): `config` (a `ModelConfig`, see llm.py) is what is sent to the model,
for example temperature or max_tokens, and depends on the model itself, not
just on it being OpenAI-compatible (a reasoning model such as o1/o3/gpt-5
rejects temperature and only it understands reasoning_effort). Subclass
`ModelConfig` per model family to declare exactly the fields that family
accepts; `ChatModelConfig` is the default, general-purpose one. `execution`
(an `ExecutionConfig`) is our own HTTP client's timeout and retry policy,
never sent to the model, so the same one works for every model.

2. MockLLM: no network call, for offline development and tests.

Why: develop and test your experiment loop, or run it in a place with no
network access, before you have a real endpoint.

Input: the same `respond(turn, system=None)` as `LLM`, above. The constructor
also takes `reply`, a function you supply to compute the text.

What it does: calls `reply` on the newest turn's text. It never contacts a
network or a real model.

Output: the text `reply` returns (or "1" to everything, by default).

Example:

    from silico import MockLLM

    llm = MockLLM(reply=lambda turn: "I'll go with 1.")
    text = llm.respond("Pick a rocket: 1 or 2.")
    # text -> "I'll go with 1."

3. make_llm(name): build one from a registered name.

Why: so your experiment code refers to a model by a short name (for example
"gpt-4.1-mini"), instead of repeating its endpoint and key everywhere it is
used.

Input: a short name, either "mock" (always available) or a name you
registered in model-registry.yaml (see registry.py).

What it does: looks up that name's backend, endpoint, and the environment
variable holding its API key.

Output: a ready `LLM` (for a real, registered model) or `MockLLM` (for
"mock").

Example:

    from silico import make_llm

    llm = make_llm("mock")   # or a name you registered, e.g. "gpt-4.1-mini"
    text = llm.respond("Pick a rocket: 1 or 2.")

Writing your own backend. `LLMBase` (see contracts.py) is an abstract base
class: inherit from it and implement `respond_with_metadata(turn, system=None,
*, config=None, execution=None) -> LLMResponse` to drop a new backend in
alongside `LLM` and `MockLLM`. You get `respond(turn, system=None) -> str` for
free: `LLMBase` already implements it as `respond_with_metadata(...).text`,
so it is not something you write yourself. You only need any of this if you
are adding a backend, not for everyday use. Example, a fixed persona wrapped
around another backend:

    @dataclass
    class MoodLLM(LLMBase):
        llm: LLMBase
        last_won: bool = True

        def respond_with_metadata(self, turn, system=None, **kwargs):
            mood = "upbeat after a win" if self.last_won else "discouraged after a loss"
            return self.llm.respond_with_metadata(
                turn, system=system or f"You feel {mood}.", **kwargs
            )
"""

from .contracts import (
    ExecutionConfig,
    LLMBase,
    LLMResponse,
    ModelConfig,
    Trajectory,
    Turn,
)
from .llm import LLM, ChatModelConfig, MockLLM
from .registry import MODELS, ModelSpec, make_llm

__all__ = [
    "LLM",
    "MODELS",
    "ChatModelConfig",
    "ExecutionConfig",
    "LLMBase",
    "LLMResponse",
    "MockLLM",
    "ModelConfig",
    "ModelSpec",
    "Trajectory",
    "Turn",
    "make_llm",
]
