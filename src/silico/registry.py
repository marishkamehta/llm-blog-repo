"""A small registry that maps a short model name to how to reach it.

You refer to a model by a short name (for example "mock"), and the registry
resolves the endpoint, the served model name, which environment variable
holds the API key, and which `ModelConfig` subclass that model expects.
`make_llm(name)` turns a name into a ready client.

Where the models come from:

  - One model, "mock", is always built in. It is an offline stand-in that needs
    no server or key, so you can test the wiring before any real model exists.
  - Real models are NOT stored in this repo. You list them in a YAML file that
    is ignored by git, so your endpoints and choices stay yours and are never
    committed. The file is found in this order:
      1. the path in the SILICO_MODELS environment variable, if set, otherwise
      2. model-registry.yaml at the repo root, if it exists.
    If neither is present, only the built-in mock is available. Copy the
    template model-registry.yaml.example to model-registry.yaml and edit it.

The YAML maps each short name to its fields. Example:

    phi-4-mini:
      backend: azure_foundry
      base_url: https://my-endpoint.models.ai.azure.com/v1
      served_name: Phi-4-mini-instruct
      key_env: AZURE_OPENAI_API_KEY

Fields per model:
  - backend:      which client talks to it. "mock" is the offline stand-in;
                  other backends (vllm, azure_foundry) are OpenAI-compatible.
  - base_url:     the endpoint address.
  - served_name:  the exact model name the endpoint expects. Defaults to the
                  short name if omitted.
  - key_env:      the environment variable that holds the API key, for example
                  AZURE_OPENAI_API_KEY. Omit it when the endpoint needs no key,
                  like a local vLLM server. Naming it per model lets several
                  providers' keys be set at the same time without clashing.
  - config:       which `ModelConfig` subclass (see llm.py) this model
                  accepts, by name, from `_CONFIG_KINDS` below. Defaults to
                  "chat" (`ChatModelConfig`, most chat-completions models) if
                  omitted. `make_llm` builds the returned `LLM` with an
                  instance of that class as its default `config`, and any
                  later per-call `config` override must match it (see
                  `LLM.respond_with_metadata` in llm.py). Add a new entry to
                  `_CONFIG_KINDS` (and a new `ModelConfig` subclass in llm.py)
                  before a model can declare a `config` other than "chat", for
                  example a reasoning model that needs its own subclass.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from .contracts import LLMBase
from .llm import LLM, ChatModelConfig, MockLLM, ModelConfig

# Every "config" name a model entry may declare in the YAML. Add a subclass to
# llm.py and a matching entry here before a model can use it.
_CONFIG_KINDS: dict[str, type[ModelConfig]] = {
    "chat": ChatModelConfig,
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything the registry needs to reach one model. See the module docstring
    for what each field means."""

    backend: str
    base_url: str = ""
    served_name: str = ""
    key_env: str | None = None
    config_kind: str = "chat"


# Always available: an offline stand-in that answers with no server or key.
_MOCK = ModelSpec(backend="mock", served_name="mock")


def _config_path() -> pathlib.Path | None:
    """The models YAML file to load, or None if there is not one.

    Prefers the SILICO_MODELS environment variable, then the default location
    model-registry.yaml at the repo root. Returns None when neither is set or
    present.
    """
    env = os.environ.get("SILICO_MODELS")
    if env:
        return pathlib.Path(env).expanduser()
    default = pathlib.Path(__file__).resolve().parents[1] / "model-registry.yaml"
    return default if default.exists() else None


def load_models(path: pathlib.Path | None = None) -> dict[str, ModelSpec]:
    """Build the model table: the built-in mock plus any models from the YAML file.

    `path` overrides where to read. When it is None, the file is found via
    SILICO_MODELS or the default location. A missing file is fine when nothing
    asked for one; you just get the mock. Stops with a clear message if the file
    lists the same model name twice, or reuses the reserved name "mock".
    """
    models: dict[str, ModelSpec] = {"mock": _MOCK}

    path = path or _config_path()
    if path is None:
        return models
    if not path.exists():
        raise SystemExit(f"models file not found: {path}")

    try:
        import yaml
    except ModuleNotFoundError:
        raise SystemExit(
            "reading a models file needs PyYAML. Install it with "
            "`pip install pyyaml`, or unset SILICO_MODELS to use only the mock."
        )

    # PyYAML silently keeps the last value when a mapping repeats a key, so a
    # duplicated model name would pass unnoticed. This loader raises instead, so
    # a typo like listing the same name twice is caught rather than hidden.
    class _NoDuplicateKeys(yaml.SafeLoader):
        pass

    def _mapping_no_dupes(loader, node, deep=False):
        seen: dict = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise SystemExit(
                    f"duplicate model name {key!r} in the models file {path}. "
                    f"Each name may appear only once."
                )
            seen[key] = loader.construct_object(value_node, deep=deep)
        return seen

    _NoDuplicateKeys.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_no_dupes
    )

    data = yaml.load(path.read_text(), Loader=_NoDuplicateKeys) or {}
    for name, entry in data.items():
        if name == "mock":
            raise SystemExit(
                f'"mock" is a built-in name and cannot be redefined in {path}. '
                f"Rename that entry."
            )
        config_kind = entry.get("config", "chat")
        if config_kind not in _CONFIG_KINDS:
            known = ", ".join(sorted(_CONFIG_KINDS))
            raise SystemExit(
                f"model {name!r} declares config: {config_kind!r}, which is "
                f"not registered. Known config kinds: {known}."
            )
        models[name] = ModelSpec(
            backend=entry["backend"],
            base_url=entry.get("base_url", ""),
            served_name=entry.get("served_name", name),
            key_env=entry.get("key_env"),
            config_kind=config_kind,
        )
    return models


# Loaded once when the package is imported. If you change the file during a run,
# call load_models() again to pick up the change.
MODELS: dict[str, ModelSpec] = load_models()


def make_llm(name: str) -> LLMBase:
    """Return a ready LLM object for a registered model name.

    Resolves the endpoint, the API key, and the right `ModelConfig` subclass
    from the registry (see the module docstring's `config` field), so the
    caller only has to know the short name. Stops with a clear message if the
    name is not registered, if the model needs a key that is not set in the
    environment, or if its `config` kind is not one this registry knows.
    """
    if name not in MODELS:
        known = ", ".join(sorted(MODELS)) or "(none registered)"
        raise SystemExit(f"unknown model {name!r}. registered models: {known}")
    spec = MODELS[name]

    # The offline mock needs no server or key.
    if spec.backend == "mock":
        return MockLLM()

    # A key is only read when the model declares one. A local vLLM server does
    # not, so it stays "EMPTY" and is ignored by the server.
    api_key = "EMPTY"
    if spec.key_env:
        api_key = os.environ.get(spec.key_env, "")
        if not api_key:
            raise SystemExit(
                f"model {name!r} needs an API key in the environment variable "
                f"{spec.key_env}, but it is not set. Set it and try again, e.g. "
                f"export {spec.key_env}=<your-key>"
            )

    # The backend decides which client to build. Every backend we support today
    # speaks the OpenAI-compatible protocol, so one client handles them all.
    # When we add Azure OpenAI (a different auth header and an api-version), add
    # a branch here that returns that client instead.
    config_class = _CONFIG_KINDS[spec.config_kind]
    return LLM(
        base_url=spec.base_url,
        model=spec.served_name,
        api_key=api_key,
        backend=spec.backend,
        config=config_class(),
    )
