from __future__ import annotations

import os
from pathlib import Path

import yaml

from .client import MockModel, ModelClient


def load_registry(path: str | Path | None = None) -> dict:
    path = Path(path or os.getenv("LLM_MODELS_FILE", "model-registry.yaml"))
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"model registry must be a YAML mapping: {path}")
    return data


def make_llm(name: str, *, registry_path: str | Path | None = None):
    if name == "mock":
        return MockModel()
    registry = load_registry(registry_path)
    if name not in registry:
        known = ", ".join(["mock", *sorted(registry)])
        raise SystemExit(f"unknown model {name!r}; registered models: {known}")
    spec = registry[name]
    key_env = spec.get("key_env")
    api_key = os.getenv(key_env, "") if key_env else "EMPTY"
    if key_env and not api_key:
        raise SystemExit(f"{name!r} requires the environment variable {key_env}")
    return ModelClient(
        base_url=spec["base_url"],
        model=spec.get("served_name", name),
        api_key=api_key,
        backend=spec.get("backend", "openai-compatible"),
    )
