# Silico

Silico is a small Python client that lets a language model sit in the agent
seat of a behavioral experiment. Your experiment loop presents a turn, Silico
returns a reply, and the same code can run against a hosted model, a local
OpenAI-compatible server, or an offline mock.

This repository is intentionally minimal. Experiment-specific workflows,
materials, audits, and reviewed results belong in the companion
`llm-behavior-experiments` repository.

## What Silico Provides

- `LLM`: an OpenAI-compatible chat client with retry and metadata capture.
- `MockLLM`: an offline stand-in for development and tests.
- `make_llm(name)`: a tiny registry loader for short model names.
- Plain conversation types: `Turn` and `Trajectory`.

## Install

Use Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, create the environment with `py -m venv .venv` and
activate it with `.venv\Scripts\Activate.ps1`; then use the same `python`
commands.

## Quick Start

Run entirely offline with the built-in mock:

```python
from silico import make_llm

llm = make_llm("mock")
print(llm.respond("Choose A or B. Reply with only the letter."))
```

Run the included smoke check:

```bash
python smokes/smoke_chat.py --model mock
python examples/growing_trajectory.py
python -m pytest tests
```

## Model Registry

Real model endpoints are configured outside source control. Copy the example
registry, edit it for your environment, and set `SILICO_MODELS` if the file is
not at the default lookup path.

```bash
cp model-registry.yaml.example model-registry.yaml
export SILICO_MODELS="$PWD/model-registry.yaml"
```

The registry maps a short name to an endpoint, served model name, optional API
key environment variable, and config kind. The built-in `mock` model is always
available and cannot be overridden. More detail is in
[`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md).

## Repository Layout

```text
src/silico/              Library code
tests/                   Offline unit tests
smokes/                  Minimal command-line smoke check
examples/                Small usage examples
docker/                  Optional local OpenAI-compatible server setup
model-registry.yaml.example
```

## License and Citation

Silico is released under the Creative Commons Attribution-NonCommercial 4.0
International license. Citation metadata is in [`CITATION.cff`](CITATION.cff).
