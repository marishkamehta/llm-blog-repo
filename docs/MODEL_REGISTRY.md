# Model Registry

Silico keeps model endpoints out of source control. The registry is a YAML file
loaded from `SILICO_MODELS` when that environment variable is set, otherwise
from `model-registry.yaml` at the package root when present.

Copy `model-registry.yaml.example` to `model-registry.yaml` and edit the
entries you actually use. `model-registry.yaml` is ignored by git.

Each entry supports:

- `backend`: a label for the serving backend, such as `vllm`, `ollama`, or
  `azure`.
- `base_url`: the OpenAI-compatible API root.
- `served_name`: the model or deployment name expected by the endpoint.
- `key_env`: optional environment variable containing the API key.
- `config`: optional config kind. Omit this for the default `chat` config.

The name `mock` is reserved for the built-in offline mock.
