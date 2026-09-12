# LLM Behavioral Pipeline Template

Use language models as participants without coupling the behavioral task to one
model provider. The model-facing code lives in `src/`, with study-specific demonstrations
in `llm-behavior-demos/`. Run the commands below from the repository root.

This template is the citable research artifact. The study-specific blog
demonstrations are in the `llm-behavior-demos/` folder in this companion checkout. Each public pipeline
version should be released on GitHub and archived through Zenodo with a DOI.

## What is included

- a tiny, typed model interface;
- an offline mock for developing without compute or cost;
- a YAML registry for Ollama, vLLM, Azure, or another chat-completions endpoint;
- exact request/response metadata capture with redacted authorization headers;
- study-level run, artifact, exclusion, and deviation record templates;
- a smoke test and a growing-trajectory example;
- an optional single-GPU vLLM Docker profile;
- unit tests, an MIT license, and `CITATION.cff`.

The distribution also retains the `silico` interface used to collect the
published blog demonstrations. This includes explicit execution settings,
retry records, and the configurable mock used by their test suites. Keeping
that interface in the pipeline distribution lets the demonstration package use the shared model clients.

## Start without a model

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python smokes/smoke_chat.py --model mock
pytest
```

## Connect a real backend

```bash
cp model-registry.yaml.example model-registry.yaml
```

Edit the copied file and run:

```bash
python smokes/smoke_chat.py --model YOUR_REGISTERED_NAME
```

Ollama, vLLM, and current Microsoft Foundry deployments can expose compatible
chat-completions routes, but they do not necessarily support identical fields or
behavior. Only send generation fields known to be supported by your model and
backend.

## Add your experiment

Put task code in a separate `study/` package. Import only `make_llm`,
`GenerationConfig`, and the `Trajectory` type from this template. Keep stimuli,
conditions, randomization, scoring, and exclusion logic outside the pipeline.

See `examples/growing_trajectory.py` for the boundary between a task loop and
the model interface.

Copy the templates in [`reproducibility/`](../reproducibility/) into the study
package before freezing a protocol. They define the minimum information needed
to trace a reported result back to its materials, request, and raw response.

## Publish and cite it

Before publishing:

1. Add the final repository URL and verified release date to `CITATION.cff`; review contributor information.
2. Choose a license deliberately; MIT is supplied as a permissive default.
3. Create a public repository and a versioned release.
4. Optionally connect the public repository to Zenodo before making the release
   so the release receives a DOI.
5. Add the resulting DOI to `CITATION.cff` as `doi: "..."` in the next release.

GitHub recognizes `CITATION.cff` and can display a **Cite this repository**
control. A DOI identifies a fixed archived release more robustly than a moving
branch.

## Safety defaults

- Registry files containing real endpoints are ignored by git.
- Keys come from environment variables and are redacted in saved request data.
- The example vLLM port binds to `127.0.0.1`, not every network interface.
- Generated outputs are ignored unless deliberately reviewed and published.
- The template never creates cloud resources or makes paid calls by itself.
