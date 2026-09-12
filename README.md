# Studying LLM Behavior — Blog Companion

Code, worked demonstrations, source audits, and reviewed results accompanying
*Studying LLM Behavior: A Practical Guide for Behavioral Researchers* by
Don K. Dennis and Marishka M. Mehta.

Read the [blog series](https://marishkamehta.github.io/blog/2026/controlled-experiments/)
and find this companion at [marishkamehta/llm-blog-repo](https://github.com/marishkamehta/llm-blog-repo).
This repository collects the articles'
supporting code and research artifacts in one checkout. The
pipeline and demonstrations remain separate Python packages. The examples
demonstrate controlled procedures; they are not replication studies.

## Start here

Use the [blog-to-code guide](docs/READING_MAP.md) to follow the articles through
the corresponding scripts, material audits, and result files.

| What you want to do | Where to go |
| --- | --- |
| Run the introductory hotel-choice experiment | [Decoy example](examples/README.md) |
| Try a model request offline | [Pipeline](docs/PIPELINE.md) |
| Configure a model backend | [Pipeline setup](docs/PIPELINE.md) and [example registry](model-registry.yaml.example) |
| Test whether answer order changes a judgment | [Position-bias workflow](llm-behavior-demos/position_bias_demo/README.md) |
| Test reasoning when logic conflicts with belief | [Belief-bias workflow](llm-behavior-demos/belief_bias_demo/README.md) |
| Inspect the reported results | [Position bias](llm-behavior-demos/results/position-bias/README.md) · [Belief bias](llm-behavior-demos/results/belief-bias/RESULTS.md) |
| Check material sources and attribution | [Source audits](llm-behavior-demos/materials/README.md) |
| Record a new study reproducibly | [Record templates](reproducibility/README.md) |

## Install and run offline checks

Use Python 3.10 or later. From the root of this checkout, on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smokes/smoke_chat.py --model mock
python examples/growing_trajectory.py
python -m pytest tests
python -m unittest discover -s llm-behavior-demos/position_bias_demo -p 'test_*.py' -v
python -m unittest discover -s llm-behavior-demos/belief_bias_demo -p 'test_*.py' -v
```

On Windows PowerShell, create the environment with `py -m venv .venv` and
activate it with `.venv\Scripts\Activate.ps1`; then use the same `python`
commands. Package installation may require internet access. The checks use
synthetic fixtures and a mock: they do not download study materials or call a
model. The smoke test should print `MOCK_RESPONSE`.

For a demonstration, change into `llm-behavior-demos/` and follow its workflow
README. Retrieve the pinned materials from the sources listed there. For a
real backend, follow the corresponding published blog setup guide from the repository root. When running demonstrations through Silico, set
`SILICO_MODELS` to the absolute path of your configured registry before starting
Python; this avoids dependence on the installed package's default lookup path.
For example, from `llm-behavior-demos/` on macOS or Linux:

```bash
export SILICO_MODELS="$PWD/../model-registry.yaml"
```

## Repository layout

```text
src/                    Model clients: silico and llm_behavior_pipeline
smokes/                 Offline smoke test
tests/                  Pipeline tests
reproducibility/        Study record templates
docker/                 Optional vLLM setup
llm-behavior-demos/      Two demonstrations, material audits, reviewed outputs
examples/               Trajectory example and introductory decoy experiment
docs/                   Blog-to-code guide and pipeline setup
requirements.txt        Installs both local packages and the existing test suite
```

Raw model responses are not bundled. MT-Bench candidate answers must be retrieved upstream;
their redistribution terms are not clear. NeuBAROCO attribution and its
CC BY 4.0 terms are recorded in the material audit. Aggregate outputs and their
original checksum manifests are included. Recreating the published aggregates requires collecting or
obtaining the underlying responses.

## License and citation

The pipeline and demonstration code retain their existing MIT licenses:
[pipeline license](LICENSE) and
[demonstration license](llm-behavior-demos/LICENSE). Upstream research materials
retain their own terms; see the [material source audits](llm-behavior-demos/materials/README.md).
Citation metadata for the pipeline is in [CITATION.cff](CITATION.cff).
