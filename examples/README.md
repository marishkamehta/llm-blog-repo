# Introductory decoy experiment

[`decoy_experiment.py`](decoy_experiment.py) is the Python code block from
the published Quick Start II article, extracted
without changing its experiment logic. It presents the same hotel choice with
and without an inferior third option.

Install the repository using the root README. Follow Quick Start II to create
the `gemini-flash` registry entry and set `GEMINI_API_KEY`. From the repository
root on macOS or Linux:

```bash
export SILICO_MODELS="$PWD/model-registry.yaml"
python examples/decoy_experiment.py
```

On PowerShell:

```powershell
$env:SILICO_MODELS = "$PWD/model-registry.yaml"
python examples/decoy_experiment.py
```

The script calls the configured hosted model. It makes ten successful requests
for a fresh run and saves `decoy_results.csv` in the current directory. Retries
can add attempts. Rerunning the script resumes after the existing CSV rows;
use a fresh working directory or move the CSV aside to start a new run. The
output is ignored by Git.

The article reports the original exploratory run and its limitations. Running
this file collects new responses; it does not reproduce the reported values
automatically. For an offline first check, use the mock smoke test in the root
README. For frozen schedules and verified resume behavior, use the
[position-bias collector](../llm-behavior-demos/position_bias_demo/README.md).
