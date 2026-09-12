# LLM Behavior Demonstrations

This repository contains the two demonstrations used in the *Studying LLM
Behavior* blog series. They show how a published experimental procedure can be
administered to a language model under controlled conditions, while preserving
the information needed to understand how each response was produced.

These are demonstrations of the pipeline rather than replication studies. They
use published materials so that the tasks, scoring rules, and differences from
the source studies can be reported openly.

## The two demonstrations

### Does answer order change an LLM judge's decision?

The position-bias demonstration presents the same pair of answers in both
possible orders. It tests whether the judge continues to prefer the same
underlying answer after that answer moves from the first position to the
second, or vice versa.

The procedure is based on:

- Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush
  Vosoughi. 2025. [*Judging the Judges: A Systematic Study of Position Bias in
  LLM-as-a-Judge*](https://arxiv.org/abs/2406.07791). AACL-IJCNLP 2025.
- Lianmin Zheng et al. 2023. [*Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena*](https://arxiv.org/abs/2306.05685). NeurIPS 2023 Datasets and
  Benchmarks Track.

The runnable workflow is in [`position_bias_demo/`](position_bias_demo/). The
source and licensing review is in
[`materials/position-bias/`](materials/position-bias/), and the reviewed
analysis outputs are in [`results/position-bias/`](results/position-bias/).

### What happens when logic conflicts with ordinary belief?

The belief-bias demonstration uses NeuBAROCO syllogistic-reasoning problems.
The logical task remains the same while the content is consistent with ordinary
belief, inconsistent with it, or symbolic.

The materials and procedure come from:

- Risako Ando, Takanobu Morishita, Hirohiko Abe, Koji Mineshima, and Mitsuhiro
  Okada. 2023. [*Evaluating Large Language Models with NeuBAROCO: Syllogistic
  Reasoning Ability and Human-like
  Biases*](https://aclanthology.org/2023.naloma-1.1/). Proceedings of the 4th
  Natural Logic Meets Machine Learning Workshop, 1–11.
- Kentaro Ozeki, Risako Ando, Takanobu Morishita, Hirohiko Abe, Koji Mineshima,
  and Mitsuhiro Okada. 2024. [*Exploring Reasoning Biases in Large Language
  Models Through Syllogism: Insights from the NeuBAROCO
  Dataset*](https://doi.org/10.18653/v1/2024.findings-acl.950). Findings of ACL
  2024, 16063–16077.

The runnable workflow is in [`belief_bias_demo/`](belief_bias_demo/). The
source audit is in [`materials/belief-bias/`](materials/belief-bias/), and the
reviewed analysis outputs are in
[`results/belief-bias/`](results/belief-bias/).

## Install the demonstrations

The demonstrations use the model clients and recording functions provided by
[`llm-behavior-pipeline`](../docs/PIPELINE.md).
From this folder in the companion checkout, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ..
python -m pip install -e .
```

The README inside each demonstration folder explains how to obtain the source
materials, build the schedule, collect responses, and run the analysis.

## What is—and is not—included

The repository includes the schedule builders, collection and analysis code,
source audits, scoring rules, aggregate results, and figures. Complete working
run directories remain local because they may contain local paths or source
text that cannot be redistributed clearly.

The MT-Bench candidate answers are not included. Their host does not provide
clear redistribution terms, so the position-bias workflow retrieves them from
the original source and verifies their SHA-256 digests before use. NeuBAROCO is
released under CC BY 4.0; its pinned source and attribution are recorded in the
material audit.

## Test the repository

After installing this repository and the pipeline dependency, run:

```bash
python -m unittest discover -s position_bias_demo -p 'test_*.py' -v
python -m unittest discover -s belief_bias_demo -p 'test_*.py' -v
```

The tests use synthetic fixtures. They do not download research materials,
contact a model, or require a GPU.

## Licensing and citation

The code in this repository is released under the [MIT License](LICENSE).
Upstream materials retain their original licenses and should be cited through
the papers listed above. The [`materials/`](materials/) directory records the
exact source revisions, checksums, and redistribution decisions used for each
demonstration.

The demonstrations do not have a separate DOI. The citable software artifact is
`llm-behavior-pipeline`; cite the version used for the study.
