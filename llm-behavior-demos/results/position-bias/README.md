# Position-Bias Demonstration Results

The completed run used `Qwen/Qwen2.5-3B-Instruct`, served with vLLM 0.22.1.
Temperature was 0, `top_p` used the server default, and the maximum response
length was 1,024 tokens. All 432 scheduled judgments were collected and parsed
successfully under the primary rule.

The files report three levels of analysis:

- `parsed-trials.csv` records the extracted verdict and selected candidate for
  every scheduled judgment without reproducing the prompt or candidate text;
- `repetition-cells.csv` and `position-pairs.csv` contain the primary
  repetition and answer-order measures;
- `summary-by-prompt.csv` contains the aggregate results for each prompt
  condition.

The corresponding `fastchat-*` files apply FastChat's literal A-first parser as
a sensitivity analysis. `analysis-manifest.json` defines both parsing rules and
records SHA-256 checksums for every table.

The MT-Bench questions and candidate answers are not redistributed here. Their
source, retrieval procedure, checksums, and licensing review are recorded in
the [`material audit`](../../materials/position-bias/AUDIT.md).
