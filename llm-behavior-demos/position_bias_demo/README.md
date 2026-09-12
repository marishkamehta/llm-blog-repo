# Position-Bias Demonstration

This study-specific code prepares the position-bias demonstration without
calling a model. It does not belong to the citable pipeline template and does
not contain MT-Bench questions or answers.

It performs four reproducibility-critical operations:

1. verifies the pinned upstream files by SHA-256;
2. selects the prespecified question IDs and answer models;
3. renders both answer orders and both prompt conditions; and
4. writes a deterministic, shuffled JSONL trial schedule plus its checksum.

## Inputs

Download or clone the pinned FastChat question and prompt files described in
[`../materials/position-bias/source-manifest.yaml`](../materials/position-bias/source-manifest.yaml).
Download the four answer files from the original MT-Bench Space. Keep the
original filenames.

Place them under one input directory:

```text
INPUT/
├── question.jsonl
├── judge_prompts.jsonl
├── alpaca-13b.jsonl
├── gpt-3.5-turbo.jsonl
├── gpt-4.jsonl
└── vicuna-13b-v1.3.jsonl
```

## Build the schedule

```bash
python3 -m position_bias_demo.build_schedule \
  --input-dir INPUT \
  --output-dir OUTPUT \
  --seed 20260815
```

The command fails before writing a schedule if a checksum differs, a selected
record is missing, a response does not have exactly two turns, or the expected
source prompt text cannot be transformed exactly.

It produces:

- `trials.jsonl`: complete rendered prompts and condition metadata;
- `build-manifest.json`: source hashes, seed, counts, and output checksum.

The schedule contains 432 trials:

```text
12 questions × 3 comparison models × 2 answer orders
× 2 prompt conditions × 3 repetitions
```

For the balanced two-item pilot used before the full collection:

```bash
python3 -m position_bias_demo.build_schedule \
  --input-dir INPUT \
  --output-dir PILOT-SCHEDULE \
  --seed 20260815 \
  --question-ids 81 82 \
  --comparison-files alpaca-13b.jsonl
```

This produces 24 calls: two source IDs, one comparison file, two answer
orders, two prompt conditions, and three repetitions. The IDs and comparison
file are selected by their prespecified source order, not by pilot outcomes.

## Test without external materials

```bash
python3 -m unittest position_bias_demo.test_build_schedule
```

The tests use synthetic questions and answers only.

## Collect judgments safely

After building a schedule, test the collector offline:

```bash
python3 -m position_bias_demo.run_trials \
  --schedule OUTPUT/trials.jsonl \
  --build-manifest OUTPUT/build-manifest.json \
  --run-dir RUN \
  --model mock
```

Each successful trial is written as a separate, atomically created JSON file
under `RUN/raw-responses/`. Existing raw records are verified and skipped on
resume; they are never overwritten. `run-manifest.json` freezes the schedule,
model name, generation settings, execution policy, and code state.
`run-status.json` is a replaceable progress summary, and `errors.jsonl`
preserves failed attempts without treating them as completed trials.

Run the collector tests with:

```bash
python3 -m unittest position_bias_demo.test_run_trials
```

Track a collection without displaying prompts or responses:

```bash
python3 -m position_bias_demo.track_run \
  --run-dir RUN \
  --verify
```

For a continuously refreshing view, add `--watch 10`. Checksum verification
reads every completed record on each refresh, so omit `--verify` for lightweight
frequent monitoring and use it for occasional integrity checks.

## Parse and summarize judgments

```bash
python3 -m position_bias_demo.analyze \
  --schedule OUTPUT/trials.jsonl \
  --run-dir RUN \
  --output-dir ANALYSIS
```

The primary parser accepts exactly one case-sensitive `[[A]]` or `[[B]]` token
anywhere in the response, including within Markdown. Missing, multiple, tie,
or lowercase verdicts remain invalid and are never inferred from prose.

The analysis also writes a complete `fastchat-*` sensitivity-analysis set. It
uses FastChat's literal published containment rule: check for `[[A]]`, then
`[[B]]`, then `[[C]]`. Keeping these outputs separate makes the consequence of
FastChat's A-first handling of multiple tokens visible.

The analysis maps the visible verdict back to `baseline` or `comparison`, takes
the unique modal verdict across the three repetitions for each order, and then
compares the underlying selected candidate across orders. It reports both
valid-only repetition stability and a conservative scheduled-trial measure in
which invalid or missing responses remain in the denominator.

Run all builder, collector, parser, and metric tests with:

```bash
python3 -m unittest discover -s position_bias_demo -p 'test_*.py' -v
```
