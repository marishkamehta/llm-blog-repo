# NeuBAROCO Cognitive Demonstration

This pipeline demonstration adapts the published English NeuBAROCO NLI
experiment. It uses original stimuli and gold labels from the authors' CC BY
4.0 release. The reported analysis includes all 366 two-premise records and
excludes nine three-premise records whose third premise is omitted by the
accompanying published prompt formatter.

The earlier `items.jsonl` file contains exploratory constructed materials and
is not an input to this demonstration.

## Obtain the pinned source

```bash
git clone https://github.com/kmineshima/NeuBAROCO.git SOURCE
git -C SOURCE checkout --detach 447929fdabe07bc3d13efae8e0c527fd458df177
```

The builder verifies the published source file's SHA-256 before reading it.

## Build the two-premise schedule

```bash
python3 -m belief_bias_demo.build_schedule \
  --source SOURCE/acl2024/NeuBAROCO_NALOMA.tsv \
  --output-dir OUTPUT \
  --seed 20260816 \
  --mode full \
  --exclude-three-premise-items
```

The schedule contains 366 items. Three repetitions produce 1,098 trials at each
temperature. The nine exclusions and their methodological reason are recorded
in the build manifest.

## Collect responses

```bash
python3 -m position_bias_demo.run_trials \
  --schedule OUTPUT/trials.jsonl \
  --build-manifest OUTPUT/build-manifest.json \
  --run-dir RUN \
  --model MODEL_NAME \
  --temperature 0 \
  --max-tokens 16
```

## Analyze responses

```bash
python3 -m belief_bias_demo.analyze \
  --schedule OUTPUT/trials.jsonl \
  --run-dir RUN \
  --output-dir ANALYSIS
```

The primary analysis reproduces the published English parser: use the first
token on the final response line, case-insensitively, and allow a terminal
period. A separate strict sensitivity analysis accepts only an exact complete
response of `entailment`, `contradiction`, or `neither` after whitespace and
case normalization.

## Test

```bash
python3 -m unittest discover -s belief_bias_demo -p 'test_*.py' -v
```

## Required citations

- Ando et al. (2023), *Evaluating Large Language Models with NeuBAROCO*.
- Ozeki et al. (2024), *Exploring Reasoning Biases in Large Language Models
  Through Syllogism*.

Use the full citations provided by the
[official NeuBAROCO repository](https://github.com/kmineshima/NeuBAROCO).
