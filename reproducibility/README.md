# Reproducibility Records

Copy these templates into each study and complete them before collecting the
confirmatory data.

## Files

- `run-manifest.yaml.example`: one record for each execution batch;
- `deviations.csv`: departures from the frozen protocol;
- `exclusions.csv`: trial-level exclusions and their prespecified rules; and
- `artifact-manifest.csv`: source, license, version, and checksum for every
  input and released output.

The examples contain no substantive study information. Add task-specific
fields in the study repository rather than coupling them to the model client.

## Raw-data rule

Write each raw response once. Never edit it in place. Parsing, normalization,
and scoring must produce new derived files linked by `trial_id` and
`raw_response_sha256`.

## Recommended release layout

```text
study/
├── protocol/
├── materials/
├── schedules/
├── prompts/
├── runs/
│   └── RUN_ID/
│       ├── run-manifest.yaml
│       ├── requests.jsonl
│       ├── raw-responses.jsonl
│       ├── parsed-responses.csv
│       └── errors.jsonl
├── analysis/
├── figures/
├── deviations.csv
├── exclusions.csv
└── artifact-manifest.csv
```
