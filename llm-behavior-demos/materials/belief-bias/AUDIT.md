# NeuBAROCO Material Audit

## Authoritative release

The official NeuBAROCO repository was retrieved and pinned at commit
`447929fdabe07bc3d13efae8e0c527fd458df177`. Its README links the 2023 NALOMA
paper and 2024 ACL Findings paper and licenses the datasets under CC BY 4.0.

The demonstration source is `acl2024/NeuBAROCO_NALOMA.tsv`, SHA-256
`b3f98f4a8f5f36079b9335c460fcea3b1044884dbb6f0ec627aa1b2b596666a5`.
It contains 375 published stimuli, complete English premises and hypotheses,
gold NLI labels, belief-content classifications, syllogism types, and bias
annotations.

## Final two-premise analysis set

The completed collection included all 375 records. During the material audit,
nine extended records were found to contain three premises even though the
accompanying prompt formatter includes only two. Omitting the third premise
can change whether the published hypothesis is entailed or contradicted.

Source rows 234–239 and 250–252 were therefore excluded from the reported
demonstration analysis. This decision was based on prompt completeness, not on
Qwen's accuracy. The final analysis contains 366 items and 12,078 responses:
three responses per item at each of eleven temperatures.

The excluded rows and reason are recorded in the version 2 series-analysis
manifest. Their collected responses remain in the raw record and were not
deleted.

## Similarities to the published experiment

- Uses the authors' original English NALOMA stimuli and gold labels.
- Retains the three-way entailment, contradiction, and neutral task.
- Retains the published content classifications.
- Uses the official zero-shot English instructions and problem formatting.
- Reports accuracy by content type and gold label.
- Uses the official final-line, first-token response parser as the primary rule.

## Differences from the published experiment

- The reported analysis uses all 366 two-premise records rather than all 375
  source records.
- Nine three-premise records are excluded because the published formatter
  omits information needed to classify their hypotheses.
- It uses Qwen-3B rather than the paper's model set.
- It collects three responses per item to measure repetition stability.
- It repeats the task at temperatures from 0.0 to 1.0 in increments of 0.1.
- It adds a stricter whole-response parser as a sensitivity analysis.

Accordingly, describe this as a pipeline demonstration adapted from the
published NeuBAROCO procedure, not as a replication test or an estimate of the
paper's model-specific effect sizes.
