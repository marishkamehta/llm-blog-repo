# Position-Bias Material Audit

## What has been verified

- The current FastChat source was inspected at commit
  `587d5cfa1609a43d192cedb8441cac3c17db105d`.
- The repository contains the 80 MT-Bench questions and the original pairwise
  judge templates.
- FastChat is distributed under Apache License 2.0.
- The four required answer files were retrieved from the archive referenced by
  FastChat. Each contains all 12 outcome-blind audited and frozen question IDs.
- File checksums and embedded model labels are recorded in
  `source-manifest.yaml`.

## License finding

The Hugging Face space that distributes the pre-generated answers declares its
license as `other` and supplies no additional terms in its README. The files
have therefore not been copied into this repository. This is a conservative
redistribution decision, not a conclusion that research use is prohibited.

This finding was checked against more than the current web page:

- The complete Hugging Face Space Git history was cloned and inspected.
- `license: other` appears in the initial Space commit (`9fa833d`, 2023-06-16)
  and remains unchanged in every later README revision.
- The Space has one branch, no tags, and no `LICENSE`, `NOTICE`, data card, or
  custom-terms file.
- The complete 29-page NeurIPS paper was searched and its data-release appendix
  inspected. It describes public availability, human-participant consent, and
  release procedures but does not name a license for the pre-generated
  MT-Bench model answers.
- The FastChat README directs users to download the answer files, but neither
  that instruction nor the downloader states that FastChat's Apache-2.0 license
  applies to the separately hosted files.

The NeurIPS 2023 Datasets and Benchmarks call required authors to confirm a data
license and hosting plan during submission. The public proceedings page does
not expose that form. An attempted audit through the OpenReview public API was
blocked with HTTP 403, so the original submission form remains unchecked.

The demonstration therefore uses the retrieval-and-checksum approach. Readers
obtain the answer files from their original host, and the builder verifies
their SHA-256 digests before constructing a schedule. Neither the source answer
files nor prompts containing their text are redistributed in this repository.

## Prompt-extension overlap found during the audit

Appendix D.1 of Zheng et al. already reports a `short` prompt condition that
removed the position-bias warning together with the request for an explanation
and other instructions. The warning-only ablation is narrower, but it
must be described as isolating one component of an earlier multi-component
prompt manipulation—not as the first test of removing the warning in any form.

## Published FastChat verdict parser

The pairwise evaluator was audited at the same pinned FastChat commit used for
the questions and prompt template:
`587d5cfa1609a43d192cedb8441cac3c17db105d`.

In `fastchat/llm_judge/common.py`, lines 282–290, FastChat classifies an output
by checking whether it contains `[[A]]`; if not, whether it contains `[[B]]`;
if not, whether it contains `[[C]]`; otherwise it returns `error`. The checks
are case-sensitive, but the token can occur anywhere in the output. There is
no requirement that it be terminal and no prohibition on surrounding Markdown.
The Qwen pilot output `**Final Verdict: [[A]]**` would therefore be accepted as
A by the published implementation.

The published parser is also ambiguous when more than one distinct verdict
token occurs. Because it checks A first, a response containing both `[[A]]`
and `[[B]]` is classified as A rather than rejected. The primary analysis
rejects this ambiguity, while a separate sensitivity analysis applies
FastChat's literal A-first rule.

FastChat runs the two answer orders separately and maps A/B back to the
underlying model identities in `common.py`, lines 325–336. Its result display
removes parser errors, then treats a tie or disagreement between the two orders
as a tie for benchmark aggregation (`show_result.py`, lines 47–75). Our study
reports order consistency directly instead, because position sensitivity is
the outcome rather than a nuisance to collapse.

The primary rule matches FastChat by accepting one
unique case-sensitive `[[A]]` or `[[B]]` token anywhere in the response,
including within Markdown, while rejecting missing tokens, `[[C]]`, and any
response containing more than one verdict token. Report a sensitivity analysis
using the literal FastChat A-first containment rule.

## Selection logic and completed material audit

The frozen subset contains the first three source IDs from four MT-Bench
categories: writing, reasoning, STEM, and humanities. Selecting by source ID
before inspecting judge outcomes avoids choosing items because they produce a
desired result.

The outcome-blind audit was completed without consulting the new Qwen judge
outcomes. It covered 36 item–comparison pairs and found:

- all 12 questions and all four candidate files have complete two-turn text;
- the question set is suitable for a general tutorial: it includes ordinary
  writing, reasoning, science, economics, mortality, and antitrust topics, but
  no private data, private study stimuli, sexual content, instructions for
  self-harm or violence, or requests for wrongdoing;
- no candidate answer contains `[[A]]`, `[[B]]`, or `[[C]]` that could be
  confused with a verdict if echoed;
- the longest rendered Qwen chat prompt is 2,949 tokens, 9.0% of the configured
  32,768-token context window (or 12.1% after reserving 1,024 output tokens);
- 14 pairs meet the prespecified clear-gap rule and 22 are close-gap pairs; and
- all four categories remain represented by three items each.

The gap rule was fixed before calculating labels: average each candidate's two
published GPT-4 single-judgment scores, then call an absolute difference of at
least 2.0 points on the 10-point scale clear and a smaller difference close.
The higher mean supplies the independent reference preference; equal means are
ties. These judgments came from the original MT-Bench archive, not the new
Qwen collection. The archive has exact pairwise judgments for the 12
Vicuna-versus-GPT-3.5 pairs but not all 36 pairs, so the single-score rule is
used uniformly.

Published answers were not rewritten to match length. Eleven of 36 pairs have
at least a 2:1 difference in total answer characters, including seven at or
above 3:1; the maximum is 4.77:1. Retain answer-length ratio as a reported
covariate or sensitivity check rather than changing the source materials.

Any excluded item and its reason must remain in a deviation log.
No item was excluded. Detailed non-text results and checksums are in
`audit-results/item-pair-audit.csv` and `audit-results/audit-manifest.json`.
