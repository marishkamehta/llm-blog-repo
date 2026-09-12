# Reviewed Results

This directory contains the outputs selected for public release. It is separate
from the working run directories, which may contain local paths, complete
requests, or third-party text that cannot be redistributed clearly.

- [`position-bias/`](position-bias/) contains trial-level parsed verdicts,
  pair-level measures, aggregate summaries, and the FastChat parser sensitivity
  analysis. It does not contain the MT-Bench questions, candidate answers, or
  complete prompts.
- [`belief-bias/`](belief-bias/) contains aggregate results and figures for the
  final 366-item, two-premise analysis across eleven temperatures.

Each results folder includes a manifest with SHA-256 checksums for its generated
outputs. The source audits and retrieval instructions are in
[`../materials/`](../materials/).

These outputs document the demonstrations described in the accompanying blog.
They are not presented as replication studies or as estimates that generalize
beyond the model, materials, and generation conditions reported here.
