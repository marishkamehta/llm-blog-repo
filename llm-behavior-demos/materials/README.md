# Demonstration Materials

This directory records the provenance and selection of the public materials
used by the two demonstrations. It deliberately separates three things:

1. upstream source files;
2. the immutable selection manifest; and
3. derived trial schedules produced later by the pipeline.

Do not manually copy text from a paper into a trial file. Retrieve released
materials from the source listed in each manifest, verify its checksum, and
select records by their source identifiers.

## Current status

| Demonstration | Source located | License checked | Subset frozen | Trial schedule |
|---|---:|---:|---:|---:|
| MT-Bench position bias | Yes | Partial | Yes | Generated and analyzed |
| NeuBAROCO belief bias | Yes | Yes, CC BY 4.0 | Yes | Generated and analyzed |

The MT-Bench license is marked partial because FastChat's repository is
Apache-2.0, but its separately hosted pre-generated-answer space declares
`license: other` without spelling out redistribution terms. The answer files
must remain on their original host unless that ambiguity is resolved. Their
URLs and checksums are recorded so the experiment remains reproducible without
repackaging them here.

The demonstrations are not part of the citable pipeline release. Their role is
to show how the separately released pipeline template can be used. The public
repository therefore records the MT-Bench retrieval procedure and checksums
without redistributing the separately hosted answer files.

The completed runs preserve their schedules, requests, responses, analyses,
and figures locally. Only reviewed outputs that do not expose credentials,
local paths, or material without clear redistribution terms belong in the
public results package.
