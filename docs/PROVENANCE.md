# Assembly provenance

Assembled on 2026-09-11 from the current working files in the workspace.
The JSON inventory records source-relative paths and SHA-256 digests, including
uncommitted source content, rather than claiming that a source commit captures
the entire snapshot.

| Source folder | Included material |
| --- | --- |
| llm-behavior-pipeline | Package sources, example registry, examples, smoke test, tests, Docker example, reproducibility templates, license, citation |
| llm-behavior-demos | Code, tests, source audits, reviewed results, license and documentation |
| llm-behavior-blog | Only the Python example extracted from Quick Start II; published pages and blog assets are not bundled |
| llm-behaviors | Inspected as background; research planning notes excluded |

The original folders remain read-only inputs. Git histories, environments,
caches, raw working runs, credentials, unpublished research plans, and
editorial drafts were not imported. Result figures in the demonstrations are
retained with their reviewed output packages and checksum manifests.

Documentation and packaging adaptations link the demos to the bundled
pipeline, describe adjacent packages, remove the old demos hosting URL, and
remove an unverified release date and placeholder repository URL from pipeline
citation metadata. Copied package code and reviewed result artifacts retain
their source bytes.

`examples/decoy_experiment.py` is extracted verbatim from the first Python
block in Quick Start II, with a final newline. Its provenance source path
identifies the original workspace article; it is not a file bundled in this
repository. No new experiment logic or research data was generated.

`source-inventory.json` records source and assembled hashes for the retained
files and marks adaptations. Root documentation, the reading map, installation
file, and GitHub Actions configuration are assembly files without source-file
entries.

The outer pipeline project folder was flattened into the repository root.
Its README is now `docs/PIPELINE.md`, and its ignore rules were merged into
the root `.gitignore`. Package imports remain `silico` and
`llm_behavior_pipeline` under `src/`. Installation commands and links use the
new layout; distribution names and package code are unchanged.
