# Validation

Validated on 2026-09-11 with Python 3.12.3 on Linux in a temporary staging
checkout. Both packages installed successfully using the root README command.
The validation environment inherited system packages; GitHub Actions is
configured for Python 3.10 and 3.12 but has not yet run remotely.

- Mock smoke test: returned `MOCK_RESPONSE`.
- Growing-trajectory example: returned two mock responses.
- Pipeline: 2 tests passed.
- Position-bias demonstration: 21 tests passed.
- Belief-bias demonstration: 10 tests passed.
- All 11 outputs listed in the two reviewed-result manifests matched SHA-256.
- Markdown relative file links resolved (external URLs and anchors not checked).
- Original source files were verified unchanged during assembly.
- All assembled files matched their recorded provenance hashes.

No real model calls, material downloads, or new experimental collections were
performed. Validation environments and generated package metadata are excluded
from the deliverable. This validates the assembled offline code, not the
scientific results or the current availability of hosted backends.

## Published-page cleanup

Removed the copied blog pages and blog assets from this repository. Checked
that all remaining Markdown relative file links resolve and all 74
retained provenance entries match their assembled hashes. Package code and
reviewed results were unchanged; the existing test suites were not rerun for
this documentation-only cleanup.
