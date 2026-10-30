# Changelog

## 0.1.0 — Initial release preparation

- Standalone `methridgeqc qc` and `methridgeqc eval` commands and Python APIs.
- Count validation, reference-aware CpG filtering, optional Ridge risk scoring,
  H/MH/ML/L reliability grades, and U for ungraded records.
- Extended bedGraph, gzip bedGraph, and internally LZ4-compressed Parquet output.
- Portable JSON models, deterministic spatial cross-validation and sampling,
  and optional explicitly selected GPU execution.
- Frozen-output evaluation against independent truth with block bootstrap.
- MIT licensing, PyPI wheel/sdist metadata, package CI,
  and `methridgeqc --version`.
- Consistent imports, four-space formatting, typed array/format boundaries,
  function documentation and named stderr diagnostics with `--log-level`.
- Regression checks for logging isolation, JSON/stdout compatibility and
  early resource-parameter validation; scientific formulas remain unchanged.

This entry describes the source release; it does not assert that the package
has already been published to PyPI or Bioconda.
