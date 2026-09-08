"""Phase 4 evaluation tooling (SPEC §6).

Judge two-prompt A/B calibration is the first slice (SPEC §6.2).  The
labeled dataset lives under ``src/docpilot/eval/dataset/`` — the root
``data/`` directory is git-ignored, but SPEC §6.4 requires the benchmark
dataset to be *committed*, so it ships inside the package.
"""