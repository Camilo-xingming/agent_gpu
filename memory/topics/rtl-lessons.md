# RTL Lessons Archive

Archived: 2026-03-06
Source: pre-cleanup STATUS and MEMORY context

## Verification Lessons
- Testbenches should fail fast on internal error paths to avoid silent pass.
- Functional end-to-end TB coverage is required in addition to decode/unit tests.
- TB and Make targets need deterministic non-zero exit codes on failure for CI reliability.

## FRM and Model Lessons
- FRM must model per-thread branch decisions; thread0-only control masks divergence bugs.
- Barrier and memory fence semantics need explicit CTA and global scope tracking.
- Warp collectives (SHFL/VOTE/REDUX) need snapshot-based lane behavior.

## Compare Harness Lessons
- Auto-detect test category and apply per-category tolerances.
- Keep tolerance policy configuration-driven; avoid a single hardcoded ULP value.
- Add category coverage checks to prevent false pass when parser misses cases.

## Process Lessons
- Keep task status in GitHub issues and milestones; local snapshots age quickly.
- Keep memory docs compact; move deep technical notes into topic files.
- Close loop in one issue thread: branch -> PR -> review -> merge.
