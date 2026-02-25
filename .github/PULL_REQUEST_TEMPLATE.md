## Summary
<!-- What does this PR do? Link to the issue it addresses. -->

Closes #

## Pre-merge Checklist

- [ ] **Lint passes**: `make lint` runs clean (0 errors on both Verilator 4.x and 5.x)
- [ ] **Tests pass**: `make test` runs clean
- [ ] **MANIFEST.md updated**: `bash tools/check-manifest.sh` passes (every new file is listed)
- [ ] **FRM compare** (if RTL changed): `python3 tools/rtl_frm_compare.py --all` passes

## Post-merge Checklist (after merging to master)

- [ ] Verify CI green on the merge commit: `gh run list --repo ssql2014/RalphGPU --branch master --limit 1`
- [ ] Run local regression: `make regression` — post result in the issue comment
- [ ] Run local lint: `make lint` — confirm no new warnings
- [ ] If CI fails on master: **immediately** open a fix PR or revert
