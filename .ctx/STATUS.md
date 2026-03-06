# RalphGPU Status (Live Snapshot)

Last Updated: 2026-03-06 (America/Los_Angeles)

## Source of Truth
Detailed execution status now lives in GitHub Issues and Milestones for `ssql2014/RalphGPU`.
This file is kept as a pointer to avoid stale local context.

## Current Sprint
- Milestone: Sprint 49
- Open:
  - #490 Texture/Surface unit e2e verification
  - #491 Update STATUS.md and archive stale context files
- Closed in Sprint 49:
  - #489 FP4/FP8 Tensor Core e2e verification
  - #474 SM v2 integration test
  - #473 TB regression for scheduler tests

## Agent Workflow
- Progress and blockers are posted as GitHub issue comments with signed prefixes (`**[Codex]**`, `**[Gemini]**`, `**[Lily]**`).
- Discord `#ralphgpu-dev` carries one-line notifications only.
- Branch naming: `issue-<num>/<agent>`.

## Why This Changed
The previous STATUS.md carried stale Sprint 16 data and duplicated task tracking.
GitHub issues are now the canonical and continuously updated status surface.
