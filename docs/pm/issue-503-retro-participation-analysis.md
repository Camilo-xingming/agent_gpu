# Issue 503: Coder Retro Participation Timing Mismatch Analysis

## 1. Problem Statement
It has been observed that `CoderGemini` (and potentially other automated coder agents) frequently fails to participate in the end-of-sprint Retro ceremonies. Specifically, the agents do not respond to the Retro issue prompts within the allocated time, leading to recurring "Retro 参与率" action items.

## 2. Root Cause Analysis
An analysis of the OpenClaw Scrum automation skill (`~/.openclaw/workspace/skills/scrum/SKILL.md`) and the heartbeat configuration reveals a fundamental timing mismatch between the ceremony execution and the coder agent heartbeats:

1. **Ceremony Phase Wait Time**: The `ceremony cron` executes the Phase 2 (RETRO) process by polling the issue comments every 30 seconds for a maximum of **5 minutes** (`每 30s，最多 5min`). If no response is received within this 5-minute window, the Phase concludes and moves to Planning.
2. **Coder Agent Heartbeat**: The coder agents (e.g., `CoderGemini`, `CoderCodex`) are driven by a heartbeat cron that executes only once every **30 minutes**.
3. **Issue Assignment**: The ceremony issues are not necessarily assigned to the coder agents, which means even when the heartbeat triggers, the coder might not natively discover the ceremony issue via the standard `gh issue list --assignee {coder.github}` query.

Because the 5-minute ceremony window is completely enveloped by the 30-minute sleep cycle of the coders, it is statistically highly unlikely (approx. 16% chance) that a heartbeat will trigger during the active Retro window.

## 3. Recommended Solutions
To resolve this structural defect in the Scrum automation, we recommend updating the OpenClaw `scrum` skill configuration or its protocol implementation with one of the following approaches:

### Option A: Extend Ceremony Timeout (Recommended)
Increase the polling timeout in the ceremony cron from `5min` to at least `35min`. This guarantees that every coder agent will experience at least one heartbeat cycle while the ceremony phase is actively waiting for input.

### Option B: Asynchronous Retro Mechanism
Decouple the Retro participation from the synchronous ceremony issue poll. Allow coder agents to continuously push their findings to a designated `RETRO.md` or a queue during their standard workflow, which the ceremony cron can later ingest instantly without waiting.

### Option C: Explicit Assignment and Mentioning
Ensure the `ceremony cron` explicitly assigns the ceremony issue to all active coders and uses `@mentions` to wake them. If an interrupt-driven wake mechanism exists, this would bypass the heartbeat delay.
