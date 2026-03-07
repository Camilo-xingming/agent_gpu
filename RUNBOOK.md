# Health Monitor Runbook

## Overview
This runbook covers the recovery steps for the Agent Health Monitor (`scripts/agent-health-monitor.sh`), which checks heartbeat problems, issue progress, and retro response windows.

## Health Signals
The health monitor checks the following signals:
1. **Stale Heartbeat**: Is the long-running task heartbeat missing?
2. **No Issue Progress**: Are assigned issues stalled without updates?
3. **Missed Retro Response**: Did the agent fail to respond in the retro window?

Evidence for these checks is stored in `~/.openclaw/shared-memory/ralphgpu/health_evidence.json`. Config knobs (like timeout thresholds) are managed via ENV vars (e.g., `HEARTBEAT_TIMEOUT_MINUTES`).

## Alert Path
When a check fails, the monitor automatically posts a one-line alert to the Discord dev channel. Evidence details are updated in the shared-memory JSON.

## Recovery Procedures

### 1. Agent Restart
If the agent is stuck in a loop or completely unresponsive:
- Log in to the build server.
- Terminate the stuck agent process.
- Trigger the watchdog or manual cron to restart the agent pipeline.

### 2. Manual Fallback
If the agent cannot recover or there's a blocking issue:
- Reassign the issue to another available Coder.
- Provide a summary of current progress to unblock them.

### 3. Reassign Task
If the issue SLA is breached but the agent is alive, the PO (Lily) can reassign the GitHub issue.
