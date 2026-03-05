# Self-Hosted CI Runner (ist-mac-s)

This document describes how to operate the GitHub Actions self-hosted runner used by `ssql2014/RalphGPU`.

## Runner Profile

- Host: `ist-mac-s`
- Runner name: `ist-mac-s`
- Labels: `self-hosted`, `macOS`, `ARM64`
- Runner root: `~/actions-runner`
- Service label (launchd): `actions.runner.ssql2014-RalphGPU.ist-mac-s`

## Verify Runner Is Online

From a machine with repo admin access:

```bash
gh api repos/ssql2014/RalphGPU/actions/runners \
  --jq '.runners[] | {name, status, busy, labels:[.labels[].name]}'
```

Expected: runner `ist-mac-s` reports `status: online` and label set includes `self-hosted/macOS/ARM64`.

## Install / Reconfigure Runner on ist-mac-s

```bash
ssh ist-mac-s
mkdir -p ~/actions-runner
cd ~/actions-runner

# Download (ARM64 macOS)
curl -L -o actions-runner.tar.gz \
  https://github.com/actions/runner/releases/download/v2.332.0/actions-runner-osx-arm64-2.332.0.tar.gz
tar xzf actions-runner.tar.gz

# Obtain a short-lived registration token from GitHub UI:
# Settings -> Actions -> Runners -> New self-hosted runner

./config.sh \
  --url https://github.com/ssql2014/RalphGPU \
  --token <REGISTRATION_TOKEN> \
  --name ist-mac-s \
  --labels self-hosted,macOS,ARM64 \
  --unattended
```

## Run as launchd Service

```bash
ssh ist-mac-s
cd ~/actions-runner
./svc.sh install
./svc.sh start
```

Useful service commands:

```bash
./svc.sh status
./svc.sh stop
./svc.sh start
./svc.sh uninstall
```

## Workflow Runner Selection / Fallback

Workflows use:

```yaml
runs-on: ${{ fromJSON(vars.CI_RUNS_ON_JSON != '' && vars.CI_RUNS_ON_JSON || '["self-hosted","macOS","ARM64"]') }}
```

Default: target the local self-hosted runner labels.

Optional fallback override: set repository variable `CI_RUNS_ON_JSON`.

Examples:

- Self-hosted default labels:
  - `["self-hosted","macOS","ARM64"]`
- Temporary GitHub-hosted fallback:
  - `["macos-14"]`

## CI Verification Checklist

1. Open a PR.
2. Confirm `CI` workflow starts automatically.
3. Confirm jobs run on expected runner labels.
4. Confirm required gates pass:
   - `make lint`
   - `make test`

## Notes

- Registration tokens are short-lived; never commit tokens/secrets.
- Keep runner software updated periodically (`config.sh remove` + reinstall latest tarball if needed).
