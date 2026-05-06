#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_RUNNER_URL:?Set GITHUB_RUNNER_URL to your repo or org URL, e.g. https://github.com/owner/repo}"
: "${GITHUB_RUNNER_TOKEN:?Set GITHUB_RUNNER_TOKEN to a fresh runner registration token}"

RUNNER_NAME="${GITHUB_RUNNER_NAME:-mlops-sonar-runner}"
RUNNER_LABELS="${GITHUB_RUNNER_LABELS:-mlops-sonar,linux,x64}"
RUNNER_WORKDIR="${GITHUB_RUNNER_WORKDIR:-_work}"
RUNNER_DIR="${GITHUB_RUNNER_DIR:-$HOME/actions-runner}"
RUNNER_VERSION="${GITHUB_RUNNER_VERSION:-2.323.0}"

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [[ ! -f ./config.sh ]]; then
  curl -fsSL -o actions-runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  tar xzf actions-runner.tar.gz
fi

if [[ ! -x ./config.sh ]]; then
  chmod +x ./config.sh ./run.sh ./bin/installdependencies.sh || true
fi

if [[ ! -f .runner ]]; then
  ./config.sh remove --unattended --token "$GITHUB_RUNNER_TOKEN" >/dev/null 2>&1 || true
  ./config.sh \
    --url "$GITHUB_RUNNER_URL" \
    --token "$GITHUB_RUNNER_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --work "$RUNNER_WORKDIR" \
    --unattended \
    --replace
fi

echo "Runner configured in $RUNNER_DIR"
if [[ "${GITHUB_RUNNER_AUTOSTART:-true}" == "true" ]]; then
  exec "$RUNNER_DIR/run.sh"
fi

echo "To start it: $RUNNER_DIR/run.sh"
