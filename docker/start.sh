#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# Day0 Krea2 Trainer — pod boot script
# Pulls the latest trainer + UI code from GitHub, then starts the web UI.
# Code updates = git push + restart pod. No image rebuild needed.
# ============================================================================

REPO_URL="${TRAINER_REPO_URL:-}"
REPO_BRANCH="${TRAINER_REPO_BRANCH:-main}"
APP_DIR="/workspace/day0-app"
UI_PORT="${UI_PORT:-8888}"

echo "[day0] Pod started."

# Optional SSH access (RunPod convention)
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p ~/.ssh && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 700 -R ~/.ssh
    ssh-keygen -A >/dev/null 2>&1
    service ssh start >/dev/null 2>&1 || true
    echo "[day0] SSH enabled."
fi

# Pull or update application code
if [[ -z "$REPO_URL" ]]; then
    echo "[day0] WARNING: TRAINER_REPO_URL is not set. Using code already in ${APP_DIR} if present."
elif [[ -d "$APP_DIR/.git" ]]; then
    echo "[day0] Updating code from ${REPO_URL} (${REPO_BRANCH})..."
    git -C "$APP_DIR" fetch origin "$REPO_BRANCH" && \
    git -C "$APP_DIR" reset --hard "origin/${REPO_BRANCH}" || \
    echo "[day0] WARNING: git update failed, running existing code."
else
    echo "[day0] Cloning ${REPO_URL} (${REPO_BRANCH})..."
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$APP_DIR" || {
        echo "[day0] ERROR: clone failed and no local code present."; sleep infinity; }
fi

mkdir -p /workspace/jobs /workspace/datasets

if [[ -z "${UI_PASSWORD:-}" ]]; then
    echo "[day0] WARNING: UI_PASSWORD is not set. The UI will be open to anyone with the pod URL."
fi

echo "[day0] Starting UI on port ${UI_PORT}..."
cd "$APP_DIR"
exec uvicorn app.main:app --host 0.0.0.0 --port "$UI_PORT"
