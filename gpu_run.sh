#!/usr/bin/env bash
# =============================================================================
# gpu_run.sh — Sync code to remote GPU and run any command there
# =============================================================================
# Usage:
#   ./gpu_run.sh python finetuning/frozenBase_customHead.py --epochs 30 ...
#   ./gpu_run.sh --attach   # re-attach after wifi drop
#   ./gpu_run.sh --pull     # just download results
#   ./gpu_run.sh --status   # GPU memory + running jobs
#   ./gpu_run.sh --kill     # stop the running job
# =============================================================================

set -euo pipefail

# ── Remote config ─────────────────────────────────────────────────────────────
REMOTE_USER="newuser"
REMOTE_HOST="100.37.41.165"
REMOTE_DIR="/home/newuser/Efficient-Distillation"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SESSION="gpu_job"
LOG_FILE="${REMOTE_DIR}/gpu_run.log"
JOB_SCRIPT="${REMOTE_DIR}/.gpu_job.sh"   # temporary script written to remote

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}▶ $*${NC}"; }
success() { echo -e "${GREEN}✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $*${NC}"; }

# ── Special flags ─────────────────────────────────────────────────────────────
case "${1:-}" in
  --attach)
    info "Re-attaching to tmux session '${SESSION}' on remote ..."
    ssh -t "${REMOTE}" "tmux attach-session -t ${SESSION} || echo 'No active session found.'"
    exit 0 ;;
  --pull)
    info "Pulling results from remote ..."
    rsync -az --progress "${REMOTE}:${REMOTE_DIR}/checkpoints/" checkpoints/ 2>/dev/null || true
    ssh "${REMOTE}" "find ${REMOTE_DIR} -maxdepth 1 -name '*.png' -o -name '*.jpg' -o -name '*.log'" \
        | while read -r remote_file; do
            fname=$(basename "${remote_file}")
            scp -q "${REMOTE}:${remote_file}" "./${fname}" && echo "  pulled: ${fname}"
        done
    success "Results pulled."
    exit 0 ;;
  --status)
    ssh "${REMOTE}" "nvidia-smi && echo '' && (tmux list-sessions 2>/dev/null || echo 'No tmux sessions.')"
    exit 0 ;;
  --kill)
    warn "Killing tmux session '${SESSION}' ..."
    ssh "${REMOTE}" "tmux kill-session -t ${SESSION} 2>/dev/null && echo 'Killed.' || echo 'Nothing to kill.'"
    exit 0 ;;
esac

[ $# -eq 0 ] && { echo "Usage: ./gpu_run.sh <command>"; exit 1; }

REMOTE_CMD="$*"

# ── Step 1: Sync code (fast — only changed .py / config files) ────────────────
info "Syncing code to remote GPU ..."
rsync -az --delete \
    --exclude=".venv/"       \
    --exclude=".git/"        \
    --exclude="__pycache__/" \
    --exclude="*.pyc"        \
    --exclude="data/"        \
    --exclude="checkpoints/" \
    --exclude="*.mat"        \
    --exclude="*.pth"        \
    --exclude="*.png"        \
    --exclude="*.jpg"        \
    --exclude=".DS_Store"    \
    ./ "${REMOTE}:${REMOTE_DIR}/"
success "Code synced."

# ── Step 2: Write the job as a shell script on the remote ─────────────────────
# This completely avoids nested quote escaping — we pipe the script over SSH
# and then simply tell tmux to run the file.
info "Preparing job script on remote ..."

ssh "${REMOTE}" "cat > ${JOB_SCRIPT}" <<SCRIPT
#!/usr/bin/env bash
cd ${REMOTE_DIR}
source .venv/bin/activate

# Force Python to flush stdout immediately — without this, prints are
# buffered inside the pipe and may not appear before a crash.
export PYTHONUNBUFFERED=1

echo "=== JOB STARTED ==="
echo "Command : ${REMOTE_CMD}"
echo "Started : \$(date)"
echo ""

${REMOTE_CMD} 2>&1 | tee ${LOG_FILE}
EXIT_CODE=\${PIPESTATUS[0]}

echo ""
if [ \$EXIT_CODE -eq 0 ]; then
    echo "=== JOB FINISHED SUCCESSFULLY ==="
else
    echo "=== JOB FAILED (exit code \$EXIT_CODE) ==="
fi
echo "Ended : \$(date)"
SCRIPT

ssh "${REMOTE}" "chmod +x ${JOB_SCRIPT}"

# ── Step 3: Run the job script inside a persistent tmux session ───────────────
info "Launching on remote GPU inside tmux session '${SESSION}' ..."

# Kill any leftover session from a previous run
ssh "${REMOTE}" "tmux kill-session -t ${SESSION} 2>/dev/null || true"

# Start a fresh tmux session and run the job script inside it
ssh "${REMOTE}" "tmux new-session -d -s ${SESSION} 'bash ${JOB_SCRIPT}'"

echo ""
success "Job is running on the GPU."
echo "  Live output below  (Ctrl+C stops streaming — job keeps running on GPU)"
echo "  Reconnect anytime: ./gpu_run.sh --attach"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

# ── Step 4: Stream the log file live ──────────────────────────────────────────
sleep 2   # give tmux a moment to create the log

ssh "${REMOTE}" "tail -f ${LOG_FILE}" &
TAIL_PID=$!

# Wait until the job-end marker appears in the log
while true; do
    sleep 5
    DONE=$(ssh "${REMOTE}" "grep -cE 'JOB FINISHED|JOB FAILED' ${LOG_FILE} 2>/dev/null || echo 0" | tr -d '[:space:]')
    [[ "${DONE}" =~ ^[0-9]+$ ]] && [ "${DONE}" -gt 0 ] && break
done

sleep 2
kill "${TAIL_PID}" 2>/dev/null || true

echo ""
echo "────────────────────────────────────────────────────────────────────────"

# ── Step 5: Pull results back to local ────────────────────────────────────────
info "Pulling results back to local machine ..."

# Pull checkpoints
rsync -az "${REMOTE}:${REMOTE_DIR}/checkpoints/" checkpoints/ 2>/dev/null || true

# Pull all .png / .jpg / .log files from the remote project root
ssh "${REMOTE}" "find ${REMOTE_DIR} -maxdepth 1 -name '*.png' -o -name '*.jpg' -o -name '*.log'" \
    | while read -r remote_file; do
        fname=$(basename "${remote_file}")
        scp -q "${REMOTE}:${remote_file}" "./${fname}" && echo "  pulled: ${fname}"
    done

success "Results are on your local machine."
echo ""

# Print final status
FINAL=$(ssh "${REMOTE}" "tail -3 ${LOG_FILE} 2>/dev/null")
if echo "${FINAL}" | grep -q "SUCCESSFULLY"; then
    success "Job completed successfully."
elif echo "${FINAL}" | grep -q "FAILED"; then
    warn "Job finished with errors — check output above."
else
    warn "Status unclear — run './gpu_run.sh --attach' to check."
fi
