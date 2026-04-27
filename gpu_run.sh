#!/usr/bin/env bash
# =============================================================================
# gpu_run.sh — Sync code to remote GPU and run any command there
# =============================================================================
# Usage:
#   ./gpu_run.sh bash customization/run_all_experiments.sh --epochs 30
#   ./gpu_run.sh --attach   # re-attach after wifi drop
#   ./gpu_run.sh --pull     # download JSON + PNG results (no .pth files)
#   ./gpu_run.sh --clean    # delete all customisation checkpoints (student safe)
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
    info "Pulling results from remote (JSON + PNG only — no model .pth files) ..."
    # Pull JSON and PNG results from checkpoints/customization/
    mkdir -p checkpoints/customization
    ssh "${REMOTE}" "find ${REMOTE_DIR}/checkpoints/customization -name '*.json' -o -name '*.png'" \
        | while read -r remote_file; do
            fname=$(basename "${remote_file}")
            scp -q "${REMOTE}:${remote_file}" "checkpoints/customization/${fname}" \
                && echo "  pulled: checkpoints/customization/${fname}"
        done
    # Also pull any .png / .log from the project root (depth preview images etc.)
    ssh "${REMOTE}" "find ${REMOTE_DIR} -maxdepth 1 -name '*.png' -o -name '*.log'" \
        | while read -r remote_file; do
            fname=$(basename "${remote_file}")
            scp -q "${REMOTE}:${remote_file}" "./${fname}" && echo "  pulled: ${fname}"
        done
    success "Results pulled."
    exit 0 ;;
  --clean)
    info "Deleting checkpoints/customization/ on remote ..."
    ssh "${REMOTE}" "rm -rf ${REMOTE_DIR}/checkpoints/customization/"
    success "Remote: checkpoints/customization/ deleted."

    info "Deleting checkpoints/customization/ locally ..."
    rm -rf checkpoints/customization/
    success "Local: checkpoints/customization/ deleted."

    # The next run recreates this directory automatically via mkdir -p
    success "Done. Student model untouched. Run the experiments to start fresh."
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
    --exclude="wandb/"       \
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
success "Job is running on the GPU inside tmux session '${SESSION}'."
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  YOUR JOB IS SAFE — YOU CAN CLOSE THIS TERMINAL OR SLEEP    ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  The job runs entirely inside tmux on the remote GPU.        ║${NC}"
echo -e "${GREEN}║  Your laptop's SSH connection does NOT affect it.            ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  To check progress : ./gpu_run.sh --attach                  ║${NC}"
echo -e "${GREEN}║  To pull results   : ./gpu_run.sh --pull                    ║${NC}"
echo -e "${GREEN}║  To check GPU      : ./gpu_run.sh --status                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Streaming live output below. Ctrl+C exits streaming safely —"
echo "  the GPU job keeps running and results are pulled before exit."
echo "────────────────────────────────────────────────────────────────────────"
echo ""

# ── Pull function — defined once, called from normal exit, Ctrl+C, and reconnect
_pull_results() {
    echo ""
    echo "────────────────────────────────────────────────────────────────────────"
    info "Pulling results to local machine (JSON + PNG only — no .pth model files) ..."
    mkdir -p checkpoints/customization
    ssh -o ConnectTimeout=15 "${REMOTE}" \
        "find ${REMOTE_DIR}/checkpoints/customization -name '*.json' -o -name '*.png' 2>/dev/null" \
        | while read -r remote_file; do
            fname=$(basename "${remote_file}")
            scp -q "${REMOTE}:${remote_file}" "checkpoints/customization/${fname}" \
                && echo "  pulled: checkpoints/customization/${fname}"
        done
    ssh -o ConnectTimeout=15 "${REMOTE}" \
        "find ${REMOTE_DIR} -maxdepth 1 -name '*.png' -o -name '*.log' 2>/dev/null" \
        | while read -r remote_file; do
            fname=$(basename "${remote_file}")
            scp -q "${REMOTE}:${remote_file}" "./${fname}" && echo "  pulled: ${fname}"
        done
    success "Results are on your local machine."
    echo ""
}

# ── Trap Ctrl+C: kill tail, pull whatever exists so far, exit cleanly ─────────
TAIL_PID=""
trap 'echo ""; warn "Streaming stopped. GPU job is still running."; \
      kill "${TAIL_PID}" 2>/dev/null || true; \
      _pull_results; \
      echo "  To reconnect: ./gpu_run.sh --attach"; \
      echo "  To pull when done: ./gpu_run.sh --pull"; \
      exit 0' INT

# ── Step 4: Stream the log file live ──────────────────────────────────────────
echo -n "  Waiting for job to start"
for i in $(seq 1 30); do
    ssh -o ConnectTimeout=5 "${REMOTE}" "[ -f ${LOG_FILE} ]" 2>/dev/null && break
    echo -n "."
    sleep 1
done
echo ""

ssh "${REMOTE}" "tail -f ${LOG_FILE}" &
TAIL_PID=$!

# Poll for job completion — 30s interval (kind to battery for overnight runs).
# Each SSH call has a 15s connect timeout so a sleeping laptop wakes and retries
# gracefully rather than hanging forever.
while true; do
    sleep 30
    DONE=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "${REMOTE}" \
        "grep -cE 'JOB FINISHED|JOB FAILED' ${LOG_FILE} 2>/dev/null || echo 0" \
        2>/dev/null | tr -d '[:space:]') || DONE="0"
    [[ "${DONE}" =~ ^[0-9]+$ ]] && [ "${DONE}" -gt 0 ] && break
done

sleep 1
kill "${TAIL_PID}" 2>/dev/null || true
trap - INT   # restore default Ctrl+C now that job is confirmed done

# ── Step 5: Pull results back to local ────────────────────────────────────────
_pull_results

# Print final status
FINAL=$(ssh -o ConnectTimeout=15 "${REMOTE}" "tail -3 ${LOG_FILE} 2>/dev/null" 2>/dev/null || true)
if echo "${FINAL}" | grep -q "SUCCESSFULLY"; then
    success "Job completed successfully."
elif echo "${FINAL}" | grep -q "FAILED"; then
    warn "Job finished with errors — check output above."
else
    warn "Status unclear — run './gpu_run.sh --attach' to check."
fi
