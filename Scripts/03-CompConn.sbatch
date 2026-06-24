i#!/bin/bash
#SBATCH --job-name=compute_conn
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

#SBATCH --output=logs/compute_conn_%j.out
#SBATCH --error=logs/compute_conn_%j.err

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  CPUs    : $(nproc)"
echo "  Started : $(date)"
echo "========================================"

# ── subject list ──────────────────────────────────────────────────────────────
# Set SUBJECTS env var at submit time, or let the script auto-discover them.
SUBJECTS="${SUBJECTS:-}"   # empty → auto-discover via SUBJECT_GLOB in config.py

# ── run ───────────────────────────────────────────────────────────────────────
if [[ -n "$SUBJECTS" ]]; then
    # shellcheck disable=SC2086  # intentional word-splitting on $SUBJECTS
    python analyze_connectivity.py --subjects $SUBJECTS
else
    python 03-CompConn.py
fi

echo "=========================================="
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

