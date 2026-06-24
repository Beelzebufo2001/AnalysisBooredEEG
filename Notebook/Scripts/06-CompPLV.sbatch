#!/bin/bash
#SBATCH --job-name=PLVcorr
#SBATCH --array=0-26
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -c 4
#SBATCH --open-mode=append
#SBATCH -o /CSNG/roblikova/Scripts/logs/plvcorr_%A_%a.out
#SBATCH -e /CSNG/roblikova/Scripts/logs/plvcorr_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail

# Create log directory if it doesn't exist
mkdir -p logs

# Activate your conda / venv environment (edit path as needed)

# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON_SCRIPT="06-CompPLV.py"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"
echo "Script      : ${PYTHON_SCRIPT}"
echo "Arguments   : $*"
echo "=========================================="

python -u "${PYTHON_SCRIPT}" 

echo "=========================================="
echo "End time    : $(date)"
echo "=========================================="
