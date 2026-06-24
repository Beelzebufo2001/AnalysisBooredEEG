#!/bin/bash
#SBATCH --job-name=
#SBATCH --array=0-26
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -c 4
#SBATCH --open-mode=append
#SBATCH -o /CSNG/roblikova/Scripts/logs/plvcorr_%A_%a.out
#SBATCH -e /CSNG/roblikova/Scripts/logs/plvcorr_%A_%a.err

# =============================================================================
# 02-CorrMatrices — sliding window correlation matrices, one job per subject
#
# --array=0-26 creates 27 parallel jobs (C01-C27, excluding known-bad C16 C24)
# Each job gets a unique $SLURM_ARRAY_TASK_ID (0, 1, 2 ... 26) which picks
# one subject from the SUBJECTS list below.
#
# Submit:
#   sbatch /CSNG/roblikova/Scripts/02-CorrMatrices.sbatch
#
# Check progress:
#   squeue -u roblikova
#
# Watch one subject live (replace 12345 and 0 with your job ID and task index):
#   tail -f /CSNG/roblikova/Scripts/logs/corr_12345_0.out
#
# Cancel everything:
#   scancel --user=roblikova
# =============================================================================

# ── Edit these if needed ──────────────────────────────────────────────────────

CONDA_ENV="BCenv"
SCRIPT="/CSNG/roblikova/Scripts/02-CompCorr.py"
OUTPUT_DIR="/CSNG/roblikova/results/corr_matrices"

# ── Print job info ────────────────────────────────────────────────────────────

echo "========================================"
echo "  Array job ID : $SLURM_ARRAY_JOB_ID"
echo "  Task index   : $SLURM_ARRAY_TASK_ID"
echo "  Node         : $(hostname)"
echo "  CPUs         : $(nproc)"
echo "  Started      : $(date)"
echo "========================================"

# ── Setup ─────────────────────────────────────────────────────────────────────

mkdir -p /CSNG/roblikova/Scripts/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python : $(which python)"
echo "Env    : $CONDA_DEFAULT_ENV"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────

python "$SCRIPT" \
    --recording RS_before \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo ""
echo "Finished: $SUBJECT at $(date) (exit code: $EXIT_CODE)"
exit $EXIT_CODE
