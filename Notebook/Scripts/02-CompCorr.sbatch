#!/bin/bash
#SBATCH --job-name eeg_corr
#SBATCH --array=0-26
#SBATCH --mem 64G
#SBATCH --time 02:00:00
#SBATCH -c 8
#SBATCH -o /CSNG/roblikova/Scripts/logs/corr_%A_%a.out
#SBATCH -e /CSNG/roblikova/Scripts/logs/corr_%A_%a.err

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

# ── Subject list ──────────────────────────────────────────────────────────────
# One subject per line — easier to read and comment out individuals.
# C16 and C24 removed (known-bad for ICA).
# --array=0-26 must match index 0 to (number of subjects - 1).

SUBJECTS=(
    C02 C03 C04 C05
    C06 C07 C08 C09 C10
    C11 C12 C13 C14 C15
    C17 C18 C19 C20 C21
    C22 C23 C25 C26 C27
)
# ↑ 27 subjects → indices 0-26 → --array=0-26
# Add more P subjects here and update --array if needed

# ── Pick this task's subject ──────────────────────────────────────────────────

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$SUBJECT" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range."
    echo "       SUBJECTS has ${#SUBJECTS[@]} entries (valid: 0 to $(( ${#SUBJECTS[@]} - 1 )))"
    echo "       Fix: update #SBATCH --array=0-$(( ${#SUBJECTS[@]} - 1 ))"
    exit 1
fi

# ── Print job info ────────────────────────────────────────────────────────────

echo "========================================"
echo "  Array job ID : $SLURM_ARRAY_JOB_ID"
echo "  Task index   : $SLURM_ARRAY_TASK_ID"
echo "  Subject      : $SUBJECT"
echo "  Node         : $(hostname)"
echo "  CPUs         : $(nproc)"
echo "  Started      : $(date)"
echo "========================================"

# ── Setup ─────────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"
mkdir -p /CSNG/roblikova/Scripts/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python : $(which python)"
echo "Env    : $CONDA_DEFAULT_ENV"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────

python "$SCRIPT" \
    --subject    "$SUBJECT" \
    --recording  RS_before \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo ""
echo "Finished: $SUBJECT at $(date) (exit code: $EXIT_CODE)"
exit $EXIT_CODE
