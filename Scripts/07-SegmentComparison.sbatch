#!/usr/bin/env bash
# =============================================================================
# submit_segment_fig.sh
# SLURM array job — one task per subject for 04_segment_comparison.py
#
# Usage:
#   sbatch submit_segment_fig.sh                        # corr, RS_before (default)
#   sbatch submit_segment_fig.sh --matrix-type plv_matrices
#   sbatch submit_segment_fig.sh --recording RS_after
#   sbatch submit_segment_fig.sh --dry-run
#
# Any extra args are forwarded verbatim to the Python script.
# =============================================================================

#SBATCH --job-name=segment_fig
#SBATCH --array=0-26          # 27 subjects (index 0-26)
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/CSNG/roblikova/Scripts/logs/segment_%A_%a.out
#SBATCH --error=/CSNG/roblikova/Scripts/logs/segment_%A_%a.err

# =============================================================================
# Environment
# =============================================================================
set -euo pipefail

mkdir -p /CSNG/roblikova/Scripts/logs

source /home/roblikova/miniconda3/etc/profile.d/conda.sh
conda activate BCenv

# =============================================================================
# Subject list  — 27 subjects, indices 0-26
# C16 and C24 are intentionally included (they were missing from the old script)
# Excluded subjects (C12, C14, C08-BF, C03-AF) are handled inside the Python
# script via eyesClosed.json — no need to remove them here.
# =============================================================================
SUBJECTS=(
    C01 C02 C03 C04 C05 C06 C07 C08 C09 C10
    C11 C12 C13 C14 C15 C16 C17 C18 C19 C20
    C21 C22 C23 C24 C25 C26 C27
)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# =============================================================================
# Info header
# =============================================================================
echo "============================================"
echo "Array job : ${SLURM_ARRAY_JOB_ID}"
echo "Task index: ${SLURM_ARRAY_TASK_ID}"
echo "Subject   : ${SUBJECT}"
echo "Host      : $(hostname)"
echo "Start     : $(date)"
echo "Extra args: $*"
echo "============================================"

# =============================================================================
# Run  (-u = unbuffered stdout so logs flush in real time)
# =============================================================================
python -u 7_SegmentComparison.py \
    --subject "$SUBJECT"                      \
    --recording    RS_before                  \
    --matrix-type  corr_matrices              \
    --params       sf250_win10_step1_bp1-45   \
    "$@"

EXIT_CODE=$?

echo "============================================"
echo "Subject   : ${SUBJECT}"
echo "Exit code : ${EXIT_CODE}"
echo "End       : $(date)"
echo "============================================"

exit $EXIT_CODE