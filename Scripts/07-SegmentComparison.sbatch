#!/bin/bash
#SBATCH --job-name=segment_fig
#SBATCH --array=0-26
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

#SBATCH --output=/CSNG/roblikova/Scripts/logs/segment_%A_%a.out
#SBATCH --error=/CSNG/roblikova/Scripts/logs/segment_%A_%a.err

# ===========================
# Environment
# ===========================
set -euo pipefail

source /home/roblikova/miniconda3/etc/profile.d/conda.sh
conda activate BCenv


# ===========================
# Subjects
# ===========================

SUBJECTS=(
    C01 C02 C03 C04 C05
    C06 C07 C08 C09 C10
    C11 C12 C13 C14 C15
    C17 C18 C19 C20 C21
    C22 C23 C25 C26 C27
)


SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}


echo "Running subject: $SUBJECT"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"


# ===========================
# Run python
# ===========================

python /CSNG/roblikova/Scripts/07-SegmentComparison.py \
    --subject "$SUBJECT" \
    --recording RS_before \
    --matrix-type corr_matrices \
    --params sf250_win10_step1_bp1-45


EXIT_CODE=$?


echo "Finished $SUBJECT"
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

exit $EXIT_CODE