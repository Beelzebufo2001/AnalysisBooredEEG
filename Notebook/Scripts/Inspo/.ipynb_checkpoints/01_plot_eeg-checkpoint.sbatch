#!/bin/bash
#SBATCH --job-name eeg_qc
#SBATCH --array=0-99              # adjust to (N_subjects - 1)
#SBATCH --mem 16G
#SBATCH --time 00:30:00
#SBATCH -c 4
#SBATCH -o logs/slurm/qc_%A_%a.out
#SBATCH -e logs/slurm/qc_%A_%a.err
#
# #SBATCH --nodelist=w16
# #SBATCH --hint=nomultithread
# #SBATCH --partition=cpu

# =============================================================================
# 01_plot_eeg — SLURM array job
# Submit:  sbatch slurm/01_plot_eeg.sbatch
# =============================================================================

CONDA_ENV="eeg"
PIPELINE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

DATA_DIR="/data/eeg"
OUTPUT_DIR="/results/qc_plots"
CLEAN_ALG="ica"

SUBJECTS=(
    C01 C02 C03 C04 C05
    C06 C07 C08 C09 C10
)

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$SUBJECT" ]]; then
    echo "ERROR: task index $SLURM_ARRAY_TASK_ID out of range (${#SUBJECTS[@]} subjects)."
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID → Subject $SUBJECT on $(hostname) at $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python "$PIPELINE_DIR/01_plot_eeg.py" \
    --subject    "$SUBJECT" \
    --data-dir   "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --conda-env  "$CONDA_ENV" \
    --log-dir    "$OUTPUT_DIR/logs"

echo "Done: $SUBJECT at $(date)"
