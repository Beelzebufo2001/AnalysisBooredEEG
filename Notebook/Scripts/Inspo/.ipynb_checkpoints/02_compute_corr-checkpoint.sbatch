#!/bin/bash
#SBATCH --job-name eeg_corr
#SBATCH --array=0-99              # one slot per subject — adjust upper bound to (N_subjects - 1)
#SBATCH --mem 64G
#SBATCH --time 08:00:00
#SBATCH -c 8                      # CPUs per task (cluster default is only 2 — always set this)
#SBATCH -o logs/slurm/corr_%A_%a.out   # %A = job array ID, %a = subject index
#SBATCH -e logs/slurm/corr_%A_%a.err
#
# Optional — uncomment and edit what you need:
# #SBATCH --nodelist=w16          # pin to a specific node
# #SBATCH --hint=nomultithread    # disable hyperthreading
# #SBATCH --partition=cpu         # if your cluster has named partitions
# #SBATCH --account=myproject     # if your cluster requires an account

# =============================================================================
# 02_compute_corr — SLURM array job
#
# Each array task processes ONE subject, identified by its index in SUBJECTS[].
# SLURM_ARRAY_TASK_ID (0, 1, 2, ...) picks the subject automatically.
#
# Submit:
#   sbatch slurm/02_compute_corr.sbatch
#
# Check:
#   squeue -u $USER
#
# Cancel all tasks of this job:
#   scancel <JOBID>
#
# Cancel one task (e.g. task 3):
#   scancel <JOBID>_3
#
# Watch live output for task 0:
#   tail -f logs/slurm/corr_<JOBID>_0.out
# =============================================================================

# ── Edit these ────────────────────────────────────────────────────────────────

CONDA_ENV="eeg"
PIPELINE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"   # one level up from slurm/

DATA_DIR="/data/eeg"
OUTPUT_DIR="/results/corr_matrices"
CLEAN_ALG="ica"
FILES_IDX="0"           # 0=RS_before  1=Task  2=RS_after

# List every subject you want to process.
# The array index in #SBATCH --array must be 0 to (length - 1).
SUBJECTS=(
    C01 C02 C03 C04 C05
    C06 C07 C08 C09 C10
    # P01 P02 P03 P04    ← uncomment or add patient subjects
)

# Subjects to exclude (checked at runtime as a safety net — also just remove
# them from SUBJECTS[] above if you want to be tidy)
EXCLUDE="P03 P04 C16 C24"

# ── Runtime — do not edit below this line ────────────────────────────────────

# Pick this task's subject from the array
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$SUBJECT" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range."
    echo "       SUBJECTS array has ${#SUBJECTS[@]} entries (indices 0–$(( ${#SUBJECTS[@]} - 1 )))."
    echo "       Update #SBATCH --array= to match."
    exit 1
fi

echo "========================================"
echo "  Job array ID : $SLURM_ARRAY_JOB_ID"
echo "  Task index   : $SLURM_ARRAY_TASK_ID"
echo "  Subject      : $SUBJECT"
echo "  Host         : $(hostname)"
echo "  CPUs         : $(nproc)"
echo "  Date         : $(date)"
echo "========================================"

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Run
python "$PIPELINE_DIR/02_compute_corr.py" \
    --subject      "$SUBJECT" \
    --data-dir     "$DATA_DIR" \
    --output-dir   "$OUTPUT_DIR" \
    --clean-alg    "$CLEAN_ALG" \
    --files-idx    $FILES_IDX \
    --exclude      $EXCLUDE \
    --conda-env    "$CONDA_ENV" \
    --log-dir      "$OUTPUT_DIR/logs"

echo "Done: $SUBJECT at $(date)"
