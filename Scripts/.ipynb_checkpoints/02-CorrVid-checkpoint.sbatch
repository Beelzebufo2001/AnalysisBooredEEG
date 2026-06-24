#!/bin/bash
#SBATCH --job-name corr_video
# #SBATCH --array=0-26
#SBATCH --mem 64G
#SBATCH --time 01:00:00
#SBATCH -c 8
#SBATCH -o /CSNG/roblikova/Scripts/logs/vidCorr_%A_%a.out
#SBATCH -e /CSNG/roblikova/Scripts/logs/vidCorr_%A_%a.err
# ── Edit these if needed ──────────────────────────────────────────────────────

CONDA_ENV="BCenv"
SCRIPT="/CSNG/roblikova/Scripts/02-CorrVid.py"
OUTPUT_DIR="/CSNG/roblikova/results/corr_matrices"

SUBJECTS=(
    C01 C02 C03 C04 C05
    C06 C07 C08 C09 C10
    C11 C12 C13 C14 C15
    C17 C18 C19 C20 C21
    C22 C23 C25 C26 C27
)
SUBJECT=(C01)
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