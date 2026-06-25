#!/bin/bash
#SBATCH --job-name eeg_qc_plots
#SBATCH --mem 16G
#SBATCH --time 01:00:00
#SBATCH -c 4
#SBATCH -o /CSNG/roblikova/Scripts/logs/qc_plots_%j.out
#SBATCH -e /CSNG/roblikova/Scripts/logs/qc_plots_%j.err

# =============================================================================
# 01_plot_eeg — QC plots for all subjects
#
# Loops over all subjects in one job (no array needed — this script is fast).
# Produces single-trace, multi-electrode, and band decomposition plots.
#
# Submit:
#   sbatch /CSNG/roblikova/Scripts/slurm/01_plot_eeg.sbatch
#
# Watch live:
#   tail -f /CSNG/roblikova/Scripts/logs/qc_plots_<JOBID>.out
# =============================================================================

# ── Edit these if needed ──────────────────────────────────────────────────────

CONDA_ENV="BCenv"
SCRIPT="/CSNG/roblikova/Scripts/01-PlotEEG.py"
OUTPUT_DIR="/CSNG/roblikova/results/qc_plots"

# ── Runtime ───────────────────────────────────────────────────────────────────

echo "========================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  CPUs    : $(nproc)"
echo "  Started : $(date)"
echo "========================================"

mkdir -p "$OUTPUT_DIR"
mkdir -p /CSNG/roblikova/Scripts/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python  : $(which python)"
echo "Env     : $CONDA_DEFAULT_ENV"
echo ""

# Runs on all subjects, RS_before recording, all plot modes, default electrode Cz.
# To change: add --subject C01 C02, --electrode Fz, --mode single, --t-min 10 --t-max 20, etc.
python "$SCRIPT" \
    --recording  RS_after \
    --mode       all \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo ""
echo "Finished at $(date) (exit code: $EXIT_CODE)"
exit $EXIT_CODE
