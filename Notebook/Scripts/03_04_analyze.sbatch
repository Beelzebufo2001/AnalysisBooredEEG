#!/bin/bash
#SBATCH --job-name eeg_analysis
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH -c 8
#SBATCH -o logs/slurm/analysis_%j.out
#SBATCH -e logs/slurm/analysis_%j.err
#
# #SBATCH --nodelist=w16
# #SBATCH --hint=nomultithread
# #SBATCH --partition=cpu

# =============================================================================
# 03_analyze_connectivity + 04_plot_pairs — single job, loops over all subjects
#
# These steps are fast (reading .npy files, no heavy signal processing),
# so a job array is overkill — one job processes everything sequentially.
#
# Run AFTER 02_compute_corr.sbatch has finished.
#
# Submit:  sbatch slurm/03_04_analyze.sbatch
# =============================================================================

CONDA_ENV="eeg"
PIPELINE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

CORR_DIR="/results/corr_matrices"
CONN_DIR="/results/connectivity"
PAIRS_DIR="/results/pair_plots"
CH_NAMES_FILE="$PIPELINE_DIR/ch_names.txt"   # generate with setup instructions in README

echo "========================================"
echo "  Job         : $SLURM_JOB_ID"
echo "  Host        : $(hostname)"
echo "  CPUs        : $(nproc)"
echo "  Date        : $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Step 3: connectivity analysis ─────────────────────────────────────────────
echo ""
echo "--- Step 3: analyze_connectivity ---"
python "$PIPELINE_DIR/03_analyze_connectivity.py" \
    --corr-dir      "$CORR_DIR" \
    --output-dir    "$CONN_DIR" \
    --ch-names-file "$CH_NAMES_FILE" \
    --conda-env     "$CONDA_ENV" \
    --log-dir       "$CONN_DIR/logs"

# ── Step 4: pair plots ────────────────────────────────────────────────────────
echo ""
echo "--- Step 4: plot_pairs ---"
python "$PIPELINE_DIR/04_plot_pairs.py" \
    --corr-dir      "$CORR_DIR" \
    --output-dir    "$PAIRS_DIR" \
    --ch-names-file "$CH_NAMES_FILE" \
    --top-pairs     50 \
    --conda-env     "$CONDA_ENV" \
    --log-dir       "$PAIRS_DIR/logs"

echo ""
echo "All done at $(date)"
