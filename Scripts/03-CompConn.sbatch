i#!/bin/bash
#SBATCH --job-name=compute_conn
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

#SBATCH --output=logs/compute_conn_%j.out
#SBATCH --error=logs/compute_conn_%j.err

set -euo pipefail

mkdir -p logs

echo "========================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  CPUs    : $(nproc)"
echo "  Started : $(date)"
echo "========================================"

# ── subject list ──────────────────────────────────────────────────────────────
# Set SUBJECTS env var at submit time, or let the script auto-discover them.

# ── run ───────────────────────────────────────────────────────────────────────
# See what param folders exist before picking one
python 03_analyze_connectivity.py --list-params

# Run on all subjects, RS_before, specific params
python -u 03_analyze_connectivity.py \
    --matrix-type corr_matrices \
    --recording RS_before \
    --params sf250_win10_step1_bp1-45

# Compare plv vs corr — run twice, outputs go to separate folders
#python 03_analyze_connectivity.py --matrix-type plv_matrices --params sf250_win10_step1_bp1-45

# Just a few subjects
#python 03_analyze_connectivity.py --subjects C01 C03 --params sf250_win10_step1_bp1-45

echo "=========================================="
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

