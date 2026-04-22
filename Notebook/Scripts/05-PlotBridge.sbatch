#!/bin/bash
#SBATCH --job-name=bridges_analysis
#SBATCH --output=logs/bridges_%A_%a.out
#SBATCH --error=logs/bridges_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH --array=0-26

echo "[SBATCH] Job started: $(date)"
echo "[SBATCH] Job ID: $SLURM_JOB_ID"
echo "[SBATCH] Task ID: $SLURM_ARRAY_TASK_ID"

mkdir -p logs

# ── run ─────────────────────────────────────────────────────────
python 05-PlotBridge.py

echo "[SBATCH] Job finished: $(date)"