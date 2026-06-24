#!/bin/bash
#SBATCH --job-name=region_conn
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/region_conn_%j.out
#SBATCH --error=logs/region_conn_%j.err

set -euo pipefail
mkdir -p logs"

echo "========================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "========================================"

python 04-RegionConn.py --subjects C01 C02 C03

echo "========================================"
echo "  Finished: $(date)"
echo "========================================"
