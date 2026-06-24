#!/bin/bash
#SBATCH --job-name=conn_summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

#SBATCH --output=logs/conn_summary_%j.out
#SBATCH --error=logs/conn_summary_%j.err

module purge
module load python/3.11

source ~/venvs/eeg/bin/activate

mkdir -p logs

python 03-CompConn.py