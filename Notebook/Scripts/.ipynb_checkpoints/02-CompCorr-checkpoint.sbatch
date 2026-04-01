#!/bin/bash
#SBATCH --job-name=corr_mat
#SBATCH --output=logs/corr_mat_%j.out
#SBATCH --error=logs/corr_mat_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# fail on errors
set -e

# create logs dir if needed
mkdir -p logs

# activate your environment
source ~/.bashrc
conda activate YOUR_ENV_NAME

# move to your project folder
cd /path/to/your/project

# run script
python 02_compute_corr.py \
    --subject C01 \
    --recording RS_before \
    --snippet-len 10 \
    --step 1 \
    --highpass 1 \
    --lowpass 45 \
    --target-sfreq 250