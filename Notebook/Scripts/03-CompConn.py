#!/usr/bin/env python3

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import argparse
from pathlib import Path
import warnings as w

import mne
import numpy as np
from scipy.signal import decimate
from elephant.signal_processing import butter

import config


# =============================================================================
# CLI
# =============================================================================
def parse_args()
    parser = argparse.ArgumentParser(description="Connectivity analysis pipeline")
    
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to process (e.g. C01 C02 C03). "
             "Defaults to all folders matching SUBJECT_GLOB in CORR_OUTPUT_DIR.",
    )
    parser.add_argument("--infoEO", default=None)
    
    return parser.parse_args()
 
# =============================================================================
# HELPERS
# =============================================================================
def load_matrices(subject_dir):
    
    files = sorted(subject_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {subject_dir}")
        
    matrices = [np.load(f) for f in files]
    connectivity = np.stack(matrices) # TIME, MATRIX
    
    return connectivity
    

def resolve_subjects(args, preferred):
    corr_dir = Path(config.CORR_OUTPUT_DIR)
    
    if not corr_dir.exists():
        print(f"ERROR: Correlation directory not found: {corr_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Recording: {recording}")

    corr_subjects = []
    file_paths = []

    for subject in sorted(corr_dir.iterdir()):
        if not subject.is_dir():
            continue
        if preferred != "all" and subject not in preferred:
            continue
        
        corr_subjects.append(subject.name)
        
        subject_file_map = dict(zip(subject.name, subject))

    if preferred != "all":
        missing = [s for s in preferred if s not in subject_file_map]
        if missing:
            print(f"ERROR: Subjects not found or missing files: {missing}", file=sys.stderr)
            sys.exit(1)

    return subjects, subject_file_map

def run_subject(subject, subject_dir, args):
    print(f"Processing {subject}")

    connectivity = load_matrices(subject_dir)

    mean_values, variability = compute_metrics(connectivity)

    save_plot(subject, mean_values, variability)
    save_summary(subject, mean_values, variability)

    print(
        f"Finished {subject} | "
        f"{len(connectivity)} windows"
    )


# =============================================================================
# MAIN
# =============================================================================
  
def main():
    args = parse_args()
    preferred = args.subject if args.subject is not None else "all"
    subjects, subject_file_map = resolve_subjects(args, preferred=preferred)

    for subject in subjects:
        run_subject(subject, subject_file_map[subject], args)
    
    
 if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main()