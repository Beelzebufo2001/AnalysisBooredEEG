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
    parser.add_argument("--change", default=None)
    
    return parser.parse_args()
 
# =============================================================================
# HELPERS
# =============================================================================
def ensure_dir_exists(dirpath):
    dirpath = Path(dirpath)
    if not dirpath.exists():
        print(f"Creating directory: {dirpath}")
        dirpath.mkdir(parents=True, exist_ok=True)


def load_matrices(file_path):
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_fif(fname=file_path, preload=False, verbose=False)

    raw.set_montage(
        mne.channels.make_standard_montage(config.DEFAULT_MONTAGE),
        on_missing="ignore"
    )
    return raw


def get_valid_subjects_and_paths(preferred, folder_path=None, clean_alg=None, recording="RS_before"):
    """
    Return matching subjects and their FIF paths.

    Rules:
    - only folders starting with 'C'
    - preferred can be 'all' or list like ['C01', 'C03']
    - choose files matching selected recording
    - ignore split files ending with '-1.fif'
    """

        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        if preferred != "all" and subject not in preferred:
            print(f"Skipping {subject}: missing folder {clean_dir}")
            continue

        matching_files = sorted(
            [
                f for f in clean_dir.iterdir()
                if (
                    f.is_file()
                    and f.suffix == ".fif"
                    and recording in f.name
                    and clean_alg in f.name
                    and not f.stem.endswith("-1")
                )
            ]
        )

        if not matching_files:
            print(f"Skipping {subject}: no matching .fif file for {recording}")
            continue

        chosen = matching_files[0]
        subjects.append(subject)
        file_paths.append(chosen)

        print(f"Adding {subject}: {chosen.name}")

    return subjects, file_paths


def resolve_subjects(args):
    preferred = args.subject if args.subject is not None else "all"

    folder_path = Path(folder_path or config.CORR_OUTPUT_DIR)

    print(f"Discovering subjects in: {folder_path}")
    print(f"Recording: {recording}")

    subjects = []
    file_paths = []

    for subject_dir in sorted(folder_path.iterdir()):
    subject_file_map = dict(zip(subjects, paths))

    if preferred != "all":
        missing = [s for s in preferred if s not in subject_file_map]
        if missing:
            print(f"ERROR: Subjects not found or missing files: {missing}", file=sys.stderr)
            sys.exit(1)

    return subjects, subject_file_map

def main() -> None:
    
 if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main()