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
import json
from scipy.signal import decimate
from elephant.signal_processing import butter

import config
import resolveSubject

# =============================================================================
# =============================================================================

def get_valid_subjects_and_paths(preferred, folder_path=None, clean_alg=None, recording="RS_before"):
    """
    Return matching subjects and their FIF paths.
    - Only folders starting with 'C'.
    - preferred can be 'all' or a list like ['C01', 'C03'].
    - Ignores split files ending with '-1.fif'.
    """
    folder_path = Path(folder_path or config.DATA_DIR)
    clean_alg = clean_alg or config.DATA_TYPE

    print(f"Discovering subjects in: {folder_path}")
    print(f"Cleaning algorithm: {clean_alg}")
    print(f"Recording: {recording}")

    subjects = []
    file_paths = []

    for subject_dir in sorted(folder_path.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        if not subject.startswith("C"):
            continue

        if preferred != "all" and subject not in preferred:
            continue

        clean_dir = subject_dir / clean_alg
        if not clean_dir.is_dir():
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

    subjects, paths = get_valid_subjects_and_paths(
        preferred=preferred,
        folder_path=config.DATA_DIR,
        clean_alg=config.DATA_TYPE,
        recording=args.recording,
    )

    subject_file_map = dict(zip(subjects, paths))

    if preferred != "all":
        missing = [s for s in preferred if s not in subject_file_map]
        if missing:
            print(f"ERROR: Subjects not found or missing files: {missing}", file=sys.stderr)
            sys.exit(1)

    return subjects, subject_file_map
