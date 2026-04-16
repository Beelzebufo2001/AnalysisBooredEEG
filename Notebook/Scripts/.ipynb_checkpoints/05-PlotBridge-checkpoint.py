#!/usr/bin/env python3

# =============================================================================
# IMPORTS
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import mne

import config.py

# =============================================================================
# PARAMS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute sliding-window EEG correlation matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--subject",
        nargs="+",
        default=None,
        help="One or more subject IDs, e.g. C01 C02. If omitted, all control subjects are processed."
    )
    parser.add_argument(
        "--lenght",
        type = int,
        default=config.DEFAULT_BRIDGE,
        help=""
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "bridges plot",
        help="Directory where output matrices will be saved."
    )
    parser.add_argument(
        "--target-freq",
        type=int,
        default=config.DEFAULT_SFREQ,
        help="Directory where output matrices will be saved."
    )

# =============================================================================
# HELPERS
# =============================================================================
def ensure_dir_exists(dirpath):
    dirpath = Path(dirpath)
    if not dirpath.exists():
        print(f"Creating directory: {dirpath}")
        dirpath.mkdir(parents=True, exist_ok=True)


def load_raw_file(file_path):
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_fif(fname=file_path, preload=True, verbose=False)

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

        chosen = matching_files[0] # THIS TODOO FOR ELSE THAN 0 >> connect together 
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

# =============================================================================
# PER SUBJECT
# =============================================================================
def run_subject(subject, subject_path, args):
    raw = load_raw_file(subject_pathct)

    #Cropping X second before
    tmax = raw.times[-1]
    seconds = args.lenght
    if(secondso > tmax):
        print("", flush=True)
        exit(1):
    tmin = tmax - seconds
    
    raw_croped = raw.copy().crop(tmin=tmin, tmax=tmax)
    
    
# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    subjects, subject_file_map = resolve_subjects(args)

    for subject in subjects:
        run_subject(subject, subject_file_map[subject], args)


if __name__ == "__main__":
    main()