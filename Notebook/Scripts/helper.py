#!/usr/bin/env python3

import re
from pathlib import Path
import numpy as np
import config


def ensure_dir_exists(dirpath):
    dirpath = Path(dirpath)
    if not dirpath.exists():
        print(f"Creating directory: {dirpath}")
        dirpath.mkdir(parents=True, exist_ok=True)


def get_valid_subjects(preferred="all", folder_path=None):
    """
    Return valid control subjects from correlation matrix folder.

    preferred:
        "all"        -> all folders starting with C
        ["C01", ...] -> selected subjects only
    """
    folder_path = Path(folder_path or config.CORR_OUTPUT_DIR)

    if isinstance(preferred, str) and preferred != "all":
        preferred = [preferred]

    subjects = []

    for subject_dir in sorted(folder_path.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        if not subject.startswith("C"):
            continue

        if preferred != "all" and subject not in preferred:
            continue

        npy_files = list(subject_dir.glob("corr_*.npy"))
        if not npy_files:
            print(f"Skipping {subject}: no corr_*.npy files found")
            continue

        subjects.append(subject)

    print(f"Found subjects: {subjects}")
    return subjects


def load_connectivity(subject, folder_path=None):
    """
    Load all correlation matrices for one subject.
    Returns:
        connectivity : np.ndarray of shape (time, channels, channels)
        files        : list of file paths
    """
    folder_path = Path(folder_path or config.CORR_OUTPUT_DIR)
    subject_path = folder_path / subject

    files = sorted(subject_path.glob("corr_*.npy"))
    if not files:
        raise ValueError(f"No correlation matrices found for subject {subject}")

    matrices = [np.load(f) for f in files]
    connectivity = np.stack(matrices)

    return connectivity, files


def extract_times_from_files(files):
    """
    Extract integer seconds from filenames like corr_0000.npy
    """
    times = []
    for f in files:
        match = re.search(r"corr_(\d+)\.npy", f.name)
        if match:
            times.append(int(match.group(1)))
        else:
            times.append(None)
    return times


def get_upper_triangle_values(matrix):
    upper_idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[upper_idx]


def classify_hemispheres(ch_names):
    left = []
    right = []
    midline = []

    for i, ch in enumerate(ch_names):
        digits = "".join(c for c in ch if c.isdigit())

        if ch.endswith("z") or digits == "":
            midline.append(i)
        else:
            num = int(digits)
            if num % 2 == 0:
                right.append(i)
            else:
                left.append(i)

    return left, right, midline


def classify_regions(ch_names):
    central   = [i for i, ch in enumerate(ch_names) if ch.startswith(("FC", "C", "CP"))]
    parietal  = [i for i, ch in enumerate(ch_names) if ch.startswith(("CP", "P"))]
    occipital = [i for i, ch in enumerate(ch_names) if ch.startswith(("PO", "O", "I"))]
    temporal  = [i for i, ch in enumerate(ch_names) if ch.startswith(("T", "FT", "TP", "TTP", "FFT"))]
    frontal   = [i for i, ch in enumerate(ch_names) if ch.startswith(("Fp", "AF", "F")) and i not in temporal]
    reference = [i for i, ch in enumerate(ch_names) if ch.startswith("M")]

    return {
        "Frontal": frontal,
        "Central": central,
        "Parietal": parietal,
        "Occipital": occipital,
        "Temporal": temporal,
        "Reference": reference,
    }


def region_connectivity_series(connectivity, region_a, region_b):
    means = []
    stds = []

    for matrix in connectivity:
        vals = matrix[np.ix_(region_a, region_b)].flatten()

        if region_a == region_b:
            sub = matrix[np.ix_(region_a, region_a)]
            vals = sub[np.triu_indices_from(sub, k=1)]

        means.append(np.mean(vals))
        stds.append(np.std(vals))

    return np.array(means), np.array(stds)