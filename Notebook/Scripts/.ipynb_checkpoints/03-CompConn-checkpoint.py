#!/usr/bin/env python3

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import argparse
from pathlib import Path
import warnings as w

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Connectivity analysis pipeline")

    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to process (e.g. C01 C02 C03). "
             "Defaults to all folders matching SUBJECT_GLOB in CORR_OUTPUT_DIR.",
    )
    

    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================
def load_matrices(subject_dir):
    """
    Load all .npy correlation matrices for one subject.

    Returns
    -------
    connectivity : ndarray
        Shape: (time, channels, channels)
    """
    files = sorted(subject_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {subject_dir}")

    matrices = [np.load(f) for f in files]
    connectivity = np.stack(matrices)  # TIME, MATRIX

    return connectivity


def resolve_subjects(args):
    """
    Return list of subject IDs and mapping subject -> directory
    """
    corr_dir = Path(config.CORR_OUTPUT_DIR)
    print(f"Openning correlation matrices directory in: {corr_dir}")

    if not corr_dir.exists():
        print(f"ERROR: Correlation directory not found: {corr_dir}", file=sys.stderr)
        sys.exit(1)

    discovered = {
        subject_dir.name: subject_dir
        for subject_dir in sorted(corr_dir.iterdir())
        if subject_dir.is_dir()
    }

    if not discovered:
        print(f"ERROR: No subject folders found in {corr_dir}", file=sys.stderr)
        sys.exit(1)

    if args.subjects is None:
        subjects = list(discovered.keys())
    else:
        missing = [s for s in args.subjects if s not in discovered]
        if missing:
            print(
                f"ERROR: Requested subjects not found: {missing}",
                file=sys.stderr
            )
            sys.exit(1)

        subjects = args.subjects

    return subjects, discovered


def compute_metrics(connectivity):
    """
    Compute:
    - mean connectivity over time
    - variability over time
    """
    upper_indices = np.triu_indices(connectivity.shape[1], k=1)

    mean_values = []
    variability = []

    for matrix in connectivity:
        pair_vals = matrix[upper_indices]

        mean_values.append(np.mean(pair_vals))
        variability.append(np.std(pair_vals))

    return np.array(mean_values), np.array(variability)


def save_plot(subject, mean_values, variability):
    out_dir = Path(config.SUMMARY_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    mid = len(mean_values) // 2

    # -------------------------------------------------------------------------
    # Mean connectivity plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(mean_values, color="darkmagenta", linewidth=2)
    ax.axvline(mid, color="black", linestyle="--", label="Condition switch")

    ax.set_title(f"Mean connectivity over time: {subject}")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Mean connectivity")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{subject}_mean_connectivity.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Variability plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(
        range(mid),
        variability[:mid],
        color="khaki",
        label="Eyes open",
    )

    ax.plot(
        range(mid, len(variability)),
        variability[mid:],
        color="powderblue",
        label="Eyes closed",
    )

    ax.axvline(
        mid,
        color="red",
        linestyle="--",
        label="Condition switch"
    )

    ax.set_title(f"Variability across electrode pairs: {subject}")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Std of pair connectivity")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{subject}_variability.png", dpi=300)
    plt.close(fig)


def save_summary(subject, mean_values, variability):
    out_dir = Path(config.SUMMARY_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    mid = len(mean_values) // 2

    first_half = np.mean(mean_values[:mid])
    second_half = np.mean(mean_values[mid:])
    delta = second_half - first_half

    var_first = np.mean(variability[:mid])
    var_second = np.mean(variability[mid:])

    text = f"""
Subject: {subject}
============================================================

Mean connectivity:
    whole recording : {np.mean(mean_values):.6f}
    first half      : {first_half:.6f}
    second half     : {second_half:.6f}
    delta           : {delta:.6f}

Variability across electrode pairs:
    first half      : {var_first:.6f}
    second half     : {var_second:.6f}
"""

    with open(out_dir / f"{subject}_summary.txt", "w") as f:
        f.write(text)

    print(text)


def run_subject(subject, subject_dir):
    print(f"Processing {subject}")

    connectivity = load_matrices(subject_dir)

    mean_values, variability = compute_metrics(connectivity)

    save_plot(subject, mean_values, variability)
    save_summary(subject, mean_values, variability)

    print(
        f"Loaded connectivity for {subject}: "
        f"{connectivity.shape[0]} windows, "
        f"{connectivity.shape[1]} channels"
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    subjects, subject_file_map = resolve_subjects(args)

    for subject in subjects:
        run_subject(subject, subject_file_map[subject])


if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main()