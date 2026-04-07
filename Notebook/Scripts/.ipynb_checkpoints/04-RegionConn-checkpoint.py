#!/usr/bin/env python3

# =============================================================================
# IMPORTS
# =============================================================================
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
    parser = argparse.ArgumentParser(description="Region connectivity analysis")

    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to process. Defaults to all folders in CORR_OUTPUT_DIR.",
    )

    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================
def load_matrices(subject_dir):
    files = sorted(subject_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {subject_dir}")
    return np.stack([np.load(f) for f in files])  # (T, N, N)


def resolve_subjects(args):
    corr_dir = Path(config.CORR_OUTPUT_DIR)

    if not corr_dir.exists():
        print(f"ERROR: {corr_dir} not found", file=sys.stderr)
        sys.exit(1)

    discovered = {
        d.name: d for d in sorted(corr_dir.iterdir()) if d.is_dir()
    }

    if not discovered:
        print(f"ERROR: No subject folders in {corr_dir}", file=sys.stderr)
        sys.exit(1)

    if args.subjects is None:
        return list(discovered.keys()), discovered

    missing = [s for s in args.subjects if s not in discovered]
    if missing:
        print(f"ERROR: Subjects not found: {missing}", file=sys.stderr)
        sys.exit(1)

    return args.subjects, discovered


def build_hemisphere_indices(ch_names):
    left, right, midline = [], [], []
    for i, ch in enumerate(ch_names):
        digits = ''.join(c for c in ch if c.isdigit())
        if ch.endswith('z') or digits == "":
            midline.append(i)
        else:
            num = int(digits)
            if num % 2 == 0:
                right.append(i)
            else:
                left.append(i)
    return left, right, midline


def build_region_indices(ch_names):
    temporal = [i for i, ch in enumerate(ch_names) if ch.startswith(("T", "FT", "TP", "TTP", "FFT"))]
    return {
        "Frontal":   [i for i, ch in enumerate(ch_names) if ch.startswith(("Fp", "AF", "F")) and i not in temporal],
        "Central":   [i for i, ch in enumerate(ch_names) if ch.startswith(("FC", "C", "CP"))],
        "Parietal":  [i for i, ch in enumerate(ch_names) if ch.startswith(("CP", "P"))],
        "Occipital": [i for i, ch in enumerate(ch_names) if ch.startswith(("PO", "O", "I"))],
        "Temporal":  temporal,
        "Reference": [i for i, ch in enumerate(ch_names) if ch.startswith("M")],
    }


def region_connectivity(connectivity, regionA, regionB):
    values = []
    stds   = []
    for matrix in connectivity:
        vals = matrix[np.ix_(regionA, regionB)]
        values.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(values), np.array(stds)


# =============================================================================
# PLOTTING
# =============================================================================
def save_mean_std_plot(mean, std, label, subject, out_dir):
    mid = len(mean) // 2
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(mean, color="darkseagreen", label="Mean connectivity")
    ax[0].axvline(mid, color="palevioletred", linestyle="--", label="Condition switch")
    ax[0].set_ylabel("Mean connectivity")
    ax[0].set_title(f"{subject} — {label}")
    ax[0].legend(loc="upper left")

    ax[1].plot(std, color="thistle", label="Std connectivity")
    ax[1].axvline(mid, color="palevioletred", linestyle="--", label="Condition switch")
    ax[1].set_xlabel("Time (10s window, 1s step)")
    ax[1].set_ylabel("Std connectivity")
    ax[1].legend(loc="upper left")

    fig.tight_layout()

    fname = label.replace(" ", "_").replace("↔", "to").replace("<->", "to")
    fig.savefig(out_dir / f"{subject}_{fname}.png", dpi=300)
    plt.close(fig)


# =============================================================================
# PER-SUBJECT
# =============================================================================
def run_subject(subject, subject_dir):
    print(f"Processing {subject}")

    connectivity = load_matrices(subject_dir)
    ch_names     = list(config.ALL_ELECTRODES)

    out_dir = Path(config.REGION_OUTPUT_DIR) / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── hemispheres ────────────────────────────────────────────────────────────
    left, right, _ = build_hemisphere_indices(ch_names)

    for label, (idxA, idxB) in {
        "left <-> left":   (left,  left),
        "left <-> right":  (left,  right),
        "right <-> right": (right, right),
    }.items():
        mean, std = region_connectivity(connectivity, idxA, idxB)
        save_mean_std_plot(mean, std, label, subject, out_dir)

    # ── brain regions ──────────────────────────────────────────────────────────
    regions = build_region_indices(ch_names)

    for nameA, regA in regions.items():
        for nameB, regB in regions.items():
            label = f"{nameA} <-> {nameB}"
            mean, std = region_connectivity(connectivity, regA, regB)
            save_mean_std_plot(mean, std, label, subject, out_dir)

    print(f"Finished {subject} | saved to {out_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    subjects, subject_dir_map = resolve_subjects(args)

    for subject in subjects:
        try:
            run_subject(subject, subject_dir_map[subject])
        except FileNotFoundError as e:
            print(f"[SKIP] {e}", file=sys.stderr)


if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main()