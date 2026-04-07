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
from matplotlib.backends.backend_pdf import PdfPages
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
    """
    TODOO: add posibility to only compute some of the connectivity of section like adding subset of electrodes or typing selected lobes
    """

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
    """
    Just the same: Path, exists, disvover all dir, None, missing some, return all and discovered 
    """
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


def build_hemisphere_indices(ch_names): # Same function as from paper
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


def build_region_indices(ch_names): # Same function as from paper
    temporal = [i for i, ch in enumerate(ch_names) if ch.startswith(("T", "FT", "TP", "TTP", "FFT"))]
    return {
        "Frontal":   [i for i, ch in enumerate(ch_names) if ch.startswith(("Fp", "AF", "F")) and i not in temporal], # this meh could use config
        "Central":   [i for i, ch in enumerate(ch_names) if ch.startswith(("FC", "C", "CP"))],
        "Parietal":  [i for i, ch in enumerate(ch_names) if ch.startswith(("CP", "P"))],
        "Occipital": [i for i, ch in enumerate(ch_names) if ch.startswith(("PO", "O", "I"))],
        "Temporal":  temporal,
        "Reference": [i for i, ch in enumerate(ch_names) if ch.startswith("M")],
    }


def region_connectivity(connectivity, regionA, regionB):
    """
    goes over matrices and computes means and standard deviation over time
    """
    values = []
    stds   = []
    for matrix in connectivity:
        vals = matrix[np.ix_(regionA, regionB)]
        values.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(values), np.array(stds)


# =============================================================================
# PLOTTING 1
# =============================================================================
def save_plot(mean, std, label, color, subject, mid, out_dir):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
 
    ax[0].plot(mean, color=color, linewidth=1.5, label="Mean")
    ax[0].axvline(mid, color="palevioletred", linestyle="--", label="Condition switch")
    ax[0].set_ylabel("Mean connectivity")
    ax[0].set_title(f"{subject} — {label}")
    ax[0].legend(loc="upper left", fontsize=8)
 
    ax[1].plot(std, color=color, linewidth=1.0, alpha=0.7, label="Std")
    ax[1].axvline(mid, color="palevioletred", linestyle="--")
    ax[1].set_xlabel("Time (10s window, 1s step)")
    ax[1].set_ylabel("Std connectivity")
    ax[1].legend(loc="upper left", fontsize=8)
 
    fig.tight_layout()
 
    fname = label.replace(" ", "_").replace("<->", "to")
    fig.savefig(out_dir / f"{subject}_{fname}.png", dpi=300)
    plt.close(fig)

# =============================================================================
# PLOTTING 2
# =============================================================================
def make_line_plot(mean, std, nameA, nameB, subject, mid):
    """Return a figure for one region pair."""
    colorA = REGION_COLORS[nameA]
    colorB = REGION_COLORS[nameB]
    # blend the two region colors for the line
    blend = tuple(
        (a + b) / 2
        for a, b in zip(
            matplotlib.colors.to_rgb(colorA),
            matplotlib.colors.to_rgb(colorB),
        )
    )
 
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    label = f"{nameA} <-> {nameB}"
 
    ax[0].plot(mean, color=blend, linewidth=1.5, label="Mean")
    ax[0].axvline(mid, color="palevioletred", linestyle="--", label="Condition switch")
    ax[0].set_ylabel("Mean connectivity")
    ax[0].set_title(f"{subject} — {label}")
    ax[0].legend(loc="upper left", fontsize=8)
 
    ax[1].plot(std, color=blend, linewidth=1.0, alpha=0.7, label="Std")
    ax[1].axvline(mid, color="palevioletred", linestyle="--")
    ax[1].set_xlabel("Time (10s window, 1s step)")
    ax[1].set_ylabel("Std connectivity")
    ax[1].legend(loc="upper left", fontsize=8)
 
    fig.tight_layout()
    return fig
 
 
def make_heatmap(avg_matrix, region_names, subject):
    """
    One figure: 6x6 heatmap of mean connectivity between all region pairs.
    Dark = weak, bright = strong.
    """
    n = len(region_names)
    fig, ax = plt.subplots(figsize=(7, 6))
 
    im = ax.imshow(avg_matrix, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean connectivity")
 
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(region_names, rotation=45, ha="right")
    ax.set_yticklabels(region_names)
 
    # annotate cells with values
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{avg_matrix[i, j]:.3f}",
                ha="center", va="center",
                fontsize=7,
                color="white" if avg_matrix[i, j] < avg_matrix.max() * 0.6 else "black",
            )
 
    ax.set_title(f"{subject} — mean connectivity heatmap")
    fig.tight_layout()
    return fig
 
# =============================================================================
# PER-SUBJECT 1
# =============================================================================
def run_subject(subject, subject_dir):
    """
    For one subject computation
    """
    print(f"Processing {subject}")

    connectivity = load_matrices(subject_dir)
    ch_names     = list(config.ALL_ELECTRODES)
    mid          = len(connectivity) // 2

    out_dir = Path(config.REGION_OUTPUT_DIR) / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── hemispheres ────────────────────────────────────────────────────────────
    left, right, _ = build_hemisphere_indices(ch_names) # Separates hemispheres 

    hemisphere_pairs = [
        ("left",    "left",    left,    left),
        ("left",    "right",   left,    right),
        ("right",   "right",   right,   right),
    ]
 
    for nameA, nameB, idxA, idxB in hemisphere_pairs:
        mean, std = region_connectivity(connectivity, idxA, idxB)
        color = config.HEMISPHERE_COLORS[nameA]
        label = f"{nameA} <-> {nameB}"
        save_plot(mean, std, label, color, subject, mid, out_dir)

    # ── brain regions ──────────────────────────────────────────────────────────
    regions = build_region_indices(ch_names)
    region_names = list(regions.keys())

    for i, nameA in enumerate(region_names):
        for j, nameB in enumerate(region_names):
            if j < i:
                continue
 
            mean, std = region_connectivity(connectivity, regions[nameA], regions[nameB])
            color = config.REGION_COLORS[nameA]
            label = f"{nameA} <-> {nameB}"
            save_plot(mean, std, label, color, subject, mid, out_dir)
 
    print(f"  Finished {subject} | saved to {out_dir}")

# =============================================================================
# PER-SUBJECT 2
# =============================================================================
"""
def run_subject(subject, subject_dir):
    print(f"Processing {subject}")
 
    connectivity = load_matrices(subject_dir)
    ch_names     = list(config.ALL_ELECTRODES)
    mid          = len(connectivity) // 2
 
    regions      = build_region_indices(ch_names)
    region_names = list(regions.keys())
    n            = len(region_names)
 
    out_dir = Path(config.REGION_OUTPUT_DIR) / subject
    out_dir.mkdir(parents=True, exist_ok=True)
 
    avg_matrix = np.zeros((n, n))
    pdf_path   = out_dir / f"{subject}_region_connectivity.pdf"
 
    with PdfPages(pdf_path) as pdf:
 
        # ── heatmap first (will be page 1) — fill avg_matrix first ────────────
        for i, nameA in enumerate(region_names):
            for j, nameB in enumerate(region_names):
                mean, _ = region_connectivity(connectivity, regions[nameA], regions[nameB])
                avg_matrix[i, j] = np.mean(mean)
 
        heatmap_fig = make_heatmap(avg_matrix, region_names, subject)
        pdf.savefig(heatmap_fig)
        heatmap_fig.savefig(out_dir / f"{subject}_heatmap.png", dpi=300)
        plt.close(heatmap_fig)
 
        # ── upper triangle line plots only (no duplicates) ────────────────────
        for i, nameA in enumerate(region_names):
            for j, nameB in enumerate(region_names):
                if j < i:          # skip lower triangle and diagonal
                    continue
 
                mean, std = region_connectivity(connectivity, regions[nameA], regions[nameB])
                fig = make_line_plot(mean, std, nameA, nameB, subject, mid)
                pdf.savefig(fig)
                plt.close(fig)
 
    print(f"  Finished {subject} | {pdf_path}")
 """
# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args() # Look at arguments 
    
    subjects, subject_dir_map = resolve_subjects(args) # Get subjects from argument and all directories

    for subject in subjects: # Just run it
        try:
            run_subject(subject, subject_dir_map[subject])
        except FileNotFoundError as e:
            print(f"[SKIP] {e}", file=sys.stderr)


if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main()