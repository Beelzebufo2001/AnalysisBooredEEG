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
    parser = argparse.ArgumentParser(description="Connectivity analysis pipeline")
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to process (e.g. C01 C02 C03).",
    )
    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================
def load_matrices(subject_dir):
    """Load all .npy correlation matrices, return (T, N, N) array."""
    files = sorted(subject_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {subject_dir}")
    return np.stack([np.load(f) for f in files])


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


def compute_metrics(connectivity):
    upper = np.triu_indices(connectivity.shape[1], k=1)
    mean_values = np.array([np.mean(m[upper]) for m in connectivity])
    variability = np.array([np.std(m[upper])  for m in connectivity])
    return mean_values, variability


# =============================================================================
# PLOT — one PNG per subject, two panels + stats in suptitle
# =============================================================================
def save_subject_figure(subject, mean_values, variability, out_dir):
    mid = len(mean_values) // 2

    first  = np.mean(mean_values[:mid])
    second = np.mean(mean_values[mid:])
    delta  = second - first

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # ── panel 1: mean connectivity ────────────────────────────────────────────
    ax1.plot(mean_values, color="darkmagenta", linewidth=1.5)
    ax1.axvline(mid, color="black", linestyle="--", linewidth=1, label="Condition switch")
    ax1.set_ylabel("Mean connectivity")
    ax1.set_xlabel("Time window")
    ax1.legend(fontsize=8)

    # ── panel 2: variability ──────────────────────────────────────────────────
    ax2.plot(range(mid),                   variability[:mid],  color="khaki",      label="Eyes open")
    ax2.plot(range(mid, len(variability)), variability[mid:],  color="steelblue",  label="Eyes closed")
    ax2.axvline(mid, color="black", linestyle="--", linewidth=1, label="Condition switch")
    ax2.set_ylabel("Std of pair connectivity")
    ax2.set_xlabel("Time window")
    ax2.legend(fontsize=8)

    # ── stats as suptitle ─────────────────────────────────────────────────────
    fig.suptitle(
        f"{subject}   |   "
        f"mean={np.mean(mean_values):.4f}   "
        f"eyes-open={first:.4f}   "
        f"eyes-closed={second:.4f}   "
        f"Δ={delta:+.4f}",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{subject}.png", dpi=200)
    plt.close(fig)


# =============================================================================
# SUMMARY — one PNG with delta bar chart across all subjects
# =============================================================================
def save_summary_figure(subjects, deltas, out_dir):
    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.8), 4))
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]
    ax.bar(subjects, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Connectivity change: eyes-closed − eyes-open (Δ)")
    ax.set_ylabel("Δ mean connectivity")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_delta.png", dpi=200)
    plt.close(fig)


# =============================================================================
# PER-SUBJECT
# =============================================================================
def run_subject(subject, subject_dir, out_dir):
    connectivity = load_matrices(subject_dir)
    mean_values, variability = compute_metrics(connectivity)
    save_subject_figure(subject, mean_values, variability, out_dir)

    mid   = len(mean_values) // 2
    delta = float(np.mean(mean_values[mid:])) - float(np.mean(mean_values[:mid]))

    print(f"  {subject}  windows={len(connectivity)}  delta={delta:+.4f}", flush=True)
    return delta


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    subjects, subject_dir_map = resolve_subjects(args)

    out_dir = Path(config.SUMMARY_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    deltas = []
    for subject in subjects:
        try:
            delta = run_subject(subject, subject_dir_map[subject], out_dir)
            deltas.append((subject, delta))
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", flush=True)

    if len(deltas) > 1:
        save_summary_figure([s for s, _ in deltas], [d for _, d in deltas], out_dir)

    print(f"\nDone. Results in: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main()