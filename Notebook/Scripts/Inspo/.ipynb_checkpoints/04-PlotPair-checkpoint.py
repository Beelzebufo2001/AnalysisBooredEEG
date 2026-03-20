#!/usr/bin/env python3
"""
04_plot_pairs.py — Per-electrode-pair connectivity plots + matrix snapshots.

Produces per-subject:
  <output-dir>/<subject>/pairs/<Ch1>_<Ch2>.png    — one file per electrode pair
  <output-dir>/<subject>/matrices/corr_XXXX.png   — heatmap of selected windows
  <output-dir>/<subject>/matrix_diff.png           — difference between two windows

Usage
-----
# All subjects, save all pairs:
python 04_plot_pairs.py \\
    --corr-dir /results/corr_matrices \\
    --output-dir /results/pair_plots

# Single subject, only top-N most variable pairs:
python 04_plot_pairs.py \\
    --subject C01 \\
    --corr-dir /results/corr_matrices \\
    --output-dir /results/pair_plots \\
    --top-pairs 20

# Skip pair plots (only matrix heatmaps):
python 04_plot_pairs.py \\
    --subject C01 \\
    --corr-dir /results/corr_matrices \\
    --output-dir /results/pair_plots \\
    --no-pairs

# Two specific windows to compare:
python 04_plot_pairs.py \\
    --subject C01 \\
    --corr-dir /results/corr_matrices \\
    --output-dir /results/pair_plots \\
    --compare-windows 5 580
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config
from utils import setup_logger, discover_subjects, save_fig, check_conda_env


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-pair connectivity plots and matrix heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corr-dir",     required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--log-dir",      default=None)
    p.add_argument("--subject",      default=None)
    p.add_argument("--conda-env",    default=None)
    p.add_argument("--ch-names-file", default=None,
                   help="Text file with one channel name per line.")
    p.add_argument("--top-pairs",    type=int, default=None,
                   help="Only plot the N most variable pairs. Default: all pairs.")
    p.add_argument("--no-pairs",     action="store_true",
                   help="Skip per-pair plots (faster; only produces matrix images).")
    p.add_argument("--compare-windows", type=int, nargs=2, default=None,
                   metavar=("WIN_A", "WIN_B"),
                   help="Indices of two windows to compare side-by-side.")
    p.add_argument("--n-matrix-plots", type=int, default=5,
                   help="Number of evenly-spaced matrix heatmaps to save.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip subjects whose pairs directory already exists.")
    p.add_argument("--dpi",  type=int, default=config.PLOT_DPI)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_matrices(corr_dir, subject, logger):
    path = os.path.join(corr_dir, subject)
    files = sorted(f for f in os.listdir(path) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {path}")
    logger.info(f"  Loading {len(files)} matrices")
    mats = [np.load(os.path.join(path, f)) for f in files]
    return np.stack(mats), files


# ── plot functions ────────────────────────────────────────────────────────────

def plot_pairs(connectivity, upper_indices, ch_names, top_n, out_dir, dpi, dry_run, logger):
    """Save one PNG per electrode pair (or top-N by std)."""
    pair_time_series = np.array([
        connectivity[:, i, j] for i, j in zip(*upper_indices)
    ])

    if top_n is not None:
        pair_std = np.std(pair_time_series, axis=1)
        indices_to_plot = np.argsort(pair_std)[-top_n:][::-1]
        logger.info(f"  Plotting top {top_n} pairs by variability")
    else:
        indices_to_plot = range(len(pair_time_series))
        logger.info(f"  Plotting all {len(pair_time_series)} pairs")

    if dry_run:
        logger.info(f"[DRY-RUN] Would write ~{len(indices_to_plot)} pair PNGs to {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    for idx in indices_to_plot:
        i = upper_indices[0][idx]
        j = upper_indices[1][idx]
        ch1, ch2 = ch_names[i], ch_names[j]
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(pair_time_series[idx], color="orange", lw=0.7)
        ax.set_title(f"{ch1} – {ch2}")
        ax.set_xlabel("Time window")
        ax.set_ylabel("Correlation")
        ax.grid(alpha=0.3)
        filepath = os.path.join(out_dir, f"{ch1}_{ch2}.png")
        save_fig(fig, filepath, dpi=dpi, logger=logger)


def plot_matrix_heatmaps(connectivity, filenames, n_plots, out_dir, dpi, dry_run, logger):
    """Save evenly-spaced matrix heatmaps."""
    T = len(connectivity)
    indices = np.linspace(0, T - 1, min(n_plots, T), dtype=int)
    if dry_run:
        logger.info(f"[DRY-RUN] Would write {len(indices)} matrix heatmaps to {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    for idx in indices:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(connectivity[idx], vmin=-1, vmax=1, cmap="RdBu_r")
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title(filenames[idx].replace(".npy", ""))
        ax.set_xlabel("Channels")
        ax.set_ylabel("Channels")
        out_path = os.path.join(out_dir, filenames[idx].replace(".npy", ".png"))
        save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_matrix_comparison(connectivity, win_a, win_b, filenames, out_path, dpi, dry_run, logger):
    """Side-by-side comparison of two windows + their difference."""
    T = len(connectivity)
    for w, label in [(win_a, "WIN_A"), (win_b, "WIN_B")]:
        if w >= T:
            logger.warning(f"--compare-windows {label}={w} >= number of windows ({T}). Skipping.")
            return
    if dry_run:
        logger.info(f"[DRY-RUN] Would save matrix comparison to {out_path}")
        return

    m1, m2 = connectivity[win_a], connectivity[win_b]
    diff = m2 - m1

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, mat, title in zip(
        axes,
        [m1, m2, diff],
        [filenames[win_a], filenames[win_b], "Difference (B − A)"],
    ):
        vabs = 1.0 if "Diff" not in title else np.abs(diff).max()
        im = ax.imshow(mat, vmin=-vabs, vmax=vabs, cmap="RdBu_r")
        plt.colorbar(im, ax=ax, label="Corr" if "Diff" not in title else "Δ Corr")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Channels")
        ax.set_ylabel("Channels")
    fig.suptitle("Matrix comparison", fontsize=11)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=dpi, logger=logger)
    logger.info(f"Mean absolute change: {np.abs(diff).mean():.4f}")


# ── per-subject runner ────────────────────────────────────────────────────────

def run_subject(subject, args, ch_names_global, logger):
    logger.info(f"{'='*60}")
    logger.info(f"Subject: {subject}")
    logger.info(f"{'='*60}")

    connectivity, filenames = load_matrices(args.corr_dir, subject, logger)
    T, C, _ = connectivity.shape
    upper_indices = np.triu_indices(C, k=1)

    # Channel names
    if ch_names_global and len(ch_names_global) == C:
        ch_names = ch_names_global
    else:
        if ch_names_global:
            logger.warning(f"ch_names count mismatch ({len(ch_names_global)} vs {C}). Using Ch labels.")
        ch_names = [f"Ch{i:03d}" for i in range(C)]

    out_sub = os.path.join(args.output_dir, subject)

    # Skip check
    pairs_dir = os.path.join(out_sub, "pairs")
    if args.skip_existing and os.path.isdir(pairs_dir):
        logger.info(f"Pairs dir exists — skipping (--skip-existing).")
        return

    # Pair plots
    if not args.no_pairs:
        plot_pairs(
            connectivity, upper_indices, ch_names,
            top_n=args.top_pairs,
            out_dir=pairs_dir,
            dpi=args.dpi, dry_run=args.dry_run, logger=logger,
        )

    # Matrix heatmaps
    if args.n_matrix_plots > 0:
        plot_matrix_heatmaps(
            connectivity, filenames,
            n_plots=args.n_matrix_plots,
            out_dir=os.path.join(out_sub, "matrices"),
            dpi=args.dpi, dry_run=args.dry_run, logger=logger,
        )

    # Comparison
    if args.compare_windows:
        win_a, win_b = args.compare_windows
        plot_matrix_comparison(
            connectivity, win_a, win_b, filenames,
            out_path=os.path.join(out_sub, "matrix_diff.png"),
            dpi=args.dpi, dry_run=args.dry_run, logger=logger,
        )

    logger.info(f"Done: {subject}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    subject_label = args.subject or "all"
    logger = setup_logger("04_plot_pairs", log_dir, subject=subject_label)

    if args.conda_env:
        check_conda_env(args.conda_env, logger=logger)
    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    ch_names_global = None
    if args.ch_names_file:
        if not os.path.isfile(args.ch_names_file):
            logger.error(f"--ch-names-file not found: {args.ch_names_file}")
            sys.exit(1)
        with open(args.ch_names_file) as f:
            ch_names_global = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(ch_names_global)} channel names")

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = discover_subjects(args.corr_dir)
        if not subjects:
            logger.error(f"No subject dirs found in: {args.corr_dir}")
            sys.exit(1)
        logger.info(f"Found {len(subjects)} subjects: {subjects}")

    errors = []
    for subj in subjects:
        try:
            run_subject(subj, args, ch_names_global, logger)
        except Exception as e:
            logger.error(f"FAILED {subj}: {e}", exc_info=True)
            errors.append(subj)

    if errors:
        logger.error(f"Finished with errors for: {errors}")
        sys.exit(1)
    else:
        logger.info("All subjects completed successfully.")


if __name__ == "__main__":