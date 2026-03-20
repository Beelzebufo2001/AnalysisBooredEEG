#!/usr/bin/env python3
"""
03_analyze_connectivity.py — Load correlation matrices and produce connectivity analyses.

Produces per-subject:
  <output-dir>/<subject>/mean_connectivity.png       — global mean over time
  <output-dir>/<subject>/condition_comparison.png    — first vs second half (eyes open/closed)
  <output-dir>/<subject>/variability_over_time.png   — std of pairs over time
  <output-dir>/<subject>/hemisphere_connectivity.png — L↔L / L↔R / R↔R
  <output-dir>/<subject>/region_connectivity.png     — frontal/central/etc. pairs
  <output-dir>/<subject>/top10_pairs.png             — most variable electrode pairs
  <output-dir>/<subject>/summary_stats.txt           — plain text summary

Produces multi-subject:
  <output-dir>/group_condition_comparison.png        — delta bar chart across subjects

Usage
-----
# All subjects:
python 03_analyze_connectivity.py \\
    --corr-dir /results/corr_matrices \\
    --output-dir /results/connectivity

# Single subject:
python 03_analyze_connectivity.py \\
    --subject C01 \\
    --corr-dir /results/corr_matrices \\
    --output-dir /results/connectivity

# You need to point --ch-names-file to a plain text file with one channel name per line.
# If omitted, channels will be labelled Ch000, Ch001, ...
# You can export channel names from MNE with:
#   python -c "import mne; r=mne.io.read_raw_fif('your.fif',verbose=False); open('ch_names.txt','w').write('\\n'.join(r.ch_names))"
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
from utils import (
    setup_logger, discover_subjects, save_fig, check_conda_env,
    map_hemispheres, map_regions,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Connectivity analysis from pre-computed correlation matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corr-dir",      required=True, help="Root dir of corr_matrices (output of 02)")
    p.add_argument("--output-dir",    required=True, help="Where to save plots and stats")
    p.add_argument("--log-dir",       default=None)
    p.add_argument("--subject",       default=None,  help="Single subject. Omit to run all.")
    p.add_argument("--conda-env",     default=None)
    p.add_argument("--ch-names-file", default=None,
                   help="Text file with one channel name per line. "
                        "If omitted, channels are labelled Ch000, Ch001 ...")
    p.add_argument("--dpi",  type=int, default=config.PLOT_DPI)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_matrices(corr_dir: str, subject: str, logger) -> np.ndarray:
    """Load all .npy files for a subject, return (T, C, C) array."""
    path = os.path.join(corr_dir, subject)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No matrix directory found: {path}")
    files = sorted(f for f in os.listdir(path) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in: {path}")
    logger.info(f"  Loading {len(files)} matrices from {path}")
    matrices = [np.load(os.path.join(path, f)) for f in files]
    return np.stack(matrices)   # (T, C, C)


def mean_upper(matrix: np.ndarray, upper_indices) -> float:
    return float(np.mean(matrix[upper_indices]))


def region_connectivity(connectivity, regionA, regionB):
    values = []
    for matrix in connectivity:
        vals = matrix[np.ix_(regionA, regionB)]
        values.append(np.mean(vals))
    return np.array(values)


# ── plot functions ────────────────────────────────────────────────────────────

def plot_mean_connectivity(values, mid, out_path, dpi, dry_run, logger):
    if dry_run:
        logger.info(f"[DRY-RUN] {out_path}")
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(values, color="darkmagenta", lw=1)
    ax.axvline(mid, color="gray", linestyle="--", label="Midpoint")
    ax.set_title("Mean global connectivity over time")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Mean connectivity")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_condition_comparison(values, mid, out_path, dpi, dry_run, logger):
    if dry_run:
        logger.info(f"[DRY-RUN] {out_path}")
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(mid),         values[:mid],  color="thistle",     label="First half")
    ax.plot(range(mid, len(values)), values[mid:], color="steelblue",  label="Second half")
    ax.axvline(mid, color="darkmagenta", linestyle="--", label="Condition switch")
    ax.set_title("Mean connectivity — condition comparison")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Mean connectivity")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_variability(variability, mid, out_path, dpi, dry_run, logger):
    if dry_run:
        logger.info(f"[DRY-RUN] {out_path}")
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(mid),               variability[:mid],  color="khaki",      label="First half")
    ax.plot(range(mid, len(variability)), variability[mid:], color="powderblue", label="Second half")
    ax.axvline(mid, color="red", linestyle="--", label="Condition switch")
    ax.set_title("Variability across electrode pairs over time")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Std of pair correlations")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_hemisphere(mean_ll, mean_lr, mean_rr, out_path, dpi, dry_run, logger):
    if dry_run:
        logger.info(f"[DRY-RUN] {out_path}")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_ll, color="steelblue",   lw=0.8, label="Left ↔ Left")
    ax.plot(mean_lr, color="darkmagenta", lw=0.8, label="Left ↔ Right")
    ax.plot(mean_rr, color="coral",       lw=0.8, label="Right ↔ Right")
    ax.set_title("Hemisphere connectivity over time")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Mean connectivity")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_regions(region_pairs, out_path, dpi, dry_run, logger):
    if dry_run:
        logger.info(f"[DRY-RUN] {out_path}")
        return
    n = len(region_pairs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = plt.cm.tab20.colors
    for ax, (label, series), color in zip(axes, region_pairs.items(), colors):
        ax.plot(series, color=color, lw=0.8)
        ax.set_ylabel("Mean corr", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time window")
    fig.suptitle("Region-to-region connectivity", fontsize=11)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_top10_pairs(pair_time_series, upper_indices, ch_names, out_path, dpi, dry_run, logger):
    if dry_run:
        logger.info(f"[DRY-RUN] {out_path}")
        return
    pair_std = np.std(pair_time_series, axis=1)
    top10 = np.argsort(pair_std)[-10:][::-1]
    fig, axes = plt.subplots(10, 1, figsize=(10, 20), sharex=True)
    colors = plt.cm.tab10.colors
    for rank, (ax, idx) in enumerate(zip(axes, top10)):
        i = upper_indices[0][idx]
        j = upper_indices[1][idx]
        label = f"{ch_names[i]}–{ch_names[j]}  (std={pair_std[idx]:.3f})"
        ax.plot(pair_time_series[idx], color=colors[rank % 10], lw=0.7)
        ax.set_title(label, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time window")
    fig.suptitle("Top 10 most variable electrode pairs", fontsize=11)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_group_bar(subjects, deltas, out_path, dpi, dry_run, logger):
    if dry_run:
        logger.info(f"[DRY-RUN] {out_path}")
        return
    colors = ["steelblue" if d >= 0 else "coral" for d in deltas]
    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 1.2), 4))
    ax.bar(subjects, deltas, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Connectivity change: second half − first half")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Δ Mean connectivity")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=dpi, logger=logger)


# ── per-subject runner ────────────────────────────────────────────────────────

def run_subject(subject, args, ch_names_global, logger):
    logger.info(f"{'='*60}")
    logger.info(f"Subject: {subject}")
    logger.info(f"{'='*60}")

    connectivity = load_matrices(args.corr_dir, subject, logger)
    T, C, _ = connectivity.shape

    # Channel names
    if ch_names_global and len(ch_names_global) == C:
        ch_names = ch_names_global
    elif ch_names_global:
        logger.warning(
            f"ch_names file has {len(ch_names_global)} names but data has {C} channels. "
            "Falling back to Ch000 labels."
        )
        ch_names = [f"Ch{i:03d}" for i in range(C)]
    else:
        ch_names = [f"Ch{i:03d}" for i in range(C)]

    upper_indices = np.triu_indices(C, k=1)
    mid = T // 2

    # ── Global mean connectivity ───────────────────────────────────────────────
    values = [mean_upper(m, upper_indices) for m in connectivity]
    values = np.array(values)
    variability = np.array([np.std(m[upper_indices]) for m in connectivity])

    out_sub = os.path.join(args.output_dir, subject)
    if not args.dry_run:
        os.makedirs(out_sub, exist_ok=True)

    plot_mean_connectivity(values, mid,
        os.path.join(out_sub, "mean_connectivity.png"), args.dpi, args.dry_run, logger)
    plot_condition_comparison(values, mid,
        os.path.join(out_sub, "condition_comparison.png"), args.dpi, args.dry_run, logger)
    plot_variability(variability, mid,
        os.path.join(out_sub, "variability_over_time.png"), args.dpi, args.dry_run, logger)

    # ── Hemisphere ────────────────────────────────────────────────────────────
    hemi = map_hemispheres(ch_names)
    left, right = hemi["left"], hemi["right"]
    if left and right:
        mean_ll = region_connectivity(connectivity, left,  left)
        mean_lr = region_connectivity(connectivity, left,  right)
        mean_rr = region_connectivity(connectivity, right, right)
        plot_hemisphere(mean_ll, mean_lr, mean_rr,
            os.path.join(out_sub, "hemisphere_connectivity.png"), args.dpi, args.dry_run, logger)
    else:
        logger.warning("Could not determine hemispheres from channel names — skipping hemisphere plot.")

    # ── Regions ───────────────────────────────────────────────────────────────
    regions = map_regions(ch_names, config.REGION_PREFIXES)
    non_empty = {k: v for k, v in regions.items() if v}
    if non_empty:
        region_pairs = {}
        for nameA, regA in non_empty.items():
            for nameB, regB in non_empty.items():
                region_pairs[f"{nameA} ↔ {nameB}"] = region_connectivity(connectivity, regA, regB)
        plot_regions(region_pairs,
            os.path.join(out_sub, "region_connectivity.png"), args.dpi, args.dry_run, logger)
    else:
        logger.warning("No known region prefixes matched — skipping region plot.")

    # ── Top-10 pairs ──────────────────────────────────────────────────────────
    pair_time_series = np.array([
        connectivity[:, i, j] for i, j in zip(*upper_indices)
    ])
    if len(pair_time_series) >= 10:
        plot_top10_pairs(pair_time_series, upper_indices, ch_names,
            os.path.join(out_sub, "top10_pairs.png"), args.dpi, args.dry_run, logger)

    # ── Summary stats ─────────────────────────────────────────────────────────
    first_half  = np.mean(values[:mid])
    second_half = np.mean(values[mid:])
    delta       = second_half - first_half
    summary = (
        f"Subject: {subject}\n"
        f"Windows: {T}\n"
        f"Channels: {C}\n"
        f"Mean global connectivity (full): {np.mean(values):.4f}\n"
        f"Std global connectivity:         {np.std(values):.4f}\n"
        f"Mean first half:                 {first_half:.4f}\n"
        f"Mean second half:                {second_half:.4f}\n"
        f"Delta (2nd - 1st):               {delta:.4f}\n"
    )
    logger.info("\n" + summary)
    if not args.dry_run:
        with open(os.path.join(out_sub, "summary_stats.txt"), "w") as f:
            f.write(summary)

    return delta


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    subject_label = args.subject or "all"
    logger = setup_logger("03_analyze_connectivity", log_dir, subject=subject_label)

    if args.conda_env:
        check_conda_env(args.conda_env, logger=logger)
    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    # Load channel names if provided
    ch_names_global = None
    if args.ch_names_file:
        if not os.path.isfile(args.ch_names_file):
            logger.error(f"--ch-names-file not found: {args.ch_names_file}")
            sys.exit(1)
        with open(args.ch_names_file) as f:
            ch_names_global = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(ch_names_global)} channel names from {args.ch_names_file}")

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = discover_subjects(args.corr_dir)
        if not subjects:
            logger.error(f"No subject directories found in: {args.corr_dir}")
            sys.exit(1)
        logger.info(f"Found {len(subjects)} subjects: {subjects}")

    errors = []
    deltas = {}
    for subj in subjects:
        try:
            delta = run_subject(subj, args, ch_names_global, logger)
            deltas[subj] = delta
        except Exception as e:
            logger.error(f"FAILED {subj}: {e}", exc_info=True)
            errors.append(subj)

    # Group plot (only when >1 subject)
    if len(deltas) > 1:
        subj_list  = list(deltas.keys())
        delta_list = [deltas[s] for s in subj_list]
        if not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)
        plot_group_bar(subj_list, delta_list,
            os.path.join(args.output_dir, "group_condition_comparison.png"),
            args.dpi, args.dry_run, logger)

    if errors:
        logger.error(f"Finished with errors for: {errors}")
        sys.exit(1)
    else:
        logger.info("All subjects completed successfully.")


if __name__ == "__main__":
    main()