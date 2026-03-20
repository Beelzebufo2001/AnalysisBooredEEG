#!/usr/bin/env python3
"""
01_plot_eeg.py — Quick-look QC plots for a single ICA-preprocessed .fif file.

Produces:
  <output_dir>/<subject>/raw_electrode_<ELEC>.png
  <output_dir>/<subject>/multi_electrode.png
  <output_dir>/<subject>/bands_<ELEC>.png

Usage
-----
# Single subject:
python 01_plot_eeg.py --subject C01 --data-dir /data/eeg --output-dir /results/qc_plots

# All subjects in data-dir:
python 01_plot_eeg.py --data-dir /data/eeg --output-dir /results/qc_plots

# Override time window and electrode:
python 01_plot_eeg.py --subject C01 --data-dir /data/eeg --output-dir /results/qc_plots \\
    --electrode Cz --t-min 10 --t-max 20

# Dry run (no files written):
python 01_plot_eeg.py --subject C01 --data-dir /data/eeg --output-dir /results/qc_plots --dry-run
"""

import argparse
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config
from utils import setup_logger, discover_subjects, find_fif_file, load_raw, save_fig, check_conda_env

try:
    from elephant.signal_processing import butter as eephant_butter
except ImportError:
    eephant_butter = None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="QC plots: single electrode, multi-electrode, and band decomposition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    required=True,  help="Root directory containing subject folders (C01, C02 ...)")
    p.add_argument("--output-dir",  required=True,  help="Where to save plots")
    p.add_argument("--log-dir",     default=None,   help="Where to save log files (default: <output-dir>/logs)")
    p.add_argument("--subject",     default=None,   help="Single subject ID (e.g. C01). Omit to run all.")
    p.add_argument("--conda-env",   default=None,   help="Expected conda environment name (for safety check)")
    p.add_argument("--electrode",   default="Cz",   help="Primary electrode for single-trace plot")
    p.add_argument("--electrodes",  default="CP1,CP2,CP3,CP4",
                                    help="Comma-separated list for multi-electrode plot")
    p.add_argument("--t-min",  type=float, default=22.0, help="Window start (s)")
    p.add_argument("--t-max",  type=float, default=23.0, help="Window end (s)")
    p.add_argument("--dpi",    type=int,   default=config.PLOT_DPI)
    p.add_argument("--dry-run", action="store_true", help="Print what would be done; write nothing")
    return p.parse_args()


# ── plotting functions ────────────────────────────────────────────────────────

def plot_single_electrode(raw, electrod, t_min, t_max, out_path, dpi, dry_run, logger):
    sfreq = raw.info["sfreq"]
    data = raw.get_data(
        picks=electrod,
        start=int(t_min * sfreq),
        stop=int(t_max * sfreq),
    )
    if dry_run:
        logger.info(f"[DRY-RUN] Would save: {out_path}")
        return
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    ax.plot(data[0], color="red", lw=0.5)
    ax.grid(alpha=0.5)
    ax.set_title(f"Electrode: {electrod}  |  {t_min}–{t_max} s")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (V)")
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_multi_electrode(raw, electrodes, t_min, t_max, out_path, dpi, dry_run, logger):
    colors = ["red", "blue", "plum", "pink", "orange", "teal", "gold", "steelblue"]
    data = raw.get_data(picks=electrodes, tmin=t_min, tmax=t_max)
    if dry_run:
        logger.info(f"[DRY-RUN] Would save: {out_path}")
        return
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    for idx, (trace, elec) in enumerate(zip(data, electrodes)):
        ax.plot(trace, color=colors[idx % len(colors)], lw=0.5, label=elec)
    ax.grid(alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Multi-electrode  |  {t_min}–{t_max} s")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (V)")
    save_fig(fig, out_path, dpi=dpi, logger=logger)


def plot_band_decomposition(raw, electrod, t_min, t_max, out_path, dpi, dry_run, logger):
    if eephant_butter is None:
        logger.warning("elephant not installed — skipping band decomposition plot.")
        return
    sfreq = raw.info["sfreq"]
    data = raw.get_data(
        picks=electrod,
        start=int(t_min * sfreq),
        stop=int(t_max * sfreq),
    )
    if dry_run:
        logger.info(f"[DRY-RUN] Would save: {out_path}")
        return
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    for band_name, (hp, lp, color) in config.FREQ_BANDS.items():
        filtered = eephant_butter(
            data,
            highpass_frequency=hp if hp else None,
            lowpass_frequency=lp if lp else None,
            filter_function="sosfiltfilt",
            sampling_frequency=sfreq,
        )
        ax.plot(filtered[0], color=color, label=band_name, lw=0.7)
    ax.legend()
    ax.grid(alpha=0.5)
    ax.set_title(f"Band decomposition — {electrod}  |  {t_min}–{t_max} s")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (V)")
    save_fig(fig, out_path, dpi=dpi, logger=logger)


# ── per-subject runner ────────────────────────────────────────────────────────

def run_subject(subject, args, logger):
    logger.info(f"{'='*60}")
    logger.info(f"Subject: {subject}")
    logger.info(f"{'='*60}")

    fif_path = find_fif_file(args.data_dir, subject)
    raw = load_raw(fif_path, preload=True, logger=logger)

    electrodes = [e.strip() for e in args.electrodes.split(",")]
    out_sub = os.path.join(args.output_dir, subject)
    os.makedirs(out_sub, exist_ok=True)

    # 1. Single electrode raw trace
    plot_single_electrode(
        raw, args.electrode, args.t_min, args.t_max,
        out_path=os.path.join(out_sub, f"raw_{args.electrode}.png"),
        dpi=args.dpi, dry_run=args.dry_run, logger=logger,
    )

    # 2. Multi-electrode
    valid_elecs = [e for e in electrodes if e in raw.ch_names]
    if len(valid_elecs) < len(electrodes):
        missing = set(electrodes) - set(valid_elecs)
        logger.warning(f"Electrodes not found in data, skipping: {missing}")
    if valid_elecs:
        plot_multi_electrode(
            raw, valid_elecs, args.t_min, args.t_max,
            out_path=os.path.join(out_sub, "multi_electrode.png"),
            dpi=args.dpi, dry_run=args.dry_run, logger=logger,
        )

    # 3. Band decomposition
    plot_band_decomposition(
        raw, args.electrode, args.t_min, args.t_max,
        out_path=os.path.join(out_sub, f"bands_{args.electrode}.png"),
        dpi=args.dpi, dry_run=args.dry_run, logger=logger,
    )

    logger.info(f"Done: {subject}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    subject_label = args.subject or "all"
    logger = setup_logger("01_plot_eeg", log_dir, subject=subject_label)

    if args.conda_env:
        check_conda_env(args.conda_env, logger=logger)

    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    # Resolve subject list
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = discover_subjects(args.data_dir)
        if not subjects:
            logger.error(f"No subject directories found in: {args.data_dir}")
            sys.exit(1)
        logger.info(f"Found {len(subjects)} subjects: {subjects}")

    errors = []
    for subj in subjects:
        try:
            run_subject(subj, args, logger)
        except Exception as e:
            logger.error(f"FAILED {subj}: {e}", exc_info=True)
            errors.append(subj)

    if errors:
        logger.error(f"Pipeline finished with errors for: {errors}")
        sys.exit(1)
    else:
        logger.info("All subjects completed successfully.")


if __name__ == "__main__":
    main()