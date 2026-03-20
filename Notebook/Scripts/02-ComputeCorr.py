#!/usr/bin/env python3
"""
02_compute_corr.py — Compute sliding-window correlation matrices from EEG data.

Uses the lab's helpers.py for subject discovery, data loading, filtering,
and downsampling — consistent with your other analysis code.

For each subject this script:
  1. Discovers files via get_valid_subjects_and_paths()
  2. Loads the .fif (or .cnt) file via load_data()
  3. Slides a window of --snippet-len seconds in --step increments
  4. Bandpass-filters each snippet via filter_signal()
  5. Downsamples to --target-sfreq via downsample_signal()
  6. Computes the full-channel Pearson correlation matrix
  7. Saves each matrix as corr_TTTT.npy (T = window start in seconds)

Output layout:
  <output-dir>/<subject>/corr_0000.npy
  <output-dir>/<subject>/corr_0001.npy
  ...

This is the HEAVY script — submit via SLURM for full datasets.

Usage examples
--------------
# Single subject:
python 02_compute_corr.py \\
    --data-dir /data/eeg \\
    --output-dir /results/corr_matrices \\
    --subject C01

# All subjects:
python 02_compute_corr.py \\
    --data-dir /data/eeg \\
    --output-dir /results/corr_matrices

# Pick file indices (0=RS_before, 1=Task, 2=RS_after):
python 02_compute_corr.py \\
    --data-dir /data/eeg --output-dir /results/corr_matrices \\
    --files-idx 0 2

# Exclude known-bad subjects (ICA: P03 P04 C16 C24 / ATAR: P04 C02 C04):
python 02_compute_corr.py \\
    --data-dir /data/eeg --output-dir /results/corr_matrices \\
    --exclude P03 P04 C16 C24

# Quick test: only 20 windows:
python 02_compute_corr.py \\
    --data-dir /data/eeg --output-dir /results/corr_matrices \\
    --subject C01 --max-windows 20

# Skip already-done subjects:
python 02_compute_corr.py \\
    --data-dir /data/eeg --output-dir /results/corr_matrices \\
    --skip-existing

# Dry run (nothing written):
python 02_compute_corr.py \\
    --data-dir /data/eeg --output-dir /results/corr_matrices \\
    --subject C01 --dry-run
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config
from utils   import setup_logger, check_conda_env
from helpers import (
    get_valid_subjects_and_paths,
    load_data,
    filter_signal,
    downsample_signal,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute sliding-window Pearson correlation matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--data-dir",    required=True,
                   help="Root directory containing subject folders")
    p.add_argument("--output-dir",  required=True,
                   help="Where to save .npy matrix files")
    p.add_argument("--log-dir",     default=None,
                   help="Log directory (default: <output-dir>/logs)")

    # Subject selection
    p.add_argument("--subject",     default=None,
                   help="Single subject ID. Omit to process all.")
    p.add_argument("--clean-alg",   default="ica",
                   choices=["raw", "ica", "atar", "preprocessed-high-freq"],
                   help="Cleaning algorithm subfolder containing .fif files.")
    p.add_argument("--files-idx",   type=int, nargs="+", default=[0],
                   help="File index per subject. 0=RS_before 1=Task 2=RS_after.")
    p.add_argument("--exclude",     nargs="*", default=[], metavar="SUBJ",
                   help="Subject IDs to skip (e.g. --exclude P03 P04 C16 C24).")

    # Processing
    p.add_argument("--snippet-len",  type=float, default=config.SNIPPET_LEN_S)
    p.add_argument("--step",         type=float, default=config.STEP_S)
    p.add_argument("--highpass",     type=float, default=config.CORR_HIGHPASS)
    p.add_argument("--lowpass",      type=float, default=config.CORR_LOWPASS)
    p.add_argument("--target-sfreq", type=int,   default=config.TARGET_SFREQ)

    # Run control
    p.add_argument("--max-windows",   type=int, default=None,
                   help="Stop after N windows (testing).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip subjects whose output dir already has .npy files.")
    p.add_argument("--conda-env",     default=None,
                   help="Expected conda env name (safety check).")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print plan; write nothing.")
    return p.parse_args()


# ── per-subject ───────────────────────────────────────────────────────────────

def run_subject(subject: str, file_path: str, args, logger) -> int:
    logger.info(f"{'='*60}")
    logger.info(f"Subject : {subject}")
    logger.info(f"File    : {file_path}")
    logger.info(f"{'='*60}")

    out_sub = os.path.join(args.output_dir, subject)

    if args.skip_existing and os.path.isdir(out_sub):
        existing = len([f for f in os.listdir(out_sub) if f.endswith(".npy")])
        if existing > 0:
            logger.info(f"  {existing} matrices already exist — skipping (--skip-existing).")
            return 0

    # Load via lab helper
    raw, _imp_start, _imp_end, sampling_rate, recording_length = load_data(
        file_path, logger=logger
    )

    n_samples      = raw.n_times
    window_samples = int(args.snippet_len * sampling_rate)
    step_samples   = int(args.step        * sampling_rate)
    decim_factor   = int(sampling_rate / args.target_sfreq)
    if decim_factor < 1:
        logger.warning(
            f"  target-sfreq ({args.target_sfreq}) >= sfreq ({sampling_rate}). "
            "No decimation."
        )
        decim_factor = 1

    total_windows = max(0, (n_samples - window_samples) // step_samples + 1)
    capped = f" (capped at {args.max_windows})" if args.max_windows else ""
    logger.info(f"  sfreq       : {sampling_rate} Hz  →  {args.target_sfreq} Hz  "
                f"(×{decim_factor})")
    logger.info(f"  Window/step : {args.snippet_len} s / {args.step} s")
    logger.info(f"  Bandpass    : {args.highpass}–{args.lowpass} Hz")
    logger.info(f"  Windows     : {total_windows}{capped}")

    if args.dry_run:
        logger.info("  [DRY-RUN] Nothing written.")
        return 0

    os.makedirs(out_sub, exist_ok=True)

    t_start   = 0
    win_count = 0

    while t_start + window_samples <= n_samples:
        t_sec = int(t_start // sampling_rate)

        # 1. Extract  (channels × samples)
        snippet = raw.get_data(start=t_start, stop=t_start + window_samples)

        # 2. Bandpass — lab's filter_signal wraps elephant.butter
        snippet_bp = filter_signal(
            snippet,
            lowpass=args.lowpass,
            highpass=args.highpass,
            sampling_rate=sampling_rate,
        )

        # 3. Downsample — lab's downsample_signal wraps scipy.decimate FIR
        snippet_ds = downsample_signal(
            snippet_bp,
            target_frequency=args.target_sfreq,
            sampling_rate=sampling_rate,
        )

        # 4. Correlation matrix
        matrix = np.corrcoef(snippet_ds)

        # 5. Save
        out_file = os.path.join(out_sub, f"corr_{t_sec:04d}.npy")
        np.save(out_file, matrix)
        logger.debug(f"  w{win_count:04d}  t={t_sec:5d}s  →  {os.path.basename(out_file)}")

        t_start   += step_samples
        win_count += 1

        if args.max_windows and win_count >= args.max_windows:
            logger.info(f"  Reached --max-windows {args.max_windows}.")
            break

    logger.info(f"  Done. Windows written: {win_count}")
    return win_count


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    logger  = setup_logger("02_compute_corr", log_dir, subject=args.subject or "all")

    if args.conda_env:
        check_conda_env(args.conda_env, logger=logger)
    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    # Discover subjects via helpers.get_valid_subjects_and_paths
    subjects, file_paths = get_valid_subjects_and_paths(
        folder_path=args.data_dir,
        clean_alg=args.clean_alg,
        exclude_subjects=args.exclude,
        files_idx=args.files_idx,
        logger=logger,
    )
    pairs = list(zip(subjects, file_paths))

    if not pairs:
        logger.error(f"No valid subjects found in '{args.data_dir}'.")
        sys.exit(1)

    # Filter to single subject if requested
    if args.subject:
        pairs = [
            (s, fp) for s, fp in pairs
            if s == args.subject or s.startswith(f"{args.subject}_")
        ]
        if not pairs:
            logger.error(
                f"Subject '{args.subject}' not found after discovery. "
                f"Check folder structure and --clean-alg value."
            )
            sys.exit(1)

    logger.info(f"Processing {len(pairs)} subject/file pair(s).")

    errors = []
    for subj, fp in pairs:
        try:
            run_subject(subj, fp, args, logger)
        except Exception as e:
            logger.error(f"FAILED {subj}: {e}", exc_info=True)
            errors.append(subj)

    if errors:
        logger.error(f"Finished with errors: {errors}")
        sys.exit(1)
    else:
        logger.info("All subjects completed successfully.")


if __name__ == "__main__":
    main()