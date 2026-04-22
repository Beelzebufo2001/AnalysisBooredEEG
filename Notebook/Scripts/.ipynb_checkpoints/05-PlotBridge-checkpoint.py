#!/usr/bin/env python3
"""
Bridges Analysis Script
Computes EEG electrode bridges for one or more subjects.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import sys
import warnings as w
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for sbatch/cluster
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
from matplotlib.backends.backend_pdf import PdfPages

import config

# =============================================================================
# PARAMS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute EEG electrode bridges for one or more subjects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        default=None,
        help="One or more subject IDs, e.g. C01 C02. If omitted, all subjects are processed.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default="RS_before",
        help="Recording type substring to match in filename.",
    )
    parser.add_argument(
        "--length",           # fixed typo: lenght -> length
        type=int,
        default=config.DEFAULT_BRIDGE,
        help="Number of seconds to crop from end of recording.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "bridges_plot",   # no space in path — bad for cluster
        help="Directory where output PDFs will be saved.",
    )
    parser.add_argument(
        "--target-freq",
        type=int,
        default=config.DEFAULT_SFREQ,
        help="Target sampling frequency after downsampling.",
    )
    parser.add_argument(
        "--lim-epochs",
        type=int,
        default=config.DEFAULT_EPOCH,
        help="epoch_threshold param for compute_bridged_electrodes.",
    )
    parser.add_argument(
        "--lim-cutoff",
        type=int,
        default=config.DEFAULT_LMCUTT,
        help="lm_cutoff param for compute_bridged_electrodes.",
    )
    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================
def ensure_dir_exists(dirpath):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)


def load_raw_file(file_path):
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_fif(fname=file_path, preload=True, verbose=False)

    raw.set_montage(
        mne.channels.make_standard_montage(config.DEFAULT_MONTAGE),
        on_missing="ignore",
        verbose=False,
    )
    return raw


def get_valid_subjects_and_paths(preferred, folder_path=None, clean_alg=None, recording="RS_before"):
    """
    Return matching subjects and their FIF paths.
    - only folders starting with 'C'
    - preferred can be 'all' or list like ['C01', 'C03']
    - match files by recording substring
    - ignore split files ending with '-1.fif'
    """
    folder_path = Path(folder_path or config.DATA_DIR)
    clean_alg = clean_alg or config.DATA_TYPE

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
            continue

        matching_files = sorted([
            f for f in clean_dir.iterdir()
            if (
                f.is_file()
                and f.suffix == ".fif"
                and recording in f.name
                and clean_alg in f.name
                and not f.stem.endswith("-1")   # skip split files
            )
        ])

        if not matching_files:
            continue

        chosen = matching_files[0]  # TODO: handle multiple matches / concat splits
        subjects.append(subject)
        file_paths.append(chosen)

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
            print(f"ERROR: subjects not found: {missing}", file=sys.stderr, flush=True)
            sys.exit(1)

    return subjects, subject_file_map


# =============================================================================
# PLOTTING
# =============================================================================
def plot_topomap(raw_crop, corr, bridged_idx, ed_matrix, subject_id="unknown", seconds=10, new_freq=100):
    """Topomap of bridged electrodes."""
    if len(bridged_idx) == 0:
        # Return a plain figure with a message — still goes into PDF
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No bridges detected", ha="center", va="center", fontsize=14)
        ax.set_title(f"Bridged Electrodes — {subject_id} | {seconds}s | {new_freq}Hz")
        ax.axis("off")
        return fig

    fig = mne.viz.plot_bridged_electrodes(
        raw_crop.info,
        bridged_idx,
        ed_matrix,
        title=f"Bridged Electrodes — {subject_id} | {seconds}s | {new_freq}Hz",
        topomap_args=dict(vlim=(None, 5))
    )
    return fig


def plot_correlation(corr, subject_id="unknown", seconds=10, new_freq=100):
    """Heatmap of Pearson correlation matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(
        f"Correlation Matrix — {subject_id} | last {seconds}s | {new_freq}Hz",
        fontsize=12,
    )
    ax.set_xlabel("Channel #")
    ax.set_ylabel("Channel #")
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    return fig


def plot_distribution(ed_matrix, subject_id="unknown"):
    """Histogram of Electrical Distance values."""
    ed_vals = ed_matrix[~np.isnan(ed_matrix)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ed_vals, bins=np.linspace(0, 500, 51), color="steelblue", edgecolor="white")
    p5 = np.percentile(ed_vals, 5)
    ax.axvline(p5, color="red", linestyle="--", label=f"5th percentile = {p5:.1f}")
    ax.set_title(f"Electrical Distance Distribution — {subject_id}")
    ax.set_xlabel(r"Electrical Distance ($\mu V^2$)")
    ax.set_ylabel("Count (channel pairs)")
    ax.legend()
    plt.tight_layout()
    return fig


# =============================================================================
# PER SUBJECT
# =============================================================================
def run_subject(subject, subject_path, args):
    print(f"\n── {subject} ──", flush=True)

    # 1. Load
    raw = load_raw_file(subject_path)

    # 2. Crop last N seconds
    tmax = raw.times[-1]
    
    if args.length is None:
        # vezmi celou nahrávku
        raw_crop = raw.copy()
        seconds = int(tmax)
    else:
        seconds = args.length
    
        if seconds > tmax:
            print(f"SKIP: recording too short ({tmax:.1f}s < {seconds}s requested)", flush=True)
            return
    
        tmin = tmax - seconds
        raw_crop = raw.copy().crop(tmin=tmin, tmax=tmax)

    # 3. EEG only
    raw_crop = raw_crop.pick_types(eeg=True, verbose=False)

    # 4. Downsample
    new_freq = args.target_freq
    if new_freq < raw_crop.info["sfreq"]:
        raw_crop.resample(new_freq, verbose=False)

    # 5. Correlation matrix
    data = raw_crop.get_data()
    corr = np.corrcoef(data)

    # 6. Bridge detection
    lc = args.lim_cutoff
    et = args.lim_epochs
    bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(
        raw_crop,
        verbose=False,
        lm_cutoff=lc,
        epoch_threshold=et,
    )
    ch_names = raw_crop.ch_names
    if len(bridged_idx) == 0:
        print(f"  bridges: none", flush=True)
    else:
        pairs = ", ".join(f"{ch_names[i]}<->{ch_names[j]}" for i, j in bridged_idx)
        print(f"  bridges ({len(bridged_idx)}): {pairs}", flush=True)

    # 7. Save PDF
    out_dir = args.output_dir / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"t{seconds}-lc{lc}-et{et}"
    pdf_path = out_dir / f"{subject}_Bridges_{tag}.pdf"

    with PdfPages(pdf_path) as pdf:
        fig_top = plot_topomap(raw_crop, corr, bridged_idx, ed_matrix, subject, seconds, new_freq)
        pdf.savefig(fig_top)
        plt.close(fig_top)

        fig_corr = plot_correlation(corr, subject_id=subject, seconds=seconds, new_freq=new_freq)
        pdf.savefig(fig_corr)
        plt.close(fig_corr)

        fig_dist = plot_distribution(ed_matrix, subject_id=subject)
        pdf.savefig(fig_dist)
        plt.close(fig_dist)

    print(f"  saved: {pdf_path}", flush=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    subjects, subject_file_map = resolve_subjects(args)

    if not subjects:
        print("No subjects found — exiting.", flush=True)
        sys.exit(1)

    print(f"Processing {len(subjects)} subjects: {subjects}", flush=True)

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if task_id is None:
        # běžíš normálně (lokálně)
        for subject in subjects:
            run_subject(subject, subject_file_map[subject], args)
    else:
        # běžíš jako array job
        task_id = int(task_id)

        if task_id >= len(subjects):
            print(f"SKIP: task_id {task_id} out of range", flush=True)
            return

        subject = subjects[task_id]
        run_subject(subject, subject_file_map[subject], args)

    print("\nDone.", flush=True)

    # if(args.array is None):
    #     for subject in subjects:
    #        run_subject(subject[], subject_file_map[], args)
    #     # ctrl lomitko 
    # else:
    #     run_subject(subject[int(sys.argv[1])], subject_file_map[subject[int(sys.argv[1])]], args)

    # print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()