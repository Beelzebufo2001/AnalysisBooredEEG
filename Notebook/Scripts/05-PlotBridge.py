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
    if not dirpath.exists():
        print(f"[DIR] Creating directory: {dirpath}", flush=True)
        dirpath.mkdir(parents=True, exist_ok=True)


def load_raw_file(file_path):
    print(f"  [LOAD] Reading: {file_path}", flush=True)
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_fif(fname=file_path, preload=True, verbose=False)

    raw.set_montage(
        mne.channels.make_standard_montage(config.DEFAULT_MONTAGE),
        on_missing="ignore",
        verbose=False,
    )
    print(f"  [LOAD] OK — {raw.n_times} samples, {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz", flush=True)
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

    print(f"[DISCOVER] Scanning: {folder_path}", flush=True)
    print(f"[DISCOVER] Clean alg: {clean_alg} | Recording: {recording}", flush=True)

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
            print(f"  [SKIP] {subject}: missing folder {clean_dir}", flush=True)
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
            print(f"  [SKIP] {subject}: no matching .fif for recording='{recording}'", flush=True)
            continue

        chosen = matching_files[0]  # TODO: handle multiple matches / concat splits
        subjects.append(subject)
        file_paths.append(chosen)
        print(f"  [ADD]  {subject}: {chosen.name}", flush=True)

    print(f"[DISCOVER] Total subjects found: {len(subjects)}", flush=True)
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
            print(f"[ERROR] Subjects not found or missing files: {missing}", file=sys.stderr, flush=True)
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
        topomap_args=dict(vlim=(None, 5)),
        show=False,
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
    print(f"\n{'='*60}", flush=True)
    print(f"[SUBJECT] {subject}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Load
    raw = load_raw_file(subject_path)   # fixed: was subject_pathct (typo)

    # 2. Crop last N seconds
    tmax = raw.times[-1]
    seconds = args.length               # fixed: was args.lenght
    if seconds > tmax:
        print(f"  [ERROR] Requested {seconds}s but recording is only {tmax:.1f}s — skipping {subject}", flush=True)
        return                          # fixed: was exit(1) which would kill all subjects
    tmin = tmax - seconds
    raw_crop = raw.copy().crop(tmin=tmin, tmax=tmax)
    print(f"  [CROP] {tmin:.1f}s — {tmax:.1f}s ({seconds}s)", flush=True)

    # 3. EEG only
    raw_crop = raw_crop.pick_types(eeg=True, verbose=False)
    print(f"  [PICK] {len(raw_crop.ch_names)} EEG channels", flush=True)

    # 4. Downsample
    new_freq = args.target_freq
    if new_freq >= raw_crop.info["sfreq"]:
        print(f"  [WARN] target_freq ({new_freq}) >= original sfreq ({raw_crop.info['sfreq']}) — skipping resample", flush=True)
    else:
        raw_crop.resample(new_freq, verbose=False)
        print(f"  [RESAMPLE] → {new_freq} Hz", flush=True)

    # 5. Correlation matrix
    data = raw_crop.get_data()
    corr = np.corrcoef(data)
    print(f"  [CORR] data shape: {data.shape} | corr shape: {corr.shape}", flush=True)

    # 6. Bridge detection
    lc = args.lim_cutoff                # fixed: was args.lim_epoch (wrong attr name)
    et = args.lim_epochs
    print(f"  [BRIDGES] Computing — lm_cutoff={lc}, epoch_threshold={et}", flush=True)
    bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(
        raw_crop,
        verbose=False,
        lm_cutoff=lc,
        epoch_threshold=et,             # fixed: was epoch_treshold (typo)
    )
    ch_names = raw_crop.ch_names
    print(f"  [BRIDGES] Found {len(bridged_idx)} bridge pairs:", flush=True)
    for (i, j) in bridged_idx:
        print(f"    {ch_names[i]} <-> {ch_names[j]}", flush=True)

    # 7. Save PDF
    out_dir = args.output_dir / subject  # fixed: was args.out_dir (wrong attr name)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"t{seconds}-lc{lc}-et{et}"
    pdf_path = out_dir / f"{subject}_Bridges_{tag}.pdf"
    print(f"  [PDF] Saving to: {pdf_path}", flush=True)

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

    print(f"  [DONE] {subject} → {pdf_path}", flush=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    print(f"[START] Bridges Analysis", flush=True)
    print(f"[ARGS]  length={args.length}s | freq={args.target_freq}Hz | lc={args.lim_cutoff} | et={args.lim_epochs}", flush=True)

    subjects, subject_file_map = resolve_subjects(args)

    if not subjects:
        print("[ERROR] No subjects to process — exiting.", flush=True)
        sys.exit(1)

    print(f"\n[QUEUE] Processing {len(subjects)} subjects: {subjects}", flush=True)

    for subject in subjects:
        run_subject(subject, subject_file_map[subject], args)

    print(f"\n[FINISH] All subjects done.", flush=True)


if __name__ == "__main__":
    main()