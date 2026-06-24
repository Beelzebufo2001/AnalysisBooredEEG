#!/usr/bin/env python3

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import argparse
from pathlib import Path
import warnings as w

import mne
import numpy as np
import json
from scipy.signal import decimate
from elephant.signal_processing import butter

import config
import resolveSubject


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute sliding-window EEG correlation matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--subject",
        nargs="+",
        default=None,
        help="One or more subject IDs, e.g. C01 C02. If omitted, all control subjects are processed."
    )

    parser.add_argument(
        "--recording",
        default="RS_before",
        choices=["RS_before", "Task", "RS_after"],
        help="Which recording to load."
    )

    parser.add_argument(
        "--snippet-len",
        type=float,
        default=config.SNIPPET_LEN_S,
        help="Window length in seconds."
    )

    parser.add_argument(
        "--step",
        type=float,
        default=config.STEP_S,
        help="Step size in seconds."
    )

    parser.add_argument(
        "--highpass",
        type=float,
        default=config.CORR_HIGHPASS,
        help="Highpass frequency in Hz."
    )

    parser.add_argument(
        "--lowpass",
        type=float,
        default=config.CORR_LOWPASS,
        help="Lowpass frequency in Hz."
    )

    parser.add_argument(
        "--target-sfreq",
        type=int,
        default=config.TARGET_SFREQ,
        help="Target sampling frequency after downsampling."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "corr_matrices",
        help="Directory where output matrices will be saved."
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Stop after N windows. Useful for testing."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, do not save files."
    )

    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================
def ensure_dir_exists(dirpath):
    dirpath = Path(dirpath)
    if not dirpath.exists():
        print(f"Creating directory: {dirpath}")
        dirpath.mkdir(parents=True, exist_ok=True)


def load_raw_file(file_path):
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_fif(fname=file_path, preload=False, verbose=False)

    raw.set_montage(
        mne.channels.make_standard_montage(config.DEFAULT_MONTAGE),
        on_missing="ignore"
    )
    return raw



def filter_and_downsample(snippet, old_freq, new_freq, highpass, lowpass):
    filtered = butter(
        snippet,
        highpass_frequency=highpass,
        lowpass_frequency=lowpass,
        filter_function="sosfiltfilt",
        sampling_frequency=old_freq,
        axis=1,
        order=6,
    )

    decim_factor = int(old_freq / new_freq)
    if decim_factor < 1:
        raise ValueError(
            f"Target sampling frequency ({new_freq}) cannot be higher than original ({old_freq})."
        )

    downsampled = decimate(
        filtered,
        q=decim_factor,
        ftype="fir",
        axis=1,
        zero_phase=True,
    )

    return downsampled


# =============================================================================
# PER SUBJECT
# =============================================================================
def run_subject(subject, file_path, args):
    print("=" * 60)
    print(f"Subject: {subject}")
    print(f"File: {file_path}")
    print("=" * 60)

    raw = load_raw_file(file_path)

    old_freq = int(raw.info["sfreq"])
    new_freq = args.target_sfreq

    len_snippet = args.snippet_len
    step = args.step

    n_samples = raw.n_times
    window_samples = int(len_snippet * old_freq)
    step_samples = int(step * old_freq)

    folder_name = (
        f"sf{new_freq:g}"
        f"_win{len_snippet:g}"
        f"_step{step:g}"
        f"_bp{args.highpass:g}-{args.lowpass:g}"
    )
    
    out_dir = (
        Path(args.output_dir)
        / subject
        / args.recording
        / folder_name
    )
    
    ensure_dir_exists(out_dir)

    metadata = {
        "subject": subject,
        "recording": args.recording,
        "original_sfreq": old_freq,
        "target_sfreq": new_freq,
        "window_length_s": len_snippet,
        "step_s": step,
        "overlap_s": len_snippet - step,
        "highpass_Hz": args.highpass,
        "lowpass_Hz": args.lowpass,
        "n_channels": len(raw.ch_names),
        "channel_names": raw.ch_names,
        "metric": "Pearson correlation",
    }
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    total_duration = n_samples / old_freq
    print(f"Sampling rate: {old_freq} Hz")
    print(f"Recording length: {total_duration:.2f} s")
    print(f"Window length: {len_snippet} s")
    print(f"Step: {step} s")
    print(f"Bandpass: {args.highpass}-{args.lowpass} Hz")
    print(f"Downsampling to: {new_freq} Hz")

    t_zero = 0
    window_count = 0

    while t_zero + window_samples <= n_samples:
        t_sec = int(t_zero // old_freq)

        snippet = raw.get_data(
            picks = "eeg",
            start=t_zero,
            stop=t_zero + window_samples
        )

        processed = filter_and_downsample( # toto se dela na kazde elektrode samostatne 
            snippet=snippet,
            old_freq=old_freq,
            new_freq=new_freq,
            highpass=args.highpass,
            lowpass=args.lowpass,
        )

        matrix = np.corrcoef(processed)

        out_file = out_dir / f"corr_{t_sec:04d}.npy"

        if args.dry_run:
            print(f"[DRY RUN] Would save: {out_file}")
        else:
            np.save(out_file, matrix)
            print(f"Saved: {out_file}")

        t_zero += step_samples
        window_count += 1

        if args.max_windows is not None and window_count >= args.max_windows:
            print(f"Reached max windows: {args.max_windows}")
            break

    print(f"Finished {subject}: {window_count} matrices created.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    subjects, subject_file_map = resolveSubject.resolve_subjects(args)

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        arr_max = int(os.environ["SLURM_ARRAY_TASK_MAX"])
        array_size = arr_max + 1

        if array_size != len(subjects):
            raise ValueError(
                f"SLURM array size ({array_size}) != subjects found ({len(subjects)}). "
                "Fix sbatch array or subject discovery."
            )

        if task_id >= len(subjects):
            raise ValueError(
                f"Task ID {task_id} but only {len(subjects)} subjects found."
            )

        subject = subjects[task_id]
        file_path = subject_file_map[subject]
        run_subject(subject, file_path, args)

    else:
        for subject in subjects:
            run_subject(subject, subject_file_map[subject], args)

if __name__ == "__main__":
    main()
