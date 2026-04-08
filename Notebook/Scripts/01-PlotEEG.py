#!/usr/bin/env python3
"""
01_plot_eeg.py — Plot EEG traces and frequency bands from ICA-cleaned FIF files.

One --electrode argument handles everything — just like --subject:
  - 1 electrode  → single-trace plot  (+ bands if mode includes it)
  - 2+ electrodes → multi-electrode plot (bands always uses the first one)
  - omitted       → defaults to config.DEFAULT_SINGLE_ELECTRODE ("Cz")

Usage examples
--------------
# All subjects, all modes, default electrode Cz:
python 01_plot_eeg.py

# Specific subjects:
python 01_plot_eeg.py --subject C01 C03 C07

# Single electrode, custom window:
python 01_plot_eeg.py --subject C01 --electrode Cz --t-min 10 --t-max 11

# Multiple electrodes:
python 01_plot_eeg.py --subject C01 --electrode CP1 CP2 CP3 CP4

# Band decomposition only:
python 01_plot_eeg.py --subject C01 --mode bands --electrode Cz

# Dry run — print what would happen, write nothing:
python 01_plot_eeg.py --subject C01 --dry-run
"""

import mne
import matplotlib
matplotlib.use("Agg")   # non-interactive — safe on cluster, no display needed
import matplotlib.pyplot as plt
from elephant.signal_processing import butter
from pathlib import Path
import warnings as w
import argparse
import sys

import config


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot EEG traces and frequency bands from FIF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        default=None,
        help="One or more subject IDs e.g. C01 C02. Omit to process all subjects."
    )
    parser.add_argument(
        "--electrode",
        nargs="+",
        default=None,
        help=(
            "One or more electrode names. "
            "1 electrode → single-trace plot. "
            "2+ electrodes → multi-electrode plot. "
            f"Default: {config.DEFAULT_SINGLE_ELECTRODE}"
        )
    )
    parser.add_argument(
        "--recording",
        default="RS_before",
        choices=["RS_before", "Task", "RS_after"],
        help="Which recording to load per subject."
    )
    parser.add_argument("--t-min", type=float, default=config.DEFAULT_T_MIN)
    parser.add_argument("--t-max", type=float, default=config.DEFAULT_T_MAX)
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "bands", "all"],
        default="all",
        help="Which plot(s) to generate."
    )
    parser.add_argument("--output-dir",  type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--output-name", type=str,  default=None,
                        help="Custom filename (only useful when generating a single plot).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done; write nothing.")
    return parser.parse_args()


def resolve_electrodes(args, raw):
    """
    Return (single_electrode, electrode_list) from args.electrode.

    - If the user passed nothing → use config default as the single electrode,
      and config.ALL_ELECTRODES as the list for multi plots.
    - If the user passed 1 name  → that's the single electrode; list = [it].
    - If the user passed 2+      → first is the single electrode; list = all of them.

    Validates every requested electrode exists in the recording.
    """
    if args.electrode is None:
        requested = [config.DEFAULT_SINGLE_ELECTRODE]
        use_all_for_multi = True   # multi plot uses full ALL_ELECTRODES list
    else:
        requested = args.electrode
        use_all_for_multi = False

    # Validate — warn and skip anything not in the recording
    valid   = [e for e in requested if e in raw.ch_names]
    missing = [e for e in requested if e not in raw.ch_names]
    if missing:
        print(f"  Warning: electrode(s) not in recording, skipping: {missing}")
    if not valid:
        raise ValueError(
            f"None of the requested electrodes {requested} exist in this recording."
        )

    single_electrode = valid[0]

    if use_all_for_multi:
        # Default multi-plot: all electrodes from config that exist in the recording
        electrode_list = [e for e in config.ALL_ELECTRODES if e in raw.ch_names]
    else:
        electrode_list = valid

    return single_electrode, electrode_list


# =============================================================================
# Subject / file helpers
# =============================================================================

def ensure_dir_exists(dirpath: Path):
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        print(f"Creating {dirpath}")
            dirpath.mkdir(parents=True, exist_ok=True)


def get_valid_subjects_and_paths(preferred, folder_path=None, clean_alg=None, recording="RS_before"):
    """
    Return (subjects, file_paths) for all valid matching subjects.

    - Only folders starting with 'C'.
    - preferred = "all"  or  list of IDs like ['C01', 'C03'].
    - Picks the first .fif matching <recording> and <clean_alg>, ignoring -1.fif splits.
    """
    folder_path = Path(folder_path or config.DATA_DIR)
    clean_alg   = clean_alg or config.DATA_TYPE

    print(f"Discovering subjects in '{folder_path}' (alg='{clean_alg}', rec='{recording}') ...")

    subjects, file_paths = [], []

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
            print(f"  Skipping {subject}: no folder '{clean_dir}'")
            continue

        matching = sorted([
            f for f in clean_dir.iterdir()
            if f.is_file()
            and f.suffix == ".fif"
            and recording in f.name
            and clean_alg in f.name
            and not f.stem.endswith("-1")   # MNE split files — ignored
        ])

        if not matching:
            print(f"  Skipping {subject}: no .fif for '{recording}'")
            continue

        subjects.append(subject)
        file_paths.append(matching[0])
        print(f"  + {subject}  →  {matching[0].name}")

    print(f"Found {len(subjects)} subjects.")
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
            print(f"ERROR: subjects not found or missing files: {missing}", file=sys.stderr)
            sys.exit(1)

    return subjects, subject_file_map


def build_output_path(output_dir, subject, mode, output_name=None, electrode=None):
    ensure_dir_exists(output_dir)
    if output_name is not None:
        filename = output_name
    elif electrode is not None:
        filename = f"{subject}_{mode}_{electrode}.png"
    else:
        filename = f"{subject}_{mode}.png"
    return Path(output_dir) / filename


# =============================================================================
# Data helpers
# =============================================================================

def load_raw_file(file_path: Path):
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_fif(fname=file_path, preload=False, verbose=False)
    raw.set_montage(
        mne.channels.make_standard_montage(config.DEFAULT_MONTAGE),
        on_missing="ignore",
    )
    return raw


def validate_time_window(raw, t_min, t_max):
    total = raw.n_times / raw.info["sfreq"]
    if t_min < 0:
        raise ValueError(f"t_min={t_min} must be >= 0")
    if t_max <= t_min:
        raise ValueError(f"t_max={t_max} must be > t_min={t_min}")
    if t_max > total:
        raise ValueError(f"t_max={t_max} exceeds recording length {total:.1f} s")


def get_data(raw, electrodes, t_min, t_max):
    """Load data for one or more electrodes. Always returns (n_channels, n_samples)."""
    sfreq = raw.info["sfreq"]
    return raw.get_data(
        picks=electrodes,
        start=int(t_min * sfreq),
        stop=int(t_max * sfreq),
    ), sfreq


# =============================================================================
# Plotting
# =============================================================================

def plot_single_electrode(data, electrode, t_min, t_max, output_path):
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    ax.plot(data[0], color="red", lw=0.5)
    ax.grid(alpha=0.5)
    ax.set_title(f"Electrode: {electrode}  |  {t_min}–{t_max} s")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (V)")
    fig.savefig(output_path, dpi=config.DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_multi_electrode(data, electrodes, t_min, t_max, output_path):
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    for i, electrode in enumerate(electrodes):
        ax.plot(data[i], lw=0.5, label=electrode)
    ax.grid(alpha=0.5)
    ax.set_title(f"Multiple electrodes  |  {t_min}–{t_max} s")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (V)")
    ax.legend(fontsize=7, ncol=4)
    fig.savefig(output_path, dpi=config.DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_bands(data, sampling_rate, electrode, t_min, t_max, output_path):
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    for band_name, (hp, lp, color) in config.FREQ_BANDS.items():
        kwargs = dict(
            highpass_frequency=hp,
            filter_function="sosfiltfilt",
            sampling_frequency=sampling_rate,
        )
        if lp is not None:
            kwargs["lowpass_frequency"] = lp
        filtered = butter(data, **kwargs)
        ax.plot(filtered[0], color=color, lw=0.7, label=band_name)
    ax.grid(alpha=0.5)
    ax.set_title(f"Frequency bands: {electrode}  |  {t_min}–{t_max} s")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (V)")
    ax.legend()
    fig.savefig(output_path, dpi=config.DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Per-subject runner
# =============================================================================

def run_subject(subject, file_path, args):
    print("=" * 60)
    print(f"Subject : {subject}")
    print(f"File    : {file_path}")
    print("=" * 60)

    raw = load_raw_file(file_path)
    validate_time_window(raw, args.t_min, args.t_max)

    # Resolve electrodes — single_electrode is always one name,
    # electrode_list is one-or-more depending on what the user passed.
    single_electrode, electrode_list = resolve_electrodes(args, raw)
    is_multi = len(electrode_list) > 1

    print(f"  Electrode(s): {electrode_list}")

    # ── Single trace ──────────────────────────────────────────────────────────
    # Shown when: mode is 'single' or 'all', AND only one electrode was given.
    # If multiple electrodes were given, 'single' mode still works on the first.
    if args.mode in ("single", "all"):
        data, sfreq = get_data(raw, [single_electrode], args.t_min, args.t_max)
        out = build_output_path(
            args.output_dir, subject, "single",
            output_name=args.output_name if args.mode == "single" else None,
            electrode=single_electrode,
        )
        if args.dry_run:
            print(f"  [DRY-RUN] single → {out}")
        else:
            plot_single_electrode(data, single_electrode, args.t_min, args.t_max, out)
            print(f"  Saved: {out}")

    # ── Multi-electrode ───────────────────────────────────────────────────────
    # Shown when: mode is 'multi' or 'all'.
    # If only 1 electrode was given and mode is 'all', we skip this gracefully.
    if args.mode in ("multi", "all"):
        if not is_multi and args.mode == "all":
            print("  Skipping multi-electrode plot (only 1 electrode specified).")
        else:
            data, _ = get_data(raw, electrode_list, args.t_min, args.t_max)
            out = build_output_path(
                args.output_dir, subject, "multi",
                output_name=args.output_name if args.mode == "multi" else None,
            )
            if args.dry_run:
                print(f"  [DRY-RUN] multi → {out}")
            else:
                plot_multi_electrode(data, electrode_list, args.t_min, args.t_max, out)
                print(f"  Saved: {out}")

    # ── Frequency bands ───────────────────────────────────────────────────────
    # Always uses single_electrode (first electrode if multiple given).
    if args.mode in ("bands", "all"):
        data, sfreq = get_data(raw, [single_electrode], args.t_min, args.t_max)
        out = build_output_path(
            args.output_dir, subject, "bands",
            output_name=args.output_name if args.mode == "bands" else None,
            electrode=single_electrode,
        )
        if args.dry_run:
            print(f"  [DRY-RUN] bands → {out}")
        else:
            plot_frequency_bands(data, sfreq, single_electrode, args.t_min, args.t_max, out)
            print(f"  Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be written.")

    subjects, subject_file_map = resolve_subjects(args)

    errors = []
    for subject in subjects:
        try:
            run_subject(subject, subject_file_map[subject], args)
        except Exception as e:
            print(f"ERROR {subject}: {e}", file=sys.stderr)
            errors.append(subject)

    if errors:
        print(f"\nFailed subjects: {errors}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
