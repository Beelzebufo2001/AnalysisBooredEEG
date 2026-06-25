#!/usr/bin/env python3

# =============================================================================
# IMPORTS
# =============================================================================
import sys
import json
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
    parser = argparse.ArgumentParser(
        description="Connectivity analysis — navigates results/<matrix_type>/<subject>/<recording>/<params>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--matrix-type",
        default="corr_matrices",
        choices=["corr_matrices", "plv_matrices"],
        help="Which matrix folder to read from.",
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject IDs (e.g. C01 C02). Default: all found.",
    )

    parser.add_argument(
        "--recording",
        default="RS_before",
        choices=["RS_before", "Task", "RS_after"],
        help="Which recording to use.",
    )

    parser.add_argument(
        "--params",
        default=None,
        metavar="FOLDER",
        help=(
            "Parameter folder name, e.g. sf250_win10_step1_bp1-45. "
            "If omitted and only one param folder exists per subject, it is selected automatically. "
            "Use --list-params to discover available folders."
        ),
    )

    parser.add_argument(
        "--list-params",
        action="store_true",
        help="Print all available parameter folders and exit.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed, do not save anything.",
    )

    return parser.parse_args()


# =============================================================================
# DISCOVERY
# =============================================================================
def get_matrix_root(matrix_type):
    root = config.OUTPUT_DIR / matrix_type
    if not root.exists():
        print(f"ERROR: Matrix directory not found: {root}", file=sys.stderr)
        sys.exit(1)
    return root


def discover_subjects(root, requested):
    found = {d.name: d for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith("C")}
    if not found:
        print(f"ERROR: No subject folders in {root}", file=sys.stderr)
        sys.exit(1)

    if requested is None:
        return list(found.keys()), found

    missing = [s for s in requested if s not in found]
    if missing:
        print(f"ERROR: Subjects not found: {missing}", file=sys.stderr)
        sys.exit(1)

    return requested, {s: found[s] for s in requested}


def resolve_param_folder(subject_dir, recording, params_arg):
    """
    Return the resolved param folder Path for one subject/recording,
    or None if missing.
    """
    rec_dir = subject_dir / recording
    if not rec_dir.exists():
        print(f"  [SKIP] {subject_dir.name}: no recording folder '{recording}'")
        return None

    available = sorted([d for d in rec_dir.iterdir() if d.is_dir()])
    if not available:
        print(f"  [SKIP] {subject_dir.name}/{recording}: no parameter folders found")
        return None

    if params_arg is not None:
        target = rec_dir / params_arg
        if not target.exists():
            print(f"  [SKIP] {subject_dir.name}/{recording}: folder '{params_arg}' not found")
            return None
        return target

    # Auto-select if unambiguous
    if len(available) == 1:
        return available[0]

    print(
        f"  [SKIP] {subject_dir.name}/{recording}: multiple param folders found, "
        f"specify --params. Options: {[d.name for d in available]}"
    )
    return None


def list_all_params(root, recording):
    print(f"\nAvailable parameter folders under {root} / <subject> / {recording}:\n")
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("C"):
            continue
        rec_dir = subject_dir / recording
        if not rec_dir.exists():
            continue
        folders = sorted([d.name for d in rec_dir.iterdir() if d.is_dir()])
        print(f"  {subject_dir.name}: {folders}")
    print()


# =============================================================================
# DATA LOADING
# =============================================================================
def load_matrices(param_dir):
    """Load all .npy matrices in order, return (T, N, N) array."""
    files = sorted(param_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {param_dir}")
    return np.stack([np.load(f) for f in files])


def load_metadata(param_dir):
    meta_path = param_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(connectivity):
    upper = np.triu_indices(connectivity.shape[1], k=1)
    mean_values  = np.array([np.mean(m[upper]) for m in connectivity])
    variability  = np.array([np.std(m[upper])  for m in connectivity])
    return mean_values, variability


# =============================================================================
# PLOTS
# =============================================================================
def save_subject_figure(subject, mean_values, variability, out_dir, params_name, matrix_type):
    mid    = len(mean_values) // 2
    first  = np.mean(mean_values[:mid])
    second = np.mean(mean_values[mid:])
    delta  = second - first

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.plot(mean_values, color="darkmagenta", linewidth=1.5)
    ax1.axvline(mid, color="black", linestyle="--", linewidth=1, label="Midpoint")
    ax1.set_ylabel("Mean connectivity")
    ax1.set_xlabel("Time window")
    ax1.legend(fontsize=8)

    ax2.plot(range(mid),                   variability[:mid],  color="khaki",     label="First half")
    ax2.plot(range(mid, len(variability)), variability[mid:],  color="steelblue", label="Second half")
    ax2.axvline(mid, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Std of pair connectivity")
    ax2.set_xlabel("Time window")
    ax2.legend(fontsize=8)

    fig.suptitle(
        f"{subject}  [{matrix_type}  {params_name}]   |   "
        f"mean={np.mean(mean_values):.4f}   "
        f"first-half={first:.4f}   second-half={second:.4f}   Δ={delta:+.4f}",
        fontsize=9,
    )
    fig.tight_layout()

    out_path = out_dir / f"{subject}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_summary_figure(subjects, deltas, out_dir, params_name, matrix_type):
    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.8), 4))
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]
    ax.bar(subjects, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Connectivity Δ (second half − first half)\n{matrix_type}  |  {params_name}")
    ax.set_ylabel("Δ mean connectivity")
    fig.tight_layout()

    # Name the summary after the parameters so multiple runs don't overwrite
    out_path = out_dir / f"summary_delta__{params_name}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\n  Summary saved: {out_path}")


# =============================================================================
# PER-SUBJECT
# =============================================================================
def run_subject(subject, param_dir, out_dir, params_name, matrix_type):
    connectivity         = load_matrices(param_dir)
    mean_values, variability = compute_metrics(connectivity)
    save_subject_figure(subject, mean_values, variability, out_dir, params_name, matrix_type)

    mid   = len(mean_values) // 2
    delta = float(np.mean(mean_values[mid:])) - float(np.mean(mean_values[:mid]))
    print(f"  {subject}  windows={len(connectivity)}  delta={delta:+.4f}", flush=True)
    return delta


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    root = get_matrix_root(args.matrix_type)

    if args.list_params:
        list_all_params(root, args.recording)
        sys.exit(0)

    subjects, subject_dir_map = discover_subjects(root, args.subjects)

    # Output goes to:
    #   results/connectivity_summary/<matrix_type>/<recording>/<params>/
    # so different parameter runs are fully separated and never overwrite.
    params_label = args.params or "auto"
    out_dir = (
        config.SUMMARY_OUTPUT_DIR
        / args.matrix_type
        / args.recording
        / params_label
    )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMatrix type : {args.matrix_type}")
    print(f"Recording   : {args.recording}")
    print(f"Params      : {args.params or '(auto)'}")
    print(f"Subjects    : {subjects}")
    print(f"Output dir  : {out_dir}\n")

    deltas       = []
    params_name  = args.params or ""   # filled in from first resolved folder if auto

    for subject in subjects:
        param_dir = resolve_param_folder(subject_dir_map[subject], args.recording, args.params)
        if param_dir is None:
            continue

        # Capture the actual folder name (important when --params was not given)
        if not params_name:
            params_name = param_dir.name

        print(f"[{subject}]  {param_dir}")

        if args.dry_run:
            print(f"  [DRY RUN] Would process {param_dir}")
            continue

        try:
            delta = run_subject(subject, param_dir, out_dir, param_dir.name, args.matrix_type)
            deltas.append((subject, delta))
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", flush=True)

    if not args.dry_run and len(deltas) > 1:
        save_summary_figure(
            [s for s, _ in deltas],
            [d for _, d in deltas],
            out_dir,
            params_name,
            args.matrix_type,
        )

    print(f"\nDone. Results in: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main()