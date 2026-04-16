#!/usr/bin/env python3

# =============================================================================
# IMPORTS
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import mne

import config.py

# =============================================================================
# PARAMS
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
        "--lenght",
        type = int,
        default=config.DEFAULT_BRIDGE,
        help=""
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "bridges plot",
        help="Directory where output matrices will be saved."
    )
    parser.add_argument(
        "--target-freq",
        type=int,
        default=config.DEFAULT_SFREQ,
        help="Directory where output matrices will be saved."
    )
    parser.add_argument(
        "--lim-epochs",
        type=int,
        default=config.DEFAULT_EPOCH
    )
    parser.add_argument(
        "--lim-cutoff",
        type=int,
        default = config.DEFAULT_LMCUTT
    )

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
        raw = mne.io.read_raw_fif(fname=file_path, preload=True, verbose=False)

    raw.set_montage(
        mne.channels.make_standard_montage(config.DEFAULT_MONTAGE),
        on_missing="ignore"
    )
    return raw


def get_valid_subjects_and_paths(preferred, folder_path=None, clean_alg=None, recording="RS_before"):
    """
    Return matching subjects and their FIF paths.

    Rules:
    - only folders starting with 'C'
    - preferred can be 'all' or list like ['C01', 'C03']
    - choose files matching selected recording
    - ignore split files ending with '-1.fif'
    """
    folder_path = Path(folder_path or config.DATA_DIR)
    clean_alg = clean_alg or config.DATA_TYPE

    print(f"Discovering subjects in: {folder_path}")
    print(f"Cleaning algorithm: {clean_alg}")
    print(f"Recording: {recording}")

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
            print(f"Skipping {subject}: missing folder {clean_dir}")
            continue

        matching_files = sorted(
            [
                f for f in clean_dir.iterdir()
                if (
                    f.is_file()
                    and f.suffix == ".fif"
                    and recording in f.name
                    and clean_alg in f.name
                    and not f.stem.endswith("-1")
                )
            ]
        )

        if not matching_files:
            print(f"Skipping {subject}: no matching .fif file for {recording}")
            continue

        chosen = matching_files[0] # THIS TODOO FOR ELSE THAN 0 >> connect together 
        subjects.append(subject)
        file_paths.append(chosen)

        print(f"Adding {subject}: {chosen.name}")

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
            print(f"ERROR: Subjects not found or missing files: {missing}", file=sys.stderr)
            sys.exit(1)

    return subjects, subject_file_map
# =============================================================================
# PLOTING & SAVING
# =============================================================================
def plot_topomap(raw_crop, data, corr, bridged_idx, ed_matrix, subject_id="unknown", seconds, new_freq):
    # want to create dictionary
    return fig
    
def plot_correlation(corr, subject_id = "unknown", seconds, new_freq):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_title(f"Korelační matice EEG kanálů\nSubject: {subject_id} | posledních {seconds}s | {new_freq}Hz", fontsize=12)
    ax.set_xlabel("Kanál #")
    ax.set_ylabel("Kanál #")
    plt.colorbar(im, ax=ax, label="Pearsonova korelace")
    plt.tight_layout()
    plt.savefig(f"corr_matrix_{subject}.png", dpi=150)
    return fig

def plot_distribution(ed_matrix):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle("Subject 6 Electrical Distance Matrix Distribution")
    ax.hist(ed_matrix[~np.isnan(ed_matrix)], bins=np.linspace(0, 500, 51))
    ax.set_xlabel(r"Electrical Distance ($\mu$$V^2$)")
    ax.set_ylabel("Count (channel pairs for all epochs)")
    return fig
    
# =============================================================================
# PER SUBJECT
# =============================================================================
def run_subject(subject, subject_path, args):
    raw = load_raw_file(subject_pathct)

    #-------Cropping X second before-------
    tmax = raw.times[-1]
    seconds = args.lenght
    if(seconds > tmax): # rovno means we dont care...
        print("", flush=True)
        exit(1):
    tmin = tmax - seconds
    
    raw_crop = raw.copy().crop(tmin=tmin, tmax=tmax)

    #-------Only EEG but do we care ? 
    raw_crop = raw_crop.pick_types(eeg=True)
    
    #-------Downsample 
    new_freq = args.target_freq
    raw_crop.resample(new_freq) #what if bigger

    #-------Correlation if we want
    data = raw_crop.get_data()
    corr = np.corrcoef(data)

    #-------Bridges issues 
    lc = args.lim_cutoff
    et = args.lim_epoch
    bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(
        raw_crop,
        verbose = False,
        lm_cutoff = lc,
        epoch_treshold = et
    )
    #-------Ploting 
    out_dir = args.out_dir / subject
    out_dir.mkdir(parents = True, exist_ok = True)

    tag = "t" + seconds + "-lc" + lc + "-et" + et
    pdf_path = out_dir / f"{subject}_Bridges_{tag}.pdf"
    with PdfPages(pdf_path) as pdf:
        
        plot_top = plot_topomap(raw_crop, corr, data, bridged_idx, ed_matrix, seconds, new_freq)
        pdf.savefig(plot_top)
        plt.close(plot_top)
        
        plot_corr = plot_correlation(corr, subject_id = subject, seconds, new_freq)
        pdf.savefig(plot_corr)
        plt.close(plot_corr)
        
        plot_dis = plot_distribution(ed_matrix)
        pdf.savefig(plot_dis)
        plt.close(plot_dis)
        
        
        plt.close(plot_top, plot_corr)
    
    print(f"  Finished {subject} | saved to {pdf_path}", flush=True)
    
# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    subjects, subject_file_map = resolve_subjects(args)

    for subject in subjects:
        run_subject(subject, subject_file_map[subject], args)


if __name__ == "__main__":
    main()