"""
helpers.py — Your lab's EEG helper functions, adapted for CLI/cluster use.

Original authors: Marek Tobias, Karolina Korvasova.
Cluster adaptation: pipeline wrapper.

These replace the generic equivalents in utils.py for all subject discovery,
data loading, filtering, and downsampling steps.
"""

import os
import numpy as np
import scipy
import scipy.signal
import warnings

import mne

# antio is optional — only needed for .cnt files
try:
    from antio import read_cnt
    from antio.parser import read_triggers
    _ANTIO_AVAILABLE = True
except ImportError:
    _ANTIO_AVAILABLE = False


# ── Directory helpers ─────────────────────────────────────────────────────────

def ensure_dir_exists(dirpath: str):
    """
    Create a folder if it does not exist already.
    Written by Karolina Korvasova.
    """
    if not os.path.isdir(dirpath):
        print("Creating", dirpath)
        os.makedirs(dirpath)


# ── Subject discovery ─────────────────────────────────────────────────────────

def get_valid_subjects_and_paths(
    folder_path: str,
    clean_alg: str = "ica",
    exclude_subjects: list = [],
    files_idx: list = [0],
    logger=None,
) -> tuple:
    """
    Get valid subjects and their file paths from the specified folder.

    Expects structure: <folder_path>/<subject>/<clean_alg>/*<clean_alg>.fif

    Parameters
    ----------
    folder_path      : root directory containing subject subdirectories
    clean_alg        : cleaning algorithm label — 'raw', 'ica', 'atar',
                       or 'preprocessed-high-freq'
    exclude_subjects : list of subject IDs to skip
    files_idx        : which file index (or indices) to pick per subject.
                       Default [0] = first file only.
    logger           : optional logger; falls back to print() if None

    Returns
    -------
    subjects   : list of subject ID strings
    file_paths : list of corresponding absolute file path strings
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    log(f"Discovering subjects in '{folder_path}' (clean_alg='{clean_alg}') ...")
    if exclude_subjects:
        log(f"  Excluding: {exclude_subjects}")

    file_paths = []
    subjects   = []

    for subject in np.sort(os.listdir(folder_path)):
        subj_dir = os.path.join(folder_path, subject)
        alg_dir  = os.path.join(subj_dir, clean_alg)

        # Skip non-directories and subjects without the algorithm subfolder
        if not os.path.isdir(subj_dir) or not os.path.isdir(alg_dir):
            continue

        if subject in exclude_subjects:
            log(f"  Skipping excluded subject: {subject}")
            continue

        files = sorted(
            f for f in os.listdir(alg_dir)
            if f"{clean_alg}.fif" in f
        )

        if len(files) != 3:
            log(f"  Skipping {subject}: expected 3 .fif files, found {len(files)}")
            continue

        if len(files_idx) == 1:
            subjects.append(subject)
            file_paths.append(os.path.join(alg_dir, files[files_idx[0]]))
            log(f"  + {subject}  →  {files[files_idx[0]]}")
        else:
            for idx in files_idx:
                if idx < len(files):
                    tag = f"{subject}_{idx}"
                    subjects.append(tag)
                    file_paths.append(os.path.join(alg_dir, files[idx]))
                    log(f"  + {tag}  →  {files[idx]}")
                else:
                    log(f"  WARNING: {subject} has only {len(files)} files — skipping index {idx}")

    log(f"Found {len(subjects)} subject/file pairs.")
    return subjects, file_paths


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(file_path: str, logger=None):
    """
    Load EEG data from a .fif or .cnt file.

    Sets the standard_1005 montage and average reference projection.

    Parameters
    ----------
    file_path : path to the EEG file (.fif or .cnt)
    logger    : optional logger

    Returns
    -------
    raw              : mne.io.Raw
    imp_start        : dict  channel→impedance at recording start  (None for .fif)
    imp_end          : dict  channel→impedance at recording end    (None for .fif)
    sampling_rate    : int   Hz
    recording_length : int   seconds
    """
    def log(msg, level="info"):
        if logger:
            getattr(logger, level)(msg)
        else:
            print(msg)

    suffix = file_path.rsplit(".", 1)[-1].lower()
    if suffix not in ("cnt", "fif"):
        raise ValueError(f"Unsupported file format: .{suffix}  (expected .fif or .cnt)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if suffix == "cnt":
            if not _ANTIO_AVAILABLE:
                raise ImportError(
                    "antio is required to load .cnt files.\n"
                    "Install with:  pip install antio"
                )
            raw = mne.io.read_raw_ant(file_path, misc="EMG", verbose=False)
        else:  # .fif
            raw = mne.io.read_raw_fif(file_path, verbose=False)
            # Tag EMG channels by name
            emg_indices = [
                i for i, name in enumerate(raw.ch_names)
                if "emg" in name.lower()
            ]
            if emg_indices:
                emg_renames = {raw.ch_names[i]: "emg" for i in emg_indices}
                raw.set_channel_types(emg_renames)
                log(f"  Tagged {len(emg_indices)} EMG channel(s): "
                    f"{[raw.ch_names[i] for i in emg_indices]}")

    # Montage + average reference
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        on_missing="warn",
        verbose=False,
    )
    raw.set_eeg_reference("average", projection=True, verbose=False)

    # Impedances (cnt only)
    imp_start = imp_end = None
    if suffix == "cnt":
        cnt = read_cnt(file_path)
        _, _, _, impedances, _ = read_triggers(cnt)
        imp_start = {ch: impedances[0][k] for k, ch in enumerate(raw.ch_names)}
        imp_end   = {ch: impedances[0][k] for k, ch in enumerate(raw.ch_names)}

    sampling_rate    = int(raw.info["sfreq"])
    recording_length = int(raw.n_times) // sampling_rate

    log(f"  Loaded  : {file_path}")
    log(f"  Channels: {len(raw.ch_names)}  |  sfreq: {sampling_rate} Hz  "
        f"|  Duration: {recording_length} s")

    return raw, imp_start, imp_end, sampling_rate, recording_length


# ── Signal processing ─────────────────────────────────────────────────────────

def downsample_signal(
    X: np.ndarray,
    target_frequency: int = 200,
    sampling_rate: int = 8000,
) -> np.ndarray:
    """
    Downsample the signal to a target frequency using FIR anti-aliasing.

    Parameters
    ----------
    X                : (channels, samples) array
    target_frequency : desired output sample rate in Hz
    sampling_rate    : input sample rate in Hz

    Returns
    -------
    np.ndarray  shape (channels, new_samples)
    """
    factor = int(sampling_rate / target_frequency)
    if factor <= 1:
        return X  # nothing to do
    return scipy.signal.decimate(X, q=factor, axis=1, ftype="fir", zero_phase=True)


def filter_signal(
    X: np.ndarray,
    lowpass: float = 50.0,
    highpass: float = 1.0,
    sampling_rate: int = 8000,
    indices: list = None,
) -> np.ndarray:
    """
    Bandpass-filter the signal using elephant's Butterworth filter.

    Applies highpass and lowpass as two sequential sosfiltfilt passes.

    Parameters
    ----------
    X             : (channels, samples) array
    lowpass       : upper cutoff frequency in Hz
    highpass      : lower cutoff frequency in Hz
    sampling_rate : sample rate in Hz
    indices       : channel indices to filter (default: all)

    Returns
    -------
    np.ndarray  same shape as X
    """
    try:
        import elephant.signal_processing
    except ImportError:
        raise ImportError(
            "elephant is required for filter_signal.\n"
            "Install with:  pip install elephant"
        )

    assert lowpass > highpass, (
        f"lowpass ({lowpass}) must be > highpass ({highpass})"
    )

    if indices is None:
        indices = np.arange(X.shape[0])
    indices = np.asarray(indices)

    assert indices.max() < X.shape[0], (
        f"Channel index {indices.max()} out of bounds for array with "
        f"{X.shape[0]} channels."
    )

    X_filtered = X.copy()

    X_filtered[indices] = elephant.signal_processing.butter(
        X_filtered[indices],
        highpass_frequency=highpass,
        sampling_frequency=sampling_rate,
        filter_function="sosfiltfilt",
        axis=1,
    )
    X_filtered[indices] = elephant.signal_processing.butter(
        X_filtered[indices],
        lowpass_frequency=lowpass,
        sampling_frequency=sampling_rate,
        filter_function="sosfiltfilt",
        axis=1,
    )

    return X_filtered


def notch(sigs: list, notch_freq: float, sampling_rate: int) -> list:
    """
    Notch filter to remove power-line noise.
    Written by Karolina Korvasova.

    Parameters
    ----------
    sigs          : list of 1-D signal arrays
    notch_freq    : frequency to notch out (Hz), typically 50 or 60
    sampling_rate : sample rate in Hz

    Returns
    -------
    list of filtered signal arrays
    """
    b, a = scipy.signal.iirnotch(
        w0=notch_freq,
        Q=30.0,   # narrow notch
        fs=sampling_rate,
    )
    return [scipy.signal.filtfilt(b, a, sig) for sig in sigs]