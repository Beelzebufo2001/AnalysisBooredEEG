from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = (BASE_DIR / "../../EEG/MS_data").resolve()
OUTPUT_DIR = (BASE_DIR / "../results").resolve()
CORR_OUTPUT_DIR = Path( OUTPUT_DIR / "corr_matrices")
VIDEO_OUTPUT_DIR = Path("./Corr_videos")
SUMMARY_OUTPUT_DIR = Path(OUTPUT_DIR / "connectivity_summary")
REGION_OUTPUT_DIR = Path(OUTPUT_DIR / "Region_connectivity")

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_TYPE    = "ica"
SUBJECT_GLOB = "C*"

# ── Plot defaults ─────────────────────────────────────────────────────────────
DEFAULT_T_MIN            = 0
DEFAULT_T_MAX            = 1
DEFAULT_SINGLE_ELECTRODE = "Cz"   # used when --electrode is not specified
FIGSIZE_WIDE             = (15, 5)
DEFAULT_DPI              = 200
DEFAULT_YLIM = (-1.0, 1.0)
DEFAULT_STD_YLIM = (0.0, 1.0)

# BUG FIX 6: MNE montage names are case-sensitive — must be lowercase 's'
DEFAULT_MONTAGE = "standard_1005"

# ── Frequency bands ───────────────────────────────────────────────────────────
# Format: band_name: (highpass_Hz, lowpass_Hz, color)
# BUG FIX 5: Gamma previously had lowpass=None which crashes elephant.butter.
# Set to 80 Hz (reasonable upper limit for 8kHz data downsampled to 250 Hz).
FREQ_BANDS = {
    "Gamma": (30,  80,  "gainsboro"),
    "Beta":  (14,  30,  "orange"),
    "Alpha": (8,   13,  "darkmagenta"),
    "Theta": (4,   7,   "darkseagreen"),
    "Delta": (0.5, 4,   "pink"),
}

# ── Correlation matrix computation (02_compute_corr.py) ──────────────────────
TARGET_SFREQ   = 250    # Hz — downsample target
CORR_HIGHPASS  = 1.0    # Hz
CORR_LOWPASS   = 45.0   # Hz
#BUTTER_ORDER   = 6
SNIPPET_LEN_S  = 10     # seconds per correlation window
STEP_S         = 1      # seconds between windows

# ── Region prefixes (03_analyze_connectivity.py) ──────────────────────────────
REGION_PREFIXES = {
    "Frontal":   ("Fp", "AF", "F"),
    "Central":   ("FC", "C", "CP"),
    "Parietal":  ("CP", "P"),
    "Occipital": ("PO", "O", "I"),
    "Temporal":  ("T", "FT", "TP", "TTP", "FFT"),
    "Reference": ("M",),
}

# ── Full electrode list (standard_1005 layout for this dataset) ───────────────
ALL_ELECTRODES = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
    'POz', 'O1', 'O2',
    'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
    'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
    'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6',
    'PO3', 'PO4', 'FT7', 'FT8', 'TP7', 'TP8',
    'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h',
    'PO9', 'PO10', 'P9', 'P10',
    'AFF1', 'AFz', 'AFF2',
    'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
    'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
    'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
    'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h',
    'PPO1', 'PPO2', 'I1', 'Iz', 'I2',
    'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h',
    'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h',
    'FTT9h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h',
    'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h',
    'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h',
    'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h',
    'POO9h', 'POO3h', 'POO4h', 'POO10h',
    'OI1h', 'OI2h',
    # 'EMG',   # intentionally excluded from EEG plots
]
# ── Region colors ─────────────────────────────────────────────────────────────
REGION_COLORS = {
    "Frontal":   "#e07b8a",   # pink
    "Central":   "#6abf8a",   # green
    "Parietal":  "#7aaddc",   # blue
    "Occipital": "#c99de0",   # purple
    "Temporal":  "#e0b86a",   # amber
    "Reference": "#a0a0a0",   # grey
    "left":    "#7aaddc",   # blue
    "right":   "#e07b8a",   # pink
    "midline": "#a0a0a0",   # grey

}
 
# ── Hemisphere colors ─────────────────────────────────────────────────────────
HEMISPHERE_COLORS = {
    "left":    "#7aaddc",   # blue
    "right":   "#e07b8a",   # pink
    "midline": "#a0a0a0",   # grey
}
