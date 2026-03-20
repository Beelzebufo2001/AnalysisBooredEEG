"""
config.py — Shared constants for EEG pipeline.
Edit this file to change global defaults without touching individual scripts.
"""

# ── Sampling ──────────────────────────────────────────────────────────────────
TARGET_SFREQ = 250          # Hz — target sampling rate after decimation

# ── Bandpass for correlation matrices ─────────────────────────────────────────
CORR_HIGHPASS  = 1.0        # Hz
CORR_LOWPASS   = 45.0       # Hz
BUTTER_ORDER   = 6

# ── Snippet / windowing ───────────────────────────────────────────────────────
SNIPPET_LEN_S  = 10         # seconds per correlation window
STEP_S         = 1          # seconds between window starts

# ── Frequency bands (for plot_eeg band decomposition) ────────────────────────
FREQ_BANDS = {
    "Delta": (0.5,  4.0,  "pink"),
    "Theta": (4.0,  7.0,  "darkseagreen"),
    "Alpha": (8.0,  13.0, "darkmagenta"),
    "Beta":  (14.0, 30.0, "orange"),
    # "Gamma": (30.0, None, "Gainsboro"),   # uncomment to enable
}

# ── EEG montage ───────────────────────────────────────────────────────────────
MONTAGE = "standard_1005"

# ── Brain regions (channel name prefix rules) ─────────────────────────────────
REGION_PREFIXES = {
    "Frontal":   ("Fp", "AF", "F"),
    "Central":   ("FC", "C", "CP"),
    "Parietal":  ("CP", "P"),
    "Occipital": ("PO", "O", "I"),
    "Temporal":  ("T", "FT", "TP", "TTP", "FFT"),
    "Reference": ("M",),
}

# ── Plot style ────────────────────────────────────────────────────────────────
FIGSIZE_WIDE   = (15, 5)
FIGSIZE_SQUARE = (10, 6)
PLOT_DPI       = 150