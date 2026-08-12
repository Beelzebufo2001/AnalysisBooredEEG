"""
segment_statistics.py
=====================
Pure statistics — no plotting, no saving, just numbers.

Functions:
  basic_stats(seg)                        — mean, median, std, Q25/Q75, IQR, range, CV
  temporal_trend(seg)                     — linear fit + Pearson/Spearman on window means
  electrode_stats(seg, ch_names)          — node strength: strongest / weakest / most variable
  difference_stats(seg_before, seg_after) — Δ mean, Δ median, Δ std, Δ node strength

Saving:
  save_statistics(segments, metadata, out_dir, subject, recording, matrix_type, params)
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


# =============================================================================
# 1. BASIC DESCRIPTIVE STATS
# =============================================================================
def basic_stats(seg) -> dict:
    """
    Mean, median, std, Q25, Q75, IQR, range, CV
    on the upper-triangle values of the mean matrix.
    """
    if not seg.ok:
        return {"status": "no_data", "segment": seg.name}

    mm    = seg.mean_matrix
    upper = np.triu_indices(mm.shape[0], k=1)   # BUG FIX: was np.triu_indices(matrix, k=1) — needs .shape[0]
    v     = mm[upper]

    mean   = float(np.mean(v))
    std    = float(np.std(v))
    q25    = float(np.percentile(v, 25))
    q75    = float(np.percentile(v, 75))
    vmin   = float(np.min(v))
    vmax   = float(np.max(v))

    return {
        "status":  "ok",
        "segment": seg.name,
        "n_pairs": int(len(v)),
        "mean":    round(mean, 6),
        "median":  round(float(np.median(v)), 6),
        "std":     round(std, 6),
        "q25":     round(q25, 6),
        "q75":     round(q75, 6),
        "iqr":     round(q75 - q25, 6),
        "min":     round(vmin, 6),
        "max":     round(vmax, 6),
        "range":   round(vmax - vmin, 6),          # BUG FIX: was min - max (negative)
        "cv":      round(std / mean, 6) if mean != 0 else None,
    }


# =============================================================================
# 2. TEMPORAL TREND
# =============================================================================
def temporal_trend(seg) -> dict:
    """
    Linear fit y = ax + b on per-window means + Pearson r + Spearman rho.
    Tells you whether connectivity rises or falls within the segment.
    """
    if not seg.ok or seg.mean_windows is None or len(seg.mean_windows) < 3:
        return {"status": "no_data", "segment": seg.name}

    y = seg.mean_windows
    x = np.arange(len(y), dtype=float)

    # BUG FIX: was `stats.linregress` (undefined) and `recording.intercept` (nonsense)
    slope, intercept, r_pearson, p_pearson, _ = sp_stats.linregress(x, y)
    r_spearman, p_spearman = sp_stats.spearmanr(x, y)

    first = float(y[0])
    last  = float(y[-1])
    delta = last - first

    return {
        "status":          "ok",
        "segment":         seg.name,
        "n_windows":       int(len(y)),
        "first_window":    round(first, 6),
        "last_window":     round(last,  6),
        "delta":           round(delta, 6),
        "percent_change":  round((delta / first) * 100, 4) if first != 0 else None,
        "slope":           round(float(slope),     8),
        "intercept":       round(float(intercept), 6),
        "direction":       "rising" if slope > 0 else ("falling" if slope < 0 else "flat"),
        "pearson_r":       round(float(r_pearson),  4),
        "pearson_p":       round(float(p_pearson),  4),
        "spearman_r":      round(float(r_spearman), 4),
        "spearman_p":      round(float(p_spearman), 4),
    }


# =============================================================================
# 3. ELECTRODE STATS
# =============================================================================
def electrode_stats(seg, ch_names: list | None = None) -> dict:
    """
    Node strength = mean of each row in mean_matrix.
    Reports: top/bottom 5 electrodes + most variable.
    """
    if not seg.ok:
        return {"status": "no_data", "segment": seg.name}

    mm       = seg.mean_matrix
    strength = np.mean(mm, axis=1)
    n        = len(strength)
    top_n    = min(5, n)

    def ch(idx):
        return ch_names[int(idx)] if ch_names else str(int(idx))

    top_idx      = np.argsort(strength)[::-1][:top_n]
    bot_idx      = np.argsort(strength)[:top_n]
    variability  = np.std(mm, axis=1)
    most_var_idx = int(np.argmax(variability))

    return {
        "status":        "ok",
        "segment":       seg.name,
        "n_electrodes":  n,
        "strength_mean": round(float(np.mean(strength)), 5),
        "strength_std":  round(float(np.std(strength)),  5),
        "strength_min":  round(float(np.min(strength)),  5),
        "strength_max":  round(float(np.max(strength)),  5),
        "strongest": [
            {"electrode": ch(i), "strength": round(float(strength[i]), 5)}
            for i in top_idx
        ],
        "weakest": [
            {"electrode": ch(i), "strength": round(float(strength[i]), 5)}
            for i in bot_idx
        ],
        "most_variable": {
            "electrode": ch(most_var_idx),
            "std": round(float(variability[most_var_idx]), 5),
        },
    }


# =============================================================================
# 4. DIFFERENCE STATS
# =============================================================================
def difference_stats(seg_before, seg_after, ch_names: list | None = None) -> dict:
    """
    Δ between two segment mean matrices (upper triangle).
    Δ mean, Δ median, Δ std, Δ node strength, strongest changed connection.
    """
    if not (seg_before.ok and seg_after.ok):
        return {
            "status":     "no_data",
            "comparison": f"{seg_after.name} − {seg_before.name}",
        }

    mb   = seg_before.mean_matrix
    ma   = seg_after.mean_matrix
    diff = ma - mb

    upper = np.triu_indices(diff.shape[0], k=1)
    v     = diff[upper]

    strength_delta = np.mean(ma, axis=1) - np.mean(mb, axis=1)

    def ch(idx):
        return ch_names[int(idx)] if ch_names else str(int(idx))

    strongest_flat = int(np.argmax(np.abs(v)))
    rows, cols     = upper
    i_max, j_max   = int(rows[strongest_flat]), int(cols[strongest_flat])
    pos_pct        = float(np.mean(v > 0) * 100)

    return {
        "status":       "ok",
        "comparison":   f"{seg_after.name} − {seg_before.name}",
        "delta_mean":   round(float(np.mean(v)),   5),
        "delta_median": round(float(np.median(v)), 5),
        "delta_std":    round(float(np.std(v)),    5),
        "delta_iqr":    round(float(np.percentile(v, 75) - np.percentile(v, 25)), 5),
        "positive_pct": round(pos_pct,       2),
        "negative_pct": round(100 - pos_pct, 2),
        "delta_node_strength": {
            "mean": round(float(np.mean(strength_delta)), 5),
            "std":  round(float(np.std(strength_delta)),  5),
            "max_increase": {
                "electrode": ch(np.argmax(strength_delta)),
                "delta":     round(float(np.max(strength_delta)), 5),
            },
            "max_decrease": {
                "electrode": ch(np.argmin(strength_delta)),
                "delta":     round(float(np.min(strength_delta)), 5),
            },
        },
        "strongest_changed_connection": {
            "electrode_i": ch(i_max),
            "electrode_j": ch(j_max),
            "delta":       round(float(v[strongest_flat]), 5),
        },
    }


# =============================================================================
# COLLECT + SAVE
# =============================================================================
DIFF_PAIRS = [
    ("FEO", "LEO"),
    ("FEC", "LEC"),
    ("LEO", "FEC"),
    ("FEO", "LEC"),
]


def collect_statistics(segments, metadata) -> dict:
    ch_names = metadata.get("channel_names", None)

    per_segment = {
        name: {
            "basic":      basic_stats(seg),
            "trend":      temporal_trend(seg),
            "electrodes": electrode_stats(seg, ch_names),
        }
        for name, seg in segments.items()
    }

    differences = {
        f"{after}_minus_{before}": difference_stats(
            segments[before], segments[after], ch_names
        )
        for before, after in DIFF_PAIRS
        if before in segments and after in segments
    }

    return {"segments": per_segment, "differences": differences}


def save_statistics(segments, metadata, out_dir: Path,
                    subject, recording, matrix_type, params,
                    overwrite: bool = False) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{subject}_{recording}_{matrix_type}_{params}_statistics.json"

    if path.exists() and not overwrite:
        print(f"  [SKIP] already exists: {path}")
        return None

    result = collect_statistics(segments, metadata)
    result["_meta"] = {
        "subject":     subject,
        "recording":   recording,
        "matrix_type": matrix_type,
        "params":      params,
        "segments_info": {
            name: {
                "start_s":  seg.start,
                "end_s":    seg.end,
                "n_windows": seg.n_windows,
                "ok":       seg.ok,
            }
            for name, seg in segments.items()
        },
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"  Stats: {path}")
    return path