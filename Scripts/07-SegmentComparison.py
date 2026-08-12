#!/usr/bin/env python3
"""
======================
Per-subject focused analysis plots. Each function produces one PNG.
 
Plots:
  1. plot_matrix_heatmap            — averaged (N×N) matrix per segment
  2. plot_difference_heatmaps       — LEO-FEO, LEC-FEC, FEC-LEO, LEC-FEO
  3. print_biggest_electrode_changes — console ranking, no PNG
  4. plot_connectivity_distribution  — histogram of ALL upper-tri values per segment
  5. plot_mean_connectivity_dist     — histogram of per-window means
  6. plot_node_strength_dist         — histogram of per-electrode strength
  7. plot_difference_distribution    — histogram of Δ values between segment pairs
  8. plot_mean_matrix_distribution   — histogram of mean-matrix upper-tri values
"""

# =============================================================================
# Imports
# =============================================================================
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from dataclasses import dataclass, field
import warnings as w
from segment_statistics import save_statistics

import config
# =============================================================================
# CONSTANTS
# =============================================================================

# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix-type",
        default="corr_matrices",
        choices=["corr_matrices", "plv_matrices"],
        help="Which matrix folder to read from.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="C01",
        help="Selected from slurm array idc for fcs",
    )
    parser.add_argument(
        "--recording",
        default="RS_before",
        choices=["RS_before", "RS_after"],
        help="Recording type",
    )
    parser.add_argument(
        "--params",
        default="sf250_win10_step1_bp1-45",
        metavar="FOLDER",
        help="Parameter folder name, e.g. sf250_win10_step1_bp1-45. ",
    )
    parser.add_argument(
        "--length",
        default = 10,
        type = int,
        help = "Length of the quantified window."
    )
    parser.add_argument("--top-n", default=10, type=int,
                    help="Top N electrodes in ranking printout.")

    return parser.parse_args()
# =============================================================================
# Segment
# =============================================================================
    
@dataclass
class Segment:
    name: str
    start: int 
    end: int 

    matrices: np.ndarray | None = field(default=None, repr=False)

    mean : np.ndarray | None = None
    std: np.ndarray | None = None
    median: np.ndarray | None = None
    q25: np.ndarray | None = None
    q75: np.ndarray | None = None
    
    @property
    def n_windows(self):
        if self.matrices is None:
            return 0
        return len(self.matrices)

    @property
    def color(self) -> str:
        return config.SEG_COLORS[self.name]

    @property 
    def ok(self):
        return self.matrices is not None and len(self.matrices) > 0 
        
    @property
    def mean_matrix(self):
        return np.mean(self.matrices, axis=0) if self.ok else None#axis = 0 pro dimenzi matice, bez toho to posle jedno cislo misto matice -> prumer pres okna
    @property
    def mean_windows(self): #np.mean(self.matrices, axis=(1,2)) umele zvedani hodnoty ig
        if not self.ok:
            return None
        upper = np.triu_indices(self.matrices.shape[1], k=1)
        values = self.matrices[:,upper[0],upper[1]]
        return np.mean(values, axis = 1)
        
    def load(self, matrices: np.ndarray, step_s: int):
        i_start = self.start // step_s
        i_end   = self.end   // step_s     # exclusive
        self.matrices = matrices[i_start:i_end]
        if len(self.matrices) == 0:
            print(f"  WARNING [{self.name}]: no windows in [{self.start}, {self.end})s")
            self.matrices = None
        
""" 
   def load_segment_matrices(self, matrices, metadata):
        #treba prevest sekundy na cislo matrice podle toho jake je meno parametru!
        #step_s = metadata["step_s"]
        step_s = metadata.get("step_s", 1)
        self.matrices = matrices[(self.start//step_s) : (self.end//step_s)]

    def compute_metrics(self):
        if self.matrices is None:
            return
        upper = np.triu_indices(self.matrices.shape[1], k=1)
        self.mean = np.array([np.mean(m[upper]) for m in self.matrices])
        self.std = np.array([np.std(m[upper])  for m in self.matrices])
        self.median = np.array([np.median(m[upper]) for m in self.matrices])
        self.q25 = np.array([np.percentile(m[upper], 25) for m in self.matrices])
        self.q75 = np.array([np.percentile(m[upper], 75) for m in self.matrices])

    def scalar(self, matric):
        #GUMBUS WERRY COOL... NICE
        arr = getattr(self,metric)
        return float(np.mean(arr)) if arr is not None else None
"""

# =============================================================================
# IO
# =============================================================================
def getRoot(m_type, subj, reco, param):
    root = config.OUTPUT_DIR/m_type/subj/reco/param
    
    #kontrola if root exist ofc
    if not root.exists():
        raise FileNotFoundError(f"Matrix forlder not found: {root}")
    return root

def load_metadata(param_dir):
    p = param_dir / "metadata.json"
    return json.load(open(p)) if p.exists() else {} # nezavre se?

def load_matrices(param_dir):
    """Load all .npy files sorted by filename → (T, N, N) array."""
    files = sorted(param_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {param_dir}")
    return np.stack([np.load(f) for f in files])

#DEPRECATED >>>>>>>>>
"""
def read_eyes_closed(subject, recording, w_size, segments):
#   CO VŠECHNO TATO FUNCKE DĚLÁ:
#   otevírá šuplík, kouká do šuplíku, třídí šuplík a vytváří nové hromádky v novém systému 
    with open("EyesClosed.json", "r") as file:
        data = json.load(file)
        entry = data[subject][recording] 
        
        subject_entry = data.get(subject, {}) #GET JE BANGER PRO CRASH PROBLEMKY
     
        # top-level exclusion (e.g. C12, C14)
        if subject_entry.get("excluded", False):
            raise ValueError(f"Subject {subject} is excluded (top-level).")
     
        rec_entry = subject_entry.get(recording, {})
        
        # recording-level exclusion (e.g. C03 RS_after, C08 RS_before)
        if rec_entry.get("excluded", False):
            raise ValueError(f"Subject {subject} recording {recording} is excluded.")
            
        change = rec_entry.get("ec_start_s", None)
        
        if change is not None:
            segments["LEO"].start = change - 2 - w_size
            segments["LEO"].end   = change - 2
            
            segments["FEC"].start = change + 2
            segments["FEC"].end   = change + 2 + w_size
            
        else:
            print(f"  EC start: unknown — LEO/FEC using config defaults (midpoint).")
            
    return segments
    """
#<<<<<<<<<<<<<<<<<<<<<<<<<  
    
def load_ec(subject, recording):
    ec_path = Path(__file__).parent / "eyesClosed.json"
    if not ec_path.exists():
        return None
        
    data = json.load(open(ec_path))
    entry = data.get(subject, {})
    if entry.get("excluded", False):
        raise ValueError(f"{subject} is excluded.")
        
    rec = entry.get(recording, {})
    
    if rec.get("excluded", False):
        raise ValueError(f"{subject}/{recording} is excluded.")
        
    return rec.get("ec_start_s", None) #, ec_start_s is not NONE je silnejsi nez v buildu >> verified

def build_segments(matrices, metadata, ec_start_s, length):
    step_s  = metadata.get("step_s", 1)
    total_s = matrices.shape[0] * step_s
 
    if ec_start_s is None:
        ec_start_s = config.DEFAULT_EC_START_S
        gap = config.UNKNOWN_EC_GAP_S
        print(f"  EC unknown → using default EC position ({ec_start_s}s)")
    else:
        gap = config.EC_GAP_S
 
    buf = config.EDGE_BUFFER_S
    segs = {
        "FEO": Segment("FEO", buf,                             buf + length),
        "LEO": Segment("LEO", ec_start_s - gap - length,   ec_start_s - gap),
        "FEC": Segment("FEC", ec_start_s + gap,            ec_start_s + gap + length),
        "LEC": Segment("LEC", total_s - buf - length,          total_s - buf),
    }
    for s in segs.values():
        s.load(matrices, step_s)
    return segs, ec_start_s
    
# =============================================================================
# Save helper
# =============================================================================
def save(fig, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
 
def suptitle(fig, subject, recording, matrix_type, params, extra=""):
    fig.suptitle(
        f"{subject}  {recording}  {matrix_type}  {params}"
        + (f"\n{extra}" if extra else ""),
        fontsize=10, fontweight="bold",
    )

def stats_text(values) -> str:
    return (
        f"Mean   : {np.mean(values):.4f}\n"
        f"Median : {np.median(values):.4f}\n"
        f"Std    : {np.std(values):.4f}\n"
        f"Q25    : {np.percentile(values, 25):.4f}\n"
        f"Q75    : {np.percentile(values, 75):.4f}\n"
        f"Min    : {np.min(values):.4f}\n"
        f"Max    : {np.max(values):.4f}"
    )

def add_stats(ax, values, extra_lines=None):
    """Put monospace stats block on an axis that has been axis('off')."""
    text = stats_text(values)
    if extra_lines:
        text += "\n" + "\n".join(extra_lines)
    ax.text(0.05, 0.5, text, fontsize=10, va="center",
            family="monospace", transform=ax.transAxes)

# =============================================================================
# 1. MATRIX HEATMAPS
# =============================================================================
def plot_matrix_heatmap(segments, metadata,
                        subject, recording, matrix_type, params,
                        out_dir):
    """2×2 grid — averaged (N×N) matrix per segment, shared color limits (5–95th pct)."""
 
    seg_names = ["FEO", "LEO", "FEC", "LEC"]
 
    # shared color scale from percentiles (robust to outliers)
    all_values = np.concatenate([
        segments[l].mean_matrix.flatten()
        for l in seg_names if segments[l].ok
    ])
    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
 
    for ax, label in zip(axes, seg_names):
        seg = segments[label]
        if not seg.ok:
            ax.set_title(f"{label} — NO DATA"); ax.axis("off"); continue
 
        im = ax.imshow(seg.mean_matrix, cmap="magma",
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label, color=seg.color, fontweight="bold", fontsize=13)
        ax.set_xlabel("Electrodes")
        ax.set_ylabel("Electrodes")
 
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="Connectivity")
    suptitle(fig, subject, recording, matrix_type, params, "averaged connectivity matrices")
    return save(fig, out_dir, f"{subject}_matrix_heatmap.png")
 
 
# =============================================================================
# 2. DIFFERENCE HEATMAPS
# =============================================================================
def plot_difference_heatmaps(segments, metadata,
                             subject, recording, matrix_type, params,
                             out_dir):
    """
    2×2 — four pairwise differences between segment mean matrices.
    Shared symmetric color scale. Coolwarm: blue = decrease, red = increase.
    """
    comparisons = [
        ("FEO", "LEO"),   # drift within EO
        ("FEC", "LEC"),   # drift within EC
        ("LEO", "FEC"),   # EO→EC transition at boundary
        ("FEO", "LEC"),   # overall change
    ]
 
    differences, valid_pairs = [], []
    for before, after in comparisons:
        sb, sa = segments.get(before), segments.get(after)
        if sb and sb.ok and sa and sa.ok:
            differences.append(sa.mean_matrix - sb.mean_matrix)
            valid_pairs.append((before, after))
 
    if not differences:
        print("  No valid pairs for difference heatmaps."); return
 
    max_abs = max(np.abs(d).max() for d in differences)
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
 
    for ax, diff, (before, after) in zip(axes, differences, valid_pairs):
        im = ax.imshow(diff, cmap="coolwarm",
                       vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_title(f"{after} − {before}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Electrodes")
        ax.set_ylabel("Electrodes")
 
    # turn off any unused axes
    for ax in axes[len(differences):]:
        ax.axis("off")
 
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="Δ Connectivity")
    suptitle(fig, subject, recording, matrix_type, params, "difference heatmaps")
    return save(fig, out_dir, f"{subject}_difference_heatmaps.png")
 
 
# =============================================================================
# 3. BIGGEST ELECTRODE CHANGES  (console only)
# =============================================================================
def print_biggest_electrode_changes(segments, metadata, top_n=10, **_):
    """Print ranked list of electrodes by average absolute change per comparison."""
    comparisons = [
        ("FEO", "LEO"),
        ("FEC", "LEC"),
        ("LEO", "FEC"),
        ("FEO", "LEC"),
    ]
    ch_names = metadata.get("channel_names", None)
 
    for before, after in comparisons:
        sb, sa = segments.get(before), segments.get(after)
        if not (sb and sb.ok and sa and sa.ok):
            print(f"\n{after} − {before}: missing data, skipping"); continue
 
        diff   = sa.mean_matrix - sb.mean_matrix
        scores = np.mean(np.abs(diff), axis=0)   # mean abs change per electrode
        top    = np.argsort(scores)[::-1][:top_n]
 
        print(f"\n{'='*50}")
        print(f"{after} − {before}")
        print(f"{'='*50}")
        for rank, idx in enumerate(top, 1):
            name = ch_names[idx] if ch_names else str(idx)
            print(f"  {rank:2d}. {name:12s}  Δ = {scores[idx]:.5f}")
 
 
# =============================================================================
# 4. CONNECTIVITY DISTRIBUTION  (all upper-tri values, all windows)
# =============================================================================
def plot_connectivity_distribution(segments, metadata,
                                   subject, recording, matrix_type, params,
                                   out_dir):
    """
    4 rows × 2 cols — one row per segment.
    Left: histogram of ALL upper-triangle values across ALL windows.
    Right: stats text box.
    Shared x-axis range across all segments.
    """
    seg_names = ["FEO", "LEO", "FEC", "LEC"]
 
    # shared x range
    all_vals = []
    for label in seg_names:
        seg = segments[label]
        if not seg.ok: continue
        upper = np.triu_indices(seg.matrices.shape[1], k=1)
        all_vals.append(seg.matrices[:, upper[0], upper[1]].flatten())
    if not all_vals:
        print("  No data for connectivity distribution."); return
    all_flat = np.concatenate(all_vals)
    xmin, xmax = all_flat.min(), all_flat.max()
 
    fig, axes = plt.subplots(4, 2, figsize=(12, 12),
                             gridspec_kw={"width_ratios": [3, 1]})
 
    for row, label in enumerate(seg_names):
        seg = segments[label]
        ax_hist  = axes[row, 0]
        ax_stats = axes[row, 1]
        ax_stats.axis("off")
 
        if not seg.ok:
            ax_hist.set_title(f"{label} — NO DATA"); ax_hist.axis("off"); continue
 
        upper  = np.triu_indices(seg.matrices.shape[1], k=1)
        values = seg.matrices[:, upper[0], upper[1]].flatten()
 
        ax_hist.hist(values, bins=60, color=seg.color, alpha=0.85, edgecolor="none")
        ax_hist.axvline(np.mean(values),   color="black",  lw=1.2, linestyle="--", label="mean")
        ax_hist.axvline(np.median(values), color="#555555", lw=1.0, linestyle=":",  label="median")
        ax_hist.set_xlim(xmin, xmax)
        ax_hist.set_title(label, color=seg.color, fontweight="bold")
        ax_hist.set_xlabel("Connectivity")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=7)
        add_stats(ax_stats, values)
 
    suptitle(fig, subject, recording, matrix_type, params,
             "connectivity distribution (all windows, upper triangle)")
    plt.tight_layout()
    return save(fig, out_dir, f"{subject}_connectivity_distribution.png")
 
 
# =============================================================================
# 5. MEAN CONNECTIVITY PER WINDOW  distribution
# =============================================================================
def plot_mean_connectivity_dist(segments, metadata,
                                subject, recording, matrix_type, params,
                                out_dir):
    """
    4 rows × 2 cols.
    Left: histogram of per-window mean connectivity values.
    Right: stats text.
    Shows whether mean connectivity is stable across windows in each segment.
    """
    seg_names = ["FEO", "LEO", "FEC", "LEC"]
 
    all_means = np.concatenate([
        segments[l].mean_windows for l in seg_names
        if segments[l].ok and segments[l].mean_windows is not None
    ])
    xmin, xmax = all_means.min(), all_means.max()
 
    fig, axes = plt.subplots(4, 2, figsize=(12, 12),
                             gridspec_kw={"width_ratios": [3, 1]})
 
    for row, label in enumerate(seg_names):
        seg = segments[label]
        ax_hist  = axes[row, 0]
        ax_stats = axes[row, 1]
        ax_stats.axis("off")
 
        if not seg.ok or seg.mean_windows is None:
            ax_hist.set_title(f"{label} — NO DATA"); ax_hist.axis("off"); continue
 
        values = seg.mean_windows
 
        ax_hist.hist(values, bins=max(5, seg.n_windows // 2),
                     color=seg.color, alpha=0.85, edgecolor="none")
        ax_hist.axvline(np.mean(values),   color="black",   lw=1.2, linestyle="--")
        ax_hist.axvline(np.median(values), color="#555555", lw=1.0, linestyle=":")
        ax_hist.set_xlim(xmin, xmax)
        ax_hist.set_title(label, color=seg.color, fontweight="bold")
        ax_hist.set_xlabel("Mean connectivity (per window)")
        ax_hist.set_ylabel("Count")
        add_stats(ax_stats, values)
     
    suptitle(fig, subject, recording, matrix_type, params,
             "mean connectivity distribution per window")
    plt.tight_layout()
    return save(fig, out_dir, f"{subject}_mean_connectivity_dist.png")
 
 
# =============================================================================
# 6. NODE STRENGTH  distribution
# =============================================================================
def plot_node_strength_dist(segments, metadata,
                            subject, recording, matrix_type, params,
                            out_dir):
    """
    4 rows × 2 cols.
    Left: histogram of node strength  (mean of each row in mean_matrix).
    Right: stats text.
    Shows which electrodes drive global connectivity in each segment.
    """
    seg_names = ["FEO", "LEO", "FEC", "LEC"]
 
    fig, axes = plt.subplots(4, 2, figsize=(12, 12),
                             gridspec_kw={"width_ratios": [3, 1]})
 
    # shared x range
    all_strengths = np.concatenate([
        np.mean(segments[l].mean_matrix, axis=1)
        for l in seg_names if segments[l].ok
    ])
    xmin, xmax = all_strengths.min(), all_strengths.max()
 
    for row, label in enumerate(seg_names):
        seg = segments[label]
        ax_hist  = axes[row, 0]
        ax_stats = axes[row, 1]
        ax_stats.axis("off")
 
        if not seg.ok:
            ax_hist.set_title(f"{label} — NO DATA"); ax_hist.axis("off"); continue
 
        strength = np.mean(seg.mean_matrix, axis=1)   # (N,)
 
        ax_hist.hist(strength, bins=30, color=seg.color, alpha=0.85, edgecolor="none")
        ax_hist.axvline(np.mean(strength),   color="black",   lw=1.2, linestyle="--")
        ax_hist.axvline(np.median(strength), color="#555555", lw=1.0, linestyle=":")
        ax_hist.set_xlim(xmin, xmax)
        ax_hist.set_title(f"{label} — node strength", color=seg.color, fontweight="bold")
        ax_hist.set_xlabel("Mean connectivity per electrode")
        ax_hist.set_ylabel("Number of electrodes")
 
        add_stats(ax_stats, strength)
        
    suptitle(fig, subject, recording, matrix_type, params,
             "node strength distribution")
    plt.tight_layout()
    return save(fig, out_dir, f"{subject}_node_strength_dist.png")
 
 
# =============================================================================
# 7. DIFFERENCE DISTRIBUTION  (upper-tri Δ values between pairs)
# =============================================================================
def plot_difference_distribution(segments, metadata,
                                 subject, recording, matrix_type, params,
                                 out_dir):
    """
    3 rows × 2 cols — histogram of Δ upper-triangle values per comparison.
    Shows whether change is mostly positive or negative and how spread it is.
    """
    comparisons = [
        ("FEO", "LEO"),
        ("FEC", "LEC"),
        ("LEO", "FEC"),
    ]
 
    valid = [(b, a) for b, a in comparisons
             if segments.get(b) and segments[b].ok
             and segments.get(a) and segments[a].ok]
 
    if not valid:
        print("  No valid pairs for difference distribution."); return
 
    fig, axes = plt.subplots(len(valid), 2, figsize=(12, 4 * len(valid)),
                             gridspec_kw={"width_ratios": [3, 1]})
    if len(valid) == 1:
        axes = [axes]   # make iterable
 
    for row, (before, after) in enumerate(valid):
        diff  = segments[after].mean_matrix - segments[before].mean_matrix
        upper = np.triu_indices(diff.shape[0], k=1)
        values = diff[upper]
        pos = np.mean(values > 0)*100
 
        ax_hist  = axes[row][0]
        ax_stats = axes[row][1]
        ax_stats.axis("off")
 
        ax_hist.hist(values, bins=60, color="mediumpurple", alpha=0.85, edgecolor="none")
        ax_hist.axvline(0,                 color="black",   lw=1.5, linestyle="--", label="zero")
        ax_hist.axvline(np.mean(values),   color="#cc3333", lw=1.2, linestyle="--", label="mean")
        ax_hist.axvline(np.median(values), color="#555555", lw=1.0, linestyle=":",  label="median")
        ax_hist.set_title(f"{after} − {before}", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel("Δ Connectivity")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=7)
 
        add_stats(ax_stats, values, extra_lines = [f"Positive: {pos:.1f}%", f"Negative: {100-pos:.1f}%"])
 
    suptitle(fig, subject, recording, matrix_type, params,
             "Δ connectivity distribution between segments")
    plt.tight_layout()
    return save(fig, out_dir, f"{subject}_difference_distribution.png")
 
 
# =============================================================================
# 8. MEAN MATRIX  distribution  (upper-tri of the single mean matrix)
# =============================================================================
def plot_mean_matrix_distribution(segments, metadata,
                                  subject, recording, matrix_type, params,
                                  out_dir):
    """
    4 rows × 2 cols.
    Left: histogram of upper-triangle values of the element-wise MEAN matrix.
    (Unlike plot_connectivity_distribution which uses all windows,
     this collapses time first and then shows the distribution of the average.)
    """
    seg_names = ["FEO", "LEO", "FEC", "LEC"]
 
    all_vals = np.concatenate([
        segments[l].mean_matrix[np.triu_indices(segments[l].mean_matrix.shape[0], k=1)]
        for l in seg_names if segments[l].ok
    ])
    xmin, xmax = all_vals.min(), all_vals.max()
 
    fig, axes = plt.subplots(4, 2, figsize=(12, 12),
                             gridspec_kw={"width_ratios": [3, 1]})
 
    for row, label in enumerate(seg_names):
        seg = segments[label]
        ax_hist  = axes[row, 0]
        ax_stats = axes[row, 1]
        ax_stats.axis("off")
 
        if not seg.ok:
            ax_hist.set_title(f"{label} — NO DATA"); ax_hist.axis("off"); continue
 
        mm     = seg.mean_matrix
        upper  = np.triu_indices(mm.shape[0], k=1)
        values = mm[upper]
 
        ax_hist.hist(values, bins=60, color=seg.color, alpha=0.85, edgecolor="none")
        ax_hist.axvline(np.mean(values),   color="black",   lw=1.2, linestyle="--")
        ax_hist.axvline(np.median(values), color="#555555", lw=1.0, linestyle=":")
        ax_hist.set_xlim(xmin, xmax)
        ax_hist.set_title(f"{label} — mean matrix", color=seg.color, fontweight="bold")
        ax_hist.set_xlabel("Connectivity")
        ax_hist.set_ylabel("Count")
 
        add_stats(ax_stats, values)
 
    suptitle(fig, subject, recording, matrix_type, params,
             "connectivity distribution of mean matrices")
    plt.tight_layout()
    return save(fig, out_dir, f"{subject}_mean_matrix_distribution.png")

# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    
    if args.subject is None:
        print("ERROR: --subject required.", file=sys.stderr); sys.exit(1)

    root = getRoot(args.matrix_type, args.subject, args.recording, args.params)
    matrices = load_matrices(root)
    metadata = load_metadata(root)
    
    try:
        ec_start_s = load_ec(args.subject, args.recording)
    except ValueError as e:
        print(f"  [SKIP] {e}"); sys.exit(0)

    segments, ec_s = build_segments(matrices, metadata, ec_start_s, args.length)

####segments = read_eyes_closed(args.subject, args.recording, args.length, segments)#####

    for label, seg in segments.items():
        status = f"OK  ({seg.n_windows} windows)" if seg.ok else "EMPTY"
        print(f"  {label}: [{seg.start}, {seg.end})s  →  {status}")


    out_dir = (
        config.SUMMARY_OUTPUT_DIR
        / "segment_figures"
        / args.matrix_type
        / args.recording
        / args.subject
    )

    
    shared = dict(
        subject=args.subject, recording=args.recording,
        matrix_type=args.matrix_type, params=args.params,
        segments=segments, metadata=metadata, out_dir=out_dir,
    )
    
    plot_matrix_heatmap(**shared)
    plot_difference_heatmaps(**shared)
    print_biggest_electrode_changes(top_n=args.top_n, **shared)
    plot_connectivity_distribution(**shared)
    plot_mean_connectivity_dist(**shared)
    plot_node_strength_dist(**shared)
    plot_difference_distribution(**shared)
    plot_mean_matrix_distribution(**shared)

    save_statistics(
        segments=segments,
        metadata=metadata,
        out_dir=out_dir,
        subject=args.subject,
        recording=args.recording,
        matrix_type=args.matrix_type,
        params=args.params,
    )

 
    print(f"\n  Done: {out_dir}")


    
    
if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main() 

