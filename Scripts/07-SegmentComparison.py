#!/usr/bin/env python3
# =============================================================================
# Imports
# =============================================================================
import sys
import json
import argparse
import matplotlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings as w

import config
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
    return parser.parse_args()
    
# =============================================================================
# Reading
# =============================================================================
def getRoot(m_type, subj, reco, param):
    root = config.OUTPUT_DIR/m_type/subj/reco/param
    
    #kontrola if root exist ofc
    if not root.exists():
        raise FileNotFoundError(f"Matrix forlder not found: {root}")
    return root

def load_metadata(param_dir):
    meta_path = param_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}

def load_matrices(param_dir):
    """Load all .npy files sorted by filename → (T, N, N) array."""
    files = sorted(param_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {param_dir}")
    return np.stack([np.load(f) for f in files])
    
# =============================================================================
# Loading
# =============================================================================
from dataclasses import dataclass

SEG_COLORS = {
    "FEO": "#7aaddc",
    "LEO": "#4a8abf",
    "FEC": "#e07b8a",
    "LEC": "#b84f5f",
}
METRICS = ["mean", "std", "median", "q25", "q75"]

@dataclass
class Segment:
    name: str
    start: int 
    end: int 

    matrices : np.ndarray | None = None

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
        return SEG_COLORS[self.name]

    @property 
    def ok(self):
        return self.matrices is not None and len(self.matrices) > 0 

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

    def mean_matrix(self):
        return np.mean(self.matrices, axis=0) if self.ok else None #axis = 0 pro dimenzi matice, bez toho to posle jedno cislo misto matice -> prumer pres okna

    def scalar(self, matric):
        #GUMBUS WERRY COOL... NICE
        arr = getattr(self,metric)
        return float(np.mean(arr)) if arr is not None else None


        
def read_eyes_closed(subject, recording, w_size, segments):
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

    
# =============================================================================
# Saving
# =============================================================================
def save_segment_figure(subject, recording, matrix_type, params, segments, metadata, output_dir):
    """
    Save one overview figure per subject.

    Shows:
    - mean connectivity timeline for each segment
    - summary statistics
    - basic metadata
    """

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        constrained_layout=True
    )

    axes = axes.flatten()

    for ax, (name, seg) in zip(axes, segments.items()):

        if seg.matrices is None:
            ax.set_title(f"{name} - NO DATA")
            ax.axis("off")
            continue

        # x axis = window number
        x = np.arange(seg.n_windows)

        ax.plot(
            x,
            seg.mean,
            linewidth=1.5,
            label="mean"
        )

        ax.fill_between(
            x,
            seg.q25,
            seg.q75,
            alpha=0.3,
            label="Q25-Q75"
        )

        ax.set_title(
            f"{name}: {seg.start}-{seg.end}s\n"
            f"{seg.n_windows} windows"
        )

        ax.set_xlabel("Window")
        ax.set_ylabel("Connectivity")

        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)


    # Metadata text
    meta_text = (
        f"Subject: {subject}\n"
        f"Recording: {recording}\n"
        f"Matrix: {matrix_type}\n"
        f"Parameters: {params}\n\n"
        f"Step: {metadata.get('step_s','?')} s\n"
        f"Window length: {metadata.get('window_length_s','?')} s\n"
        f"Channels: {metadata.get('n_channels','?')}"
    )

    fig.text(
        0.01,
        0.01,
        meta_text,
        fontsize=9,
        verticalalignment="bottom"
    )


    fig.suptitle(
        f"{subject} | {recording} | {matrix_type}",
        fontsize=14
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{subject}_{recording}_{matrix_type}_{params}.png"
    )

    path = output_dir / filename

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved figure: {path}")

    return path

def save_segment_figure_hard(subject, recording, matrix_type, params,
                        segments, metadata, output_dir):
    """
    4-column comparison figure — one column per segment (FEO, LEO, FEC, LEC).
 
    Row 0  — averaged connectivity heatmap
    Row 1  — violin + boxplot of all upper-triangle values across windows
    Row 2  — mean connectivity timeline within the segment (window-by-window)
    Row 3  — statistics table (mean / std / median / Q25 / Q75 / min / max)
    """
    seg_names = ["FEO", "LEO", "FEC", "LEC"]
    n_cols    = len(seg_names)
 
    is_plv  = matrix_type == "plv_matrices"
    cmap    = "magma" if is_plv else "RdBu_r"
 
    # shared color limits from actual data
    all_means = [segments[n].mean_matrix() for n in seg_names
                 if segments[n].ok]
    if all_means:
        vmin = min(m.min() for m in all_means)
        vmax = max(m.max() for m in all_means)
    else:
        vmin, vmax = (-1, 1)
 
    fig = plt.figure(figsize=(5 * n_cols, 18), constrained_layout=False)
    fig.patch.set_facecolor("#fafafa")
 
    # grid: 4 rows, proportional heights
    gs = fig.add_gridspec(
        4, n_cols,
        height_ratios=[3, 2.5, 1.8, 1.6],
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.03,
    )
 
    STAT_ROWS = ["mean", "std", "median", "Q25", "Q75", "min", "max"]
 
    for col, name in enumerate(seg_names):
        seg  = segments[name]
        color = SEG_COLORS[name]
 
        # ── column header bar ─────────────────────────────────────────────────
        # attach a colored label just above row 0
        header_ax = fig.add_axes([
            gs[0, col].get_position(fig).x0,
            gs[0, col].get_position(fig).y1 + 0.002,
            gs[0, col].get_position(fig).width,
            0.022,
        ])
        header_ax.set_facecolor(color)
        header_ax.axis("off")
        label = name if not seg.ok else (
            f"{name}   [{seg.start}–{seg.end})s   n={seg.n_windows}"
        )
        header_ax.text(0.5, 0.5, label, ha="center", va="center",
                       fontsize=11, fontweight="bold", color="white",
                       transform=header_ax.transAxes)
 
        # ── ROW 0 — heatmap ───────────────────────────────────────────────────
        ax_heat = fig.add_subplot(gs[0, col])
        if not seg.ok:
            ax_heat.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                         fontsize=14, color="#aaaaaa",
                         transform=ax_heat.transAxes)
            ax_heat.axis("off")
        else:
            mm = seg.mean_matrix()
            im = ax_heat.imshow(mm, cmap=cmap, aspect="auto",
                                vmin=vmin, vmax=vmax)
            print(name, mm.shape, flush=True)
            ax_heat.axis("off")
            plt.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.02,
                         label="connectivity")
            ax_heat.set_title("avg matrix", fontsize=9, pad=4, color="#555555")
 
        # ── ROW 1 — violin + box ──────────────────────────────────────────────
        ax_vio = fig.add_subplot(gs[1, col])
        if not seg.ok:
            ax_vio.axis("off")
        else:
            upper = np.triu_indices(seg.matrices.shape[1], k=1)
            all_vals = np.concatenate([m[upper] for m in seg.matrices])
 
            # violin
            parts = ax_vio.violinplot(
                [all_vals], positions=[0],
                showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.45)
 
            # box
            bp = ax_vio.boxplot(
                [all_vals], positions=[0],
                widths=0.18,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                boxprops=dict(facecolor=color, alpha=0.7),
                whiskerprops=dict(color="#444444"),
                capprops=dict(color="#444444"),
                flierprops=dict(marker=".", markersize=2,
                                markerfacecolor="#888888", alpha=0.4),
            )
 
            # annotate key stats on the side
            q25, med, q75 = (np.percentile(all_vals, p) for p in (25, 50, 75))
            mn, mx = all_vals.min(), all_vals.max()
            for val, lbl, ha in [
                (mx,  f"max {mx:.3f}",  "left"),
                (q75, f"Q75 {q75:.3f}", "left"),
                (med, f"med {med:.3f}", "left"),
                (q25, f"Q25 {q25:.3f}", "left"),
                (mn,  f"min {mn:.3f}",  "left"),
            ]:
                ax_vio.axhline(val, color=color, linewidth=0.6,
                               linestyle="--", alpha=0.6)
                ax_vio.text(0.52, val, lbl, va="center", ha=ha,
                            fontsize=7, color="#333333",
                            transform=ax_vio.get_yaxis_transform())
 
            ax_vio.set_xlim(-0.45, 0.9)
            ax_vio.set_xticks([])
            ax_vio.set_ylabel("connectivity", fontsize=8)
            ax_vio.set_title("distribution", fontsize=9, color="#555555")
            ax_vio.grid(axis="y", alpha=0.2)
 
        # ── ROW 2 — within-segment timeline ───────────────────────────────────
        ax_tl = fig.add_subplot(gs[2, col])
        if not seg.ok:
            ax_tl.axis("off")
        else:
            x = np.arange(seg.n_windows)
            ax_tl.plot(x, seg.mean, color=color, linewidth=1.6)
            ax_tl.fill_between(x, seg.q25, seg.q75,
                               color=color, alpha=0.2, label="Q25–Q75")
            ax_tl.axhline(np.mean(seg.mean), color="#444444",
                          linewidth=0.9, linestyle="--", label="mean")
            ax_tl.set_xlabel("window index", fontsize=8)
            ax_tl.set_ylabel("mean conn.", fontsize=8)
            ax_tl.set_title("within-segment timeline", fontsize=9, color="#555555")
            ax_tl.tick_params(labelsize=7)
            ax_tl.legend(fontsize=7, loc="upper right")
            ax_tl.grid(alpha=0.18)
 
        # ── ROW 3 — stats table ───────────────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[3, col])
        ax_tbl.axis("off")
        if seg.ok:
            upper  = np.triu_indices(seg.matrices.shape[1], k=1)
            all_v  = np.concatenate([m[upper] for m in seg.matrices])
            values = [
                f"{np.mean(all_v):.5f}",
                f"{np.std(all_v):.5f}",
                f"{np.median(all_v):.5f}",
                f"{np.percentile(all_v, 25):.5f}",
                f"{np.percentile(all_v, 75):.5f}",
                f"{all_v.min():.5f}",
                f"{all_v.max():.5f}",
            ]
            tbl = ax_tbl.table(
                cellText=[[v] for v in values],
                rowLabels=STAT_ROWS,
                colLabels=["value"],
                cellLoc="center",
                rowLoc="right",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1.1, 1.55)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0 or c == -1:
                    cell.set_facecolor("#333333")
                    cell.get_text().set_color("white")
                    cell.get_text().set_fontweight("bold")
                else:
                    cell.set_facecolor(
                        matplotlib.colors.to_rgba(color, alpha=0.15))
 
    # ── main title ────────────────────────────────────────────────────────────
    step = metadata.get("step_s", "?")
    win  = metadata.get("window_length_s", "?")
    bp   = f"{metadata.get('highpass_Hz','?')}–{metadata.get('lowpass_Hz','?')} Hz"
    ch   = metadata.get("n_channels", "?")
 
    fig.suptitle(
        f"{subject}   {recording}   {matrix_type}   {params}\n"
        f"step={step}s   window={win}s   bandpass={bp}   channels={ch}",
        fontsize=12, fontweight="bold", y=0.975, color="#1a1a1a",
    )
 
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{subject}_{recording}_{matrix_type}_{params}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

        
# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    root = getRoot(args.matrix_type, args.subject, args.recording, args.params)
    matrices = load_matrices(root)
    metadata = load_metadata(root)

    segments = {
        "FEO": Segment("FEO", config.STATE_WINDOWS["FEO"][0], config.STATE_WINDOWS["FEO"][1]),
        
        "LEO": Segment("LEO", config.STATE_WINDOWS["LEO"][0], config.STATE_WINDOWS["LEO"][1]),
    
        "FEC": Segment("FEC", config.STATE_WINDOWS["FEC"][0], config.STATE_WINDOWS["FEC"][1]),
    
        "LEC": Segment("LEC", config.STATE_WINDOWS["LEC"][0], config.STATE_WINDOWS["LEC"][1]),

    }

    segments = read_eyes_closed(args.subject, args.recording, args.length, segments)

    for label, seg in segments.items():
        seg.load_segment_matrices(matrices, metadata) #tuna mame actual segmenty matic
        seg.compute_metrics()

    output_dir = (
        config.SUMMARY_OUTPUT_DIR
        / "segment_figures"
        / args.matrix_type
        / args.recording
        / args.subject
    )
 
    save_segment_figure_hard(
        subject=args.subject,
        recording=args.recording,
        matrix_type=args.matrix_type,
        params=args.params,
        segments=segments,
        metadata=metadata,
        output_dir=output_dir,
    )

    
    
if __name__ == "__main__":
    with w.catch_warnings():
        w.simplefilter("ignore", RuntimeWarning)
        main() 

