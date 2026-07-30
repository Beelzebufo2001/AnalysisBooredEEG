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
        default=None,
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
    return root

def load_metadata(param_dir):
    meta_path = param_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}

def load_matrices(param_dir):
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

    def load_segment_matrices(self, matrices, metadata):
        #treba prevest sekundy na cislo matrice podle toho jake je meno parametru!
        step_s = metadata["step_s"]
        self.matrices = matrices[(self.start//step_s) : (self.end//step_s)+1]

    def compute_metrics(self):
        if self.matrices is None:
            return
        upper = np.triu_indices(self.matrices.shape[1], k=1)
        self.mean = np.array([np.mean(m[upper]) for m in self.matrices])
        self.std = np.array([np.std(m[upper])  for m in self.matrices])
        self.median = np.array([np.median(m[upper]) for m in self.matrices])
        self.q25 = np.array([np.percentile(m[upper], 25) for m in self.matrices])
        self.q75 = np.array([np.percentile(m[upper], 75) for m in self.matrices])


        
def read_eyes_closed(subject, recording, w_size, segments):
    with open("EyesCLosed.json", "r") as file:
        data = json.load(file)
        entry = data[subject][recording] 
    
        change = entry["ec_start_s"]
        
        if entry["excluded"]: # treba predelat json aby povedal ze pacienta prcam 
            print(f"Subject's {subject} recording {recording} is excluded from quantification")
            raise ValueError("I dont wanna by by~")
    
        elif change is not None:
            segments["LEO"].start = change - 2 - w_size
            segments["LEO"].end   = change - 2
            
            segments["FEC"].start = change + 2
            segments["FEC"].end   = change + 2 + w_size
            
            
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


    save_segment_figure(
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

