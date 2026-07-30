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
`
import config
# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        desctription="",
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
        nargs="+",
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

    def load_segment_matrices(self, matrices, metadata):
        #treba prevest sekundy na cislo matrice podle toho jake je meno parametru!
        step_s = metadata["step_s"]
        self.matrices = matrices[(self.start//step_s) : (self.end//step_s)+1]

    def compute_metrics(self):
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

    
    
if __name__ == "__main__":
    main() 
