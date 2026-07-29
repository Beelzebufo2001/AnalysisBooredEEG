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

def stateWindows(subject, recording, w_size):
    #koukni jestli neexistuje zaznam v EyesClosed.json -> apply config or apply Eyesclosed 
    with open("EyesCLosed.json", "r") as file:
        data[subject][recording] = json.load(file)
        
        if ["excluded"] == "true": # treba predelat json aby povedal ze pacienta prcam 
            print(f"Subject's {subject} recording {recording} is excluded from quantification")
            break;

        feo = config.STATE_WINDOWS["FEO"]
        lec = config.STATE_WINDOWS["LEC"]

        change = ["ec_start_s"]
        if change == "null"
            leo = config.STATE_WINDOWS["LEO"]
            fec = config.STATE_WINDOWS["FEC"]
        else
            leo = (change -2 - w_size ,change-2)
            fec = (chage + 2, chage + 2 + w_size)

        
        #select our subject from data
    return feo, leo, fec, lec

def loadSegments(matrices, metadata, feo, leo, fec, lec):
    #treba prevest sekundy na cislo matrice podle toho jake je meno parametru!
    step_s = metadata["step_s"]

    seg1 = matrices[(feo(0)/step_s): (feo(1)/step_s)+1]
    seg2 = matrices[(leo(0)/step_s): (leo(1)/step_s)+1]
    seg3 = matrices[(fec(0)/step_s): (fec(1)/step_s)+1]
    seg4 = matrices[(lec(0)/step_s): (lec(1)/step_s)+1]
    
    return seg1, seg2, seg3, seg4
    
# =============================================================================
# Saving
# =============================================================================
    #!/usr/bin/env python3
# =============================================================================
# Processing data
# =============================================================================

def compute_metrics(segments):
    mean = []
    std = []
    median = []
    q25 = []
    q75 = []
    for seg in segments:
        upper = np.triu_indices(seg.shape[1], k=1)
        mean.append(np.array([np.mean(m[upper]) for m in seg]))
        std.append(np.array([np.std(m[upper])  for m in seg]))
        median.append(np.array([np.median(m[upper]) for m in seg]))
        q25.append(np.array([np.percentile(m[upper], 25) for m in seg]))
        q75.append(np.array([np.percentile(m[upper], 75) for m in seg]))
        
    return mean, std, median, q25, q75


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    root = getRoot(args.matrix_type, args.subject, args.recording, args.params)
    matrices = load_matrices(root)
    metadata = load_metadata(root)

    feo, leo, fec, lec = stateWindows(args.subject, args.recording, args.length) # tuna mame useky co budeme v npy hledat 
    seg1, seg2, seg3, seg4 = loadSegments(matrices, metadata, feo, leo, fec, lec) #tuna mame actual segmenty matic

    mean, std, median, q25, q75 = compute_metrics([seg1, seg2, seg3, seg4])

    
    
if __name__ == "__main__":
    main() 