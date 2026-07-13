#!/usr/bin/env python3
# =============================================================================
# Imports
# =============================================================================
import sys
import json
import argparse
import matplotlib
import mat

import config
# =============================================================================
# CLI
# =============================================================================
def argparse():
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
# Saving
# =============================================================================
def saveMetadata():

# =============================================================================
# Processing data
# =============================================================================
def stateWindows(matrices, subject):
    #koukni jestli neexistuje zaznam v EyesClosed.json -> apply config or apply Eyesclosed 
    with open("EyesCLosed.json", "r") as file:
        data[subject] = json.load(file)
        if ["excluded"] == "false":
            return smutek;

        feo = config.STATE_WINDOWS["FEO"]
        lec = config.STATE_WINDOWS["LEC"]
        before =     "RS_before": { "ec_start_s" },

        after =     "RS_after": { "ec_start_s": },

        if before == null 
        if after == null 

        
        #select our subject from data
    return feo, leo, fec, lec
# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    root = getRoot(args.matrix_type, args.subject, args.recording, args.params)
    matrices = load_matrices(root)
    metadata = load_metadata(root)

    feo, leo, fec, lec = stateWindows(matrices, args.subject)
    
if __name__ == "__main__":
    main()