#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — SLURM master wrapper for the EEG connectivity pipeline
#
# Submits each of the 4 pipeline steps as SLURM jobs with proper
# afterok dependencies (step N only runs if step N-1 succeeded).
#
# Cluster notes (from your sinfo / sbatch docs):
#   - Use sbatch for non-interactive batch jobs (preferred)
#   - Use srun --cpus-per-task=N for interactive / direct commands
#   - Check jobs with:  squeue
#   - Cancel a job:     scancel <JOBID>
#   - See nodes:        sinfo
#   - Default CPUs if unspecified: only 2 — always set --cpus-per-task
#
# Usage
# -----
#   # Full pipeline, all subjects:
#   bash run_pipeline.sh --data-dir /data/eeg --output-dir /results/my_study
#
#   # Single subject:
#   bash run_pipeline.sh --data-dir /data/eeg --output-dir /results/my_study --subject C01
#
#   # Exclude bad subjects:
#   bash run_pipeline.sh --data-dir /data/eeg --output-dir /results/my_study \
#       --exclude P03 P04 C16 C24
#
#   # Only run step 2 (the heavy compute step):
#   bash run_pipeline.sh --data-dir /data/eeg --output-dir /results/my_study --step 2
#
#   # Quick test — 20 windows, single subject:
#   bash run_pipeline.sh --data-dir /data/eeg --output-dir /results/test \
#       --subject C01 --step 2 --max-windows 20
#
#   # Dry run (print sbatch commands, submit nothing):
#   bash run_pipeline.sh --data-dir /data/eeg --output-dir /results/my_study --dry-run
#
#   # Interactive debugging on a node (blocks until you exit):
#   srun --pty --cpus-per-task=4 bash
#   then:  conda activate eeg && python 02_compute_corr.py --subject C01 ...
# =============================================================================

set -euo pipefail

# =============================================================================
# ── CONFIG — EDIT BEFORE FIRST RUN ───────────────────────────────────────────
# =============================================================================

CONDA_ENV="eeg"           # your conda environment name
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SLURM settings — adjust to your cluster
PARTITION="cpu"           # check available partitions with: sinfo
ACCOUNT=""                # your SLURM account, leave empty if not required
EMAIL=""                  # job notifications — leave empty to disable
EMAIL_TYPE="END,FAIL"     # SLURM --mail-type value

# Per-step resources
# Step 2 (compute_corr) is the heavy one — give it the most RAM and time.
# All scripts reserve CPUs explicitly (cluster default is only 2).
declare -A STEP_CPUS=(  [1]=4   [2]=16   [3]=4    [4]=4    )
declare -A STEP_MEM=(   [1]="8G" [2]="64G" [3]="16G" [4]="16G" )
declare -A STEP_TIME=(  [1]="00:30:00" [2]="08:00:00" [3]="01:00:00" [4]="02:00:00" )
declare -A STEP_NAME=(  [1]="01_plot_eeg" [2]="02_compute_corr" [3]="03_connectivity" [4]="04_plot_pairs" )

# =============================================================================
# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
# =============================================================================

DATA_DIR=""
OUTPUT_DIR=""
SUBJECT=""
STEP=""
CLEAN_ALG="ica"
FILES_IDX="0"           # space-separated indices, e.g. "0 2"
EXCLUDE=""              # space-separated subject IDs to exclude
DRY_RUN=0
SKIP_EXISTING=0
MAX_WINDOWS=""
CH_NAMES_FILE=""

usage() {
    grep "^#" "$0" | grep -v "^#!/" | sed 's/^# \{0,3\}//' | head -40
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)      DATA_DIR="$2";        shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";      shift 2 ;;
        --subject)       SUBJECT="$2";         shift 2 ;;
        --step)          STEP="$2";            shift 2 ;;
        --clean-alg)     CLEAN_ALG="$2";       shift 2 ;;
        --files-idx)     FILES_IDX="$2";       shift 2 ;;
        --exclude)
            shift
            EXCLUDE=""
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXCLUDE="$EXCLUDE $1"; shift
            done
            ;;
        --max-windows)   MAX_WINDOWS="$2";     shift 2 ;;
        --ch-names-file) CH_NAMES_FILE="$2";   shift 2 ;;
        --skip-existing) SKIP_EXISTING=1;      shift ;;
        --dry-run)       DRY_RUN=1;            shift ;;
        --help|-h)       usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$DATA_DIR"   ]] && { echo "ERROR: --data-dir is required";   exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "ERROR: --output-dir is required"; exit 1; }

# Derived paths
CORR_DIR="${OUTPUT_DIR}/corr_matrices"
QC_DIR="${OUTPUT_DIR}/qc_plots"
CONN_DIR="${OUTPUT_DIR}/connectivity"
PAIRS_DIR="${OUTPUT_DIR}/pair_plots"
LOG_DIR="${OUTPUT_DIR}/logs/slurm"
mkdir -p "$LOG_DIR"

# =============================================================================
# ── HELPER: build and submit one sbatch job ───────────────────────────────────
# =============================================================================

# submit <step_num> <py_script> <extra_python_args> <dep_jobid>
# Returns the new SLURM job ID on stdout.
submit() {
    local step_num="$1"
    local py_script="$2"
    local extra_args="$3"
    local dep_jobid="$4"

    local job_name="${STEP_NAME[$step_num]}"
    local log_out="${LOG_DIR}/${job_name}_%j.out"
    local log_err="${LOG_DIR}/${job_name}_%j.err"

    # Build optional flags
    local subj_flag="";    [[ -n "$SUBJECT"       ]] && subj_flag="--subject $SUBJECT"
    local dep_flag="";     [[ -n "$dep_jobid"      ]] && dep_flag="--dependency=afterok:${dep_jobid}"
    local account_flag=""; [[ -n "$ACCOUNT"        ]] && account_flag="--account=${ACCOUNT}"
    local mail_flag="";    [[ -n "$EMAIL"          ]] && mail_flag="--mail-user=${EMAIL} --mail-type=${EMAIL_TYPE}"

    # The actual python command that will run inside the job
    local py_cmd="python ${PIPELINE_DIR}/${py_script} \
        ${subj_flag} \
        --conda-env ${CONDA_ENV} \
        --log-dir ${OUTPUT_DIR}/logs \
        ${extra_args}"

    # Wrap in conda activation
    local wrapped="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
echo \"[JOB] Host: \$(hostname)  CPUs: \$(nproc)  Date: \$(date)\"
echo \"[JOB] Command: ${py_cmd}\"
${py_cmd}
"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo ""
        echo "[DRY-RUN] Step ${step_num} — ${job_name}"
        echo "  sbatch --job-name=${job_name} --partition=${PARTITION}"
        echo "         --cpus-per-task=${STEP_CPUS[$step_num]}"
        echo "         --mem=${STEP_MEM[$step_num]}"
        echo "         --time=${STEP_TIME[$step_num]}"
        echo "         ${dep_flag:-(no dependency)}"
        echo "  Python: ${py_cmd}"
        echo "  Log: ${log_out}"
        # Return fake ID so dependency chaining still prints correctly
        echo "DRY${step_num}"
        return
    fi

    local jobid
    jobid=$(sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --cpus-per-task="${STEP_CPUS[$step_num]}" \
        --mem="${STEP_MEM[$step_num]}" \
        --time="${STEP_TIME[$step_num]}" \
        --output="${log_out}" \
        --error="${log_err}" \
        ${dep_flag} \
        ${account_flag} \
        ${mail_flag} \
        --wrap="${wrapped}" \
        | awk '{print $NF}')

    echo "  Submitted step ${step_num} (${job_name}): SLURM job ${jobid}" >&2
    echo "$jobid"
}

# =============================================================================
# ── BUILD EXTRA ARGS PER STEP ─────────────────────────────────────────────────
# =============================================================================

SKIP_FLAG="";   [[ "$SKIP_EXISTING" -eq 1 ]] && SKIP_FLAG="--skip-existing"
MAXWIN_FLAG=""; [[ -n "$MAX_WINDOWS"      ]] && MAXWIN_FLAG="--max-windows ${MAX_WINDOWS}"
EXCL_FLAG="";  [[ -n "$EXCLUDE"          ]] && EXCL_FLAG="--exclude ${EXCLUDE}"
CHNAMES_FLAG=""; [[ -n "$CH_NAMES_FILE"   ]] && CHNAMES_FLAG="--ch-names-file ${CH_NAMES_FILE}"

# Step 1 — QC plots (doesn't use helpers subject discovery, uses utils)
STEP1_ARGS="--data-dir ${DATA_DIR} --output-dir ${QC_DIR}"

# Step 2 — The heavy correlation matrix computation (uses helpers.py fully)
STEP2_ARGS="--data-dir ${DATA_DIR} --output-dir ${CORR_DIR} \
    --clean-alg ${CLEAN_ALG} \
    --files-idx ${FILES_IDX} \
    ${EXCL_FLAG} ${SKIP_FLAG} ${MAXWIN_FLAG}"

# Steps 3 & 4 — Post-processing (reads .npy files, no FIF needed)
STEP3_ARGS="--corr-dir ${CORR_DIR} --output-dir ${CONN_DIR} ${CHNAMES_FLAG}"
STEP4_ARGS="--corr-dir ${CORR_DIR} --output-dir ${PAIRS_DIR} ${CHNAMES_FLAG} ${SKIP_FLAG}"

# =============================================================================
# ── SUBMIT ────────────────────────────────────────────────────────────────────
# =============================================================================

echo "============================================================"
echo "  EEG Pipeline — SLURM submission"
echo "  Data dir    : ${DATA_DIR}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "  Subject     : ${SUBJECT:-ALL}"
echo "  Step(s)     : ${STEP:-ALL}"
echo "  Clean alg   : ${CLEAN_ALG}"
echo "  Files idx   : ${FILES_IDX}"
echo "  Exclude     : ${EXCLUDE:-(none)}"
echo "  Conda env   : ${CONDA_ENV}"
echo "  Partition   : ${PARTITION}"
echo "  Dry run     : ${DRY_RUN}"
echo "============================================================"

JOB1="" JOB2="" JOB3="" JOB4=""

run_step_1() { JOB1=$(submit 1 "01_plot_eeg.py"             "$STEP1_ARGS" "");      }
run_step_2() { JOB2=$(submit 2 "02_compute_corr.py"         "$STEP2_ARGS" "$JOB1"); }
run_step_3() { JOB3=$(submit 3 "03_analyze_connectivity.py" "$STEP3_ARGS" "$JOB2"); }
run_step_4() { JOB4=$(submit 4 "04_plot_pairs.py"           "$STEP4_ARGS" "$JOB3"); }

if [[ -z "$STEP" ]]; then
    run_step_1; run_step_2; run_step_3; run_step_4
else
    case "$STEP" in
        1) run_step_1 ;;
        2) run_step_2 ;;
        3) run_step_3 ;;
        4) run_step_4 ;;
        *) echo "Unknown step: $STEP (valid: 1 2 3 4)"; exit 1 ;;
    esac
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel <JOBID>"
echo "Node info: sinfo"
echo "Logs:     ${LOG_DIR}/"
echo ""
echo "Interactive debug on a node:"
echo "  srun --pty --cpus-per-task=4 bash"
echo "  conda activate ${CONDA_ENV}"
echo "  python ${PIPELINE_DIR}/02_compute_corr.py --subject C01 --max-windows 5 ..."