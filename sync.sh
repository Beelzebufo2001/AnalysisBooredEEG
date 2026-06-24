#!/usr/bin/env bash
# =============================================================================
# sync.sh  –  rsync between local workspace and cluster
#
# Usage:
#   ./sync.sh push   # local → cluster
#   ./sync.sh pull   # cluster → local
# =============================================================================

set -euo pipefail

REMOTE="roblikova@wintermute.ms.mff.cuni.cz:./../../CSNG/roblikova/"
LOCAL="./"

RSYNC_OPTS="-avz --progress --prune-empty-dirs"

# Rule order matters — rsync stops at the first match.
# 1. Explicitly include the top-level folders we want to descend into
# 2. Exclude things we never want (matrices, jupyter checkpoints)
# 3. Include everything else inside those folders
# 4. Exclude everything else at the root level
RULES=(
    # ── Folders to sync ───────────────────────────────────────────────────────
    "--include=Scripts/"
    "--include=results/"

    # ── Exclusions inside those folders ───────────────────────────────────────
    "--exclude=results/corr_matrices/"
    "--exclude=results/plv_matrices/"
    "--exclude=.ipynb_checkpoints/"
    "--exclude=__pycache__/"
    "--exclude=*.pyc"

    # ── Include everything remaining inside the allowed folders ───────────────
    "--include=Scripts/**"
    "--include=results/**"

    # ── Drop everything else ──────────────────────────────────────────────────
    "--exclude=*"
)

if [ "${1:-}" == "push" ]; then
    echo "PUSH: local → cluster"
    rsync $RSYNC_OPTS "${RULES[@]}" "$LOCAL" "$REMOTE"

elif [ "${1:-}" == "pull" ]; then
    echo "PULL: cluster → local"
    rsync $RSYNC_OPTS "${RULES[@]}" "$REMOTE" "$LOCAL"

else
    echo "Usage: $0 [push|pull]"
    exit 1
fi