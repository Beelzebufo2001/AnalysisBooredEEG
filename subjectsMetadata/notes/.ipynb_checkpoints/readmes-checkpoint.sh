#!/bin/bash

REMOTE="roblikova@wintermute.ms.mff.cuni.cz"
BASE="/CSNG/EEG/MS_data"

mkdir -p EEG_READMES

for i in {01..27}; do
  SRC="${BASE}/C${i}/README.md"
  DEST="./EEG_READMES/C${i}_README.md"

  echo "Pulling C${i}..."

  rsync -avz --progress \
    "${REMOTE}:${SRC}" \
    "${DEST}" \
    || echo "FAILED: C${i}" >> rsync_failures.log

done
