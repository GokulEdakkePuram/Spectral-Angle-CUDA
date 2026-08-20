#!/usr/bin/env bash
# Fetch the HyperBlood dataset: 14 hyperspectral cubes of blood alongside
# substances chosen to look like it (artificial blood, tomato concentrate,
# beetroot juice), with per-pixel ground truth.
#
# It is the right dataset for this pipeline because separating blood from its
# visual lookalikes is exactly the case where colour fails and the spectral
# angle does not - and because it ships annotations, so detection quality is
# measurable rather than eyeballed.
#
#   Romaszewski, Glomb, Cholewa, Sochan - Institute of Theoretical and Applied
#   Informatics, Polish Academy of Sciences. CC-BY-4.0.
#   https://doi.org/10.5281/zenodo.3984905
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${HSI_DATA_DIR:-$ROOT/data}"
ARCHIVE="$DATA_DIR/HyperBlood.zip"
URL="https://zenodo.org/api/records/3984905/files/HyperBlood.zip/content"

mkdir -p "$DATA_DIR"

if [[ -d "$DATA_DIR/HyperBlood" ]]; then
  echo "already extracted: $DATA_DIR/HyperBlood"
  exit 0
fi

# -C - resumes, which matters for a 2.3 GB download over a flaky link.
echo "downloading HyperBlood (~2.3 GB) to $ARCHIVE"
curl -L --retry 5 --retry-delay 5 -C - -o "$ARCHIVE" "$URL"

echo "extracting"
unzip -q -o "$ARCHIVE" -d "$DATA_DIR"

echo "done:"
find "$DATA_DIR" -name '*.hdr' | head -20
