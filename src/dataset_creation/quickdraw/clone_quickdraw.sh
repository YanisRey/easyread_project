#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Quick, Draw! dataset fetcher + SVG conversion (default ON)
# ------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

OUT_DIR="${OUT_DIR:-"$SCRIPT_DIR/../../../data/quickdraw"}"
PARALLEL="${PARALLEL:-8}"

RAW="${RAW:-true}"
SIMPLIFIED="${SIMPLIFIED:-false}"
BITMAP_NPY="${BITMAP_NPY:-false}"
BITMAP_BIN="${BITMAP_BIN:-false}"

DEFAULT_CLASSES_FILE="$SCRIPT_DIR/categories.txt"
CLASSES_FILE="${CLASSES_FILE:-}"
if [[ -z "$CLASSES_FILE" && -f "$DEFAULT_CLASSES_FILE" ]]; then
  CLASSES_FILE="$DEFAULT_CLASSES_FILE"
fi

# If no file, you can still override via env: CLASSES="cat dog airplane"
# (ignored if CLASSES_FILE is set)
CLASSES=(${CLASSES:-cat dog airplane})

CONVERT_TO_SVG="${CONVERT_TO_SVG:-true}"
SVG_SIZE="${SVG_SIZE:-256}"
SVG_PAD="${SVG_PAD:-8}"
SVG_STROKE="${SVG_STROKE:-2.0}"
SVG_COLOR="${SVG_COLOR:-"#000000"}"
SVG_BG="${SVG_BG:-"transparent"}"

LOCAL_CONVERTER="$SCRIPT_DIR/ndjson_to_svg.py"
GCS_BASE="https://storage.googleapis.com/quickdraw_dataset"

echo "==> Output dir: $OUT_DIR"
mkdir -p "$OUT_DIR"
pushd "$OUT_DIR" >/dev/null

# -------- collect & normalize classes (lowercase, trim blanks) --------
if [[ -n "$CLASSES_FILE" ]]; then
  if [[ ! -f "$CLASSES_FILE" ]]; then
    echo "ERROR: CLASSES_FILE not found: $CLASSES_FILE" >&2
    exit 1
  fi
  mapfile -t CLASSES < <(sed 's/\r$//' "$CLASSES_FILE" | awk 'NF' | tr '[:upper:]' '[:lower:]')
else
  # normalize env-provided list too
  mapfile -t CLASSES < <(printf "%s\n" "${CLASSES[@]}" | tr '[:upper:]' '[:lower:]')
fi

if [[ ${#CLASSES[@]} -eq 0 ]]; then
  echo "ERROR: No classes specified." >&2
  exit 1
fi

echo "==> Classes: ${CLASSES[*]}"
echo "==> Formats: RAW=$RAW, SIMPLIFIED=$SIMPLIFIED, BITMAP_NPY=$BITMAP_NPY, BITMAP_BIN=$BITMAP_BIN"
echo "==> Convert to SVG: $CONVERT_TO_SVG"

# ---------------------- helpers ----------------------
url_encode() {
  local s="$*"
  s=${s// /%20}
  echo "$s"
}
curl_dl() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
    return 0
  fi
  curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
}

download_raw() {
  local c="$1"
  local ce; ce="$(url_encode "$c")"
  mkdir -p raw
  curl_dl "$GCS_BASE/full/raw/${ce}.ndjson" "raw/${c}.ndjson"
}
download_simplified() {
  local c="$1"
  local ce; ce="$(url_encode "$c")"
  mkdir -p simplified
  curl_dl "$GCS_BASE/full/simplified/${ce}.ndjson" "simplified/${c}.ndjson"
}
download_bitmap_npy() {
  local c="$1"
  local ce; ce="$(url_encode "$c")"
  mkdir -p bitmap_numpy
  curl_dl "$GCS_BASE/full/numpy_bitmap/${ce}.npy" "bitmap_numpy/${c}.npy"
}
download_bitmap_bin() {
  local c="$1"
  local ce; ce="$(url_encode "$c")"
  mkdir -p bitmap_binary
  curl_dl "$GCS_BASE/full/binary/${ce}.bin" "bitmap_binary/${c}.bin"
}

# simple concurrency limiter (works with function calls without leaving the shell)
run_with_limit() {
  local max_jobs="$1"; shift
  while (( $(jobs -pr | wc -l) >= max_jobs )); do
    wait -n || true
  done
  "$@" &
}

# ---------------------- downloads ----------------------
echo "==> Starting downloads…"
for c in "${CLASSES[@]}"; do
  [[ "$RAW" == "true" ]]          && run_with_limit "$PARALLEL" download_raw "$c"
  [[ "$SIMPLIFIED" == "true" ]]   && run_with_limit "$PARALLEL" download_simplified "$c"
  [[ "$BITMAP_NPY" == "true" ]]   && run_with_limit "$PARALLEL" download_bitmap_npy "$c"
  [[ "$BITMAP_BIN" == "true" ]]   && run_with_limit "$PARALLEL" download_bitmap_bin "$c"
done
wait
echo "==> Done downloading."

# ---------------------- SVG conversion (default ON) ----------------------
if [[ "$CONVERT_TO_SVG" == "true" ]]; then
  echo "==> Converting NDJSON to SVG…"
  CONVERTER=""
  if [[ -f "$LOCAL_CONVERTER" ]]; then
    CONVERTER="$LOCAL_CONVERTER"
  elif command -v ndjson_to_svg.py >/dev/null 2>&1; then
    CONVERTER="ndjson_to_svg.py"
  fi

  if [[ -z "$CONVERTER" ]]; then
    echo "WARNING: ndjson_to_svg.py not found next to script or on PATH."
    echo "         Skipping SVG conversion. Place it at: $LOCAL_CONVERTER"
  else
    mkdir -p "svg_raw" "svg_simplified"

    if [[ "$RAW" == "true" && -d raw ]]; then
      for f in raw/*.ndjson; do
        [[ -e "$f" ]] || continue
        outdir="svg_raw/$(basename "${f%.ndjson}")"
        mkdir -p "$outdir"
        python "$CONVERTER" -i "$f" -o "$outdir" --size "$SVG_SIZE" --padding "$SVG_PAD" \
          --stroke_width "$SVG_STROKE" --stroke_color "$SVG_COLOR" --background "$SVG_BG"
      done
    fi

    if [[ "$SIMPLIFIED" == "true" && -d simplified ]]; then
      for f in simplified/*.ndjson; do
        [[ -e "$f" ]] || continue
        outdir="svg_simplified/$(basename "${f%.ndjson}")"
        mkdir -p "$outdir"
        python "$CONVERTER" -i "$f" -o "$outdir" --size "$SVG_SIZE" --padding "$SVG_PAD" \
          --stroke_width "$SVG_STROKE" --stroke_color "$SVG_COLOR" --background "$SVG_BG"
      done
    fi
    echo "==> SVG conversion finished."
  fi
fi

popd >/dev/null

echo "✅ QuickDraw fetch complete in: $OUT_DIR"
echo "   Folders (depending on settings): raw/, simplified/, bitmap_numpy/, bitmap_binary/"
echo "   SVGs (by default): svg_raw/, svg_simplified/"
