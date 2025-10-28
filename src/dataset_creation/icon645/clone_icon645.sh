#!/usr/bin/env bash
set -euo pipefail

# Resolve this script's directory so the relative path works from anywhere
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# SPECIFY output dir
OUT_DIR="/work/scratch/ndickenmann/icon645"
ZIP_URL="https://iconqa2021.s3.us-west-1.amazonaws.com/icon645.zip"
ZIP_PATH="$OUT_DIR/icon645.zip"

echo "==> Creating output directory: $OUT_DIR"
mkdir -p "$OUT_DIR"

echo "==> Downloading: $ZIP_URL"
# Use curl if available, otherwise try wget
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 --retry-delay 2 -o "$ZIP_PATH" "$ZIP_URL"
else
  wget -O "$ZIP_PATH" "$ZIP_URL"
fi

echo "==> Unzipping to: $OUT_DIR"
# Ensure unzip is available
if ! command -v unzip >/dev/null 2>&1; then
  echo "!! 'unzip' not found. Install it (e.g., 'sudo apt-get install -y unzip') and re-run." >&2
  exit 1
fi

unzip -o "$ZIP_PATH" -d "$OUT_DIR"

echo "==> Cleaning up ZIP"
rm -f "$ZIP_PATH"

echo "✅ Done. Files are in: $OUT_DIR"
