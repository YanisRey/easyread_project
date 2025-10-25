#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Removes all icon645 images and metadata entries from training_data.

Actions:
- Deletes all image files whose dataset == "icon645"
- Removes those entries from metadata.json and metadata.csv
- Creates backups of the metadata files before rewriting
"""

import json
import csv
from pathlib import Path
import shutil

# --- Paths (match your main setup) ---
DATA_DIR = (Path(__file__).resolve().parent / "../../data").resolve()
TRAINING_DIR = DATA_DIR / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
META_JSON = TRAINING_DIR / "metadata.json"
META_CSV = TRAINING_DIR / "metadata.csv"

def backup_file(path: Path):
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"[INFO] Backup created: {backup.name}")

def remove_icon645_entries():
    if not META_JSON.exists() and not META_CSV.exists():
        print("[ERROR] No metadata.json or metadata.csv found.")
        return

    # --- Load JSON metadata ---
    if META_JSON.exists():
        with open(META_JSON, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    # --- Filter out icon645 entries ---
    filtered = [m for m in metadata if m.get("dataset") != "icon645"]
    removed_count = len(metadata) - len(filtered)

    print(f"[INFO] Found {len(metadata):,} total entries in JSON.")
    print(f"[INFO] Removing {removed_count:,} icon645 entries...")

    # --- Delete image files ---
    deleted_images = 0
    for m in metadata:
        if m.get("dataset") == "icon645":
            img_file = m.get("image_file")
            if img_file:
                path = IMAGES_DIR / img_file
                try:
                    if path.exists():
                        path.unlink()
                        deleted_images += 1
                except Exception as e:
                    print(f"[WARN] Could not delete {path}: {e}")

    print(f"[INFO] Deleted {deleted_images:,} image files from icon645 dataset.")

    # --- Save cleaned JSON metadata ---
    backup_file(META_JSON)
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Updated {META_JSON.name} with {len(filtered):,} entries.")

    # --- Clean CSV metadata ---
    if META_CSV.exists():
        backup_file(META_CSV)
        with open(META_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("dataset") != "icon645"]

        with open(META_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[INFO] Updated {META_CSV.name} with {len(rows):,} entries (removed {removed_count:,}).")

    print("[SUCCESS] icon645 dataset fully removed.")


if __name__ == "__main__":
    remove_icon645_entries()
