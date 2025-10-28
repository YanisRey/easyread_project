#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safely remove one dataset from training_data.

Deletes image files whose metadata has dataset == <name>
AND (optionally) whose filename starts with <prefix>.
Then rewrites metadata.json and metadata.csv without those entries.

Features:
- --dry-run to preview
- --force to bypass 90% safety guard
- Backups: metadata.json.bak, metadata.csv.bak (if present)
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import List, Dict, Any, Tuple

# ---- Paths (match your project layout) ----
DATA_DIR = (Path(__file__).resolve().parent / "../../data").resolve()
TRAINING_DIR = DATA_DIR / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
META_JSON = TRAINING_DIR / "metadata.json"
META_CSV = TRAINING_DIR / "metadata.csv"

# ---- Utils ----
def backup(p: Path) -> None:
    if p.exists():
        b = p.with_suffix(p.suffix + ".bak")
        shutil.copy2(p, b)
        print(f"[INFO] Backup created: {b.name}")

def load_metadata_json() -> List[Dict[str, Any]]:
    if not META_JSON.exists():
        raise FileNotFoundError(f"{META_JSON} not found")
    with META_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either list-of-rows or dict keyed by id
    if isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        raise ValueError("metadata.json is not a list or dict of entries")
    return data

def split_rows(
    rows: List[Dict[str, Any]],
    dataset: str,
    prefix: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (kept_rows, filenames_to_delete)."""
    to_delete_files: List[str] = []
    kept: List[Dict[str, Any]] = []
    use_prefix = bool(prefix)

    for m in rows:
        ds = (m.get("dataset") or "").strip()
        img = (m.get("image_file") or "").strip()
        if not img:
            kept.append(m)
            continue

        match_dataset = (ds == dataset)
        match_prefix = (img.startswith(prefix) if use_prefix else True)

        if match_dataset and match_prefix:
            to_delete_files.append(img)
        else:
            kept.append(m)

    return kept, to_delete_files

def rewrite_metadata_csv(kept_rows: List[Dict[str, Any]]) -> None:
    if not kept_rows and not META_CSV.exists():
        return  # nothing to write, nothing to back up

    # Determine header (preserve original if present)
    fieldnames = ["dataset", "image_file", "id", "title", "keywords", "categories", "license"]
    if META_CSV.exists():
        backup(META_CSV)
        with META_CSV.open("r", encoding="utf-8") as f:
            try:
                r = csv.DictReader(f)
                if r.fieldnames:
                    fieldnames = r.fieldnames
            except Exception:
                pass

    with META_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in kept_rows:
            out = dict(row)
            # If keywords/categories are lists, join them (match your writer)
            if isinstance(out.get("keywords"), list):
                out["keywords"] = "|".join(out["keywords"])
            if isinstance(out.get("categories"), list):
                out["categories"] = "|".join(out["categories"])
            # Only keep known columns
            out = {k: out.get(k, "") for k in fieldnames}
            w.writerow(out)

def main():
    ap = argparse.ArgumentParser(description="Safely remove a dataset from training_data.")
    ap.add_argument("--dataset", required=True, help="Dataset name to remove (e.g., icon645)")
    ap.add_argument("--prefix", default="", help="Optional filename prefix to restrict deletions (e.g., icon645_)")
    ap.add_argument("--dry-run", action="store_true", help="Preview deletions; make no changes.")
    ap.add_argument("--force", action="store_true", help="Bypass 90%% safety guard.")
    args = ap.parse_args()

    dataset = args.dataset
    prefix = args.prefix

    # Load metadata
    try:
        rows = load_metadata_json()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    total = len(rows)
    kept, to_delete_files = split_rows(rows, dataset=dataset, prefix=prefix)
    removal_count = len(to_delete_files)
    removal_ratio = removal_count / max(1, total)

    print("=" * 70)
    print(f"[INFO] metadata.json entries: {total:,}")
    print(f"[INFO] Requested removal: dataset='{dataset}'"
          f"{' AND prefix='+prefix if prefix else ''}")
    print(f"[PLAN] Entries to remove: {removal_count:,} ({removal_ratio:.1%})")
    if to_delete_files:
        preview = ", ".join(to_delete_files[:5])
        print(f"[PLAN] Example files: {preview}{' ...' if removal_count > 5 else ''}")
    else:
        print("[WARN] No matching entries found. Nothing to do.")
        print("=" * 70)
        return

    # Safety guard
    if (removal_ratio > 0.90) and not (args.dry_run or args.force):
        print("[ABORT] This would remove >90% of metadata. Use --dry-run to inspect or --force to proceed.")
        print("=" * 70)
        return

    if args.dry_run:
        print("[DRY-RUN] No changes made.")
        print("=" * 70)
        return

    # Delete image files
    deleted, missing = 0, 0
    for fname in to_delete_files:
        p = IMAGES_DIR / fname
        try:
            if p.exists():
                p.unlink()
                deleted += 1
            else:
                missing += 1
        except Exception as e:
            print(f"[WARN] Could not delete {p}: {e}")

    print(f"[INFO] Deleted files: {deleted:,} (missing: {missing:,})")

    # Rewrite metadata.json
    backup(META_JSON)
    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Updated metadata.json: {len(kept):,} entries remain.")

    # Rewrite metadata.csv to match
    rewrite_metadata_csv(kept)

    print("=" * 70)
    print("[SUCCESS] Removal complete.")

if __name__ == "__main__":
    main()
