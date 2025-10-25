#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize training_data:
- totals and per-dataset counts
- image sizes (WxH) and counts (overall and per dataset)
- file extensions, licenses, categories (top N)
- sanity checks: missing files, unreadable images, duplicate ids/filenames
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -------- Paths (same layout as your main script) --------
DATA_DIR = (Path(__file__).resolve().parent / "../../data").resolve()
TRAINING_DIR = DATA_DIR / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
META_JSON = TRAINING_DIR / "metadata.json"

# -------- Optional: lightweight image size readers --------
def _png_size(path: Path) -> Optional[Tuple[int, int]]:
    # PNG: 8-byte signature + IHDR chunk: width/height at bytes 16..23
    try:
        with path.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            f.read(8)  # IHDR length+type
            w = int.from_bytes(f.read(4), "big")
            h = int.from_bytes(f.read(4), "big")
            return w, h
    except Exception:
        return None

def _jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    # Minimal JPEG SOF scanner
    try:
        with path.open("rb") as f:
            data = f.read()
        if not (data[:2] == b"\xff\xd8"):
            return None
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            # skip fill bytes
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i >= len(data):
                break
            marker = data[i]
            i += 1
            # standalone markers without length
            if marker in (0xD8, 0xD9):  # SOI/EOI
                continue
            if i + 2 > len(data):
                break
            seglen = int.from_bytes(data[i:i+2], "big")
            if seglen < 2:
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                # SOF: [len][precision][height][width]...
                if i + 5 < len(data):
                    precision = data[i+2]
                    h = int.from_bytes(data[i+3:i+5], "big")
                    w = int.from_bytes(data[i+5:i+7], "big")
                    return w, h
                else:
                    break
            i += seglen
        return None
    except Exception:
        return None

def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    ext = path.suffix.lower()
    if ext == ".png":
        sz = _png_size(path)
        if sz:
            return sz
    if ext in (".jpg", ".jpeg"):
        sz = _jpeg_size(path)
        if sz:
            return sz
    # Fallback to Pillow if available
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None

# -------- Helpers --------
def human(n: int) -> str:
    return f"{n:,}"

def top_n(counter: Counter, n: int = 20) -> List[Tuple[str, int]]:
    return counter.most_common(n)

# -------- Main summary --------
def main():
    if not META_JSON.exists():
        print(f"[ERROR] metadata.json not found at {META_JSON}")
        return
    if not IMAGES_DIR.exists():
        print(f"[ERROR] images directory not found at {IMAGES_DIR}")
        return

    with META_JSON.open("r", encoding="utf-8") as f:
        meta: List[Dict] = json.load(f)

    total_meta = len(meta)
    print("=" * 70)
    print("TRAINING DATA SUMMARY")
    print("=" * 70)
    print(f"[INFO] Metadata entries: {human(total_meta)}")
    print(f"[INFO] Images dir:        {IMAGES_DIR}")

    # Counters
    by_dataset = Counter()
    by_size_overall = Counter()
    by_size_per_dataset: Dict[str, Counter] = defaultdict(Counter)
    by_ext = Counter()
    by_license = Counter()
    categories_counter = Counter()

    # Sanity
    missing_files = []
    unreadable = []
    duplicate_filenames = Counter()
    duplicate_ids = Counter()

    seen_filenames = set()
    seen_ids_per_dataset: Dict[str, set] = defaultdict(set)

    for e in meta:
        ds = e.get("dataset", "unknown")
        img = e.get("image_file")
        lic = e.get("license", "unknown")
        cats = e.get("categories", [])
        iid = e.get("id")

        by_dataset[ds] += 1
        by_license[lic] += 1
        if isinstance(cats, list):
            categories_counter.update(cats)

        if img is None:
            continue
        p = IMAGES_DIR / img
        if not p.exists():
            missing_files.append(str(p))
            continue

        ext = p.suffix.lower() or "<noext>"
        by_ext[ext] += 1

        # Sizes
        size = get_image_size(p)
        if size is None:
            unreadable.append(str(p))
        else:
            w, h = size
            key = f"{w}x{h}"
            by_size_overall[key] += 1
            by_size_per_dataset[ds][key] += 1

        # Duplicates
        fname = p.name
        if fname in seen_filenames:
            duplicate_filenames[fname] += 1
        else:
            seen_filenames.add(fname)

        if iid is not None:
            if iid in seen_ids_per_dataset[ds]:
                duplicate_ids[(ds, str(iid))] += 1
            else:
                seen_ids_per_dataset[ds].add(iid)

    # -------- Print summary --------
    print("\n--- Totals ---")
    print(f"Total entries in metadata: {human(total_meta)}")
    print(f"Existing image files:      {human(total_meta - len(missing_files))}")
    print(f"Missing files:             {human(len(missing_files))}")
    print(f"Unreadable images:         {human(len(unreadable))}")

    print("\n--- By dataset ---")
    for ds, cnt in sorted(by_dataset.items()):
        print(f"{ds:15s}: {human(cnt)}")

    print("\n--- Image sizes (overall) ---")
    for size, cnt in top_n(by_size_overall, n=len(by_size_overall)):
        print(f"{size:>9s}: {human(cnt)}")

    print("\n--- Image sizes per dataset (top 10 per dataset) ---")
    for ds in sorted(by_size_per_dataset.keys()):
        print(f"[{ds}]")
        for size, cnt in top_n(by_size_per_dataset[ds], n=10):
            print(f"  {size:>9s}: {human(cnt)}")

    print("\n--- File extensions ---")
    for ext, cnt in sorted(by_ext.items(), key=lambda x: (-x[1], x[0])):
        print(f"{ext:8s}: {human(cnt)}")

    print("\n--- Licenses ---")
    for lic, cnt in sorted(by_license.items(), key=lambda x: (-x[1], x[0])):
        print(f"{lic}: {human(cnt)}")

    print("\n--- Top categories (up to 20) ---")
    for cat, cnt in top_n(categories_counter, n=20):
        print(f"{cat}: {human(cnt)}")

    # Sanity / diagnostics
    if missing_files:
        print("\n[WARN] Missing files (showing up to 10):")
        for p in missing_files[:10]:
            print(f"  - {p}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    if unreadable:
        print("\n[WARN] Unreadable images (showing up to 10):")
        for p in unreadable[:10]:
            print(f"  - {p}")
        if len(unreadable) > 10:
            print(f"  ... and {len(unreadable) - 10} more")

    if duplicate_filenames:
        print("\n[INFO] Duplicate image filenames (same name appears multiple times):")
        for fn, extra in top_n(duplicate_filenames, n=10):
            print(f"  {fn}: +{extra}")

    if duplicate_ids:
        print("\n[INFO] Duplicate IDs within a dataset (same (dataset,id) appears multiple times):")
        shown = 0
        for (ds, iid), extra in duplicate_ids.items():
            print(f"  ({ds}, id={iid}): +{extra}")
            shown += 1
            if shown >= 10:
                remaining = len(duplicate_ids) - shown
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break

    print("\n" + "=" * 70)
    print("[DONE] Summary complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
