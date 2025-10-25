#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Script
Integrates multiple icon/pictogram datasets (arasaac, icon645, lds, openmoji)
into a unified training_data directory with standardized naming.
"""

import json
import shutil
from pathlib import Path
import csv
from typing import Dict, List, Any, Optional, Tuple
import os

# Base paths
DATA_DIR = (Path(__file__).resolve().parent / "../../data").resolve()
TRAINING_DIR = DATA_DIR / "training_data"

# Dataset source directories
ARASAAC_DIR = DATA_DIR / "arasaac"
ICON645_DIR = DATA_DIR / "icon645"
LDS_DIR = DATA_DIR / "lds"
OPENMOJI_DIR = DATA_DIR / "openmoji" / "openmoji-lite"


def move_preserve_meta(src: Path, dst: Path) -> None:
    """
    Move file from src to dst.
    If src and dst are on different filesystems, shutil.move falls back to copy2+unlink,
    preserving metadata. Raises on failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def setup_training_directory():
    """Create training_data directory structure."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    (TRAINING_DIR / "images").mkdir(exist_ok=True)
    print(f"[INFO] Training directory set up at: {TRAINING_DIR}")

def process_arasaac(all_metadata: List[Dict[str, Any]], save_interval: int = 100) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process ARASAAC dataset.
    Returns (list of metadata entries, copied_count).
    """
    print("\n[ARASAAC] Processing dataset...")

    metadata_file = ARASAAC_DIR / "metadata.json"
    if not metadata_file.exists():
        print(f"[WARN] ARASAAC metadata not found at {metadata_file}")
        return [], 0  # <-- return tuple

    # Load metadata (mapping id -> record)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    processed_data: List[Dict[str, Any]] = []
    copied_count = 0

    for i, (pic_id, meta) in enumerate(metadata.items(), 1):
        # ARASAAC uses "file_name"
        image_file = meta.get("image_file") or meta.get("file_name") or meta.get("filename")
        if not image_file:
            continue

        source_path = ARASAAC_DIR / image_file
        if not source_path.exists():
            # Some dumps place images in a subdir; try one fallback
            alt = ARASAAC_DIR / "images" / image_file
            if alt.exists():
                source_path = alt
            else:
                continue

        # Prepend dataset name for uniqueness
        new_filename = f"arasaac_{Path(image_file).stem}{Path(image_file).suffix}"
        dest_path = TRAINING_DIR / "images" / new_filename

        try:
            move_preserve_meta(source_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"[WARN] Failed to transfer {source_path} -> {dest_path}: {e}")
            continue

        # Normalize fields
        keywords = meta.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split("|") if k.strip()]

        categories = meta.get("categories", [])
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split("|") if c.strip()]

        entry = {
            "dataset": "arasaac",
            "image_file": new_filename,
            "id": pic_id,                              # keep the outer key
            "title": meta.get("title"),
            "keywords": keywords,
            "categories": categories,
            "license": meta.get("license", "CC BY-NC-SA 4.0"),
        }
        processed_data.append(entry)

        if copied_count % save_interval == 0:
            save_metadata(all_metadata + processed_data)
            print(f"[ARASAAC] Checkpoint: {copied_count} images processed")

    print(f"[ARASAAC] Processed {copied_count} images")
    return processed_data, copied_count


def process_icon645(all_metadata: List[Dict[str, Any]], save_interval: int = 100) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process ICON645 dataset.
    Returns list of metadata entries in standardized format.
    """
    print("\n[ICON645] Processing dataset...")

    icons_dir = ICON645_DIR / "colored_icons_final"
    if not icons_dir.exists():
        print(f"[WARN] ICON645 directory not found at {icons_dir}")
        return [], 0

    processed_data = []
    copied_count = 0

    # Iterate through category directories
    for category_dir in icons_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        # Process all PNG images in the category
        for image_file in category_dir.glob("*.png"):
            # Prepend dataset name to keep filenames unique
            new_filename = f"icon645_{image_file.name}"
            dest_path = TRAINING_DIR / "images" / new_filename

            # Copy image
            try:
                move_preserve_meta(image_file, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"[WARN] Failed to transfer {image_file} -> {dest_path}: {e}")
                continue


            # Extract label from filename (icon ID)
            icon_id = image_file.stem

            entry = {
                "dataset": "icon645",
                "image_file": new_filename,
                "id": icon_id,
                "title": category_name,
                "keywords": [category_name],
                "categories": [category_name],
                "license": "CC BY-NC-SA 4.0"
            }
            processed_data.append(entry)

            # Periodic save
            if copied_count % save_interval == 0:
                save_metadata(all_metadata + processed_data)
                print(f"[ICON645] Checkpoint: {copied_count} images processed")

    print(f"[ICON645] Processed {copied_count} images")
    return processed_data, copied_count


def process_lds(all_metadata: List[Dict[str, Any]], save_interval: int = 100) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process LDS (Learning Design Symbols) dataset.
    Returns list of metadata entries in standardized format.
    """
    print("\n[LDS] Processing dataset...")

    if not LDS_DIR.exists():
        print(f"[WARN] LDS directory not found at {LDS_DIR}")
        return [], 0

    processed_data = []
    copied_count = 0

    # Process all PNG images in LDS directory
    for image_file in LDS_DIR.glob("*.png"):
        # Prepend dataset name to keep filenames unique
        new_filename = f"lds_{image_file.name}"
        dest_path = TRAINING_DIR / "images" / new_filename

        # Copy image
        try:
            move_preserve_meta(image_file, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"[WARN] Failed to transfer {image_file} -> {dest_path}: {e}")
            continue


        # Extract label from filename (remove .png and clean up)
        label = image_file.stem
        # Convert filename to readable keywords (e.g., "1-hour" -> ["1", "hour"])
        keywords = [kw.strip() for kw in label.replace("-", " ").replace("_", " ").split()]

        entry = {
            "dataset": "lds",
            "image_file": new_filename,
            "id": label,
            "title": label.replace("-", " ").replace("_", " "),
            "keywords": keywords,
            "categories": ["lds"],
            "license": "Learning Design Symbols"
        }
        processed_data.append(entry)

        # Periodic save
        if copied_count % save_interval == 0:
            save_metadata(all_metadata + processed_data)
            print(f"[LDS] Checkpoint: {copied_count} images processed")

    print(f"[LDS] Processed {copied_count} images")
    return processed_data, copied_count


def process_openmoji(all_metadata: List[Dict[str, Any]], save_interval: int = 100) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process OpenMoji dataset.
    Returns list of metadata entries in standardized format.
    """
    print("\n[OPENMOJI] Processing dataset...")

    # Load OpenMoji metadata
    metadata_file = OPENMOJI_DIR / "data" / "openmoji.json"
    if not metadata_file.exists():
        print(f"[WARN] OpenMoji metadata not found at {metadata_file}")
        return []

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    # We'll use the 618x618 PNG color images
    images_dir = OPENMOJI_DIR / "color" / "618x618"
    if not images_dir.exists():
        # Try alternative path
        images_dir = OPENMOJI_DIR / "color" / "72x72"
        if not images_dir.exists():
            print(f"[WARN] OpenMoji images directory not found")
            return [], 0
            
    processed_data = []
    copied_count = 0

    # Create a mapping from hexcode to metadata
    meta_map = {}
    for item in metadata_list:
        hexcode = item.get("hexcode")
        if hexcode:
            meta_map[hexcode] = item

    # Process all PNG images
    for image_file in images_dir.glob("*.png"):
        hexcode = image_file.stem

        # Prepend dataset name to keep filenames unique
        new_filename = f"openmoji_{image_file.name}"
        dest_path = TRAINING_DIR / "images" / new_filename

        # Copy image
        try:
            move_preserve_meta(image_file, dest_path)   # moves, i.e., copy then delete original
            copied_count += 1
        except Exception as e:
            print(f"[WARN] Failed to transfer {image_file} -> {dest_path}: {e}")
            continue


        # Get metadata if available
        meta = meta_map.get(hexcode, {})
        annotation = meta.get("annotation", hexcode)
        tags = meta.get("tags", "")
        openmoji_tags = meta.get("openmoji_tags", "")
        group = meta.get("group", "")
        subgroups = meta.get("subgroups", "")

        # Combine all tags
        all_tags = []
        if tags:
            all_tags.extend([t.strip() for t in tags.split(",") if t.strip()])
        if openmoji_tags:
            all_tags.extend([t.strip() for t in openmoji_tags.split(",") if t.strip()])

        # Add group/subgroup info
        categories = []
        if group:
            categories.append(group)
        if subgroups:
            categories.append(subgroups)

        entry = {
            "dataset": "openmoji",
            "image_file": new_filename,
            "id": hexcode,
            "title": annotation,
            "keywords": all_tags if all_tags else [annotation],
            "categories": categories if categories else ["emoji"],
            "license": "CC BY-SA 4.0"
        }
        processed_data.append(entry)

        # Periodic save
        if copied_count % save_interval == 0:
            save_metadata(all_metadata + processed_data)
            print(f"[OPENMOJI] Checkpoint: {copied_count} images processed")

    print(f"[OPENMOJI] Processed {copied_count} images")
    return processed_data, copied_count


def save_metadata(all_metadata: List[Dict[str, Any]]):
    """Save combined metadata to JSON and CSV formats."""

    # Save as JSON
    json_path = TRAINING_DIR / "metadata.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Saved metadata JSON to: {json_path}")

    # Save as CSV
    csv_path = TRAINING_DIR / "metadata.csv"
    if all_metadata:
        fieldnames = ["dataset", "image_file", "id", "title", "keywords", "categories", "license"]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in all_metadata:
                row = entry.copy()
                # Convert lists to pipe-separated strings for CSV
                if isinstance(row.get("keywords"), list):
                    row["keywords"] = "|".join(row["keywords"])
                if isinstance(row.get("categories"), list):
                    row["categories"] = "|".join(row["categories"])
                writer.writerow(row)

        print(f"[INFO] Saved metadata CSV to: {csv_path}")


def generate_statistics(all_metadata: List[Dict[str, Any]]):
    """Generate and print statistics about the combined dataset."""

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    # Count by dataset
    dataset_counts = {}
    for entry in all_metadata:
        dataset = entry.get("dataset", "unknown")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    print(f"\nTotal images: {len(all_metadata)}")
    print("\nBreakdown by dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset:15s}: {count:6,d} images")

    # Collect all unique categories
    all_categories = set()
    for entry in all_metadata:
        cats = entry.get("categories", [])
        if isinstance(cats, list):
            all_categories.update(cats)

    print(f"\nTotal unique categories: {len(all_categories)}")
    print("="*60 + "\n")


def main():
    """Main function to orchestrate dataset preparation and space-saving moves."""
    print("=" * 60)
    print("PREPARING COMBINED DATASET")
    print("=" * 60)

    # Setup output directory
    setup_training_directory()

    # Accumulate metadata and total transfer count
    all_metadata: List[Dict[str, Any]] = []
    grand_total = 0

    # ---- Process ARASAAC ----
    dataset_data, n = process_arasaac(all_metadata, save_interval=100)
    all_metadata.extend(dataset_data)
    grand_total += n
    save_metadata(all_metadata)

    # ---- Process ICON645 ----
    dataset_data, n = process_icon645(all_metadata, save_interval=100)
    all_metadata.extend(dataset_data)
    grand_total += n
    save_metadata(all_metadata)

    # ---- Process LDS ----
    dataset_data, n = process_lds(all_metadata, save_interval=100)
    all_metadata.extend(dataset_data)
    grand_total += n
    save_metadata(all_metadata)

    # ---- Process OPENMOJI ----
    dataset_data, n = process_openmoji(all_metadata, save_interval=100)
    all_metadata.extend(dataset_data)
    grand_total += n

    # ---- Final save and summary ----
    save_metadata(all_metadata)
    generate_statistics(all_metadata)

    print("=" * 60)
    print(f"[SUCCESS] Dataset preparation complete!")
    print(f"[INFO] Training data location: {TRAINING_DIR}")
    print(f"[INFO] Total images transferred (originals deleted): {grand_total:,}")
    print("=" * 60)



if __name__ == "__main__":
    main()
