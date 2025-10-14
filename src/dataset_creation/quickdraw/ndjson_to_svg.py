#!/usr/bin/env python3
"""
ndjson_to_svg.py
----------------
Convert Google Quick, Draw! NDJSON stroke data into per-sample SVG files.

Usage:
  python ndjson_to_svg.py --input path/to/cat.ndjson --out out_dir --size 256 --limit 100

Notes:
- Works with both RAW ndjson (with timing t) and simplified ndjson (no timing).
- Automatically rescales drawing to fit a square canvas with padding.
- Produces one SVG per line (sample) named: <word>_<key_id>.svg (falls back to row index if missing).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Path to NDJSON file (e.g., apple.ndjson)")
    ap.add_argument("--out", "-o", required=True, help="Output directory for SVG files")
    ap.add_argument("--size", type=int, default=256, help="Output canvas size in pixels (width=height)")
    ap.add_argument("--padding", type=int, default=8, help="Padding around drawing in pixels")
    ap.add_argument("--stroke_width", type=float, default=2.0, help="SVG stroke width in pixels")
    ap.add_argument("--stroke_color", default="#000000", help="SVG stroke color")
    ap.add_argument("--background", default="transparent", help="Background color (e.g. '#ffffff' or 'transparent')")
    ap.add_argument("--limit", type=int, default=0, help="If >0, stop after this many samples")
    return ap.parse_args()

def compute_bounds(strokes: List[List[List[float]]]) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) across all points."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for s in strokes:
        if len(s) < 2:
            continue
        xs, ys = s[0], s[1]
        if not xs or not ys:
            continue
        minx = min(minx, min(xs))
        miny = min(miny, min(ys))
        maxx = max(maxx, max(xs))
        maxy = max(maxy, max(ys))
    if minx == float("inf"):
        minx = miny = 0.0
        maxx = maxy = 1.0
    return minx, miny, maxx, maxy

def scale_translate_point(x, y, minx, miny, scale, pad):
    return ( (x - minx) * scale + pad, (y - miny) * scale + pad )

def strokes_to_svg_path(strokes: List[List[List[float]]], minx, miny, scale, pad) -> str:
    """Build a single SVG path string from strokes."""
    cmds = []
    for s in strokes:
        if len(s) < 2:
            continue
        xs, ys = s[0], s[1]
        if not xs or not ys:
            continue
        # Move to first point of stroke
        x0, y0 = scale_translate_point(xs[0], ys[0], minx, miny, scale, pad)
        cmds.append(f"M {x0:.3f} {y0:.3f}")
        # Line to each subsequent point
        for x, y in zip(xs[1:], ys[1:]):
            sx, sy = scale_translate_point(x, y, minx, miny, scale, pad)
            cmds.append(f"L {sx:.3f} {sy:.3f}")
    return " ".join(cmds)

def make_svg(content_path: Path, size: int, background: str, path_d: str, stroke_color: str, stroke_width: float):
    with open(content_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">\n')
        if background != "transparent":
            f.write(f'  <rect x="0" y="0" width="{size}" height="{size}" fill="{background}"/>\n')
        f.write(f'  <path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>\n')
        f.write('</svg>\n')

def sanitize(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"

def process_file(in_path: Path, out_dir: Path, size: int, pad: int, stroke_color: str, stroke_width: float, background: str, limit: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(in_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # some files can contain trailing commas or minor issues; skip
                continue
            strokes = rec.get("drawing")
            if not strokes:
                continue

            minx, miny, maxx, maxy = compute_bounds(strokes)
            w = maxx - minx
            h = maxy - miny
            inner = max(size - 2 * pad, 1)
            scale = inner / max(w if w > 0 else 1, h if h > 0 else 1)

            path_d = strokes_to_svg_path(strokes, minx, miny, scale, pad)

            word = sanitize(str(rec.get("word", "drawing")))
            key_id = sanitize(str(rec.get("key_id", idx)))
            fname = f"{word}_{key_id}.svg"
            make_svg(out_dir / fname, size, background, path_d, stroke_color, stroke_width)
            count += 1

            if limit and count >= limit:
                break
    print(f"Wrote {count} SVGs to {out_dir}")

def main():
    args = parse_args()
    process_file(
        in_path=Path(args.input),
        out_dir=Path(args.out),
        size=args.size,
        pad=args.padding,
        stroke_color=args.stroke_color,
        stroke_width=args.stroke_width,
        background=args.background,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()
