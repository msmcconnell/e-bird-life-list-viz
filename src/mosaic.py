"""Build a photo mosaic of a reference image from a life list's bird photos.

The reference is cropped and downscaled so it has exactly one pixel per
species. Each species is then placed on exactly one pixel, using whichever of
its photos fits best, so that the total color difference between photos and
pixels is as small as possible (an optimal assignment, not a greedy one).

Usage:
    python src/mosaic.py data/mosiac_reference/Broad-billed-motmot_268880490.jpg

Output (in <out>/<reference name>/):
    mosaic.jpg      the rendered mosaic (mosaic_debug.jpg with --debug, each
                    tile labeled "row,col")
    reference.png   the downscaled reference, one pixel per species
    layout.csv      row, col, species and photo for every cell
    layout.json     grid size and settings plus the same per-cell data
"""
import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from life_list import load_life_list

RGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """Convert sRGB values in 0-255 (shape (..., 3)) to CIE Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ RGB_TO_XYZ.T / WHITE_D65
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def center_square(img):
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


def open_rgb(path, min_size=None):
    img = Image.open(path)
    if min_size:
        img.draft("RGB", (min_size, min_size))  # fast JPEG decode at reduced size
    return img.convert("RGB")


def mean_rgb(path):
    """Average color of the square tile that will be cut from a photo."""
    return center_square(open_rgb(path, 64)).resize((1, 1), Image.BOX).getpixel((0, 0))


def load_species_photos(images_dir, life_list):
    """Return [(meta, [photo paths])] in life list order; error if any species has no photos."""
    entries, missing = [], []
    for species in life_list:
        species_dir = images_dir / species.slug
        meta_path = species_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        paths = [species_dir / p["file"] for p in meta["photos"] if not p.get("rejected")] if meta else []
        paths = [p for p in paths if p.exists()]
        if paths:
            entries.append((meta, paths))
        else:
            missing.append(species)
    if missing:
        names = ", ".join(s.common_name for s in missing[:10])
        raise SystemExit(f"{len(missing)} species have no photos in {images_dir}: {names}")
    return entries


def tile_colors(images_dir, paths):
    """Lab color of every photo, cached in <images_dir>/tile_colors.json."""
    cache_path = images_dir / "tile_colors.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    keys = [p.relative_to(images_dir).as_posix() for p in paths]
    mtimes = [p.stat().st_mtime for p in paths]
    stale = [i for i, (k, m) in enumerate(zip(keys, mtimes)) if cache.get(k, {}).get("mtime") != m]
    if stale:
        print(f"Measuring colors of {len(stale)} photos...")
        with ThreadPoolExecutor() as pool:
            rgbs = list(pool.map(mean_rgb, [paths[i] for i in stale]))
        for i, lab in zip(stale, srgb_to_lab(rgbs)):
            cache[keys[i]] = {"mtime": mtimes[i], "lab": lab.round(3).tolist()}
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return np.array([cache[k]["lab"] for k in keys])


def choose_grid(n, aspect):
    """Return (rows, cols) with rows * cols == n and cols / rows closest to aspect."""
    pairs = [(r, n // r) for r in range(1, n + 1) if n % r == 0]
    return min(pairs, key=lambda rc: abs(math.log(rc[1] / rc[0] / aspect)))


def downscale_reference(path, n):
    """Center-crop the reference to an exact rows x cols grid and average each cell."""
    img = open_rgb(path)
    rows, cols = choose_grid(n, img.width / img.height)
    if cols / rows < img.width / img.height:
        width, height = round(img.height * cols / rows), img.height
    else:
        width, height = img.width, round(img.width * rows / cols)
    left, top = (img.width - width) // 2, (img.height - height) // 2
    crop = (left, top, left + width, top + height)
    small = img.crop(crop).resize((cols, rows), Image.BOX)
    return small, rows, cols, crop


def assign(target_lab, species_tiles):
    """Optimally place every species on one cell, choosing its best photo.

    target_lab: (cells, 3). species_tiles: per species, an (n_photos, 3) array.
    Returns (species index, photo index, squared color distance) per cell.
    """
    n = len(species_tiles)
    max_photos = max(len(t) for t in species_tiles)
    best_cost = np.full((n, n), np.inf)
    best_photo = np.zeros((n, n), dtype=np.int32)
    for k in range(max_photos):
        has_k = np.array([len(t) > k for t in species_tiles])
        lab_k = np.array([t[k] if len(t) > k else (0, 0, 0) for t in species_tiles])
        cost = ((target_lab[:, None, :] - lab_k[None, :, :]) ** 2).sum(-1)
        cost[:, ~has_k] = np.inf
        better = cost < best_cost
        best_cost[better] = cost[better]
        best_photo[better] = k
    cells, species = linear_sum_assignment(best_cost)
    order = np.argsort(cells)
    species = species[order]
    return species, best_photo[cells[order], species], best_cost[cells[order], species]


def render(paths, rows, cols, tile_size, labels=False):
    mosaic = Image.new("RGB", (cols * tile_size, rows * tile_size))
    draw = ImageDraw.Draw(mosaic)
    font = ImageFont.load_default(size=max(8, tile_size // 5))

    def tile(path):
        return center_square(open_rgb(path, tile_size)).resize((tile_size, tile_size), Image.LANCZOS)

    with ThreadPoolExecutor() as pool:
        for i, img in enumerate(pool.map(tile, paths)):
            row, col = divmod(i, cols)
            x, y = col * tile_size, row * tile_size
            mosaic.paste(img, (x, y))
            if labels:
                box = draw.textbbox((x + 2, y + 2), f"{row},{col}", font=font)
                draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), fill="black")
                draw.text((x + 2, y + 2), f"{row},{col}", font=font, fill="white")
    return mosaic


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", help="reference image to recreate")
    parser.add_argument("--life-list", default="data/list_actual/bird_list.csv",
                        help="life list CSV (default: data/list_actual/bird_list.csv)")
    parser.add_argument("--images", default="data/images", help="photos from fetch_inat.py (default: data/images)")
    parser.add_argument("--out", default="data/mosaic", help="output directory (default: data/mosaic)")
    parser.add_argument("--tile-size", type=int, default=40, help="pixels per tile in mosaic.jpg (default: 40)")
    parser.add_argument("--debug", action="store_true",
                        help='label each tile with its "row,col" and write mosaic_debug.jpg instead of mosaic.jpg')
    args = parser.parse_args()

    images_dir = Path(args.images)
    entries = load_species_photos(images_dir, load_life_list(args.life_list))
    all_paths = [p for _, paths in entries for p in paths]
    all_lab = tile_colors(images_dir, all_paths)
    species_tiles, start = [], 0
    for _, paths in entries:
        species_tiles.append(all_lab[start:start + len(paths)])
        start += len(paths)

    small, rows, cols, crop = downscale_reference(args.reference, len(entries))
    print(f"{len(entries)} species -> {rows} x {cols} grid (reference cropped to {crop})")
    target_lab = srgb_to_lab(np.asarray(small)).reshape(-1, 3)

    print("Assigning species to cells...")
    species_idx, photo_idx, cost = assign(target_lab, species_tiles)
    delta_e = np.sqrt(cost)
    print(f"Color difference (CIE76 dE): mean {delta_e.mean():.1f}, median {np.median(delta_e):.1f}, max {delta_e.max():.1f}")

    out_dir = Path(args.out) / Path(args.reference).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    small.save(out_dir / "reference.png")

    cells = []
    for cell, (s, k) in enumerate(zip(species_idx, photo_idx)):
        meta, paths = entries[s]
        row, col = divmod(cell, cols)
        cells.append({
            "row": row,
            "col": col,
            "common_name": meta["common_name"],
            "scientific_name": meta["scientific_name"],
            "slug": meta["slug"],
            "file": paths[k].relative_to(images_dir).as_posix(),
            "target_lab": target_lab[cell].round(2).tolist(),
            "tile_lab": species_tiles[s][k].round(2).tolist(),
            "delta_e": round(float(delta_e[cell]), 2),
        })
    layout = {
        "reference": Path(args.reference).as_posix(),
        "reference_crop": crop,
        "rows": rows,
        "cols": cols,
        "images_dir": images_dir.as_posix(),
        "tile_size": args.tile_size,
        "cells": cells,
    }
    write_layout(out_dir, layout)
    save_mosaic(out_dir, layout, args.debug)
    print(f"Wrote {out_dir}")


def write_layout(out_dir, layout):
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
    with open(out_dir / "layout.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["row", "col", "common_name", "scientific_name", "file", "delta_e"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(layout["cells"])


def save_mosaic(out_dir, layout, debug=False):
    rows, cols, tile_size = layout["rows"], layout["cols"], layout["tile_size"]
    print(f"Rendering {cols * tile_size} x {rows * tile_size} mosaic...")
    images_dir = Path(layout["images_dir"])
    mosaic = render([images_dir / c["file"] for c in layout["cells"]], rows, cols, tile_size, labels=debug)
    mosaic.save(out_dir / ("mosaic_debug.jpg" if debug else "mosaic.jpg"), quality=92)


if __name__ == "__main__":
    main()
