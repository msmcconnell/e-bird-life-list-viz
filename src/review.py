"""Propose replacement photos for mosaic cells flagged for review.

Each flagged cell keeps its species; the other photos of that species are
ranked by how closely their color matches the cell, and the closest becomes
the proposal.

Usage:
    python src/review.py data/mosaic/Broad-billed-motmot_268880490 to_review.txt

to_review.txt has one "row,col" per line. Output, next to layout.json:
    replacements.csv   per flagged cell: current photo and ranked alternatives.
                       "choice" is the alternative to use (1 = best color
                       match by default); set it to another number, or 0 to
                       keep the current photo.
    review_sheet.jpg   current tile (red) and alternatives (green = choice),
                       each labeled "row,col"
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mosaic import center_square, open_rgb, tile_colors

TILE = 96
GAP = 6
INFO_WIDTH = 190
ENTRIES_PER_COLUMN = 30


def read_cells(path):
    cells = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row, col = (int(v) for v in line.split(","))
            if (row, col) not in cells:
                cells.append((row, col))
    return cells


def propose(layout, flagged):
    """Return one entry per flagged cell with its alternatives ranked by color."""
    images_dir = Path(layout["images_dir"])
    by_position = {(c["row"], c["col"]): c for c in layout["cells"]}
    entries = []
    for position in flagged:
        cell = by_position.get(position)
        if cell is None:
            raise SystemExit(f"{position} is outside the {layout['rows']} x {layout['cols']} grid")
        meta = json.loads((images_dir / cell["slug"] / "meta.json").read_text(encoding="utf-8"))
        files = [f"{cell['slug']}/{p['file']}" for p in meta["photos"] if not p.get("rejected")]
        files = [f for f in files if f != cell["file"] and (images_dir / f).exists()]
        alternatives = []
        if files:
            lab = tile_colors(images_dir, [images_dir / f for f in files])
            delta_e = np.sqrt(((lab - np.array(cell["target_lab"])) ** 2).sum(axis=1))
            alternatives = [(files[i], round(float(delta_e[i]), 2)) for i in np.argsort(delta_e)]
        entries.append({"cell": cell, "alternatives": alternatives})
    return entries


def write_csv(path, entries):
    max_alts = max(len(e["alternatives"]) for e in entries)
    fields = ["row", "col", "common_name", "scientific_name", "current_file", "current_delta_e", "choice"]
    for k in range(1, max_alts + 1):
        fields += [f"alt{k}_file", f"alt{k}_delta_e"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for e in entries:
            c = e["cell"]
            row = [c["row"], c["col"], c["common_name"], c["scientific_name"], c["file"], c["delta_e"],
                   1 if e["alternatives"] else 0]
            for file, delta_e in e["alternatives"]:
                row += [file, delta_e]
            writer.writerow(row)


def render_sheet(entries, images_dir):
    max_alts = max(len(e["alternatives"]) for e in entries)
    entry_width = INFO_WIDTH + (1 + max_alts) * (TILE + GAP) + 3 * GAP
    entry_height = TILE + 2 * GAP
    n_columns = -(-len(entries) // ENTRIES_PER_COLUMN)
    n_rows = min(len(entries), ENTRIES_PER_COLUMN)
    sheet = Image.new("RGB", (n_columns * entry_width, n_rows * entry_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=13)
    tag_font = ImageFont.load_default(size=TILE // 5)

    def wrap(text, width):
        lines = [""]
        for word in text.split():
            candidate = f"{lines[-1]} {word}".strip()
            if lines[-1] and draw.textlength(candidate, font=font) > width:
                lines.append(word)
            else:
                lines[-1] = candidate
        return "\n".join(lines)

    def tile(file, x, y, tag, border):
        img = center_square(open_rgb(images_dir / file, TILE)).resize((TILE, TILE), Image.LANCZOS)
        sheet.paste(img, (x, y))
        draw.rectangle((x - 3, y - 3, x + TILE + 2, y + TILE + 2), outline=border, width=3)
        box = draw.textbbox((x + 2, y + 2), tag, font=tag_font)
        draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), fill="black")
        draw.text((x + 2, y + 2), tag, font=tag_font, fill="white")

    for i, e in enumerate(entries):
        c = e["cell"]
        x0 = (i // ENTRIES_PER_COLUMN) * entry_width + GAP
        y0 = (i % ENTRIES_PER_COLUMN) * entry_height + GAP
        target = tuple(int(v) for v in np.clip(lab_to_srgb(c["target_lab"]), 0, 255))
        draw.rectangle((x0, y0, x0 + 24, y0 + TILE), fill=target)
        name = wrap(c["common_name"], INFO_WIDTH - 40)
        draw.multiline_text((x0 + 32, y0), f"{c['row']},{c['col']}\n{name}\n\ncurrent dE {c['delta_e']:.1f}",
                            font=font, fill="black", spacing=4)
        x = x0 + INFO_WIDTH
        tile(c["file"], x, y0, f"{c['row']},{c['col']}", "red")
        for k, (file, delta_e) in enumerate(e["alternatives"], 1):
            x += TILE + GAP + (GAP if k == 1 else 0)
            tile(file, x, y0, f"{c['row']},{c['col']} #{k}", "limegreen" if k == 1 else "white")
            draw.text((x + 2, y0 + TILE - 16), f"dE {delta_e:.1f}", font=font, fill="white",
                      stroke_width=2, stroke_fill="black")
    return sheet


def lab_to_srgb(lab):
    """Inverse of mosaic.srgb_to_lab for a single color, for the target swatch."""
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    delta = 6 / 29
    xyz = np.where(f > delta, f**3, 3 * delta**2 * (f - 4 / 29)) * np.array([0.95047, 1.0, 1.08883])
    linear = np.linalg.solve(np.array([[0.4124, 0.3576, 0.1805],
                                       [0.2126, 0.7152, 0.0722],
                                       [0.0193, 0.1192, 0.9505]]), xyz).clip(0, 1)
    return 255 * np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mosaic_dir", help="mosaic output directory containing layout.json")
    parser.add_argument("cells", help='text file with one "row,col" per line')
    args = parser.parse_args()

    mosaic_dir = Path(args.mosaic_dir)
    layout = json.loads((mosaic_dir / "layout.json").read_text(encoding="utf-8"))
    entries = propose(layout, read_cells(args.cells))
    without = [e["cell"]["common_name"] for e in entries if not e["alternatives"]]
    if without:
        print(f"No alternative photos for: {', '.join(without)}")

    write_csv(mosaic_dir / "replacements.csv", entries)
    render_sheet(entries, Path(layout["images_dir"])).save(mosaic_dir / "review_sheet.jpg", quality=90)
    print(f"{len(entries)} cells -> {mosaic_dir / 'replacements.csv'}, {mosaic_dir / 'review_sheet.jpg'}")


if __name__ == "__main__":
    main()
