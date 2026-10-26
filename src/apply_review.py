"""Apply the choices in replacements.csv (from review.py) to a mosaic.

Usage:
    python src/apply_review.py data/mosaic/Broad-billed-motmot_268880490 [--debug]

For every row with choice > 0, the cell's photo is swapped for that alternative
and the replaced photo is marked "rejected" in its species' meta.json, so
mosaic.py and review.py never use it again. layout.json, layout.csv and
mosaic.jpg are rewritten; --debug also rewrites mosaic_debug.jpg.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mosaic import save_mosaic, tile_colors, write_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mosaic_dir", help="mosaic output directory containing layout.json and replacements.csv")
    parser.add_argument("--debug", action="store_true", help='also write mosaic_debug.jpg with "row,col" labels')
    args = parser.parse_args()

    mosaic_dir = Path(args.mosaic_dir)
    layout = json.loads((mosaic_dir / "layout.json").read_text(encoding="utf-8"))
    images_dir = Path(layout["images_dir"])
    by_position = {(c["row"], c["col"]): c for c in layout["cells"]}

    changes = []  # (cell, chosen file)
    with open(mosaic_dir / "replacements.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            where = f"{row['row']},{row['col']}"
            choice = int(row["choice"] or 0)
            if choice == 0:
                continue
            chosen = row.get(f"alt{choice}_file")
            if not chosen:
                raise SystemExit(f"{where}: choice {choice} has no alternative")
            cell = by_position[(int(row["row"]), int(row["col"]))]
            if cell["file"] == chosen:
                continue  # already applied
            if cell["file"] != row["current_file"]:
                raise SystemExit(f"{where}: layout has {cell['file']}, not {row['current_file']}; "
                                 "re-run review.py to refresh replacements.csv")
            changes.append((cell, chosen))
    if not changes:
        print("Nothing to apply")
        return

    labs = tile_colors(images_dir, [images_dir / chosen for _, chosen in changes])
    rejected = {}  # slug -> set of rejected file names
    before, after = [], []
    for (cell, chosen), lab in zip(changes, labs):
        rejected.setdefault(cell["slug"], set()).add(Path(cell["file"]).name)
        before.append(cell["delta_e"])
        cell["file"] = chosen
        cell["tile_lab"] = lab.round(2).tolist()
        cell["delta_e"] = round(float(np.sqrt(((lab - np.array(cell["target_lab"])) ** 2).sum())), 2)
        after.append(cell["delta_e"])

    for slug, files in rejected.items():
        meta_path = images_dir / slug / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for photo in meta["photos"]:
            if photo["file"] in files:
                photo["rejected"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    write_layout(mosaic_dir, layout)
    print(f"Replaced {len(changes)} photos (mean dE {np.mean(before):.1f} -> {np.mean(after):.1f}); "
          f"marked {len(changes)} photos rejected")
    save_mosaic(mosaic_dir, layout)
    if args.debug:
        save_mosaic(mosaic_dir, layout, debug=True)
    print(f"Wrote {mosaic_dir}")


if __name__ == "__main__":
    main()
