"""Re-render an existing mosaic layout, e.g. at a larger tile size.

Tiles stay exactly where layout.json puts them (including review changes);
nothing is re-solved.

Usage:
    python src/render.py data/mosaic/Broad-billed-motmot_268880490 --tile-size 100 [--debug]

Rewrites mosaic.jpg (and mosaic_debug.jpg with --debug) and records the new
tile size in layout.json.
"""
import argparse
import json
from pathlib import Path

from mosaic import save_mosaic, write_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mosaic_dir", help="mosaic output directory containing layout.json")
    parser.add_argument("--tile-size", type=int, help="pixels per tile (default: the size in layout.json)")
    parser.add_argument("--debug", action="store_true", help='also write mosaic_debug.jpg with "row,col" labels')
    args = parser.parse_args()

    mosaic_dir = Path(args.mosaic_dir)
    layout = json.loads((mosaic_dir / "layout.json").read_text(encoding="utf-8"))
    if args.tile_size and args.tile_size != layout["tile_size"]:
        layout["tile_size"] = args.tile_size
        write_layout(mosaic_dir, layout)
    save_mosaic(mosaic_dir, layout)
    if args.debug:
        save_mosaic(mosaic_dir, layout, debug=True)
    print(f"Wrote {mosaic_dir}")


if __name__ == "__main__":
    main()
