"""Tessa Tales.

  uv run tessa.py                    open the shelf (plug in the camera card any time: the shelf notices it)
  uv run tessa.py import [SOURCE]    copy new pictures off the SD card (found automatically), let Olaf sort them, open the page
  uv run tessa.py sieve [IMPORT_ID]  let Olaf sort again after tuning config.toml (books you've made are kept)
"""

import argparse
import sys
from pathlib import Path

from tessa_tales import sieve, stories
from tessa_tales.config import load_config
from tessa_tales.library import Library
from tessa_tales.server import serve

PORT = 4747


def find_sd_card() -> Path | None:
    vols = Path("/Volumes")
    cards = [v for v in vols.iterdir() if (v / "DCIM").is_dir()] if vols.exists() else []
    if len(cards) > 1:
        sys.exit("I found more than one camera card: " + ", ".join(str(c) for c in cards) + ". Say which one.")
    return cards[0] if cards else None


def sort_pictures(lib: Library, cfg: dict, import_id: str) -> None:
    print("Olaf is sorting the pictures…")
    man = sieve.run(lib, cfg, import_id)
    sv = man["sieve"]
    print(f"  {len(man['photos'])} pictures → {sv['counts']['treasure']} warm hugs in {len(sv['moments'])} moments, "
          f"{sv['photo_counts']['snowdrift']} buried in Marshmallow's pile")
    made = stories.ensure_day_stories(lib, import_id)
    if made:
        print(f"  new stories on the shelf: {', '.join(made)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tessa Tales")
    sub = ap.add_subparsers(dest="cmd")
    imp = sub.add_parser("import", help="copy pictures off the camera card and sort them")
    imp.add_argument("source", nargs="?", help="folder to import (default: the SD card with a DCIM folder)")
    imp.add_argument("--no-open", action="store_true")
    sv = sub.add_parser("sieve", help="sort an import again")
    sv.add_argument("import_id", nargs="?")
    sub.add_parser("serve", help="open the book-making page")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-open", action="store_true", dest="no_open_top")
    args = ap.parse_args()

    cfg = load_config()
    lib = Library(cfg["home"])

    if args.cmd == "import":
        source = Path(args.source).expanduser() if args.source else find_sd_card()
        if not source or not source.is_dir():
            sys.exit("No camera card found. Plug in the SD card, or pass a folder: uv run tessa.py import ~/Desktop/photos")
        print(f"Copying pictures from {source} into {lib.originals}…")
        import_id = lib.import_card(source)
        if not import_id:
            return
        sort_pictures(lib, cfg, import_id)
        if not args.no_open:
            serve(lib, cfg, args.port, "/")
    elif args.cmd == "sieve":
        import_id = args.import_id or lib.latest_import()
        if not import_id:
            sys.exit("Nothing imported yet. Try: uv run tessa.py import")
        sort_pictures(lib, cfg, import_id)
    else:
        serve(lib, cfg, args.port, None if args.no_open_top else "/")


if __name__ == "__main__":
    main()
