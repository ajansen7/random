"""A book draft: Olaf's first guess at pages, which the grown-up then rearranges.

Pages keep their own picture lists, so re-running the sieve never breaks a book.
"""

import json
import re
from datetime import datetime

from .library import Library

MAX_PAGES = 12


def _page_score(st: dict) -> float:
    sc = st["scores"]
    return (3.0 * (st["gold"] and sc["junk"] <= 0.35) + 2.5 * st["flipbook"] + 2.0 * sc["people"] + (1 - sc["junk"])
            + min(sc["sharpness"], 1500) / 1500)


def page_from_stack(st: dict, photo: str | None = None, as_flipbook: bool | None = None) -> dict:
    flip = st["flipbook"] if as_flipbook is None else as_flipbook
    return {
        "id": f"p-{st['id']}-{datetime.now().strftime('%H%M%S%f')}",
        "type": "flipbook" if flip else "photo",
        "photo": photo or st["cover"],
        "frames": (st.get("highlights") or st["members"]) if flip else [photo or st["cover"]],
        "from_stack": st["id"],
        "star": False,
    }


def draft(man: dict) -> dict:
    """Pick the warmest hugs from every moment, in time order."""
    sv = man["sieve"]
    stacks = {s["id"]: s for s in sv["stacks"]}
    picks = []
    for m in sv["moments"]:
        hugs = [stacks[i] for i in m["stacks"] if stacks[i]["pile"] == "treasure"]
        k = 1 if len(hugs) <= 2 else 2 if len(hugs) <= 5 else 3
        picks += sorted(hugs, key=_page_score, reverse=True)[:k]
    if len(picks) > MAX_PAGES:
        keep = {s["id"] for s in sorted(picks, key=_page_score, reverse=True)[:MAX_PAGES]}
        picks = [s for s in picks if s["id"] in keep]
    picks.sort(key=lambda s: s["start"])
    return {
        "id": man["id"],
        "import": man["id"],
        "title": "Coming home",
        "moment_names": {},
        "nope": [],
        "pages": [page_from_stack(s) for s in picks],
        "created": datetime.now().isoformat(timespec="seconds"),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }


SAFE_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def book_path(lib: Library, book_id: str):
    if not SAFE_ID.match(book_id):
        raise ValueError("bad book id")
    return lib.home / "stories" / book_id / "story.json"


def load_or_draft(lib: Library, import_id: str) -> dict:
    path = book_path(lib, import_id)
    if path.exists():
        return json.loads(path.read_text())
    book = draft(lib.load_manifest(import_id))
    save(lib, book)
    return book


def save(lib: Library, book: dict) -> None:
    path = book_path(lib, book["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    book["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(book, indent=1))
    tmp.replace(path)
