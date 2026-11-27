"""Stories: many per card. By default one per day of pictures; tiny days fold into "Little things".

A story owns a pool of pictures (by sha1, so re-sorting never breaks it) and its pages.
  ~/TessaTales/stories/<id>/story.json
  { id, title, import, kind: "day"|"little"|"all", day, pool: [sha], pages: [page],
    hint, moment_names, woven_at, weave_note, new, created, updated }
  page: { id, type: "photo"|"flipbook", photo, frames, from_stack, bridge }
Thumbs-down is global (junk is junk in every story): ~/TessaTales/nope.json
"""

import json
import re
import shutil
from collections import defaultdict
from datetime import datetime

from .library import Library

MIN_DAY_PICTURES = 4     # a day with fewer warm hugs than this joins "Little things"
MAX_DRAFT_PAGES = 12
STORY_ID = re.compile(r"^[a-z0-9-]{1,64}$")


# ---------- files ----------------------------------------------------------------
def _path(lib: Library, sid: str):
    if not STORY_ID.match(sid):
        raise ValueError("bad story id")
    return lib.home / "stories" / sid / "story.json"


def exists(lib: Library, sid: str) -> bool:
    return STORY_ID.match(sid) is not None and _path(lib, sid).exists()


def load(lib: Library, sid: str) -> dict:
    return json.loads(_path(lib, sid).read_text())


def save(lib: Library, story: dict) -> None:
    path = _path(lib, story["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    story["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(story, indent=1))
    tmp.replace(path)


def list_all(lib: Library) -> list[dict]:
    out = []
    for p in sorted((lib.home / "stories").glob("*/story.json")):
        s = json.loads(p.read_text())
        if "pool" in s:
            out.append(s)
    return sorted(out, key=lambda s: (s.get("import", ""), s.get("day") if s.get("day") is not None else 999))


def load_nope(lib: Library) -> set[str]:
    p = lib.home / "nope.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def save_nope(lib: Library, nope: set[str]) -> None:
    (lib.home / "nope.json").write_text(json.dumps(sorted(nope)))


# ---------- drafting pages from pictures ---------------------------------------
def _page_score(st: dict) -> float:
    sc = st["scores"]
    return (3.0 * (st["gold"] and sc["junk"] <= 0.35) + 2.5 * st["flipbook"] + 2.0 * sc["people"] + (1 - sc["junk"])
            + min(sc["sharpness"], 1500) / 1500)


def page_from_stack(st: dict, photo: str | None = None, flip: bool | None = None) -> dict:
    flip = st["flipbook"] if flip is None else flip
    return {
        "id": f"p-{st['id']}-{datetime.now().strftime('%H%M%S%f')}",
        "type": "flipbook" if flip else "photo",
        "photo": photo or st["cover"],
        "frames": (st.get("highlights") or st["members"]) if flip else [photo or st["cover"]],
        "from_stack": st["id"],
        "bridge": None,
    }


def _draft_pages(man: dict, stack_ids: set[str]) -> list[dict]:
    sv = man["sieve"]
    stacks = {s["id"]: s for s in sv["stacks"]}
    picks = []
    for m in sv["moments"]:
        hugs = [stacks[i] for i in m["stacks"] if i in stack_ids and stacks[i]["pile"] == "treasure"]
        k = 1 if len(hugs) <= 2 else 2 if len(hugs) <= 5 else 3
        picks += sorted(hugs, key=_page_score, reverse=True)[:k]
    if len(picks) > MAX_DRAFT_PAGES:
        keep = {s["id"] for s in sorted(picks, key=_page_score, reverse=True)[:MAX_DRAFT_PAGES]}
        picks = [s for s in picks if s["id"] in keep]
    picks.sort(key=lambda s: s["start"])
    return [page_from_stack(s) for s in picks]


def _new_story(sid, title, import_id, kind, day, stacks, man) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": sid, "title": title, "import": import_id, "kind": kind, "day": day,
        "pool": [h for st in stacks for h in st["members"]],
        "pages": _draft_pages(man, {st["id"] for st in stacks}),
        "hint": "", "moment_names": {}, "woven_at": None, "weave_note": None,
        "new": True, "created": now, "updated": now,
    }


# ---------- one story per day ----------------------------------------------------
def ensure_day_stories(lib: Library, import_id: str) -> list[str]:
    """Make the default stories for a card: one per day, tiny days folded together."""
    man = lib.load_manifest(import_id)
    sv = man["sieve"]
    kept = [s for s in sv["stacks"] if s["pile"] != "snowdrift"]
    if not kept:
        return []
    made = []

    def add(sid, *args):
        if not exists(lib, sid):
            save(lib, _new_story(sid, *args, man))
            made.append(sid)

    if not sv["clock_reliable"]:
        add(f"{import_id}-all", "All the pictures", import_id, "all", None, kept)
        return made
    by_day = defaultdict(list)
    for st in kept:
        by_day[datetime.fromisoformat(st["start"]).date()].append(st)
    first = min(by_day)
    little = []
    for day in sorted(by_day):
        stacks = by_day[day]
        n = (day - first).days + 1
        if sum(1 for s in stacks if s["pile"] == "treasure") >= MIN_DAY_PICTURES:
            add(f"{import_id}-d{n}", f"Day {n}", import_id, "day", n, stacks)
        else:
            little += stacks
    if little:
        add(f"{import_id}-little", "Little things", import_id, "little", None, little)
    return made


def migrate_old_books(lib: Library) -> None:
    """The first version kept one book per card at stories/<import-id>/. Fold it into that card's day 1."""
    for p in list((lib.home / "stories").glob("*/story.json")):
        old = json.loads(p.read_text())
        if "pool" in old or not lib.manifest_path(old.get("import", "")).exists():
            continue
        import_id = old["import"]
        made = ensure_day_stories(lib, import_id)
        target = f"{import_id}-d1"
        if exists(lib, target):
            st = load(lib, target)
            st["title"] = old.get("title") or st["title"]
            pages = old.get("pages", [])
            for pg in pages:
                pg.setdefault("bridge", None)
            st["pages"] = pages
            st["moment_names"] = old.get("moment_names", {})
            st["pool"] = list(dict.fromkeys(st["pool"] + [h for pg in pages for h in pg.get("frames", [])]))
            st["new"] = False
            save(lib, st)
            old_voice = p.parent / "voice.json"
            if old_voice.exists():
                (lib.home / "stories" / target / "voice.json").write_text(old_voice.read_text())
                if (p.parent / "audio").exists():
                    shutil.copytree(p.parent / "audio", lib.home / "stories" / target / "audio", dirs_exist_ok=True)
        if old.get("nope"):
            save_nope(lib, load_nope(lib) | set(old["nope"]))
        (p.parent / "story.json").rename(p.parent / "story.v1.json")  # kept, just out of the way
        print(f"  moved the old book into {target} (+{len(made)} day stories)")


# ---------- what a story page needs ----------------------------------------------
def view(lib: Library, story: dict) -> dict:
    """The stacks and moments behind a story, plus the card's buried pictures for Marshmallow's pile."""
    man = lib.load_manifest(story["import"])
    sv = man["sieve"]
    pool = set(story["pool"])
    stacks = [s for s in sv["stacks"] if s["pile"] != "snowdrift" and pool.intersection(s["members"])]
    ids = {s["id"] for s in stacks}
    moments = []
    for m in sv["moments"]:
        mine = [i for i in m["stacks"] if i in ids]
        if mine:
            moments.append({**m, "stacks": mine})
    drift = [s for s in sv["stacks"] if s["pile"] == "snowdrift"]
    return {
        "story": {**story, "nope": sorted(load_nope(lib))},
        "total": len(pool),
        "sieve": {"stacks": stacks + drift, "moments": moments, "clock_reliable": sv["clock_reliable"]},
    }


def save_from_page(lib: Library, sid: str, data: dict) -> dict:
    """Save what the story page sends. Thumbs-down goes to the global list."""
    story = load(lib, sid)
    for k in ("title", "pages", "moment_names", "hint"):
        if k in data:
            story[k] = data[k]
    story["new"] = False
    if "nope" in data:
        save_nope(lib, set(data["nope"]))
    save(lib, story)
    return story
