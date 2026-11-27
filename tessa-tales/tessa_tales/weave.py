"""Weave: turn Tessa's chatter about her pictures into a little book.

Her recordings are the pages. The weaver picks which pictures, puts them in order, suggests a
title, and writes short narrator "bridges" so a listener can follow. It never writes words for her.

It runs free on the Mac: a small open model through Apple's MLX. Nothing leaves the Mac.
"""

import json
import threading
from datetime import datetime

from . import stories
from .library import Library

DEFAULT_TITLES = {"Day", "Little things", "All the pictures"}

SYSTEM = """You help a parent turn their three-year-old daughter Tessa's chatter about her own photos into a small picture book that the family plays back together.

Each page of the book is one of her photos, played with her own recording of what she said about it. Her words are the heart of the book: never change, correct, finish or invent words for her. You do not write her lines; they play as her real voice.

Your job:
- Choose the pictures (usually 4 to 12) that tell the clearest little story. A picture she talked about should almost always be in the book. Strongly prefer pictures a grown-up hearted. Leave out near-duplicates.
- Put them in an order that flows. Usually that is time order, unless her words clearly tell the story another way.
- Write a few optional "bridges": a short narrator line (at most 12 words) that plays BEFORE her recording, so someone listening can follow the story. A bridge sets up the page; it never says what she is about to say.
  Rules for bridges:
  * Never repeat or reword Tessa's words. Her recording comes right after the bridge.
  * Never speak as Tessa. Say "Tessa" or "we", never "I" or "me".
  * Never invent actions, places, names or times. Only use the grown-up's hint, the time of day, and the order of events.
  * Most pages need no bridge. Give the first page an opening bridge, then add one only where the story jumps (a new place, a new day, or a big moment). Leave the bridge "" otherwise.
  Example. Hint: "we went to the beach". Tessa said on page 1: "big wave!" and on page 2: "sandy toes".
  Good: page 1 bridge "One sunny day, we went to the beach.", page 2 bridge "".
  Bad: page 1 bridge "Tessa sees a big wave." (repeats her), page 2 bridge "Then we built a sandcastle." (invented).
- Suggest a title of 2 to 5 words, built from her own words when you can.
- Add one short note for the grown-up: what you chose, or which picture would be worth asking her about.
- "flipbook": true only for pictures whose "frames" count is more than 2."""

JSON_RULES = """

Reply with only a JSON object and nothing else, shaped like this:
{"title": "...", "pages": [{"picture": 3, "bridge": "...", "flipbook": false}], "note_for_grownup": "..."}
"picture" is the picture's number from the list. Use "" for an empty bridge."""

_local = {}
_local_lock = threading.Lock()


def _ask_local(prompt: str, repo: str) -> str:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    with _local_lock:  # one weave at a time; the model is big
        if repo not in _local:
            _local[repo] = load(repo)
        model, tok = _local[repo]
        msgs = [{"role": "system", "content": SYSTEM + JSON_RULES}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=text, max_tokens=1500, sampler=make_sampler(temp=0.3, top_p=0.9))


def _parse(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("pages"), list) else None


def ask(prompt: str, cfg: dict) -> dict:
    for _ in range(2):  # small local models occasionally fumble the JSON; one retry
        text = _ask_local(prompt, cfg.get("local_model", "mlx-community/Qwen2.5-7B-Instruct-4bit"))
        woven = _parse(text)
        if woven:
            return woven
    raise WeaveError("The weaver's answer came back garbled twice. Try again.")


class WeaveError(Exception):
    """Something the grown-up can act on, already in plain words."""


def _words(s: str) -> set[str]:
    small = {"a", "an", "the", "and", "is", "are", "in", "on", "to", "of", "her", "she", "tessa", "we", "it", "there"}
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if w not in small}


def _echoes(bridge: str, said: str) -> bool:
    """True when a bridge mostly repeats what she said (the bridge would steal her line)."""
    b, s = _words(bridge), _words(said)
    return bool(b) and bool(s) and len(b & s) / len(b) >= 0.4


GLUE = set("""then next and so one day days morning afternoon evening night later after before time it was is are
were we us our all together first last finally home came went go goes going got get see saw look looked here there
this that now very little big new two three four five six seven eight nine ten with at to for from by of up out
back again had has have be been when while soon""".split())


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _grounded(bridge: str, known: set[str]) -> bool:
    """True when every meaningful word of a bridge comes from what we actually know."""
    return all(_stem(w) in known for w in _words(bridge) if w not in GLUE and not w.isdigit())


def _when(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%-I:%M %p").lower()


def _candidates(lib: Library, story: dict, talk: dict, cover_words: str | None):
    v = stories.view(lib, story)
    stacks = [s for s in v["sieve"]["stacks"] if s["pile"] != "snowdrift"]
    nope = set(v["story"]["nope"])
    page_by_stack = {}
    for pg in story["pages"]:
        for s in stacks:
            if pg["photo"] in s["members"]:
                page_by_stack[s["id"]] = pg
    moment_of = {sid: m.get("name", "") for m in v["sieve"]["moments"] for sid in m["stacks"]}
    numbered = []
    lines = []
    for s in sorted(stacks, key=lambda s: s["start"]):
        if all(h in nope for h in s["members"]):
            continue
        said = next((talk[h] for h in [s["cover"], *s["members"]] if h in talk and talk[h].get("words")), None)
        page = page_by_stack.get(s["id"])
        if s["pile"] != "treasure" and not said and not page:
            continue
        numbered.append(s)
        n = len(numbered)
        bits = [f"#{n}", f"time {_when(s['start'])}", f"moment: {moment_of.get(s['id'], '')}",
                f"rough guess: {s['scores']['label']}", f"frames: {len(s.get('highlights') or [s['cover']])}"]
        if page:
            bits.append("grown-up hearted it")
        if s.get("gold"):
            bits.append("she took this one on purpose")
        bits.append(f'Tessa said: "{said["words"]}"' if said else "Tessa hasn't talked about this one")
        lines.append(" | ".join(bits))
    prompt = [f"Story so far is called: {story['title']}"]
    if cover_words:
        prompt.append(f'When asked what the story is called, Tessa said: "{cover_words}"')
    if story.get("hint"):
        prompt.append(f"The grown-up's hint about what happened: {story['hint']}")
    prompt.append("Her pictures, in the order she took them:\n" + "\n".join(lines))
    return numbered, "\n\n".join(prompt)


def weave(lib: Library, sid: str, talk: dict, story_voice: dict, cfg: dict | None = None) -> dict:
    story = stories.load(lib, sid)
    cover = story_voice.get("cover") or {}
    numbered, prompt = _candidates(lib, story, talk, cover.get("words"))
    if not numbered:
        raise WeaveError("There are no pictures to weave yet.")
    if not any("Tessa said" in line for line in prompt.splitlines()):
        raise WeaveError("Tessa hasn't talked about any of these pictures yet. Do Tessa's turn first.")

    woven = ask(prompt, cfg or {})

    old_by_stack = {}
    for pg in story["pages"]:
        for s in numbered:
            if pg["photo"] in s["members"]:
                old_by_stack[s["id"]] = pg
    said_by_stack = {s["id"]: next((talk[h]["words"] for h in [s["cover"], *s["members"]]
                                    if h in talk and talk[h].get("words")), "") for s in numbered}
    picked = []
    for wp in woven.get("pages", []):
        try:
            i = int(wp.get("picture", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(numbered) and i not in [j for j, _ in picked]:
            picked.append((i, wp))
    # every picture she talked about belongs in the book; add any the weaver skipped, in time order
    for i, s in enumerate(numbered):
        if said_by_stack[s["id"]] and i not in [j for j, _ in picked]:
            at = next((k for k, (j, _) in enumerate(picked) if j > i), len(picked))
            picked.insert(at, (i, {"bridge": ""}))
    v = stories.view(lib, story)
    moment_of = {sid: m for m in v["sieve"]["moments"] for sid in m["stacks"]}
    names = story.get("moment_names", {})
    known_text = " ".join([story.get("hint", ""), story["title"], *said_by_stack.values(),
                           *[names.get(m["id"], m["name"]) for m in v["sieve"]["moments"]]])
    known = {_stem(w) for w in _words(known_text)}
    pages, seen, last_day = [], set(), None
    for i, wp in picked:
        seen.add(i)
        st = numbered[i]
        old = old_by_stack.get(st["id"])
        flip = bool(st.get("flipbook")) and len(st.get("highlights") or []) >= 3
        page = stories.page_from_stack(st, photo=old["photo"] if old and not flip else None, flip=flip)
        bridge = (wp.get("bridge") or "").strip()[:140]
        if bridge and (_echoes(bridge, said_by_stack[st["id"]]) or bridge.lower().startswith(("i ", "me ", "my "))
                       or not _grounded(bridge, known)):
            bridge = ""  # it repeated her, spoke as her, or made something up
        day = st["start"][:10]
        if not bridge and pages and day != last_day and st["id"] in moment_of:
            m = moment_of[st["id"]]
            name = names.get(m["id"], m["name"])
            bridge = name[0].upper() + name[1:] + "."  # a true fact: "The next morning."
        last_day = day
        page["bridge"] = bridge or None
        pages.append(page)
    if not pages:
        raise WeaveError("The weaver didn't pick any pictures. Try again.")

    story["previous_pages"] = story["pages"]  # one step of undo
    story["pages"] = pages
    title = (woven.get("title") or "").strip()[:60]
    story["suggested_title"] = title
    if title and (story["title"].split(" ")[0] in DEFAULT_TITLES or story["title"] in DEFAULT_TITLES):
        story["title"] = title
    story["weave_note"] = (woven.get("note_for_grownup") or "").strip()[:300]
    story["woven_at"] = datetime.now().isoformat(timespec="seconds")
    stories.save(lib, story)
    return story
