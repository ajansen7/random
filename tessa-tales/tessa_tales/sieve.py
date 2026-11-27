"""The snow sieve: bury the junk, stack what's left, find the moments.

1. Every picture gets scored on its own (CLIP + brightness math).
2. Junk frames are buried in the snowdrift first, so a burst interrupted by black frames
   (a screen in a dark car) still stacks into one run.
3. Kept pictures stack while the scene looks the same, then neighbouring look-alike stacks merge.
4. Moments: a long stretch with nothing worth keeping starts a new moment.

Promises (never buried): gold-star pictures (a lone click with quiet before and after)
and anything that might have a person in it. Nothing is ever deleted.
"""

import time
from datetime import datetime

import numpy as np

from .library import Library
from .photos import natural_key, open_small, phash_distance

PILES = ("treasure", "maybe", "snowdrift")
CLIP_INPUT_SIDE = 256


def pretty(prompt: str) -> str:
    for lead in ("a blurry photo of ", "a photo of the ", "a photo of ", "a "):
        if prompt.startswith(lead):
            return prompt[len(lead):]
    return prompt


def _taken(feat: dict, photo: dict) -> datetime:
    if feat.get("exif_time"):
        return datetime.fromisoformat(feat["exif_time"])
    return datetime.fromtimestamp(photo["mtime"])


def clock_is_reliable(times: list[datetime]) -> bool:
    if len(times) < 2:
        return True
    years = {t.year for t in times}
    if min(years) < 2015 or max(years) > datetime.now().year + 1:
        return False
    return len(set(times)) > max(1, len(times) // 10)


def moment_name(start: datetime, first: datetime) -> str:
    h = start.hour
    part = "night" if h < 5 else "morning" if h < 12 else "afternoon" if h < 17 else "evening" if h < 21 else "night"
    days = (start.date() - first.date()).days
    if days == 0:
        return f"in the {part}"
    if days == 1:
        return f"the next {part}"
    return f"{days} days later, in the {part}"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def run(lib: Library, cfg: dict, import_id: str, clip=None, log=print) -> dict:
    man = lib.load_manifest(import_id)
    photos = man["photos"]
    b, s, sv, mo = cfg["bursts"], cfg["stacks"], cfg["sieve"], cfg["moments"]
    t0 = time.time()

    # -- features and CLIP numbers (both cached) --------------------------------
    feats = {p["sha1"]: lib.features(p) for p in photos}
    if clip is None:
        from .clip import Clip
        clip = Clip(sv["model"], sv["pretrained"])
    missing = [p for p in photos if lib.load_embedding(p["sha1"], clip.tag) is None]
    if missing:
        imgs = [open_small(lib.thumb_path(p["sha1"]), CLIP_INPUT_SIDE)[0] for p in missing]
        for p, vec in zip(missing, clip.embed_images(imgs)):
            lib.save_embedding(p["sha1"], clip.tag, vec)
    emb = {p["sha1"]: lib.load_embedding(p["sha1"], clip.tag) for p in photos}
    log(f"  looked at {len(photos)} pictures ({len(missing)} new for CLIP on {clip.device})")

    # -- score every picture on its own -----------------------------------------
    pr = cfg["prompts"]
    prompts = pr["junk"] + pr["people"] + pr["things"]
    kinds = np.array(["junk"] * len(pr["junk"]) + ["people"] * len(pr["people"]) + ["things"] * len(pr["things"]))
    text = clip.embed_texts(prompts)
    score = {}
    for sha, vec in emb.items():
        logits = 100.0 * (text @ vec)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        top = int(prob.argmax())
        f = feats[sha]
        blank = ("too dark to see" if f["brightness"] < sv["too_dark"] else
                 "all white" if f["brightness"] > sv["too_bright"] else
                 "nothing there (a thumb on the lens?)" if f["spread"] < sv["too_flat"] else None)
        score[sha] = {
            "junk": round(float(prob[kinds == "junk"].sum()), 3),
            "people": round(float(prob[kinds == "people"].sum()), 3),
            "label": pretty(prompts[top]),
            "label_kind": str(kinds[top]),
            "blank": blank,
            "sharpness": f["sharpness"],
            "brightness": f["brightness"],
        }

    # -- time order ---------------------------------------------------------------
    taken = {p["sha1"]: _taken(feats[p["sha1"]], p) for p in photos}
    reliable = clock_is_reliable(list(taken.values()))
    key = (lambda p: (taken[p["sha1"]], natural_key(p["name"]))) if reliable else (lambda p: natural_key(p["name"]))
    order = [p["sha1"] for p in sorted(photos, key=key)]
    secs = {h: (taken[h] - taken[order[0]]).total_seconds() if reliable else i * 5.0 for i, h in enumerate(order)}

    # -- gold stars: lone clicks with quiet on both sides (judged on every frame) --
    gold = set()
    if reliable:
        bursts = [[order[0]]] if order else []
        for h in order[1:]:
            if secs[h] - secs[bursts[-1][-1]] <= b["gap_seconds"]:
                bursts[-1].append(h)
            else:
                bursts.append([h])
        for i, burst in enumerate(bursts):
            before = secs[burst[0]] - secs[bursts[i - 1][-1]] if i else float("inf")
            after = secs[bursts[i + 1][0]] - secs[burst[-1]] if i + 1 < len(bursts) else float("inf")
            if len(burst) <= b["standout_max_photos"] and min(before, after) >= b["standout_quiet_seconds"]:
                gold.update(burst)

    # -- bury junk first ----------------------------------------------------------
    def buried(h):
        sc = score[h]
        if h in gold:
            return False
        if sc["blank"]:
            return True  # a black/white frame can't show a person, whatever CLIP guesses
        return sc["junk"] >= sv["junk_high"] and sc["people"] < sv["person_protect"]

    kept = [h for h in order if not buried(h)]
    drift = [h for h in order if buried(h)]

    # -- stack the kept pictures --------------------------------------------------
    stacks: list[list[str]] = []
    cent: list[np.ndarray] = []
    for h in kept:
        v = emb[h]
        if stacks and h not in gold and stacks[-1][-1] not in gold:
            last = stacks[-1][-1]
            close = secs[h] - secs[last] <= s["stack_gap_seconds"]
            alike = float(_unit(cent[-1]) @ v) >= s["same_scene"] or \
                phash_distance(feats[last]["phash"], feats[h]["phash"]) <= s["same_look_bits"]
            if close and alike:
                stacks[-1].append(h)
                cent[-1] = cent[-1] + v
                continue
        stacks.append([h])
        cent.append(v.copy())

    # merge neighbouring stacks that look alike (a cartoon screen filmed in bursts)
    merged = True
    while merged:
        merged = False
        for i in range(len(stacks) - 1):
            a, c = stacks[i], stacks[i + 1]
            if a[0] in gold or c[0] in gold:
                continue
            if secs[c[0]] - secs[a[-1]] <= s["merge_gap_seconds"] and \
                    float(_unit(cent[i]) @ _unit(cent[i + 1])) >= s["merge_scene"]:
                stacks[i] = a + c
                cent[i] = cent[i] + cent[i + 1]
                del stacks[i + 1], cent[i + 1]
                merged = True
                break

    # -- describe each stack ------------------------------------------------------
    def cover_of(members):
        return max(members, key=lambda h: (score[h]["blank"] is None, score[h]["sharpness"] * (1.2 - score[h]["junk"])))

    def highlights_of(members):
        """The different-looking pictures in a stack, sharpest of each look, in order."""
        same, most = s.get("highlight_same", 0.93), s.get("highlight_max", 10)
        runs: list[list[str]] = []
        for h in members:
            if runs and float(emb[runs[-1][0]] @ emb[h]) >= same:
                runs[-1].append(h)
            else:
                runs.append([h])
        picks = [max(r, key=lambda h: score[h]["sharpness"] * (1.2 - score[h]["junk"])) for r in runs]
        if len(picks) > most:
            picks = [picks[round(i * (len(picks) - 1) / (most - 1))] for i in range(most)]
        return picks

    out = []
    for n, members in enumerate(stacks):
        cover = cover_of(members)
        sc = score[cover]
        is_gold = cover in gold
        reasons = []
        if is_gold:
            reasons.append("quiet before and after, so she meant this one")
        reasons.append(f"looks like {sc['label']}" if sc["label_kind"] == "junk" else f"I think I see {sc['label']}")
        if sc["blank"]:
            reasons.append(sc["blank"])
        treasure = (is_gold and not sc["blank"] and sc["junk"] < sv["junk_high"]) or \
            sc["junk"] <= sv["junk_low"] or sc["people"] >= sv["person_protect"]
        hl = highlights_of(members)
        flip = len(members) >= b["flipbook_min_photos"] and len(hl) >= 3
        if flip:
            reasons.append(f"{len(members)} in a row: flip-book!")
        out.append({
            "id": f"s{n + 1}", "pile": "treasure" if treasure else "maybe", "gold": is_gold, "flipbook": flip,
            "cover": cover, "members": members, "highlights": hl,
            "start": taken[members[0]].isoformat(), "end": taken[members[-1]].isoformat(),
            "reasons": reasons, "scores": sc,
        })

    # snowdrift: buried frames, bunched by time just for browsing
    runs: list[list[str]] = []
    for h in drift:
        if runs and secs[h] - secs[runs[-1][-1]] <= b["gap_seconds"]:
            runs[-1].append(h)
        else:
            runs.append([h])
    for n, members in enumerate(runs):
        cover = cover_of(members)
        sc = score[cover]
        out.append({
            "id": f"d{n + 1}", "pile": "snowdrift", "gold": False, "flipbook": False,
            "cover": cover, "members": members,
            "start": taken[members[0]].isoformat(), "end": taken[members[-1]].isoformat(),
            "reasons": [sc["blank"] or f"looks like {sc['label']}"], "scores": sc,
        })

    # -- moments: a long stretch with nothing kept starts a new one ---------------
    kept_stacks = [st for st in out if st["pile"] != "snowdrift"]
    moments = []
    for st in kept_stacks:
        start = datetime.fromisoformat(st["start"])
        if moments and (start - datetime.fromisoformat(moments[-1]["end"])).total_seconds() < mo["gap_minutes"] * 60:
            moments[-1]["stacks"].append(st["id"])
            moments[-1]["end"] = st["end"]
            continue
        moments.append({"id": f"m{len(moments) + 1}", "start": st["start"], "end": st["end"], "stacks": [st["id"]]})
    first = taken[order[0]] if order else datetime.now()
    for m in moments:
        m["name"] = moment_name(datetime.fromisoformat(m["start"]), first) if reliable else f"part {m['id'][1:]}"

    man["sieve"] = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "model": clip.tag,
        "clock_reliable": reliable,
        "counts": {k: sum(1 for st in out if st["pile"] == k) for k in PILES},
        "photo_counts": {k: sum(len(st["members"]) for st in out if st["pile"] == k) for k in PILES},
        "stacks": out,
        "moments": moments,
    }
    lib.save_manifest(man)
    log(f"  sieved in {time.time() - t0:.1f}s")
    return man
