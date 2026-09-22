"""Turn the ranking into a shortlist: drop what is not actually a new name.

taste_match.py ranks well and its top thirty is useless, for two reasons that
are both facts about the SSA data rather than anything to do with taste:

  - The source lists every spelling that cleared the top 1000 in any year, so
    Margaret arrives with Margarett, Margarete, Margeret, Margaretha,
    Margarite and Margarette behind it. They are one name.
  - Ranking a name they have already rated is not a recommendation. Margaret
    at the top is the validation passing, not a suggestion.

So names are grouped into families, only the commonest spelling of each
family survives, and any family containing a rated name is dropped whole.
Both are done in code because both are lookups, and both are checked against
the names rated 2 or 3 before being applied - the rule in this project is
that a filter which deletes a name they liked is a wrong filter.

Usage:
    python shortlist.py                     # taste.json -> the shortlist
    python shortlist.py --top 40 --show-dropped
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein, given up on once it passes `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def spelling_key(name: str) -> str:
    """Collapse the spellings English uses interchangeably in given names.

    Not a phonetic algorithm - Soundex and friends are far too aggressive
    here, merging Clara with Claire and Nora with Norah *and* Nara. This only
    folds the substitutions that actually generate SSA variant spellings.
    """
    n = name.lower()
    for a, b in (("ph", "f"), ("ck", "k"), ("qu", "kw"), ("x", "ks"),
                 ("th", "t"), ("ae", "e"), ("oe", "e")):
        n = n.replace(a, b)
    out = []
    for ch in n:
        if ch in "yj":
            ch = "i"
        elif ch == "c":
            ch = "k"
        elif ch == "z":
            ch = "s"
        elif ch == "w":
            ch = "v"
        if not (out and out[-1] == ch):       # Anne -> ane, Elissa -> elisa
            out.append(ch)
    n = "".join(out)
    if n.endswith("e") and len(n) > 4:        # Madeline / Madelyn -> madelin
        n = n[:-1]
    return n


def families(names: list[str]) -> dict[str, str]:
    """{name: family id}. Same collapsed spelling, or one edit apart on it."""
    by_key = collections.defaultdict(list)
    for n in names:
        by_key[spelling_key(n)].append(n)

    keys = sorted(by_key)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    # One edit apart on the collapsed key, and only for keys long enough that
    # one edit is a spelling slip rather than a different name. Ava/Ada/Ana
    # are three names; Madelin/Madalin are one.
    by_len = collections.defaultdict(list)
    for k in keys:
        for L in (len(k) - 1, len(k), len(k) + 1):
            by_len[L].append(k)
    for k in keys:
        if len(k) < 6:
            continue
        for other in by_len[len(k)]:
            if other > k and len(other) >= 6 and edit_distance(k, other, 1) <= 1:
                union(k, other)

    return {n: find(spelling_key(n)) for n in names}


def good_shape(nickname: str) -> bool:
    """Two syllables, ending in a vowel sound. Tessa, Maggie, Millie, Nora."""
    import build_corpus as bc
    return bc.syllables(nickname) == 2 and nickname[-1].lower() in "aeiouy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taste", default=str(HERE / "taste.json"))
    ap.add_argument("--named", default=str(HERE / "named.json"))
    ap.add_argument("--scored", default=str(HERE / "scored.json"))
    ap.add_argument("--ratings", default=str(HERE / "ratings.json"))
    ap.add_argument("--confounds", default=str(HERE / "confounds.json"))
    ap.add_argument("--out", default=str(HERE / "shortlist.json"))
    ap.add_argument("--top", type=int, default=30)
    # The SSA lists record whatever went on the birth certificate, so Maggie,
    # Meg, Tess, Trudy and Tammy are all in the corpus as formal names. They
    # rank high for the obvious reason - they sound like the short forms this
    # family likes - and they are not candidates, because the thing being
    # chosen is a long formal name with a short form hanging off it.
    #
    # Checked below rather than assumed: all nine names rated 2 or 3 have a
    # formal name of three syllables or more, and every name that goes down
    # to two was rated 1 or 0. The floor deletes nothing they liked.
    ap.add_argument("--min-syllables", type=int, default=3,
                    help="floor on the formal name; 0 disables it")
    # A requirement, not a preference, and the difference matters. Every one
    # of the nine names rated 2 or 3 is called something two syllables long
    # ending in a vowel - Tessa, Maggie, Evie, Nora, Millie, Becca, Vera,
    # Dora. So is most of what they rejected (29 of 33), so this predicts
    # nothing and cannot be used to rank; what it does is throw out Mel, Mo,
    # Lu, Marj, Mer and Nic, which are real short forms of the wrong shape.
    ap.add_argument("--any-nickname-shape", action="store_true",
                    help="keep short forms that are not two syllables ending in a vowel")
    # Probability the Choice stage left on "none of these options is a real
    # name", which tops out at 0.01 across the names they liked. Distinct
    # from low confidence, which is spread across several real options and
    # must not be filtered - see the Magdalena note below.
    ap.add_argument("--max-no-real", type=float, default=0.25)
    ap.add_argument("--show-dropped", action="store_true")
    args = ap.parse_args()

    import build_corpus as bc
    import fit_weights as fw

    taste = json.loads(pathlib.Path(args.taste).read_text())
    named = {r["id"]: r for r in json.loads(pathlib.Path(args.named).read_text())}
    scored = {r["id"]: r for r in json.loads(pathlib.Path(args.scored).read_text())}
    ratings = fw.load_ratings(pathlib.Path(args.ratings))
    mean_rating = {n: sum(v.values()) / len(v) for n, v in ratings.items()}

    fam = families([r["formal"] for r in taste] + [s["formal"] for s in scored.values()])

    # --- the check this project insists on -------------------------------
    # Every filter below gets run against the names rated 2 or 3 first. If
    # one of them would be deleted for a reason other than "they already
    # rated it", the filter is wrong and does not ship.
    liked = {nid for nid, r in mean_rating.items() if r >= 2}
    print(f"Checking the filters against the {len(liked)} names rated >= 2")
    worst = min(bc.syllables(scored[n]["formal"]) for n in liked if n in scored)
    print(f"  fewest syllables among them: {worst}   (floor is {args.min_syllables})")
    if args.min_syllables > worst:
        print("  -> the floor would delete a name they liked. Refusing to apply it.")
        return 1
    # Not applied, and this is why: the Choice stage is least sure exactly
    # where it matters most. Magdalena -> Maggie came back at 0.42 and is a
    # rated 3; Genevieve -> Evie at 0.49 and is a rated 2. Any confidence
    # floor above 0.4 deletes one of their favourites, so nickname
    # confidence is carried through as information and never as a filter.
    floor = min(named[n]["nickname_confidence"] for n in liked if n in named)
    print(f"  lowest nickname confidence among them: {floor:.2f}  "
          f"-> no confidence filter (it would cut Magdalena)")
    shaped = sum(good_shape(scored[n]["nickname"]) for n in liked if n in scored)
    print(f"  two-syllable vowel-final short form: {shaped}/{len(liked)}"
          f"   (floor {'off' if args.any_nickname_shape else 'on'})")
    if not args.any_nickname_shape and shaped < len(liked):
        print("  -> the shape rule would delete a name they liked. Refusing to apply it.")
        return 1
    worst_no_real = max(named[n]["no_real_nickname"] for n in liked if n in named)
    print(f"  highest no_real_nickname among them: {worst_no_real:.2f} "
          f"  (cut at {args.max_no_real})")
    if worst_no_real > args.max_no_real:
        print("  -> that cut would delete a name they liked. Refusing to apply it.")
        return 1

    # --- families that contain something already rated -------------------
    rated_families = {fam[scored[nid]["formal"]] for nid in mean_rating if nid in scored}

    # --- commonest spelling wins -----------------------------------------
    best_in_family: dict[str, dict] = {}
    for row in taste:
        rec = named.get(row["id"])
        peak = rec.get("peak_pct", 0) if rec else 0
        f = fam[row["formal"]]
        if f not in best_in_family or peak > best_in_family[f]["_peak"]:
            best_in_family[f] = {**row, "_peak": peak}

    kept, dropped = [], collections.Counter()
    seen_family = set()
    for row in sorted(taste, key=lambda r: -r["taste_fit"]):
        f = fam[row["formal"]]
        if args.min_syllables and bc.syllables(row["formal"]) < args.min_syllables:
            dropped["a short form, not a formal name"] += 1
            continue
        if not args.any_nickname_shape and not good_shape(row["nickname"]):
            dropped["short form is the wrong shape (Mel, Mo, Marj)"] += 1
            continue
        if named.get(row["id"], {}).get("no_real_nickname", 0) > args.max_no_real:
            dropped["no real short form was on offer"] += 1
            continue
        if f in rated_families:
            dropped["already rated, or a respelling of one"] += 1
            continue
        if best_in_family[f]["id"] != row["id"]:
            dropped["a rarer spelling of another candidate"] += 1
            continue
        if f in seen_family:
            dropped["a rarer spelling of another candidate"] += 1
            continue
        seen_family.add(f)
        rec = named.get(row["id"], {})
        kept.append({**row, "peak_pct": rec.get("peak_pct"),
                     "rank_2024": rec.get("rank_2024"),
                     "nickname_confidence": rec.get("nickname_confidence")})

    pathlib.Path(args.out).write_text(json.dumps(kept, indent=2))
    print(f"\n{len(taste)} ranked  ->  {len(kept)} distinct new names")
    for reason, n in dropped.most_common():
        print(f"  -{n:>5}  {reason}")

    print(f"\nTop {args.top}")
    print(f"  {'fit':>4}  {'called':<10} {'formal':<14} {'2024':>6}  nearest favourite")
    for row in kept[: args.top]:
        rank = f"#{row['rank_2024']}" if row.get("rank_2024") else "-"
        print(f"  {row['taste_fit']:.2f}  {row['nickname']:<10} {row['formal']:<14}"
              f" {rank:>6}  {row['nearest_favourite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
