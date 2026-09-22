"""Drop what is not a candidate, then print the ranking.

This file used to be two hundred and fifty lines of rules: a syllable floor
on the formal name, a required shape for the short form, and a spelling-key
plus edit-distance plus union-find family grouper. All of it was written by
looking at what came out wrong and inventing a rule to stop it, and all of
it was fitted to the nine names rated 2 or 3 - the same mistake as fitting
eight weights to forty-eight ratings, made somewhere nothing would catch it.

The syllable floor was the worst. It deleted 1,485 names the model considers
perfectly good formal names - Vera, Grace, Miriam, Marian, Talia - on the
strength of nine examples.

So the rules are gone, and the two data-quality facts they were groping at
are asked instead, in taste_match.py, as questions riding in the request
each name already costs. Both are safe on the names rated 2 or 3:
is_short_form tops out at 0.18 there and is_respelling at 0.36, so a 0.5
cut cannot delete a favourite. That is re-checked here at run time rather
than trusted.

What stays in code is bookkeeping rather than judgment: a name they have
already rated is not a recommendation.

Usage:
    python shortlist.py
    python shortlist.py --top 40 --keep-everything    # no filtering at all
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).parent

# Each is a probability from taste_match.py, cut well above where the names
# they liked actually sit. The check below refuses any cut that would delete
# one of them.
CUTS = [
    ("is_short_form", 0.5, "the formal name is itself a pet form"),
    ("is_respelling", 0.5, "a variant spelling of a more standard name"),
    ("no_real_nickname", 0.25, "no real short form was on offer"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taste", default=str(HERE / "taste.json"))
    ap.add_argument("--named", default=str(HERE / "named.json"))
    ap.add_argument("--ratings", default=str(HERE / "ratings.json"))
    ap.add_argument("--out", default=str(HERE / "shortlist.json"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--keep-everything", action="store_true",
                    help="apply nothing; show the raw ranking")
    args = ap.parse_args()

    import fit_weights as fw

    taste = json.loads(pathlib.Path(args.taste).read_text())
    named = {r["id"]: r for r in json.loads(pathlib.Path(args.named).read_text())}
    ratings = fw.load_ratings(pathlib.Path(args.ratings))
    mean_rating = {n: sum(v.values()) / len(v) for n, v in ratings.items()}

    for row in taste:                   # no_real_nickname lives in named.json
        row.setdefault("no_real_nickname",
                       named.get(row["id"], {}).get("no_real_nickname", 0.0))
    by_id = {r["id"]: r for r in taste}

    # --- the check this project insists on -------------------------------
    liked = [n for n, r in mean_rating.items() if r >= 2 and n in by_id]
    print(f"Checking every cut against the {len(liked)} names rated >= 2 in the pool")
    for field, thr, why in CUTS:
        worst = max(by_id[n][field] for n in liked)
        print(f"  {field:<18} highest among them {worst:.2f}   cut at {thr}   "
              f"{'ok' if worst <= thr else 'REFUSED'}")
        if worst > thr:
            print(f"  -> would delete {[n for n in liked if by_id[n][field] > thr]}")
            return 1

    kept, dropped = [], collections.Counter()
    for row in sorted(taste, key=lambda r: -r["taste_fit"]):
        if row["id"] in mean_rating:
            dropped["already rated - not a recommendation"] += 1
            continue
        hit = None if args.keep_everything else next(
            (why for f, thr, why in CUTS if row[f] > thr), None)
        if hit:
            dropped[hit] += 1
            continue
        rec = named.get(row["id"], {})
        kept.append({**row, "peak_pct": rec.get("peak_pct"),
                     "rank_2024": rec.get("rank_2024"),
                     "nickname_confidence": rec.get("nickname_confidence")})

    pathlib.Path(args.out).write_text(json.dumps(kept, indent=2))
    print(f"\n{len(taste)} ranked  ->  {len(kept)} candidates")
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
