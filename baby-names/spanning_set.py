"""Choose the next names to rate so that the ratings can carry signal.

The fifty names in names.json were all picked by hand as plausible, which
means they were all picked out of one corner of the space. Across them every
trait sits within a sliver of its range - sd 0.07 to 0.21 on 0-1 - and a
trait that does not vary cannot explain a rating that does. That is why the
weight fit came out worse than a constant, and no amount of modelling fixes
it from this side: the information is not in the data.

So the next batch is chosen to spread out rather than to look promising. But
spreading out as far as possible is also wrong: the first attempt maximised
distance in trait space and came back with John, Timothy, Princess, Diamond
and Treasure. Those would all be rated 0, which re-creates the original
problem the other way up - no variance in the ratings this time instead of
none in the traits, and an hour of someone's attention spent confirming that
they do not want to call their daughter Diamond.

What is wanted is variance in both. So the pool is cut into bands by
predicted reaction, and an equal number is drawn from each band, picking
within a band the names least like the ones already taken. That guarantees
a spread of likely ratings, a spread of traits, and every name in the batch
still a real girl's name someone could plausibly say yes to.

Three things are pinned in regardless:

  - the anchors, so the new batch stays comparable to the old one;
  - a handful of already-rated names spread across the scale, so drift
    between rating sessions is measurable rather than assumed;
  - anything already rated 2 or 3, so the batch contains enough positives
    to learn from at all.

Usage:
    python spanning_set.py --n 72 --out names_v2.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

TRAITS = [
    "era", "nickname_transformation", "sound_texture", "child_adult_range",
    "currently_common", "recognisable", "clashes_with_sibling", "register_match",
]
ANCHORS = ["theresa", "margaret", "rebecca"]
# The family's own names. The first sampling run offered Hannah, which is the
# mother's - not a name to put in front of them for a blind gut reaction.
FAMILY = {"hannah", "tessa", "theresa", "alex", "alexandra"}


def spread(X: np.ndarray, n: int, seeded: list[int]) -> list[int]:
    """Farthest-point sampling: each pick is the one least like what we have."""
    chosen = list(seeded)
    if not chosen:
        chosen = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    d = np.min(np.linalg.norm(X[:, None, :] - X[chosen][None, :, :], axis=2), axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(d))
        if d[nxt] <= 0:
            break
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default=str(HERE / "scored_corpus.json"))
    ap.add_argument("--named", default=str(HERE / "named.json"))
    ap.add_argument("--rated-scored", default=str(HERE / "scored.json"))
    ap.add_argument("--ratings", default=str(HERE / "ratings.json"))
    ap.add_argument("--taste", default=str(HERE / "taste.json"))
    ap.add_argument("--out", default=str(HERE / "names_v2.json"))
    ap.add_argument("--n", type=int, default=72)
    ap.add_argument("--carry-over", type=int, default=8,
                    help="already-rated names to repeat, for drift")
    ap.add_argument("--bands", type=int, default=6,
                    help="predicted-reaction bands to draw equally from")
    # Stricter than shortlist.py's 0.25, and it is allowed to be: this
    # chooses what to *ask* about, not what to recommend, and a rating
    # collected on Hope/"Hop" or Fatima/"Fat" is a wasted rating. The bad
    # pairs sit at 0.08-0.25 and the good ones at 0.00-0.10, so 0.05 splits
    # them - and the names rated 2 or 3 top out at 0.01, so it still clears
    # the rule that a filter must not delete one of those.
    ap.add_argument("--max-no-real", type=float, default=0.05)
    args = ap.parse_args()

    import fit_weights as fw

    corpus = json.loads(pathlib.Path(args.scored).read_text())
    named = {r["id"]: r for r in json.loads(pathlib.Path(args.named).read_text())}
    rated_scored = {r["id"]: r for r in json.loads(pathlib.Path(args.rated_scored).read_text())}
    ratings = fw.load_ratings(pathlib.Path(args.ratings))
    mean_rating = {n: sum(v.values()) / len(v) for n, v in ratings.items()}
    taste = {r["id"]: r for r in json.loads(pathlib.Path(args.taste).read_text())}

    # Junk out: the data-quality questions already answered per name, none of
    # which is a judgment about taste.
    pool = [r for r in corpus
            if taste.get(r["id"], {}).get("is_short_form", 0) <= 0.5
            and taste.get(r["id"], {}).get("is_respelling", 0) <= 0.5
            and taste.get(r["id"], {}).get("is_boys_name", 0) <= 0.5
            and named.get(r["id"], {}).get("no_real_nickname", 0) <= args.max_no_real
            and r["id"] in taste and r["id"] not in mean_rating
            and r["id"] not in FAMILY]
    X = np.asarray([[r[t] for t in TRAITS] for r in pool])
    fit = np.asarray([taste[r["id"]]["taste_fit"] for r in pool])
    worst = max(named[n]["no_real_nickname"] for n in mean_rating
                if mean_rating[n] >= 2 and n in named)
    print(f"{len(pool)} candidates to draw from, {X.shape[1]} trait dimensions")
    print(f"  no_real_nickname cut {args.max_no_real}; highest among the names "
          f"rated >= 2 is {worst:.2f}")

    # Equal draw from each band of predicted reaction, most trait-diverse
    # first within the band.
    edges = np.quantile(fit, np.linspace(0, 1, args.bands + 1))
    edges[-1] += 1e-9
    per = args.n // args.bands
    picked, taken = [], []
    for b in range(args.bands):
        idx = [i for i in range(len(pool)) if edges[b] <= fit[i] < edges[b + 1]]
        if not idx:
            continue
        sub = spread(X[idx], min(per, len(idx)),
                     seeded=[] if not taken else [])
        # keep successive bands from repeating each other's corner of the space
        chosen = [idx[i] for i in sub]
        picked.extend(pool[i] for i in chosen)
        taken.extend(chosen)
        print(f"  band {b + 1}: predicted {edges[b]:.2f}-{edges[b + 1]:.2f}  "
              f"{len(idx):>4} available  ->  {len(chosen)} drawn")

    # --- pinned ----------------------------------------------------------
    keep_rated = [nid for nid in ANCHORS if nid in rated_scored]
    keep_rated += [nid for nid, r in sorted(mean_rating.items(), key=lambda p: -p[1])
                   if r >= 2 and nid not in keep_rated]
    others = [nid for nid in mean_rating if nid not in keep_rated and nid in rated_scored]
    step = max(len(others) // max(args.carry_over, 1), 1)
    keep_rated += others[::step][: args.carry_over]

    out = []
    for nid in keep_rated:
        row = rated_scored[nid]
        out.append({"id": nid, "formal": row["formal"], "nickname": row["nickname"],
                    "note": "rated before - repeated to measure drift",
                    "repeat": True,
                    **({"anchor": True} if nid in ANCHORS else {})})
    for r in picked:
        rec = named.get(r["id"], {})
        out.append({"id": r["id"], "formal": r["formal"], "nickname": r["nickname"],
                    "note": f"peak {rec.get('peak_decade','?')}s"
                            + (f", US #{rec['rank_2024']} in 2024" if rec.get("rank_2024")
                               else ", not in the 2024 top 1000")})

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))

    # --- did it actually work? -------------------------------------------
    old = np.asarray([[rated_scored[n][t] for t in TRAITS] for n in rated_scored])
    new = np.asarray([[r[t] for t in TRAITS] for r in picked])
    print(f"\nwrote {len(out)} names to {args.out} "
          f"({len(keep_rated)} repeats, {len(picked)} new)\n")
    print(f"  {'trait':<26} {'sd now':>7} {'sd new':>7}  {'range now':>11} {'range new':>11}")
    for j, t in enumerate(TRAITS):
        print(f"  {t:<26} {old[:, j].std():>7.2f} {new[:, j].std():>7.2f}"
              f"  {old[:, j].min():.2f}-{old[:, j].max():.2f}"
              f"  {new[:, j].min():>6.2f}-{new[:, j].max():.2f}")
    print(f"\n  mean trait sd:  now {old.std(0).mean():.2f}   "
          f"new batch {new.std(0).mean():.2f}   "
          f"({new.std(0).mean() / old.std(0).mean():.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
