"""Stage 3, second attempt: compare a candidate to the ratings themselves.

The trait fit does not work, and the reason is structural rather than a bug.
The eight traits in classify.py were designed to explain why Theresa and
Margaret beat Rebecca, but names.json was then built entirely out of names
*like* Theresa and Margaret. Across the 48 usable ratings every trait is
nearly constant - sd 0.07 to 0.21 on a 0-1 scale, with sound_texture sitting
at 0.77 for the names rated 0 and 0.77 for the names rated 2 and 3. Traits
that describe what all fifty names share cannot separate them, so least
squares fits noise: leave-one-out RMSE 1.10 against 0.91 for predicting the
mean, and a corpus ranking topped by Elizbeth, Avah and Madie.

Eight numbers per name is the wrong compression of this evidence. Three names
were loved, five liked, thirty-three rejected, and which is which is the whole
signal - so the ratings go into the state and the judgment is a direct
comparison against them. That also carries what no trait rubric can: that
Georgiana and Magdalena score identically on every trait and one is a 3 and
the other a 0.

Validation is leave-one-out and uses the same code path: a candidate that
appears in the ratings is removed from the exemplars before it is asked
about, so a rated name is always judged against the other 47.

Usage:
    export TYPESAFE_API_KEY=...
    python taste_match.py --validate                  # honest test on the 48
    python taste_match.py --names named.json          # rank the corpus
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from typesafe_sdk import Choice, Noul, Score

from parallel import run_all

HERE = pathlib.Path(__file__).parent
SIBLING = {"nickname": "Tessa", "formal": "Theresa"}

# The rating scale the artifact collected on, restated as situations. The
# answer lands on the same 0-3 scale as a real rating, so a prediction and a
# rating are directly comparable and the validation needs no rescaling.
BANDS = [
    "This family would reject it out of hand. It is the kind of name they have "
    "turned down repeatedly: nothing in it matches what they have chosen before.",
    "They would not dislike it, but nothing about it would make them stop. It is "
    "plausible for them and unmemorable, the sort of name that stays on a list "
    "without ever moving up it.",
    "They would like it and keep it in contention, with something holding it back "
    "from the very top - it does much of what their favourites do, but not all.",
    "They would love it. It does the same work for them as the names they rated "
    "highest, and would be a serious contender for the actual name.",
]


def exemplars(rated: list[dict], exclude_id: str | None) -> dict:
    """The ratings as state: what they loved, liked and rejected, by name.

    `exclude_id` drops the candidate itself, so asking about a rated name is
    a genuine held-out prediction rather than a lookup.
    """
    buckets: dict[str, list[str]] = {
        "loved_rated_3": [], "liked_rated_2": [],
        "lukewarm_rated_1": [], "rejected_rated_0": [],
    }
    key = {3: "loved_rated_3", 2: "liked_rated_2",
           1: "lukewarm_rated_1", 0: "rejected_rated_0"}
    for r in rated:
        if r["id"] == exclude_id:
            continue
        buckets[key[round(r["rating"])]].append(f"{r['formal']}, called {r['nickname']}")
    return buckets


QUESTIONS = {
    "taste_fit": Score(
        instructions=(
            "`ratings` records how these particular parents reacted to fifty "
            "girls' names, sorted by what they gave each one. They are naming a "
            "second daughter; the first is `sibling.formal`, called "
            "`sibling.nickname`. Judge `candidate.formal_name`, which they would "
            "call `candidate.nickname`, against that record. How would these "
            "parents react to it?\n\n"
            "Every name in `ratings` was hand-picked as a plausible choice "
            "for them before being rated, and `rejected_rated_0` is still "
            "longer than the other three lists together. Resembling their "
            "favourites in length, age or style is therefore not enough to "
            "earn the top bands, since most of what they rejected does that "
            "too. Weigh the rejected list as heavily as the loved one."
        ),
        criteria=BANDS,
    ),
    # The two things shortlist.py currently handles with hand-written rules,
    # asked instead. Both are independent of the score and of each other, so
    # all four questions ride in the one request the name already costs -
    # which is the test: if these work, the rules are dead weight and worse,
    # since a rule derived from nine liked names is fitted to nine points.
    "is_short_form": Noul(
        instructions=(
            "The SSA lists record whatever was written on a birth "
            "certificate, so pet names appear alongside full given names. Is "
            "`candidate.formal_name` itself a diminutive or short form rather "
            "than a full formal name?"
        ),
        criteria={
            "true": "A nickname or pet form that happens to get registered as "
                    "a legal name - Maggie, Meg, Tess, Trudy, Betty, Sadie.",
            "false": "A full given name in its own right, whatever its length, "
                     "and whether or not it has short forms of its own.",
        },
    ),
    "is_boys_name": Noul(
        instructions=(
            "The source data records a name under every sex it was registered "
            "under, so a handful of boys' names appear in the girls' file from "
            "rare cross-registrations. Is `candidate.formal_name` "
            "predominantly a boy's name in the English-speaking world?"
        ),
        criteria={
            "true": "Overwhelmingly given to boys - John, Timothy, Joseph, "
                    "Kyle - whatever a few records say.",
            "false": "Given to girls, or genuinely used for both.",
        },
    ),
    "is_respelling": Noul(
        instructions=(
            "Is `candidate.formal_name` a variant or misspelling of a more "
            "standard English given name, rather than the standard form?"
        ),
        criteria={
            "true": "A rarer respelling of a name whose usual spelling is "
                    "different - Elizbeth, Margarett, Madelyn, Rebeka.",
            "false": "The standard spelling of the name, or a distinct name in "
                     "its own right rather than a variant of another.",
        },
    ),
    # Independent of the score and asked in the same request: which of their
    # favourites this is nearest to. It costs nothing extra and it is what
    # makes a shortlist entry arguable rather than just ranked.
    "nearest_favourite": Choice(
        instructions=(
            "Of the names in `ratings.loved_rated_3`, which one does "
            "`candidate.formal_name` (called `candidate.nickname`) most resemble "
            "in the way that would matter to these parents - how it sounds said "
            "aloud, and what the short form does?"
        ),
        criteria={
            "Theresa": "Nearest to Theresa, called Tessa.",
            "Margaret": "Nearest to Margaret, called Maggie.",
            "Magdalena": "Nearest to Magdalena, called Maggie.",
            "none_of_these": "It does not resemble any of their favourites.",
        },
    ),
}


def load_rated(scored_path: pathlib.Path, ratings_path: pathlib.Path,
               confounds: dict, rater: str | None = None) -> list[dict]:
    """The rated names with the exact formal/nickname pair that was shown.

    Confounded ratings are about a dog or a friend rather than the name, so
    they are no more usable as exemplars than they were as fit targets.
    """
    import fit_weights as fw

    # The formal/nickname pair exactly as the artifact showed it. Both name
    # files are needed now: names.json is batch one, names_v2.json batch two
    # plus the repeats. Taking the pair from a scored file instead would
    # silently substitute whatever short form pick_nicknames.py chose, which
    # is not what anyone rated.
    scored = {}
    for f in ("names.json", "names_v2.json"):
        q = scored_path.parent / f
        if q.exists():
            scored.update({r["id"]: r for r in json.loads(q.read_text())})
    scored.update({r["id"]: r for r in json.loads(scored_path.read_text())
                   if r["id"] not in scored})
    ratings = fw.load_ratings(ratings_path)
    out = []
    for nid, per_rater in ratings.items():
        if nid in confounds or nid not in scored:
            continue
        if rater:
            if rater not in per_rater:
                continue
            value = per_rater[rater]
        else:
            value = sum(per_rater.values()) / len(per_rater)
        out.append({
            "id": nid,
            "formal": scored[nid]["formal"],
            "nickname": scored[nid]["nickname"],
            "rating": value,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", default=None,
                    help="pool to judge (default: validate on the rated names)")
    ap.add_argument("--scored", default=str(HERE / "scored.json"))
    ap.add_argument("--ratings", default=str(HERE / "ratings.json"))
    ap.add_argument("--confounds", default=str(HERE / "confounds.json"))
    ap.add_argument("--unavailable", default=str(HERE / "unavailable.json"))
    ap.add_argument("--out", default=str(HERE / "taste.json"))
    ap.add_argument("--validate", action="store_true",
                    help="predict the rated names held out one at a time")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--top", type=int, default=25)
    # Whose taste to model. Averaging the two raters looked natural and is
    # not: Hannah rejects 88% of the pool and Alex 68%, so the mean squashes
    # her 0s against his 2s and predicts a scale nobody actually used. It
    # also hides which of them is the constraint. Modelling each separately
    # and intersecting afterwards is both sharper and more useful, because
    # the scarce thing is a name Hannah likes.
    ap.add_argument("--rater", default=None,
                    help="model one rater (Alex / Hannah); default averages them")
    args = ap.parse_args()

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set.", file=sys.stderr)
        return 2

    confounds = json.loads(pathlib.Path(args.confounds).read_text())
    unavailable = json.loads(pathlib.Path(args.unavailable).read_text())
    rated = load_rated(pathlib.Path(args.scored), pathlib.Path(args.ratings),
                       confounds, rater=args.rater)
    print(f"{len(rated)} usable ratings as exemplars "
          f"({sum(1 for r in rated if r['rating'] >= 3)} loved, "
          f"{sum(1 for r in rated if r['rating'] == 0)} rejected)")

    if args.validate or not args.names:
        pool = [dict(r) for r in rated]
    else:
        pool = json.loads(pathlib.Path(args.names).read_text())
        pool = [r for r in pool if r["id"] not in unavailable]
    if args.limit:
        pool = pool[: args.limit]

    def state_for(rec: dict) -> dict:
        return {
            "candidate": {
                "formal_name": rec["formal"],
                "nickname": rec["nickname"],
            },
            "sibling": SIBLING,
            "ratings": exemplars(rated, rec["id"]),
        }

    answered, failures = run_all(
        pool, state_for=state_for, questions_for=lambda _r: QUESTIONS,
        concurrency=args.concurrency, progress_every=250,
    )

    rows = []
    for rec, resp in answered:
        fit = resp.answers["taste_fit"]
        near = resp.answers["nearest_favourite"]
        # ScoreAnswer.score is already the probability-weighted position on
        # the band scale rather than the argmax band, which is what puts 2,900
        # names in order instead of into four piles.
        rows.append({
            "id": rec["id"], "formal": rec["formal"], "nickname": rec["nickname"],
            "taste_fit": round(fit.score, 3),
            "taste_top_band": max(fit.probabilities, key=fit.probabilities.get),
            "taste_confidence": round(fit.confidence, 3),
            "nearest_favourite": near.choice,
            "is_short_form": round(resp.answers["is_short_form"].noul, 3),
            "is_respelling": round(resp.answers["is_respelling"].noul, 3),
            "is_boys_name": round(resp.answers["is_boys_name"].noul, 3),
            **({"rating": rec["rating"]} if "rating" in rec else {}),
        })
    rows.sort(key=lambda r: -r["taste_fit"])
    pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} rows to {args.out}")
    if failures:
        print(f"{len(failures)} failed; rerun to fill them in", file=sys.stderr)

    if args.validate:
        import numpy as np
        pred = np.array([r["taste_fit"] for r in rows])
        true = np.array([r["rating"] for r in rows])
        rmse = float(np.sqrt(((pred - true) ** 2).mean()))
        base = float(true.std())
        r = float(np.corrcoef(pred, true)[0, 1])
        print(f"\nHeld-out prediction of {len(rows)} ratings")
        print(f"  RMSE              {rmse:.2f}   (predicting the mean: {base:.2f})")
        print(f"  correlation       r={r:+.2f}")
        print(f"  {'BEATS' if rmse < base else 'DOES NOT BEAT'} the mean baseline")
        print("\n  by what they actually said")
        for band in sorted(set(true.tolist()), reverse=True):
            v = pred[true == band]
            print(f"    rated {band:>4.1f}  n={len(v):>3}  predicted "
                  f"{v.mean():.2f}  (range {v.min():.2f}-{v.max():.2f})")
        order = np.argsort(-pred)
        top = int(max(1, len(rows) // 10))
        hits = sum(1 for i in order[:top] if true[i] >= 2)
        tot = int((true >= 2).sum())
        rk = np.argsort(np.argsort(pred))
        sp = float(np.corrcoef(rk, np.argsort(np.argsort(true)))[0, 1])
        print(f"\n  spearman          {sp:+.2f}")
        print(f"  top {top}: {hits} of the {tot} names rated >= 2"
              f"   (chance {top * tot / len(rows):.1f})")
        print("\n  worst misses")
        for row in sorted(rows, key=lambda x: -abs(x["taste_fit"] - x["rating"]))[:6]:
            print(f"    rated {row['rating']:.0f}  predicted {row['taste_fit']:.2f}   "
                  f"{row['nickname']:<10} {row['formal']}")
    else:
        print(f"\nTop {args.top} of {len(rows)}")
        for row in rows[: args.top]:
            print(f"  {row['taste_fit']:.2f}  {row['nickname']:<10} {row['formal']:<14}"
                  f" like {row['nearest_favourite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
