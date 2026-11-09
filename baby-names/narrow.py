"""Narrow 7,200 names to a pool worth reading, without pretending to decide.

The previous version of this reported "zero of 1,646 candidates clear both
held-out bars" and stopped, which is a decision rule masquerading as a
search. It is also the wrong instrument: Hannah rated 107 of 122 names zero,
so an absolute bar set by her two 2s rejects essentially everything, and a
name she would have called a 1 - already her top 12% - gets thrown out
alongside names she would have hated.

So nothing is thresholded here. The two raters are put on a common scale by
rank rather than by value, because their scales are not comparable: Alex
averages 0.48 and Hannah 0.16, and her 1 means roughly what his 2 does.
Percentile fixes that without inventing a conversion.

The pool is then everything in the top slice for both, which is a funnel
rather than a verdict: it is meant to be read, said aloud and argued about,
and `facets.py` is what breaks it into sets afterwards.

Usage:
    python narrow.py --keep 600
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--named", default=str(HERE / "named.json"))
    ap.add_argument("--ratings", default=str(HERE / "ratings.json"))
    ap.add_argument("--out", default=str(HERE / "pool.json"))
    ap.add_argument("--keep", type=int, default=600)
    args = ap.parse_args()

    import fit_weights as fw

    named = {r["id"]: r for r in json.loads(pathlib.Path(args.named).read_text())}
    ratings = fw.load_ratings(pathlib.Path(args.ratings))
    taste = {p: {r["id"]: r for r in json.loads((HERE / f"taste_{p}.json").read_text())}
             for p in ("Alex", "Hannah")}

    # The same data-quality cuts as before, all of them answers from the
    # model rather than rules, and none of them about taste.
    CUTS = [("is_short_form", 0.5), ("is_respelling", 0.5), ("is_boys_name", 0.5)]

    def established(i: str) -> bool:
        """Used by enough people to be a name rather than a one-off entry.

        ONS Table 6 goes down to three girls in a year, which is thin enough
        that registration one-offs get in - Gracious, Gracey, Maylie, Estee.
        This is not a rarity filter and must not become one: the bar is
        "appears in the SSA history at all, OR at least ten girls in England
        and Wales". Every name either parent rated 2 or 3 clears it,
        including Verity, which is absent from SSA entirely and sits at 92.
        """
        rec = named.get(i, {})
        return rec.get("source") in ("ssa", "both") or (rec.get("uk_count") or 0) >= 10

    ids = [i for i in taste["Alex"]
           if i not in ratings
           and all(taste["Alex"][i].get(f, 0) <= t for f, t in CUTS)
           and named.get(i, {}).get("no_real_nickname", 0) <= 0.25
           and established(i)]

    # the check this project insists on, run before the cut is applied
    import build_corpus_uk  # noqa: F401  (ensures the merged corpus is the one loaded)
    liked = [i for i, v in ratings.items() if max(v.values()) >= 2 and i in named]
    fails = [i for i in liked if not established(i)]
    print(f"establishment bar checked against {len(liked)} names rated >= 2: "
          + ("all clear" if not fails else f"WOULD DELETE {fails}"))
    if fails:
        return 1

    rows = []
    pct = {}
    for p in ("Alex", "Hannah"):
        v = np.asarray([taste[p][i]["taste_fit"] for i in ids])
        # percentile within the pool: the only scale on which a rater who
        # says no to 88% and one who says no to 68% can be compared
        pct[p] = 100.0 * np.argsort(np.argsort(v)) / max(len(v) - 1, 1)

    for k, i in enumerate(ids):
        rec = named.get(i, {})
        rows.append({
            "id": i,
            "formal": taste["Alex"][i]["formal"],
            "nickname": taste["Alex"][i]["nickname"],
            "pct_Alex": round(float(pct["Alex"][k]), 1),
            "pct_Hannah": round(float(pct["Hannah"][k]), 1),
            # The weaker of the two percentiles. Ranking on this puts a name
            # both of them find plausible above one he loves and she does
            # not, which is the actual constraint in this house.
            "fit_both": round(float(min(pct["Alex"][k], pct["Hannah"][k])), 1),
            "fit_Alex": taste["Alex"][i]["taste_fit"],
            "fit_Hannah": taste["Hannah"][i]["taste_fit"],
            "source": rec.get("source", "ssa"),
            "uk_count": rec.get("uk_count"),
            "rank_2024": rec.get("rank_2024"),
        })

    rows.sort(key=lambda r: -r["fit_both"])
    keep = rows[: args.keep]
    pathlib.Path(args.out).write_text(json.dumps(keep, indent=2))

    src = {}
    for r in keep:
        src[r["source"]] = src.get(r["source"], 0) + 1
    print(f"{len(ids)} candidates -> {len(keep)} in the pool")
    print(f"  weakest percentile kept: {keep[-1]['fit_both']:.0f}th")
    print("  by source: " + ", ".join(
        f"{n} {{'ons':'UK only','ssa':'US only','both':'US + UK'}}"[0:0] or
        f"{n} " + {"ons": "UK only", "ssa": "US only", "both": "US + UK"}[k]
        for k, n in sorted(src.items(), key=lambda kv: -kv[1])))
    print(f"\n  the new UK source contributes {src.get('ons', 0)} of {len(keep)} "
          f"({100 * src.get('ons', 0) / len(keep):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
