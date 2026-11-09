"""Names worth another look, for reasons the main ranking cannot produce.

The pool in `narrow.py` answers "what else is like what you liked". This
answers the harder question: where is the current picture likely to be
wrong? Four sources, each grounded in something the 244 ratings established
rather than in a hunch.

**Rejected on the short form, not the name.** The ratings' strongest finding
is that the nickname carries the decision: Margaret and Magdalena both
scored 3 behind Maggie while Marguerite scored 0 behind Daisy. A name rated
0 was therefore rated as one formal/short-form *pair*, and a different pair
is a different candidate. `pick_nicknames.py` has been keeping the runners-up
since the beginning for exactly this - the comment says "a name can be worth
rescoring under its second-best short form, which is exactly the Marguerite
case" - and this is the first time anything has used them.

**Stated and revealed preference disagree about popularity.** "Nothing too
common" was the stated rule. Measured on both raters independently,
`currently_common` correlates *positively* with liking: +0.37 for Hannah,
+0.33 for Alex. Names dismissed for being popular are the single largest
known blind spot, so they get surfaced deliberately.

**Disagreement.** Where one of them places a name far above the other, the
average buries it. Those are the conversations worth having, not the noise.

**Ordinary in Britain, invisible in America.** Both raters have only ever
seen names through a US frame, because until the ONS source was added that
was the only corpus. A name given to sixty girls a year in England and Wales
and absent from the American top thousand is outside the frame by
construction, not by taste.

Low model confidence was tried as a fifth source and dropped: 1,451 names
sit at exactly 0.00 and the *highest*-confidence names are the obvious
rejections (Stormi at 0.97). Confidence here reports "certainly no", not
"uncertain", so it cannot find the interesting middle.

Usage:
    python second_look.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import taste_match as tm
from parallel import run_all

HERE = pathlib.Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "second_look.json"))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--per-list", type=int, default=25)
    args = ap.parse_args()

    import fit_weights as fw

    named = {r["id"]: r for r in json.loads((HERE / "named.json").read_text())}
    ratings = fw.load_ratings(HERE / "ratings.json")
    pool = {r["id"]: r for r in json.loads((HERE / "pool.json").read_text())}
    confounds = json.loads((HERE / "confounds.json").read_text())
    unavail = json.loads((HERE / "unavailable.json").read_text())
    taste = {p: {r["id"]: r for r in json.loads((HERE / f"taste_{p}.json").read_text())}
             for p in ("Alex", "Hannah")}

    rated_pairs = {}
    for f in ("names.json", "names_v2.json"):
        for r in json.loads((HERE / f).read_text()):
            rated_pairs.setdefault(r["id"], r)

    out: dict[str, list] = {}

    # --- 1. rejected on the short form -----------------------------------
    exemplars = {p: tm.load_rated(HERE / "scored.json", HERE / "ratings.json",
                                  confounds, rater=p) for p in ("Alex", "Hannah")}
    jobs = []
    for nid, per in ratings.items():
        if nid in unavail or nid not in named or max(per.values()) > 1:
            continue
        was = rated_pairs.get(nid, {}).get("nickname", named[nid]["nickname"])
        for alt in named[nid].get("nickname_alternatives", [])[:3]:
            if alt.lower() == was.lower() or len(alt) < 3:
                continue
            jobs.append({"id": f"{nid}::{alt}", "base": nid, "formal": named[nid]["formal"],
                         "nickname": alt, "rated_as": was,
                         "was": (per.get("Alex", 0) + per.get("Hannah", 0)) / 2})
    print(f"re-pairing {len(jobs)} formal/short-form combinations "
          f"across {len({j['base'] for j in jobs})} rejected names")

    lifted = {}
    for who in ("Alex", "Hannah"):
        rated = exemplars[who]

        def state_for(rec, _rated=rated):
            return {
                "candidate": {"formal_name": rec["formal"], "nickname": rec["nickname"]},
                "sibling": tm.SIBLING,
                "ratings": tm.exemplars(_rated, rec["base"]),
            }

        answered, _ = run_all(jobs, state_for=state_for,
                              questions_for=lambda _r: {"taste_fit": tm.QUESTIONS["taste_fit"]},
                              concurrency=args.concurrency, progress_every=250)
        for rec, resp in answered:
            lifted.setdefault(rec["id"], dict(rec))[f"fit_{who}"] = resp.answers["taste_fit"].score

    # Compare against the same name under the short form they actually saw.
    base_fit = {nid: (taste["Alex"].get(nid, {}).get("taste_fit"),
                      taste["Hannah"].get(nid, {}).get("taste_fit"))
                for nid in {j["base"] for j in jobs}}
    rows = []
    for k, r in lifted.items():
        ba, bh = base_fit.get(r["base"], (None, None))
        if ba is None or bh is None:
            continue
        lift = min(r["fit_Alex"] - ba, r["fit_Hannah"] - bh)
        rows.append({**r, "base_Alex": round(ba, 2), "base_Hannah": round(bh, 2),
                     "fit_Alex": round(r["fit_Alex"], 2),
                     "fit_Hannah": round(r["fit_Hannah"], 2),
                     "lift": round(lift, 2)})
    best: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: -r["lift"]):
        best.setdefault(r["base"], r)
    out["nickname_was_the_problem"] = [r for r in best.values() if r["lift"] > 0][: args.per_list]

    # --- 2. the popularity blind spot ------------------------------------
    common = [r for r in pool.values() if (r.get("rank_2024") or 9999) <= 400]
    out["too_common_to_have_considered"] = sorted(
        common, key=lambda r: -r["fit_both"])[: args.per_list]

    # --- 3. where they disagree ------------------------------------------
    split = sorted(pool.values(), key=lambda r: -abs(r["pct_Alex"] - r["pct_Hannah"]))
    out["you_two_disagree"] = split[: args.per_list]

    # --- 4. ordinary in Britain, invisible in America --------------------
    uk = [r for r in pool.values()
          if (r.get("uk_count") or 0) >= 25 and not r.get("rank_2024")]
    out["british_not_american"] = sorted(uk, key=lambda r: -r["fit_both"])[: args.per_list]

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"\n{k}  ({len(v)})")
        for r in v[:8]:
            if "lift" in r:
                print(f"  +{r['lift']:.2f}  {r['formal']:<13} {r['rated_as']} -> {r['nickname']}"
                      f"   (rated {r['was']:.1f} as {r['rated_as']})")
            else:
                extra = (f"US #{r['rank_2024']}" if r.get("rank_2024") else
                         f"{r['uk_count']} girls in E&W" if r.get("uk_count") else
                         f"A{r['pct_Alex']:.0f}/H{r['pct_Hannah']:.0f}")
                print(f"  {r['fit_both']:>5.0f}  {r['nickname']:<11} {r['formal']:<14} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
