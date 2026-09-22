"""Fit the trait weights to what we actually said, instead of guessing them.

classify.py produces raw per-trait scores. Those are just measurements; they
say nothing yet about which traits we care about. This script takes the
ratings collected in the artifact and works out which traits predict them.

It does three separate things, in increasing order of how much you should
trust them:

1. Anchor check. Theresa, Margaret and Rebecca were rated blind alongside
   everything else. We already know roughly how those three landed, so if
   the fitted model disagrees with them the traits are wrong and nothing
   below is worth reading.

2. Which traits carry signal: correlation of each trait against the ratings,
   one at a time. This is the part that answers "what do we actually like".

3. A ranking of the unrated names. Least squares over seven traits and fifty
   names will fit noise as happily as taste, so the leave-one-out error is
   printed next to the in-sample error. If they diverge, the ranking is
   decoration and the correlations in (2) are the real output.

Usage:
    python fit_weights.py --ratings ratings.json --scored scored.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

TRAITS = [
    "era",
    "nickname_transformation",
    "sound_texture",
    "child_adult_range",
    "currently_common",
    "recognisable",
    "clashes_with_sibling",
    "register_match",
]

# Matches HIGHER_IS_BETTER in classify.py; traits we want LOW get flipped so
# every column points the same way before fitting.
INVERT = {"currently_common", "clashes_with_sibling"}


def load_ratings(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """{name_id: {rater: rating}} from the artifact's exported rows."""
    raw = json.loads(path.read_text())
    rows = raw.values() if isinstance(raw, dict) else raw
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in rows:
        if row.get("nameId") is not None and row.get("rating") is not None:
            out[row["nameId"]][row.get("rater", "?")] = float(row["rating"])
    return dict(out)


def design_matrix(scored: list[dict]) -> tuple[np.ndarray, list[str]]:
    feats = []
    for row in scored:
        feats.append([
            (1.0 - row[t]) if t in INVERT else row[t]
            for t in TRAITS
        ])
    return np.asarray(feats, dtype=float), [r["id"] for r in scored]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default=str(HERE / "scored.json"))
    ap.add_argument("--ratings", default=str(HERE / "ratings.json"))
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    scored = json.loads(pathlib.Path(args.scored).read_text())
    ratings = load_ratings(pathlib.Path(args.ratings))
    by_id = {r["id"]: r for r in scored}

    X, ids = design_matrix(scored)
    mean_rating = {n: sum(v.values()) / len(v) for n, v in ratings.items()}

    labelled = [i for i, nid in enumerate(ids) if nid in mean_rating]
    if len(labelled) < len(TRAITS) + 2:
        print(f"Only {len(labelled)} rated names. Rate more before fitting.")
        return 1

    Xl = X[labelled]
    yl = np.asarray([mean_rating[ids[i]] for i in labelled])

    # --- 1. anchors ------------------------------------------------------
    print("Anchor check")
    for nid in ("theresa", "margaret", "rebecca"):
        row = by_id.get(nid)
        if not row:
            continue
        got = mean_rating.get(nid)
        print(
            f"  {row['nickname']:<9} rated {got if got is not None else '--':>4}"
            f"   era {row['era']:.2f}  transform {row['nickname_transformation']:.2f}"
        )

    # --- 2. per-trait signal --------------------------------------------
    print("\nWhich traits track the ratings")
    sd_y = yl.std()
    signal = []
    for j, trait in enumerate(TRAITS):
        col = Xl[:, j]
        r = 0.0 if col.std() == 0 or sd_y == 0 else float(np.corrcoef(col, yl)[0, 1])
        signal.append((abs(r), r, trait))
    for _, r, trait in sorted(signal, reverse=True):
        bar = "#" * int(abs(r) * 28)
        print(f"  {trait:<26} r={r:+.2f}  {bar}")

    # --- 3. weights and honesty about them -------------------------------
    Xd = np.column_stack([Xl, np.ones(len(Xl))])
    w, *_ = np.linalg.lstsq(Xd, yl, rcond=None)
    in_sample = float(np.sqrt(((Xd @ w - yl) ** 2).mean()))

    errs = []
    for k in range(len(Xl)):
        keep = [i for i in range(len(Xl)) if i != k]
        wk, *_ = np.linalg.lstsq(Xd[keep], yl[keep], rcond=None)
        errs.append((Xd[k] @ wk - yl[k]) ** 2)
    loo = float(np.sqrt(np.mean(errs)))

    print(f"\nFit over {len(labelled)} rated names")
    print(f"  in-sample RMSE   {in_sample:.2f}  (on a 0-3 scale)")
    print(f"  leave-one-out    {loo:.2f}")
    if loo > in_sample * 1.8:
        print("  -> overfit. Trust the correlations above, not the ranking below.")

    print("\nFitted weights")
    for trait, weight in sorted(zip(TRAITS, w[:-1]), key=lambda p: -abs(p[1])):
        print(f"  {trait:<26} {weight:+.2f}")

    # --- ranking of what we have not rated -------------------------------
    unrated = [i for i, nid in enumerate(ids) if nid not in mean_rating]
    if unrated:
        preds = np.column_stack([X[unrated], np.ones(len(unrated))]) @ w
        order = np.argsort(-preds)
        print(f"\nPredicted best of the {len(unrated)} unrated names")
        for k in order[: args.top]:
            row = by_id[ids[unrated[k]]]
            print(f"  {preds[k]:4.2f}  {row['nickname']:<10} {row['formal']}")
    else:
        print("\nEverything scored has been rated - add more names to names.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
