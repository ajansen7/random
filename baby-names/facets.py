"""Ask what *kind* of name each one is, then group by the answers.

Everything up to here produced one ranked column, which answers "which is
best" and hides the more useful question: what are the different reasons a
name could work for these two, and which names cluster under each reason.

So this asks a wider set of facets than the original eight traits, and
picks them for what the 244 ratings actually established rather than for
what sounded plausible at the start:

  - the nickname carries the decision, so four facets are about the short
    form and how it relates to the formal name;
  - `era` is dead and `sound_texture` was worse than dead, so neither is
    here;
  - Margaret won on a deep nickname bench and Marguerite lost on a shallow
    one, so the bench gets its own facet;
  - the Daniel Tiger worry turned out to be unfounded, but "does this name
    drag one loud association behind it" is still worth knowing per name.

The sets are not written by hand. Names are clustered in facet space, and
each cluster is then labelled by whichever facets it is most extreme on
relative to the pool - so a set exists because the names in it genuinely
share something, and the label reports what that something is rather than
asserting it up front. That is the point: the grouping can surface a reason
nobody had articulated.

Usage:
    python facets.py --limit 600          # score the strongest candidates
    python facets.py --sets 8             # and cut them into eight sets
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from typesafe_sdk import Noul, Score

from parallel import run_all

HERE = pathlib.Path(__file__).parent
SIBLING = {"nickname": "Tessa", "formal": "Theresa"}

# Every facet is one narrow judgment, all are independent, and all are asked
# over the same state - so they cost one request per name however many there
# are. None of them is a preference: each is a property of the name, and
# what the family wants out of it is applied afterwards in code.
FACETS = {
    "nickname_reshapes": Score(
        instructions=(
            "Compare `candidate.nickname` to `candidate.formal_name`. How much "
            "does the short form change the sound of the full name, rather than "
            "just shortening it?"
        ),
        criteria=[
            "No change at all: the short form is the formal name, or a syllable removed from the end.",
            "The front or back is clipped and nothing else moves.",
            "Clipped, with a conventional -y or -ie stuck on the end.",
            "Sounds inside the name change, so the short form is audibly a different word.",
            "A different word entirely, connected to the formal name by tradition rather than spelling.",
        ],
    ),
    "nickname_bench": Score(
        instructions=(
            "Ignore `candidate.nickname`. How many genuinely different, "
            "established short forms does `candidate.formal_name` have - so that "
            "she could pick a different one later and it would still be her name?"
        ),
        criteria=[
            "None. The name is used in full or not at all.",
            "One, and only one.",
            "Two that are real, though one is much more usual.",
            "Three or more real short forms, several of them in common use.",
            "An unusually deep bench: many established short forms that sound nothing like each other.",
        ],
    ),
    "formal_gravity": Score(
        instructions=(
            "Said in full, how much weight does `candidate.formal_name` carry - "
            "the version she would use on a document or be introduced by in a "
            "room of adults?"
        ),
        criteria=[
            "None: it reads as a child's name however old she gets.",
            "Light. Pleasant, but it would not add anything in a formal setting.",
            "Ordinary: neither adds nor costs her anything.",
            "Substantial. It would carry her in a professional room.",
            "Considerable, to the point that it could feel heavy for a small child.",
        ],
    ),
    "everyday_warmth": Score(
        instructions=(
            "Think of `candidate.nickname` called across a playground or said a "
            "hundred times a day at home. How warm does it sound in that use?"
        ),
        criteria=[
            "Cold or severe - it lands as a reprimand.",
            "Neutral and functional, with no particular affection in it.",
            "Pleasant enough, unremarkable either way.",
            "Warm. It sounds like someone is fond of her.",
            "Unusually warm and easy in the mouth, the kind of name that gets sung.",
        ],
    ),
    "pairs_with_sibling": Score(
        instructions=(
            "Say `sibling.nickname` and `candidate.nickname` aloud together, the "
            "way a parent calls two children at once. How does the pair sound?"
        ),
        criteria=[
            "Bad: they rhyme or blur into each other and would be misheard.",
            "Awkward - too close, or jarringly mismatched.",
            "Fine. They work together without any particular relationship.",
            "Good: clearly distinct, and obviously chosen by the same people.",
            "Excellent - they sound like a deliberate pair without being matchy.",
        ],
    ),
    "one_loud_association": Noul(
        instructions=(
            "Does `candidate.formal_name` or `candidate.nickname` carry one "
            "dominant association - a character, celebrity, song or brand - "
            "strong enough that most people would think of it first?"
        ),
        criteria={
            "true": "One referent dominates and most people would name it immediately.",
            "false": "No single association dominates; it is just a name.",
        },
    ),
    "spelling_friction": Noul(
        instructions=(
            "Would `candidate.formal_name` have to be spelled out or corrected "
            "most times she gives it?"
        ),
        criteria={
            "true": "Ambiguous spelling or pronunciation; she would repeat it constantly.",
            "false": "Heard once and written correctly by most people.",
        },
    ),
    "short_form_stands_alone": Noul(
        instructions=(
            "Could `candidate.nickname` work as her entire name, with nobody "
            "ever needing to know the formal one?"
        ),
        criteria={
            "true": "A complete name in its own right that happens to have a longer version.",
            "false": "Audibly a short form; it needs the full name standing behind it.",
        },
    ),
    "travels": Noul(
        instructions=(
            "Outside the English-speaking world, would `candidate.formal_name` "
            "still be recognised and said roughly the same way?"
        ),
        criteria={
            "true": "Recognised across European languages, whatever the local spelling.",
            "false": "Distinctly Anglophone, or it becomes a different name abroad.",
        },
    ),
}

SCORES = [k for k, v in FACETS.items() if isinstance(v, Score)]
NOULS = [k for k, v in FACETS.items() if isinstance(v, Noul)]
ALL = SCORES + NOULS


def state_for(rec: dict) -> dict:
    return {
        "candidate": {"formal_name": rec["formal"], "nickname": rec["nickname"]},
        "sibling": SIBLING,
    }


def row_for(rec: dict, resp) -> dict:
    out = {"id": rec["id"], "formal": rec["formal"], "nickname": rec["nickname"]}
    for k in SCORES:
        out[k] = round(resp.answers[k].score / (len(FACETS[k].criteria) - 1), 3)
    for k in NOULS:
        out[k] = round(resp.answers[k].noul, 3)
    return out


def label_set(members: np.ndarray, pool_mean: np.ndarray, pool_sd: np.ndarray) -> str:
    """Name a cluster after whatever it is most extreme on."""
    z = (members.mean(0) - pool_mean) / np.where(pool_sd > 0, pool_sd, 1)
    order = np.argsort(-np.abs(z))[:2]
    phrase = {
        "nickname_reshapes":       ("the short form reshapes the name", "the short form only clips it"),
        "nickname_bench":          ("a deep bench of short forms", "one short form or none"),
        "formal_gravity":          ("the full name carries weight", "the full name is light"),
        "everyday_warmth":         ("warm to say every day", "cool and formal to say"),
        "pairs_with_sibling":      ("sits well beside Tessa", "sits awkwardly beside Tessa"),
        "one_loud_association":    ("drags one loud association", "no dominant association"),
        "spelling_friction":       ("will be spelled out constantly", "spells itself"),
        "short_form_stands_alone": ("the short form is a whole name", "the short form needs the long one"),
        "travels":                 ("travels outside English", "stays Anglophone"),
    }
    return ", ".join(phrase[ALL[j]][0 if z[j] > 0 else 1] for j in order)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--named", default=str(HERE / "named.json"))
    ap.add_argument("--out", default=str(HERE / "facets.json"))
    ap.add_argument("--pool", default=str(HERE / "pool.json"),
                    help="candidates to score, from narrow.py")
    ap.add_argument("--sets", type=int, default=8)
    ap.add_argument("--per-set", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rescore", action="store_true",
                    help="re-ask the facets instead of reusing facets.json")
    args = ap.parse_args()

    pool = json.loads(pathlib.Path(args.pool).read_text())
    out_path = pathlib.Path(args.out)

    if args.rescore or not out_path.exists():
        answered, failures = run_all(
            pool, state_for=state_for, questions_for=lambda _r: FACETS,
            concurrency=args.concurrency, progress_every=200,
        )
        rows = [row_for(rec, resp) for rec, resp in answered]
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"scored {len(rows)} names on {len(FACETS)} facets"
              + (f", {len(failures)} failed" if failures else ""))
    else:
        rows = json.loads(out_path.read_text())
        print(f"reusing {len(rows)} scored names from {args.out}")

    by_id = {r["id"]: r for r in pool}
    X = np.asarray([[r[f] for f in ALL] for r in rows])
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)

    # k-means, seeded spread out so the sets do not collapse together
    rng = np.random.default_rng(7)
    cent = [Z[rng.integers(len(Z))]]
    while len(cent) < args.sets:
        d = np.min(np.linalg.norm(Z[:, None] - np.asarray(cent)[None], axis=2), axis=1)
        cent.append(Z[int(np.argmax(d))])
    C = np.asarray(cent)
    for _ in range(60):
        lab = np.argmin(np.linalg.norm(Z[:, None] - C[None], axis=2), axis=1)
        new = np.asarray([Z[lab == k].mean(0) if (lab == k).any() else C[k]
                          for k in range(args.sets)])
        if np.allclose(new, C):
            break
        C = new

    print()
    order = sorted(range(args.sets),
                   key=lambda k: -np.mean([by_id[rows[i]["id"]]["fit_both"]
                                           for i in np.where(lab == k)[0]] or [0]))
    sets_out = []
    for k in order:
        idx = np.where(lab == k)[0]
        if len(idx) == 0:
            continue
        name = label_set(Z[idx], Z.mean(0), Z.std(0))
        members = sorted(idx, key=lambda i: -by_id[rows[i]["id"]]["fit_both"])
        print(f"\n\033[1m{name}\033[0m  ({len(idx)} names)")
        print(f"  {'Alex':>4} {'Han':>4}  {'called':<11} {'formal':<15} where it's from")
        for i in members[: args.per_set]:
            r = rows[i]
            p = by_id[r["id"]]
            src = {"ons": "UK only", "ssa": "US only", "both": "US + UK"}.get(
                p.get("source"), "?")
            extra = f"{src}"
            if p.get("uk_count"):
                extra += f", {p['uk_count']} girls in E&W"
            print(f"  {p['pct_Alex']:>4.0f} {p['pct_Hannah']:>4.0f}  "
                  f"{r['nickname']:<11} {r['formal']:<15} {extra}")
        sets_out.append({"label": name,
                         "names": [{**rows[i], **by_id[rows[i]["id"]]} for i in members]})

    pathlib.Path(HERE / "sets.json").write_text(json.dumps(sets_out, indent=2))
    print(f"\nwrote {len(sets_out)} sets to sets.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
