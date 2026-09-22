"""Stage 1 of the funnel: decide what each candidate would actually be called.

build_corpus.py generates short forms mechanically, and mechanical generation
produces non-words - "Ntine" for Clementine, "Therrie" for Theresa. No rule
set fixes that reliably, because whether a short form is real is a fact about
usage, not about spelling.

So the code over-generates and a judgment selects, with a no-match outcome for
the names where nothing offered is real. The model can only choose what it is
shown, so build_corpus.py errs toward offering too much.

This matters more than it looks. The ratings say the nickname carries the
decision: Margaret and Magdalena both scored 3 behind the same Maggie, while
their sibling Marguerite scored 0 behind Daisy. Scoring a name on the wrong
short form scores the wrong name.

Names where nothing fits are dropped rather than guessed at - a candidate we
cannot name is not a candidate.

Usage:
    export TYPESAFE_API_KEY=...
    python pick_nicknames.py                  # corpus.json -> named.json
    python pick_nicknames.py --limit 20       # try it on twenty first
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from typesafe_sdk import Choice

from parallel import run_all

HERE = pathlib.Path(__file__).parent
NO_MATCH = "none_of_these"


def question_for(record: dict) -> dict:
    """A Choice over this name's generated short forms, plus a way out."""
    criteria = {
        cand: f"She would be called {cand}, and {cand} is a short form of "
              f"{record['formal']} that English speakers actually use."
        for cand in record["nickname_candidates"]
    }
    criteria[NO_MATCH] = (
        "None of the options is a real short form of this name. Either the "
        "name has no established short form, or every option listed is a "
        "misspelling or an invented word rather than a name."
    )
    return {
        "nickname": Choice(
            instructions=(
                "`candidate.formal_name` is a girl's name. The options are "
                "possible short forms of it, generated mechanically, so some "
                "may not be real words. Which one would a family most "
                "naturally use as her everyday name?"
            ),
            criteria=criteria,
        )
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(HERE / "corpus.json"))
    ap.add_argument("--out", default=str(HERE / "named.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop picks below this confidence instead of keeping them")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="requests in flight at once")
    args = ap.parse_args()

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set.", file=sys.stderr)
        return 2

    records = json.loads(pathlib.Path(args.corpus).read_text())
    if args.limit:
        records = records[: args.limit]

    answered, failures = run_all(
        records,
        state_for=lambda rec: {"candidate": {"formal_name": rec["formal"]}},
        questions_for=question_for,
        concurrency=args.concurrency,
    )

    kept, no_match, low_conf = [], 0, 0
    for rec, resp in answered:
        ans = resp.answers["nickname"]
        if ans.choice == NO_MATCH:
            no_match += 1
            continue
        if ans.confidence < args.min_confidence:
            low_conf += 1
            continue

        out = dict(rec)
        out["nickname"] = ans.choice
        out["nickname_confidence"] = round(ans.confidence, 4)
        # The probability mass left on the no-match outcome even though a
        # real option won. This is NOT the same as low confidence: Magdalena
        # spreads 0.42 across Maggie, Magda and Lena, which are all real, and
        # must not be filtered. Meredith splits between Mer and "none of
        # these are real words", which is the case worth dropping. Kept as a
        # number so the threshold can be set from the ratings downstream.
        out["no_real_nickname"] = round(ans.probabilities.get(NO_MATCH, 0.0), 4)
        # Keep the runners-up: a name can be worth rescoring under its
        # second-best short form, which is exactly the Marguerite case.
        out["nickname_alternatives"] = [
            c for c, _ in sorted(
                ((c, p) for c, p in ans.probabilities.items() if c != NO_MATCH and c != ans.choice),
                key=lambda kv: -kv[1],
            )[:3]
        ]
        kept.append(out)

    pathlib.Path(args.out).write_text(json.dumps(kept, indent=2))
    print(f"\n{len(kept)} names written to {args.out}")
    print(f"  {no_match} had no real short form and were dropped")
    if low_conf:
        print(f"  {low_conf} dropped below confidence {args.min_confidence}")
    if failures:
        print(f"  {len(failures)} failed; rerun to fill them in", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
