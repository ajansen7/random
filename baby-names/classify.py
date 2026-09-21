"""Score candidate girls' names on the traits that separate the ones we like.

The trait set is reverse-engineered from three known reactions:

    Theresa / Tessa    loved
    Margaret / Maggie  liked
    Rebecca / Becca    "doesn't quite land"

All three share the surface shape (a long classic formal name, a two-syllable
nickname stressed on the first syllable). So the surface shape is not what
separates them, and the questions below go after the things that might:
how old the formal name feels, and how much work the nickname does to get
from the formal name to itself.

Every question is independent and asked over the same state, so they go in a
single request per name and run in parallel. Raw per-trait scores are written
out untouched; turning them into a ranking is a separate, cheap step in
fit_weights.py, which fits the weights to real ratings instead of guessing.

Usage:
    export TYPESAFE_API_KEY=...
    python classify.py                  # scores names.json -> scored.json
    python classify.py --limit 5        # smoke-test on five names first
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from typesafe_sdk import (
    Noul,
    Score,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
)

HERE = pathlib.Path(__file__).parent
SIBLING = {"nickname": "Tessa", "formal": "Theresa"}


# --- the judgments -------------------------------------------------------
#
# Score criteria are an ordered rubric: each level describes a concrete
# situation and has to stand on its own, because the model sees the level
# text and not the variable name. Noul criteria describe the yes and no
# outcomes for a single condition.

QUESTIONS = {
    # The leading hypothesis for why Becca misses. Theresa and Margaret sit
    # further back than Rebecca, which reads as the parents' own generation
    # rather than a grandparent's.
    "era": Score(
        instructions=(
            "Consider only `candidate.formal_name`. Which generation does it "
            "sound like it belongs to, judged by when it was widely given to "
            "babies in the English-speaking world?"
        ),
        criteria=[
            "Newly coined, or a name that surged into fashion within the last ten years.",
            "A name given widely to women now in their thirties and forties.",
            "A name given widely to women now in their fifties and sixties.",
            "A name given widely to women born in the early twentieth century, old enough that it reads as inherited rather than dated.",
            "A name with no single modern peak at all: medieval, literary, or biblical, in continuous use across centuries.",
        ],
    ),
    # The second hypothesis. Theresa->Tessa and Margaret->Maggie reshape the
    # sound; Rebecca->Becca just removes a syllable from the front.
    "nickname_transformation": Score(
        instructions=(
            "Compare `candidate.nickname` to `candidate.formal_name`. How much "
            "work does the short form do to get from one to the other?"
        ),
        criteria=[
            "The short form is identical to the formal name, or the name has no real short form.",
            "The short form is the formal name with syllables removed from one end and nothing else changed.",
            "The short form removes syllables and adds a conventional -y or -ie ending.",
            "The short form changes or drops sounds inside the name, so it is audibly reshaped rather than trimmed.",
            "The short form is a substantially different word, linked to the formal name by naming tradition rather than by spelling.",
        ],
    ),
    "child_adult_range": Score(
        instructions=(
            "Consider `candidate.formal_name` and `candidate.nickname` together "
            "as one name a person carries for life. How well does the pair cover "
            "both ends of a life?"
        ),
        criteria=[
            "Works at only one age: either the short form is unusable on an adult or the formal name is unusable on a child.",
            "Usable across a life, but one end is a clear stretch.",
            "Both forms work at both ends without being especially apt at either.",
            "The short form suits a child well and the formal name carries an adult professionally.",
            "The pair is unusually well matched: a warm name for a child and a dignified one for an adult, with a natural moment to switch.",
        ],
    ),
    "currently_common": Noul(
        instructions=(
            "Is `candidate.formal_name` or `candidate.nickname` common among "
            "girls born in the United States in the last five years?"
        ),
        criteria={
            "true": "Either form is in frequent current use; a child would likely share it with a classmate.",
            "false": "Both forms are rare among children today, whatever their history.",
        },
    ),
    "recognisable": Noul(
        instructions=(
            "Would a typical American English speaker recognise "
            "`candidate.formal_name` as an established name and spell it "
            "correctly after hearing it once?"
        ),
        criteria={
            "true": "An established name most people have met before and can spell from hearing it.",
            "false": "Unfamiliar, easily misspelled, or likely to need repeating and explaining.",
        },
    ),
    "clashes_with_sibling": Noul(
        instructions=(
            "The candidate would be a younger sister to a girl already called "
            "`sibling.nickname` (full name `sibling.formal`). Said aloud together "
            "as a pair, would the two names be awkward or easily confused?"
        ),
        criteria={
            "true": "The names rhyme, share most of their sounds, or are mistakable for one another when called across a room.",
            "false": "The names are comfortably distinct spoken together, though they may share a style.",
        },
    ),
    "register_match": Noul(
        instructions=(
            "Do `candidate.formal_name` and `candidate.nickname` belong to the "
            "same naming world as `sibling.formal` and `sibling.nickname` - the "
            "kind of choice the same parents would plausibly make twice?"
        ),
        criteria={
            "true": "The same taste is visible in both: they sit together as a set without one looking out of place.",
            "false": "A noticeably different taste - more modern, more ornate, more invented, or more plain than the sibling's name.",
        },
    ),
}

# Traits where a HIGHER answer is what we want. The rest are inverted when
# composed into a single number; keeping the raw answers un-flipped here
# means a change of mind about direction costs no inference.
HIGHER_IS_BETTER = {
    "era": True,
    "nickname_transformation": True,
    "child_adult_range": True,
    "currently_common": False,
    "recognisable": True,
    "clashes_with_sibling": False,
    "register_match": True,
}


def state_for(name: dict) -> dict:
    """The evidence one judgment gets. Named fields, since it has several parts."""
    return {
        "candidate": {
            "formal_name": name["formal"],
            "nickname": name["nickname"],
            "other_nicknames": name.get("other_nicknames", []),
            "background": name.get("note", ""),
        },
        "sibling": {"nickname": SIBLING["nickname"], "formal": SIBLING["formal"]},
    }


def answers_to_row(name: dict, response) -> dict:
    """Flatten one response into a row, keeping confidence alongside each value."""
    row = {"id": name["id"], "formal": name["formal"], "nickname": name["nickname"]}
    if name.get("anchor"):
        row["anchor"] = True
    for qid, ans in response.answers.items():
        if ans.type == "noul":
            row[qid] = ans.noul
        elif ans.type == "score":
            # Normalise to 0..1 so traits compose on one scale regardless of
            # how many levels a rubric happens to have.
            levels = max(len(QUESTIONS[qid].criteria) - 1, 1)
            row[qid] = ans.score / levels
            row[qid + "__confidence"] = ans.confidence
        else:
            row[qid] = ans.choice
            row[qid + "__confidence"] = ans.confidence
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", default=str(HERE / "names.json"))
    ap.add_argument("--out", default=str(HERE / "scored.json"))
    ap.add_argument("--limit", type=int, default=0, help="score only the first N names")
    ap.add_argument("--model", default=None, help="override the default model")
    args = ap.parse_args()

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set.", file=sys.stderr)
        return 2

    names = json.loads(pathlib.Path(args.names).read_text())
    if args.limit:
        names = names[: args.limit]

    rows, failures = [], []
    with TypeSafeClient(model=args.model) as client:
        for i, name in enumerate(names, 1):
            label = f"{name['nickname']} ({name['formal']})"
            try:
                response = client.system_one(
                    state=state_for(name), questions=QUESTIONS
                )
            except (TypeSafeAPIConnectionError, TypeSafeAPIError, TypeSafeError) as exc:
                print(f"  [{i}/{len(names)}] {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failures.append({"id": name["id"], "error": str(exc)})
                continue
            rows.append(answers_to_row(name, response))
            print(f"  [{i}/{len(names)}] {label}")

    pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} scored names to {args.out}")
    if failures:
        print(f"{len(failures)} failed; rerun to fill them in", file=sys.stderr)
    return 1 if failures and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
