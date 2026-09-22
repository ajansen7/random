"""Build the candidate pool: thousands of real names, not the fifty we rated.

Rating fifty names taught us what we like. This turns that into a search over
every girls' name the US has actually used, so the shortlist contains names we
have never thought of - which was the point of the exercise.

Two public datasets, both fetched from GitHub:

  - SSA top-1000 per year, 1880-2008 (hadley/data-baby-names): 4,018 distinct
    girls' names with year-by-year frequency.
  - SSA top-1000 for 2024 (aruljohn/popular-baby-names): what is common now.

Everything this script computes is a fact, not a judgment: peak decade,
current rank, whether a name is reviving, syllable count, whether it rhymes
with Tessa. None of that needs a model, and asking one to guess at numbers we
can look up would be slower, costlier and wrong more often. The model's job
starts after this, on the things that genuinely need taste.

The code filters here are deliberately blunt and only cut on near-certainties
(currently too common, clashes with the sister's name, no usable short form).
Anything resembling taste is left for the classifier.

Usage:
    python build_corpus.py                 # -> corpus.json
    python build_corpus.py --keep-common   # skip the popularity filter
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import pathlib
import re
import urllib.request

HERE = pathlib.Path(__file__).parent

HISTORIC_URL = "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv"
CURRENT_URL = "https://raw.githubusercontent.com/aruljohn/popular-baby-names/master/2024/girl_names_2024.json"

SISTER = "Tessa"

# Short forms that no mechanical rule would ever produce. Only the ones that
# are genuinely established in English - this is not a place to invent.
IRREGULAR_NICKNAMES = {
    "Margaret": ["Maggie", "Meg", "Greta", "Daisy", "Peggy"],
    "Eleanor": ["Nell", "Nellie", "Ellie"],
    "Elizabeth": ["Libby", "Bess", "Betsy", "Eliza", "Beth"],
    "Theresa": ["Tessa", "Tess", "Terry"],
    "Teresa": ["Tessa", "Tess"],
    "Magdalena": ["Maggie", "Lena", "Magda"],
    "Marguerite": ["Daisy", "Margo", "Rita"],
    "Katherine": ["Kitty", "Kate", "Katie"],
    "Catherine": ["Kitty", "Kate", "Katie"],
    "Henrietta": ["Etta", "Hettie", "Retta"],
    "Harriet": ["Hattie", "Hettie"],
    "Sarah": ["Sadie", "Sally"],
    "Susannah": ["Sukie", "Susie", "Zanna"],
    "Susan": ["Sukie", "Susie"],
    "Mary": ["Molly", "Mamie", "Polly"],
    "Dorothy": ["Dot", "Dottie", "Dolly"],
    "Dorothea": ["Thea", "Dot", "Dodie"],
    "Bridget": ["Birdie", "Biddy"],
    "Alberta": ["Bertie", "Albie"],
    "Roberta": ["Bobbie", "Birdie"],
    "Florence": ["Flossie", "Flo"],
    "Beatrice": ["Bea", "Trixie"],
    "Rosalind": ["Roz", "Rosie"],
    "Wilhelmina": ["Mina", "Willa", "Billie"],
    "Gwendolyn": ["Gwen", "Winnie"],
    "Frances": ["Frankie", "Fran"],
    "Josephine": ["Josie", "Jo", "Posy"],
    "Philippa": ["Pippa", "Pip"],
    "Augusta": ["Gussie", "Gus"],
    "Constance": ["Connie"],
    "Priscilla": ["Cilla", "Pris"],
    "Millicent": ["Millie"],
    "Winifred": ["Winnie", "Freddie"],
    "Genevieve": ["Evie", "Vivi", "Gen"],
    "Clementine": ["Clemmie", "Clem"],
    "Anastasia": ["Stasia", "Ana", "Tasia"],
    "Isadora": ["Dora", "Izzy", "Sadie"],
    "Leonora": ["Nora", "Leo"],
    "Georgiana": ["Georgie", "Gigi"],
    "Cordelia": ["Delia", "Cora"],
    "Matilda": ["Tilly", "Mattie", "Tilda"],
    "Adelaide": ["Addie", "Della"],
    "Emmeline": ["Emmy", "Lina"],
    "Seraphina": ["Sera", "Fina"],
    "Theodora": ["Teddy", "Thea", "Dora"],
    "Antonia": ["Toni", "Nia"],
    "Ottilie": ["Tillie", "Ottie"],
    "Verity": ["Vera"],
    "Rosemary": ["Romy", "Rosie"],
    "Cecilia": ["Cece", "Cilla"],
    "Juliana": ["Jules", "Lia"],
    "Imogen": ["Immy", "Midge"],
    "Edith": ["Edie"],
    "Agatha": ["Aggie"],
    "Rebecca": ["Becca", "Bex"],
    "Charlotte": ["Lottie", "Charlie"],
    "Eloise": ["Elle", "Weezie"],
    "Vanessa": ["Nessa"],
    "Clarissa": ["Rissa", "Clara"],
}

VOWELS = "aeiouy"


def syllables(name: str) -> int:
    """Rough syllable count: vowel groups, with a silent trailing -e."""
    n = name.lower()
    groups = re.findall(r"[aeiouy]+", n)
    count = len(groups)
    if n.endswith("e") and count > 1 and not n.endswith(("le", "ie", "ee")):
        count -= 1
    return max(count, 1)


def rhymes_with_sister(name: str, sister: str = SISTER) -> bool:
    """Would it be confusing next to the sister's name, said aloud?"""
    a, b = name.lower(), sister.lower()
    if a == b:
        return True
    # shared final two-vowel-and-after chunk, e.g. -essa / -esa
    for tail in (b[-4:], b[-3:]):
        if len(tail) >= 3 and a.endswith(tail):
            return True
    # same first syllable AND same ending is too close (Tessa / Tessie)
    if a[:3] == b[:3] and a[-1] == b[-1]:
        return True
    return False


def nickname_candidates(name: str) -> list[str]:
    """Short forms a person might actually use, generated mechanically.

    Deliberately over-generates. Picking the one people really use is a
    judgment, so the classifier selects from this list rather than the code
    guessing - the model can only choose what we put in front of it, so the
    list errs toward including too much.
    """
    out: list[str] = list(IRREGULAR_NICKNAMES.get(name, []))
    low = name.lower()
    groups = list(re.finditer(r"[aeiouy]+", low))

    # Prefixes cut at real syllable boundaries: the end of a vowel group, and
    # that plus its following consonants. Anything else produced non-words
    # ("Millicie", "Therie", "Isadie") - a stem has to be short and end where
    # a syllable ends before a diminutive can hang off it.
    stems = set()
    for g in groups[:2]:
        stems.add(low[: g.end()])
        end = g.end()
        while end < len(low) and low[end] not in VOWELS:
            end += 1
        stems.add(low[:end])

    for stem in sorted(stems):
        if not 2 <= len(stem) <= 6:
            continue
        out.append(stem.capitalize())
        # -ie / -y only off a SHORT consonant-ending stem, doubling the final
        # consonant the way English does: Mil -> Millie, not Milie.
        if 2 <= len(stem) <= 4 and stem[-1] not in VOWELS:
            base = stem if stem[-1] == stem[-2:-1] else stem + stem[-1]
            out.append((base + "ie").capitalize())

    # Trailing element of a long name: Isadora -> Dora, Leonora -> Nora.
    if len(groups) >= 3:
        start = groups[-2].start()
        while start > 0 and low[start - 1] not in VOWELS:
            start -= 1
        tail = low[start:]
        if 3 <= len(tail) <= 5:
            out.append(tail.capitalize())

    seen, uniq = set(), []
    for c in out:
        if c.lower() != low and c.lower() not in seen and 2 <= len(c) <= 8:
            seen.add(c.lower())
            uniq.append(c)
    return uniq[:8]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "corpus.json"))
    ap.add_argument("--cache", default=str(HERE / ".cache"))
    # Popularity is a FEATURE, not a filter, and off by default. "Nothing too
    # common" was the stated preference, but the revealed one disagrees:
    # Margaret is rank 119 in 2024 and rated 3, Genevieve is 165 and rated 2.
    # A rank cutoff of 250 deleted both. rank_2024 rides along on every record
    # so the fit can price it; the code does not get to make that call.
    ap.add_argument("--max-current-rank", type=int, default=0,
                    help="drop names inside this 2024 rank; 0 (default) disables it")
    # Set from the ratings, not from taste. The names rated 2 or 3 run down to
    # Isadora at 0.00008 of births, so any higher floor deletes names already
    # known to be liked - Magdalena, a 3, sits at 0.00028 and an earlier 0.005
    # floor dropped it. This is a noise floor only; rarity is not a filter here.
    ap.add_argument("--min-peak-pct", type=float, default=0.00005,
                    help="noise floor only; set from the lowest-peaking name rated >=2")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    cache.mkdir(exist_ok=True)

    hist_file = cache / "baby-names.csv"
    if not hist_file.exists():
        print("fetching historical SSA data ...")
        hist_file.write_bytes(fetch(HISTORIC_URL))
    cur_file = cache / "girl_names_2024.json"
    if not cur_file.exists():
        print("fetching 2024 SSA top 1000 ...")
        cur_file.write_bytes(fetch(CURRENT_URL))

    # --- historical frequency -------------------------------------------
    by_name: dict[str, dict[int, float]] = collections.defaultdict(dict)
    with io.StringIO(hist_file.read_text()) as fh:
        for row in csv.DictReader(fh):
            if not row["sex"].lower().startswith(("g", "f")):
                continue
            by_name[row["name"]][int(row["year"])] = float(row["percent"])

    current = json.loads(cur_file.read_text())
    current_names = current["names"] if isinstance(current, dict) else current
    current_rank = {n: i + 1 for i, n in enumerate(current_names)}

    # --- assemble --------------------------------------------------------
    records, dropped = [], collections.Counter()
    for name, series in sorted(by_name.items()):
        peak_year = max(series, key=series.get)
        peak_pct = series[peak_year]
        rank_now = current_rank.get(name)

        if peak_pct < args.min_peak_pct:
            dropped["too rare to be recognisable"] += 1
            continue
        if rhymes_with_sister(name):
            dropped[f"clashes with {SISTER}"] += 1
            continue
        if args.max_current_rank and rank_now and rank_now <= args.max_current_rank:
            dropped["too common right now"] += 1
            continue

        nicks = nickname_candidates(name)
        if not nicks:
            dropped["no usable short form"] += 1
            continue

        recent = [p for y, p in series.items() if y >= 1990]
        early = [p for y, p in series.items() if y <= 1950]
        records.append({
            "id": name.lower(),
            "formal": name,
            "nickname_candidates": nicks,
            "syllables": syllables(name),
            "peak_year": peak_year,
            "peak_decade": (peak_year // 10) * 10,
            "peak_pct": round(peak_pct, 5),
            "rank_2024": rank_now,
            # rising again after a long absence - the "ready to come back" shape
            "reviving": bool(recent and early and max(recent) > 0
                             and max(recent) < max(early) * 0.5 and rank_now),
        })

    pathlib.Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\n{len(by_name)} distinct girls' names in the source data")
    for reason, n in dropped.most_common():
        print(f"  -{n:>5}  {reason}")
    print(f"\n{len(records)} candidates written to {args.out}")
    by_decade = collections.Counter(r["peak_decade"] for r in records)
    print("peak decade spread:", dict(sorted(by_decade.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
