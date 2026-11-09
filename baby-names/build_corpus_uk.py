"""Second source: every girl's name registered in England and Wales.

Known gap #1 was that the SSA corpus is top-1000-per-year, so a name that
never cracked the US top thousand cannot appear in it at all. Verity, rated
2 by Alex, was simply not there to be found. Neither were Isadora, Araminta,
Perdita or Clemency - which is to say, most of the register this family
actually likes.

ONS Table 6 fixes that. It is every name given to at least three girls in
England and Wales in one year, 7,617 of them, and it reaches exactly the
rare British and European end the SSA list truncates:

    Verity 92    Millicent 33    Isadora 34    Genevieve 83
    Leonora 71   Cordelia 59     Araminta 11   Perdita 5    Clemency 3

Two things it gives beyond coverage. It is the girls' file, so a name in it
is a girl's name by construction and the boys'-name contamination the SSA
data needed a question to clean up cannot occur. And the count is a real
count of babies rather than a rank, so rarity is measurable rather than
merely off the bottom of the list.

No frequency floor is applied. The instinct to add one is the same instinct
that produced the syllable rule, and the names they like run down to
Theresa at 21 and Magdalena at 23 - fitting a floor to nine liked names is
how this project has gone wrong twice. The model's own is_short_form,
is_respelling and no_real_nickname questions do that work downstream.

Usage:
    python build_corpus_uk.py                 # -> corpus_uk.json
    python build_corpus_uk.py --merge         # -> corpus.json, both sources
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request

import build_corpus as bc

HERE = pathlib.Path(__file__).parent
ONS_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
    "birthsdeathsandmarriages/livebirths/datasets/"
    "babynamesenglandandwalesbabynamesstatisticsgirls/2025/2025girlsbabynames.xlsx"
)
TOTAL_BIRTHS = 253281      # sum of Table 6, used to put counts on the SSA scale


def fetch_ons(cache: pathlib.Path) -> dict[str, int]:
    """{name: girls given it in England and Wales, 2025} from ONS Table 6."""
    import openpyxl

    xlsx = cache / "ons_girls_2025.xlsx"
    if not xlsx.exists():
        print("fetching ONS baby names ...")
        req = urllib.request.Request(ONS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            xlsx.write_bytes(r.read())

    rows = list(openpyxl.load_workbook(xlsx, read_only=True)["Table_6"]
                .iter_rows(values_only=True))
    head = next(i for i, r in enumerate(rows)
                if r and len(r) > 1 and str(r[1]).strip().lower() == "name")
    out: dict[str, int] = {}
    for r in rows[head + 1:]:
        if not r or len(r) < 3 or not r[1]:
            continue
        try:
            out[str(r[1]).strip()] = int(r[2])
        except (TypeError, ValueError):
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "corpus_uk.json"))
    ap.add_argument("--cache", default=str(HERE / ".cache"))
    ap.add_argument("--merge", action="store_true",
                    help="also fold the SSA corpus in and write corpus.json")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    cache.mkdir(exist_ok=True)
    counts = fetch_ons(cache)
    print(f"{len(counts)} names in ONS Table 6, {sum(counts.values()):,} births, "
          f"rarest {min(counts.values())}")

    records, dropped = [], {"clashes with Tessa": 0, "no usable short form": 0,
                            "a compound registration, not one name": 0}
    for name, count in sorted(counts.items()):
        # England and Wales register hyphenated double names as single
        # entries - Gracie-Mae, Lillie-Grace, Ophelia-Mae, Ivy-May - and
        # there are hundreds of them. They are two names with a hyphen, not
        # the long-formal-plus-short-form shape being chosen here, and they
        # swamped two of the eight sets on the first run. Mechanical to spot,
        # so it belongs in code; no name either parent rated is compound.
        if "-" in name or " " in name or "'" in name:
            dropped["a compound registration, not one name"] += 1
            continue
        if bc.rhymes_with_sister(name):
            dropped["clashes with Tessa"] += 1
            continue
        nicks = bc.nickname_candidates(name)
        if not nicks:
            dropped["no usable short form"] += 1
            continue
        records.append({
            "id": name.lower(),
            "formal": name,
            "nickname_candidates": nicks,
            "syllables": bc.syllables(name),
            "source": "ons",
            "uk_count": count,
            # Share of girls born, so it sits on the same scale as the SSA
            # peak_pct and the two sources can be compared at all.
            "uk_pct": round(count / TOTAL_BIRTHS, 6),
        })

    pathlib.Path(args.out).write_text(json.dumps(records, indent=2))
    for reason, n in dropped.items():
        print(f"  -{n:>5}  {reason}")
    print(f"\n{len(records)} UK candidates written to {args.out}")

    if args.merge:
        ssa_path = HERE / "corpus.json"
        ssa = json.loads(ssa_path.read_text()) if ssa_path.exists() else []
        for r in ssa:
            r.setdefault("source", "ssa")
        by_id = {r["id"]: r for r in ssa}
        added = 0
        for r in records:
            if r["id"] in by_id:
                # Seen in both: keep the SSA record for its history, and carry
                # the UK count across so a name that is rare in America and
                # ordinary in Britain is visible as such.
                by_id[r["id"]]["uk_count"] = r["uk_count"]
                by_id[r["id"]]["uk_pct"] = r["uk_pct"]
                by_id[r["id"]]["source"] = "both"
            else:
                by_id[r["id"]] = r
                added += 1
        merged = sorted(by_id.values(), key=lambda r: r["id"])
        ssa_path.write_text(json.dumps(merged, indent=2))
        both = sum(1 for r in merged if r.get("source") == "both")
        print(f"\nmerged: {len(ssa)} SSA + {added} new from ONS = {len(merged)}")
        print(f"  {both} names appear in both sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
