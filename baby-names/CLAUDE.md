# Context for this project

Picking a name for a second daughter. The first is **Tessa** (full name
Theresa). The parents are **Alex** and **Hannah**.

## Where things stand

Alex has rated all 50 names in `names.json`. Hannah has not rated yet — her
ratings are the held-out test for a hypothesis formed on Alex's (see
`sound_texture` below), so they are worth more than a second opinion.

The rating tool is an artifact:
https://claude.ai/artifact/YDJY6apBPXZpE7SuKZCkUn
Ratings live in its `db` under the `ratings` collection and are read back with
the `ArtifactData` tool. `ratings.json` is gitignored — if it is missing, pull
it from the artifact rather than reconstructing it.

## What the ratings established

**The nickname carries the decision.** Margaret → Maggie and Magdalena →
Maggie both scored 3. Two unrelated formal names, one shared short form,
identical score. This also killed the worry that Maggie was contaminated by
Daniel Tiger (Baby Margaret is Daniel's little sister) — Magdalena is not on
the show and scored the same.

**The era hypothesis was wrong.** The original theory was that Theresa and
Margaret read as grandmother-era while Rebecca read as too recent. The data
says otherwise: archetypal grandmother names averaged 0.21 (28 of 33 scored
zero — Hattie, Winnie, Nell, Etta, Josie), longer flowing Latinate names
averaged 1.17, and Rebecca — the supposedly too-recent control — scored 2.
`era` stays in the question set so the fit reports this rather than assuming
it.

**What the winners share is sound, not age.** Theresa, Magdalena, Genevieve,
Isadora, Leonora, Millicent, Rebecca: long, vowel-led, soft-consonanted. The
clipped Anglo-Germanic register (Edith, Harriet, Agatha, Winifred, Bridget)
was rejected almost uniformly. `sound_texture` was added for this — but it was
derived by looking at Alex's ratings, so fitting it back to them proves
nothing. Hannah's ratings are the honest test.

## Two kinds of excluded name — do not conflate them

**`confounds.json` — the rating is untrustworthy.** The rating is about a
real-world association the model cannot see (Izzy is close friends' dog, Daisy
is Tessa's best friend). Excluded from the fit entirely. Trying to rate around
an association does not work; the effect is not introspectable.

**`unavailable.json` — the rating is accurate but the name cannot be chosen.**
Theresa is the clearest case: genuinely loved, correctly rated 3, and
unpickable because it is the older sister's name. These stay *in* the fit —
Theresa is the single purest example of the target taste — but are kept out of
recommendations. Removing Theresa from the fit would delete the best evidence
of what they like.

## Filters must be set from the ratings, not from assumptions

This was got wrong twice, in the same way, and is the main thing to be careful
about here:

- A `min-peak-pct` of 0.005 deleted Magdalena (peak 0.00028), rated **3**. The
  ratings run down to Isadora at 0.00008, so rarity is effectively not a
  filter — this family's taste runs rare.
- A `max-current-rank` of 250 deleted Margaret (2024 rank 119, rated **3**)
  and Genevieve (rank 165, rated **2**). Stated preference was "nothing too
  common"; revealed preference contradicts it. Popularity is now a feature,
  not a filter.

Before adding any filter, check it against the names already rated ≥ 2. If it
drops one, it is wrong.

## Division of labour

Facts go in code, judgments go to the model. `build_corpus.py` computes peak
decade, current rank, revival, syllables and Tessa-clash from SSA data — none
of that needs a model. The model handles sound, nickname quality, register
and clash-by-ear.

## The pipeline

```
build_corpus.py    SSA data -> corpus.json      3,726 candidates   (no API)
pick_nicknames.py  corpus.json -> named.json    Choice + no-match  (API)
classify.py        named.json -> scored.json    8 traits per name  (API)
fit_weights.py     + ratings.json               weights + ranking  (no API)
```

`classify.py` still defaults to `names.json` (the rated 50). For the real
search, point it at `named.json`.

**Scale warning:** 3,726 names is one request each in `pick_nicknames.py` and
again in `classify.py`. Use `--limit` first and measure cost before running
the whole corpus.

## Known gaps

- **Corpus coverage.** It is SSA top-1000-per-year, so names that never
  reached the US top 1000 cannot appear. Verity, rated **2**, is not in it.
  A British or literary name list would fill this in.
- **Mechanical nicknames are still imperfect** — `Ntine` for Clementine,
  `Therrie` for Theresa survive into `corpus.json`. This is by design:
  `pick_nicknames.py` is what rejects them, via its no-match outcome.
- **No tags.** Alex left every "what decided it?" tag blank, so which trait
  drove each rating is inferred rather than stated.
- **Other confounds may be hiding in the zeros.** 34 of 50 names scored 0;
  any that are taken by a friend, relative or pet are indistinguishable from
  genuine rejections and should be added to `confounds.json`.

## Environment note

This project was started in a Claude Code web session where `api.typesafe.ai`
and `docs.typesafe.ai` were both blocked by the egress proxy, so nothing that
needs the API was ever executed there — only validated offline against the
SDK's types. Locally with open egress that constraint is gone, and the live
TypeSafe docs (which the skill asks you to read) become available too.
