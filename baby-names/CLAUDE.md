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

## What the first full corpus run established (supersedes the above)

The pipeline has now been run end to end against the live API. Three things
that were written down as settled are not.

**The trait fit does not work, and cannot.** `fit_weights.py` gives a
leave-one-out RMSE of 1.10 against 0.91 for predicting the mean - worse than
a constant. The reason is structural rather than a bug: across the 48 usable
ratings every one of the eight traits is nearly constant (sd 0.07 to 0.21 on
a 0-1 scale). `names.json` was hand-built out of fifty long flowing classics,
and the traits describe exactly what all fifty share, so they cannot separate
them. Eight weights fitted to that produce noise, and the noise ranked the
corpus with Elizbeth, Avah and Madie on top.

**`sound_texture` is flat, and the README's claim for it is wrong.** It
averages 0.77 on the names rated 0 and 0.77 on the names rated 2 and 3.
Georgiana scores 0.91 and was rated 0; Magdalena scores 0.88 and was rated 3.
Whatever separates them is finer than that rubric. This was not Hannah's
held-out test finding it wanting - it fails on Alex's own ratings, the ones
it was derived from.

**What replaced it.** `taste_match.py` puts the 48 ratings into the *state*
and asks for a direct comparison, instead of compressing them into eight
numbers. Validated leave-one-out, with each name judged against the other 47:
**Spearman +0.48**, and Margaret and Magdalena - the only two 3s - come back
at ranks 1 and 2 of 47. RMSE stays above the mean baseline because the model
over-predicts the level, but a shortlist only needs the ordering.

That number was first reported as +0.57, which was wrong and worth recording
why. The instructions had been given a calibration paragraph naming Eleanor,
Charlotte, Beatrice, Philippa and Josephine as names they rejected - written
after reading the first validation's worst misses. The exemplar list excluded
the held-out name, but the *instructions* did not, so five of the forty-seven
had their answers in the prompt. Nothing in the leave-one-out machinery can
catch that; the only defence is that no rated name appears anywhere in a
question. It does not now.

**The corpus is the binding constraint, and now it is measured.** Held out,
the names rated 3 score 2.64 and 2.79. The best of the 179 filtered corpus
candidates is Honora at 2.01, nothing reaches either 3, and only 16 of the
179 are in the 4+ syllable Latinate register they demonstrably prefer. This
is known gap #1 - SSA top-1000-only - quantified. The next real move is a
British, literary or European name list, not more modelling.

## Rules were the wrong instinct - ask instead

`shortlist.py` first grew a syllable floor on the formal name, a required
shape for the short form, and a spelling-key plus edit-distance plus
union-find family grouper. Every one of them was written by looking at bad
output and inventing a rule, and every one was fitted to the nine names
rated 2 or 3 - the same overfitting as the weight fit, in a place where the
project's own "check it against the names rated >= 2" ritual waved it
through, because a rule tuned on nine names trivially passes a check
against those nine names.

The syllable floor deleted **1,485 names the model considers perfectly good
formal names** - Vera, Grace, Miriam, Marian, Talia - to remove Maggie and
Meg.

All of it is gone, replaced by three Nouls in `taste_match.py` that ride in
the request each name already costs:

| question | catches | highest among the names rated >= 2 |
| --- | --- | --- |
| `is_short_form` | Maggie, Meg, Tess, Trudy as formal names | 0.19 |
| `is_respelling` | Margarett, Elizbeth, Madelyn | 0.38 |
| `is_boys_name` | John, Timothy, Kyle in the girls' file | 0.03 |

Cut at 0.5, none of them can delete a favourite, and `is_respelling` alone
reproduces the whole family grouper - it keeps exactly one canonical
spelling of Margaret, Rebecca and Elizabeth. `shortlist.py` went from 254
lines to 110 and from 179 candidates to 1,804.

**Still refused: any floor on nickname confidence.** Magdalena -> Maggie
comes back at 0.44 and is a rated 3; Genevieve -> Evie at 0.49 and a rated
2. Low confidence there is probability spread across several *real* short
forms. The safe signal is `no_real_nickname`, the mass the Choice leaves on
its no-match outcome, which tops out at 0.01 across the liked names.

## The ground truth is the real bottleneck

`spanning_set.py` builds the next rating batch. The fifty in `names.json`
were each picked by hand as plausible, so they all came out of one corner:
mean trait sd 0.13, and a trait that does not vary cannot explain a rating
that does.

Two wrong turns worth not repeating. Farthest-point sampling in trait space
maximises distance, so it returns outliers - the first batch was John,
Timothy, Princess, Diamond and Treasure, all of which would be rated 0,
which swaps no variance in the traits for no variance in the ratings. And
the first batch offered **Hannah**, the mother's name.

What it does now: cut the pool into six bands by predicted reaction, draw
twelve from each, spreading within the band. Mean trait sd 0.17 against
0.13, with the two that matter most roughly doubled - `era` 0.10 -> 0.24,
`sound_texture` 0.11 -> 0.20. 72 new names plus 17 repeats, the repeats
being the anchors, everything rated >= 2, and a spread of others so drift
between sessions is measurable.

`nickname_transformation` will not move: its sd across the whole 3,050-name
corpus is 0.07. That is the rubric, not the sample.

## Running it now

A `.venv` with `typesafe-sdk` and `numpy` is in the project directory, and
`TYPESAFE_API_KEY` is already in the environment. Both API stages go through
`parallel.py`, which runs them concurrently - the whole 3,726-name corpus is
about 20 seconds a stage, not the twenty minutes the scale warning assumed.

```bash
.venv/bin/python build_corpus.py                       # 3,726 candidates, no API
.venv/bin/python pick_nicknames.py                     # -> named.json     ~20s
.venv/bin/python taste_match.py --validate             # the honest test    ~2s
.venv/bin/python taste_match.py --names named.json     # -> taste.json     ~20s
.venv/bin/python shortlist.py                          # -> shortlist.json
```

`classify.py` and `fit_weights.py` still run and are still worth keeping -
the traits are the diagnostic that showed why the fit fails - but they are no
longer the path to a recommendation.

## Environment note

This project was started in a Claude Code web session where `api.typesafe.ai`
and `docs.typesafe.ai` were both blocked by the egress proxy, so nothing that
needs the API was ever executed there — only validated offline against the
SDK's types. Locally with open egress that constraint is gone, and the live
TypeSafe docs (which the skill asks you to read) become available too.
