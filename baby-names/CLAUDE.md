# Context for this project

Picking a name for a second daughter. The first is **Tessa** (full name
Theresa). The parents are **Alex** and **Hannah**.

## Where things stand

Alex has rated all 50 names in `names.json`. Hannah has not rated yet — her
ratings are the held-out test for a hypothesis formed on Alex's (see
`sound_texture` below), so they are worth more than a second opinion.

The rating tool is an artifact:
https://claude.ai/artifact/YDJY6apBPXZpE7SuKZCkUn

It now holds **122 names in two batches**. Batch one is the original fifty,
in their original order, so the fifty ratings already collected still key
correctly and Hannah rates them first - they are the held-out test of
everything fitted to Alex's. Batch two is the 72 from `spanning_set.py`,
shuffled and appended. The names live in the page source, not the db, so
adding more means republishing the artifact; ratings key off `nameId`, so
that is safe as long as existing ids do not change.
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

## Both raters are in. 244 ratings, and the search is over.

Hannah rated all 122 on 2026-09-21, so the held-out test finally ran.

**`sound_texture` is dead, and this is the clean kill.** Against Hannah's
ratings on the original fifty - the pure held-out test, a hypothesis formed
on Alex's ratings meeting data it never saw - it scores **r = -0.15**. The
wrong sign. Remove Theresa, whose 3 distorts everything, and it is -0.08. It
never worked on Alex's either (+0.03). The theory that "what the winners
share is sound, long and vowel-led and soft-consonanted" is finished: the
model says Georgiana and Magdalena have the same sound, and one is a 3 and
the other a 0.

**`nickname_transformation` was right all along.** It is the strongest
surviving trait - +0.21 against Hannah's held-out fifty, +0.33 across her
122, +0.27 joint. That is the README's *original* second hypothesis, the one
abandoned when the era theory collapsed and `sound_texture` was invented to
replace it. Theresa to Tessa and Margaret to Maggie reshape the sound;
Rebecca to Becca clips it. That was the answer.

**Confirmed on both raters independently:** `era` is noise (+0.09 / -0.05).
`currently_common` is *positively* correlated with liking (+0.37 Hannah,
+0.33 Alex) - popularity is a feature, and now twice measured.
`clashes_with_sibling` reads +0.37 until Theresa is removed and then -0.15,
which is the correct sign and a reminder to always drop her first.

**Margaret and Magdalena scored identically for both raters, separately**
(Alex 3, Hannah 2, both). The nickname-carries-the-decision finding now
replicates on a rater who had never seen it.

## Hannah is the constraint, and she is harsh

Mean rating 0.16 against Alex's 0.48. She gave one 3 and two 2s in 122
names; he gave three 3s and fourteen 2s. Agreement is r = +0.49, 68% exact,
91% within a point.

Averaging the two is the wrong move and was tried: it squashes her 0s
against his 2s and predicts a scale neither used. `taste_match.py --rater`
models them separately. Alex is predictable (Spearman +0.53, 4 of his 16
liked names in the top 12 against 1.6 by chance). Hannah is barely
modellable - Spearman +0.39 but only **three** positive examples, so treat
any claim about her taste as provisional.

## The answer

Of 1,646 filtered corpus candidates, **zero clear both held-out bars**
(Alex's best 3 scored 2.61 held out, Hannah's best 2 scored 1.88). The best
joint candidate is Katherine/Kate at 1.74, short of Hannah's bar. The
corpus - SSA top-1000, known gap #1 - has been searched and does not
contain a better name than the one they already have.

What the ratings say on their own:

| | Alex | Hannah | |
| --- | --- | --- | --- |
| Theresa / Tessa | 3 | 3 | the older sister's name, unpickable |
| **Magdalena / Maggie** | **3** | **2** | |
| **Margaret / Maggie** | **3** | **2** | |
| Alivia / Allie | 2 | 1 | the only others Hannah did not reject |
| Elodie / Ellie | 2 | 1 | |
| Rebecca / Becca | 2 | 1 | |

Two names clear a joint 2.0, and they are the same name twice: **Maggie**,
reached through Margaret or through Magdalena. The remaining decision is
which formal name to put on the certificate, which is a question about
Margaret's popularity (US #126 in 2024) against Magdalena's rarity - not a
question any of this machinery can answer.

If they want more options, the move is not more modelling. It is a British,
literary or European name list, which is the one thing that would put names
like Verity - rated 2, absent from the corpus - back in play.

## Second source: ONS England and Wales

Known gap #1 is closed. `build_corpus_uk.py` pulls ONS Table 6 - every name
given to at least three girls in England and Wales in a year, 7,617 of them -
and merges it with the SSA corpus. It reaches exactly the register the SSA
top-1000 truncates:

    Verity 92   Millicent 33   Isadora 34   Genevieve 83   Leonora 71
    Cordelia 59   Araminta 11   Perdita 5   Clemency 3

Verity, rated 2 and previously impossible to find, is now in the corpus.
Merged: 3,725 SSA + 4,929 new = 8,654, with 1,537 in both.

Two facts handled in code because both are lookups. England and Wales
register hyphenated double names as single entries and there are 905 of them
(Gracie-Mae, Lillie-Grace, Ivy-May); they swamped two of the eight sets on
the first run and are dropped. And Table 6's floor of three is thin enough
to admit registration one-offs, so `narrow.py` requires a name to appear in
SSA at all **or** reach ten girls in England and Wales - checked against the
17 names rated >= 2, all of which clear it, Verity included.

## The output is sets, not a ranking

Two corrections from Alex, both right.

**A bar is the wrong instrument.** The previous run reported "zero of 1,646
candidates clear both held-out bars" and stopped. That is a decision rule
pretending to be a search, and it is especially wrong here: Hannah rated 107
of 122 names zero, so a bar set by her two 2s rejects everything, and a name
she would call a 1 - already her top 12% - dies alongside names she hated.
Nothing is thresholded now. `narrow.py` puts the two raters on a common
scale by **percentile**, because their scales genuinely differ (he averages
0.48, she 0.16, and her 1 means about what his 2 does), then keeps the top
slice on the weaker of the two. A funnel, not a verdict.

**One ranked column hides the useful question.** `facets.py` asks nine wider
facets per name - all independent, so all nine ride in the single request
each name already costs - then clusters the pool in facet space and labels
each cluster by whatever it is most extreme on. The labels are computed, not
written: a set exists because its names share something measurable, and the
label reports what. The facets were chosen from what the 244 ratings
established, so `era` and `sound_texture` are absent and four of the nine are
about the short form.

This surfaces things no ranking would. The clearest is a 35-name cluster
labelled *sits awkwardly beside Tessa* - Tabitha, Talia, Tamsin, Tamar,
Talya, Tiana - every one of which scores high on its own and is a trap next
to the sister. A second cluster, *drags one loud association*, collects
Amelie, Aretha, Mercedes, Ophelia, Madonna, Zelda: the Daniel Tiger problem
generalised and made per-name.

The set to read first is **the short form reshapes the name, warm to say
every day** - Madeleine, Grace, Mercy, Letitia, Lucile, Domenica - because
`nickname_reshapes` is the descendant of the one trait that survived the
held-out test.

## Where TypeSafe actually does the work

Worth stating plainly, because from the outside this looks like statistics.
Every judgment in the pipeline is a Jev call and none of the ranking is
Python:

| stage | questions per name | what code does |
| --- | --- | --- |
| `pick_nicknames.py` | 1 Choice over generated short forms + a no-match | generates candidates, tallies |
| `taste_match.py` | 1 Score + 3 Nouls + 1 Choice | assembles state from the ratings |
| `facets.py` | 5 Scores + 4 Nouls | clusters the answers, labels the clusters |
| `classify.py` | 4 Scores + 4 Nouls | now a diagnostic only |

A full run is roughly 30,000 Jev calls and about two minutes. Code fetches
data, computes lookups (syllables, peak decade, counts, hyphens) and does
arithmetic on the answers. It never decides whether a name is any good.

## The explorer artifact

https://claude.ai/artifact/8tHGteyTPZFz34v6wf94RD  (source kept at `explorer.html`)

Three tabs: the seven computed sets, the second-look lists, and what has been
marked. Marks are shortlist / maybe / no, stored in the artifact's `db` under
`marks` keyed `raterSlug__nameId` - the same shape the rating tool uses, so
`ArtifactData` reads them back the same way. Both raters see each other's
marks live.

It carries Fraunces and Karla from the rating artifact so the two read as a
pair, on a cooler palette since this one is the output rather than the input.

## Second look: where the picture is most likely wrong

`second_look.py` builds lists the main ranking cannot, each grounded in
something the 244 ratings established.

- **Too common to have considered.** Stated rule was "nothing too common";
  `currently_common` correlates *positively* with liking on both raters
  independently (+0.37 Hannah, +0.33 Alex). This is the largest known blind
  spot. Margot, Katherine, Grace, Leah, Gianna, Ruth.
- **Ordinary in Britain, invisible here.** Only reachable since the ONS
  source: Margaux, Meabh, Martina, Viola, Maja, Lowenna.
- **Where you two split.** Alethea, Talulah, Allegra, Elvira, Roisin.
- **BLOCKED: rejected on the short form, not the name.** The strongest of the
  four and the one that did not run. `pick_nicknames.py` has been storing
  `nickname_alternatives` since the first commit for exactly this - Marguerite
  scored 0 behind Daisy and Margo is a different candidate - and 265
  re-pairings across 102 rejected names were built and submitted. Every call
  returned **402: no available TypeSafe API credits**. The code is correct and
  runs on `python second_look.py` once credits are topped up.

**Dropped: model confidence as a signal.** Low `taste_confidence` looked like
a way to find names the evidence does not settle. It is not: 1,451 names sit
at exactly 0.00 and the *highest*-confidence names are the obvious rejections
(Stormi 0.97). Confidence here reports "certainly no", not "uncertain", so it
cannot find the interesting middle. Do not reach for it again.

## Allie, and the bug in ranking on the minimum

Alex spotted by eye that Allie was missing. He was right, and the reason is a
flaw worth keeping in mind.

**Alivia -> Allie is joint 4th of the 122**, behind only Theresa and the two
roads to Maggie, and one of only fifteen names Hannah did not rate zero. It
never appeared because `narrow.py` ranks on `fit_both`, the **weaker** of the
two percentiles, and Jev puts every road to Allie squarely in Hannah's middle
(~1.05-1.20). That is defensible on the evidence - Allie has one example to
learn from, Alivia at H1, against Maggie's two at H2 - but the effect is that
a real signal with thin support is invisible next to one with thick support.

The same mistake then repeated one level down: the first "roads to Allie"
list sorted by the minimum too, which buried **Alice** - the single
highest-scoring road to Allie (Alex 1.93) and a completely clean name
(is_respelling 0.09, no_real_nickname 0.01). The roads lists now sort on the
sum. Ranking on the minimum is right for a joint shortlist and wrong for
finding anything; do not use it twice in the same pipeline.

## The short form is the unit, so count evidence per short form

Margaret and Magdalena are unrelated names that scored identically for both
raters because they share Maggie. That makes the short form a candidate in
its own right, and the useful question is how much evidence sits behind each.

| short form | tried | Alex total | Hannah total | still live |
| --- | --- | --- | --- | --- |
| **Maggie** | 2 | 6 | 4 | Margaret, Magdalena |
| **Ellie** | 3 | 5 | 1 | Elodie |
| **Allie** | 1 | 2 | 1 | Alivia |
| Becca | 1 | 2 | 1 | Rebecca |
| Addie | 2 | 1 | 1 | Adelaide |

Read the columns against each other. **Ellie is better tested and Allie is
better confirmed**: Ellie was tried three times and took 1 of a possible 9
from Hannah (Elodie 1, Elianna 0, Elisabeth 0), while Allie went 1 of 3 on
its single try. Ellie is Alex's short form; Allie is the one that landed for
both. Maggie, tried twice and scoring for both every time, is still in a
class of its own.

**Alivia and Elodie are already-rated names sitting joint 4th and 5th**, and
every other view hides them because a rated name is not a recommendation.
They are still candidates and the artifact now says so.

Unrated roads: to Allie - Alice, Alessandra, Alia, Alyce, Alesia, Allegra,
Allyson. To Ellie - Ellodie, Elea, Ellora, Elayne, Elyssia, Eleia.

**A hint that is not a finding.** 10 of Hannah's 15 non-zero names have a
short form ending in -ie or -y, against 43% among the ones she rejected -
z = +1.72 on fifteen names, which is not significant, and flat for Alex (44%
against 47%). It is the right shape for the next rating batch to test
deliberately. It is not established and must not be filtered on.

## Environment note

This project was started in a Claude Code web session where `api.typesafe.ai`
and `docs.typesafe.ai` were both blocked by the egress proxy, so nothing that
needs the API was ever executed there — only validated offline against the
SDK's types. Locally with open egress that constraint is gone, and the live
TypeSafe docs (which the skill asks you to read) become available too.
