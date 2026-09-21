# Baby names

Working out *why* we like the names we like, instead of guessing at more of them.

## The problem

Three data points, and only three:

| Name | Reaction |
| --- | --- |
| Theresa / Tessa | loved |
| Margaret / Maggie | liked, but we watch a lot of Daniel Tiger |
| Rebecca / Becca | "doesn't quite land" |

All three are long classic formal names with a two-syllable nickname stressed
on the first syllable. **So the obvious shape is not what separates them** —
whatever makes Becca miss is something else, and that something else is what
we're trying to name.

Two hypotheses, both testable:

- **Era.** Theresa and Margaret read as inherited from a grandparent.
  Rebecca reads as our own generation — familiar rather than old.
- **How hard the nickname works.** Theresa → Tessa and Margaret → Maggie
  reshape the sound. Rebecca → Becca just removes a syllable from the front.

## The Daniel Tiger problem

Baby Margaret is Daniel Tiger's little sister, and we are naming a little
sister. That is about as strong a priming effect as you could design, so the
worry is fair.

But it's testable rather than something to agonise over, and the name pool is
built to test it. It contains **Magdalena → Maggie** and **Marguerite → Daisy**:
a second road to the same nickname, and a Margaret variant with a completely
different one. If Margaret scores high and its structural neighbours score low,
it's the television. If the whole neighbourhood scores high, the taste is real
and Margaret just happens to sit in it.

## How it works

1. **`names.json`** — 50 candidates, each with its formal name, the short form
   you'd actually call her, and a note. Three of them are the anchors above,
   mixed into the deck so they get rated blind alongside everything else.

2. **Rate them.** The [rating artifact](https://claude.ai/artifact/YDJY6apBPXZpE7SuKZCkUn)
   presents each name on its own and takes a gut reaction. Both parents rate
   separately, then it shows where they agree and where they split. Export the
   rows to `ratings.json`.

3. **`classify.py`** — scores every name on seven traits with TypeSafe, one
   parallel request per name. Writes `scored.json`.

4. **`fit_weights.py`** — fits the traits to the real ratings. Reports which
   traits actually predict what we said, checks the fit against the three
   anchors, and ranks the names nobody has rated yet.

The order matters: the ratings come *first* and are what the trait scores get
validated against. Without them the seven weights would just be my opinion
wearing a number.

## The seven traits

| Trait | Kind | Want |
| --- | --- | --- |
| `era` | Score, 5 levels | high — inherited, not our generation |
| `nickname_transformation` | Score, 5 levels | high — reshaped, not clipped |
| `child_adult_range` | Score, 5 levels | high |
| `currently_common` | Noul | low |
| `recognisable` | Noul | high |
| `clashes_with_sibling` | Noul | low — it has to work next to Tessa |
| `register_match` | Noul | high |

Each is one narrow judgment, and all seven are independent, so they go in a
single request per name and run in parallel. The raw scores are stored
un-flipped; direction and weighting are applied later in `fit_weights.py`, so
changing our minds about what we want costs no inference.

## Running it

```bash
pip install typesafe-sdk numpy          # NB: the package is typesafe-sdk.
                                        # Plain `typesafe` on PyPI is an
                                        # unrelated decorators library.
export TYPESAFE_API_KEY=...

python classify.py --limit 5            # smoke-test on five names
python classify.py                      # score all 50 -> scored.json
python fit_weights.py                   # needs ratings.json
```

`classify.py` needs to reach `api.typesafe.ai`, which is blocked from inside
the Claude Code web sandbox — run it somewhere with open egress.

## Reading the output

`fit_weights.py` prints the leave-one-out error next to the in-sample error.
Seven traits fitted to fifty names will happily fit noise, so if those two
numbers diverge, believe the per-trait correlations and ignore the ranking.
The correlations are the actual deliverable — they're the answer to "what do
we like" — and the ranking is a convenience on top.

The anchor check is the circuit breaker. Theresa, Margaret and Rebecca were
rated blind, and we already know roughly where they should land. If the fitted
model disagrees with them, the traits are wrong and nothing else on the page
is worth reading.
