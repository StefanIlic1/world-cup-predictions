# World Cup 2026 Predictions

A from-scratch statistical forecast of the 2026 FIFA World Cup. The project builds its own
**form** metric on top of FIFA-style ELO ratings, trains three classifiers on World Cup matches
from 1998–2018, validates them against the 2022 tournament, and then runs **10,000 Monte Carlo
simulations** of the full 48-team 2026 tournament.

Everything lives in four Jupyter notebooks under [`src/`](src), each one a stage in the pipeline.

---

## Headline results

Champion probability over 10,000 simulated tournaments, per model:

| Rank | Logistic Reg. (N=10) | XGBoost (N=10) | XGBoost (N=20) |
|---|---|---|---|
| 1 | 🇪🇸 Spain **28.0%** | 🇪🇸 Spain **34.0%** | 🇦🇷 Argentina **29.4%** |
| 2 | 🇦🇷 Argentina 16.4% | 🇦🇷 Argentina 20.5% | 🇪🇸 Spain 21.4% |
| 3 | 🇫🇷 France 14.9% | 🇫🇷 France 14.4% | 🇫🇷 France 15.4% |
| 4 | 🇨🇴 Colombia 12.8% | 🇨🇴 Colombia 7.3% | 🇧🇷 Brazil 5.1% |
| 5 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England 7.8% | 🇧🇷 Brazil 3.0% | 🇳🇱 Netherlands 4.7% |

All three models independently converge on the same **Spain vs. Argentina final** — they only
disagree on who lifts the trophy (LR and XGB N=10 favour Spain at 59.7% / 65.9%; XGB N=20 favours
Argentina at 77.3%).

---

## How it works

```
  raw data                 notebook                          artifact
─────────────────────────────────────────────────────────────────────────────────
 results.csv          ┌─ 01_data_cleaning_and_elo ─┐   output/*_clean.csv
 former_names.csv  ───┤   normalise team names,    ├──  (49k matches, ELO
 *_elo_ratings.csv    └─ fix dates, align names ───┘    time series, 48 teams)
                                   │
                      ┌─ 02_functions_and_data ────┐   model_data/
 matches_1930_2022 ───┤   get_form(), ELO-at-date, ├──  training_data_{5,10,15,20}
 wc2026_fixtures   ───┤   per-team-per-year        │    validation_data_{…}
                      └─ profiles → feature rows ──┘    profiles_2026_{10,20}
                                   │                     fixtures_2026_n{10,20}
                      ┌─ 03_training_and_validation ┐
                      │   LR / GBT / XGBoost ×      ├──  model comparison table
                      └─  N ∈ {5,10,15,20} ────────┘
                                   │
                      ┌─ 04_simulation ────────────┐   group tables, brackets,
                      │   10k group-stage sims,    ├──  champion probabilities
                      └─  10k knockout sims ───────┘
```

### The form feature

The core idea is that a team's ELO tells you how good they *have been*, and form tells you
whether they are currently **beating or missing their own expectation**. Form is the net ELO delta
a team accumulated over their last *N* matches, using FIFA's SUM formula:

$$\text{form} = \sum_{i=1}^{N} I_i \times (W_i - W_{e,i})$$

- **I** — match importance weight, on FIFA's official scale (friendly `1.0` → Nations League `1.5`
  → qualifier `2.5` → confederation final `3.5` → World Cup `5.0`)
- **W** — actual result: win `1`, draw `0.5`, loss `0`
- **Wₑ** — expected result from the ELO gap, `1 / (10^(-Δ/600) + 1)`

Positive form means a team has consistently outperformed expectations; negative means they have
underperformed. It is computed two ways — **overall form** (all matches) and **competitive form**
(friendlies excluded) — and can be evaluated *as of* any historical date so that training rows
only ever use information available before that tournament kicked off.

### Model features

Early versions fed raw `home_elo`, `away_elo`, `home_form`, … to the models and overfit badly. The
final feature set is three symmetric differences only:

| Feature | Meaning |
|---|---|
| `elo_diff` | home ELO − away ELO at the tournament snapshot date |
| `form_diff` | home form − away form over the last *N* matches |
| `comp_form_diff` | same, friendlies excluded |

Label: `1` home win, `0` draw, `-1` away win, from the 90-minute score.

### Model selection

Trained on World Cups 1998–2018 (384 matches), validated on 2022 (64 matches, held out entirely):

| Model | N=5 | N=10 | N=15 | N=20 |
|---|---|---|---|---|
| Logistic Regression | .563 | **.578** | .547 | .547 |
| Gradient Boosted Trees | .422 | .547 | .484 | .547 |
| XGBoost | .484 | **.609** | .516 | **.625** |

*(2022 validation accuracy; log-loss tracked alongside in notebook 03.)*

Takeaways baked into the project:

- **N=10 is the sweet spot.** Ten matches reaches back far enough to cover late qualifiers and
  warm-up friendlies without diluting current form.
- **Plain GBT overfits hard** — 0.98 train vs. 0.42–0.55 validation. Soccer has too much variance
  and too few World Cup rows for unregularised trees.
- **XGBoost's regularisation (`subsample=0.8`, `reg_lambda=2.0`) survives it**, and beats logistic
  regression at both N=10 and N=20.

The three survivors — LR N=10, XGB N=10, XGB N=20 — are each simulated separately rather than
ensembled, so the spread between them acts as a rough honesty check on the forecast.

### Simulation

**Group stage.** Each of the 72 group fixtures gets a `(p_away, p_draw, p_home)` triple from the
model; 10,000 simulations sample an outcome per match and accumulate points. Teams are ranked by
expected points, top two per group advance, plus the eight best third-place finishers.

**Knockouts.** The Round of 32 bracket is seeded from each model's own group projection. Knockout
matches cannot be drawn, so the draw probability is dropped and the remaining two renormalised:
`P(team1) = p_home / (p_home + p_away)`. Another 10,000 full-bracket simulations give each team's
probability of reaching every round and of winning the tournament.

---

## Repo layout

```
src/
├── 01_data_cleaning_and_elo.ipynb   name normalisation, date fixing, ELO cleanup
├── 02_functions_and_data.ipynb      form engine + training/validation/2026 feature sets
├── 03_training_and_validation.ipynb LR vs GBT vs XGBoost across N ∈ {5,10,15,20}
├── 04_simulation.ipynb              group-stage + knockout Monte Carlo, bracket plots
│
├── international_matches/           raw: results.csv (49k matches, 1872–2026), former_names.csv
├── world_cup_matches/               raw: matches_1930_2022.csv, wc2026_group_fixtures.csv, FIFA rankings
├── elos/                            raw: historical_elo_ratings.csv (6.6k snapshots), wc2026_elo_ratings.csv
├── output/                          cleaned data written by notebook 01 (+ recent_elo_clean.csv)
└── model_data/                      model-ready feature tables written by notebook 02
```

### Key files

| File | Rows | What it is |
|---|---|---|
| `international_matches/results.csv` | 49,450 | every international match, 1872–2026 |
| `elos/historical_elo_ratings.csv` | 6,678 | ELO per team per date snapshot, 1872–2025 |
| `elos/wc2026_elo_ratings.csv` | 48 | ELO for the qualified 2026 teams (27 May 2026 snapshot) |
| `output/recent_elo_clean.csv` | 243 | current ELO baseline used for 2026 form |
| `model_data/training_data_10.csv` | 384 | 1998–2018 WC matches with features + label |
| `model_data/validation_data_10.csv` | 64 | 2022 WC matches, same schema |
| `model_data/fixtures_2026_n10.csv` | 72 | the 72 group fixtures with feature differences attached |

---

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost matplotlib jupyterlab
jupyter lab
```

Built against pandas 3.0, numpy 2.4, scikit-learn 1.9, xgboost 3.2, matplotlib 3.10.

Notebooks read and write paths **relative to `src/`**, so run Jupyter with `src/` as the working
directory and execute them in order — `01 → 02 → 03 → 04`. Each stage consumes the previous
stage's CSVs, which are committed, so 03 and 04 can be re-run on their own.

Notebook 04 is the slow one: 10,000 group-stage simulations plus 10,000 bracket simulations per
model, three models, all in pure Python loops.

---

## Caveats

- **Elos calculated on tens of thousands of matches, but only ~400 pure training matches due to scarcity of world cups.**
  validation accuracy on a single 64-match tournament carries wide error bars.
- **`home_team` in a World Cup is a fixture-listing convention**, not real home advantage (except
  for the actual hosts). The models learn whatever asymmetry is in the data regardless.
- **Form uses a fixed ELO baseline** for all N matches in a window rather than walking the rating
  forward match by match — a deliberate simplification, close enough at this horizon.
- **Brackets are hard-coded** in notebook 04 from each model's group projection, so the knockout
  simulation assumes the most likely group outcome instead of propagating group uncertainty into
  the bracket.
- **Scores, not just results.** The models predict win/draw/loss only; goal difference, the first
  group-stage tiebreaker, is not modelled, so group ranking is by expected points alone.
- `output/recent_elo_clean.csv` is an input to notebook 02 but is not regenerated by notebook 01 —
  it ships with the repo as a committed snapshot.
