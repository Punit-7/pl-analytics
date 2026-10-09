# Premier League Stats Hub

Every Premier League match since 1993/94, loaded by a tested Python pipeline into PostgreSQL and explored in Power BI and Excel.

![League table page](docs/league_table.png)

## What it shows

- **League table**: standings for any season and venue, with form and a cumulative points race for chosen teams.
- **Team profile**: a team's points, goals, shot conversion and goals minus xG, plus goals vs xG per game and shots by venue.
- **Match centre**: the score, referee and side-by-side stats for any single match.
- **History**: final position of every club in every season, the all-time table, and head-to-head records.

<details>
<summary>Screenshots of the other pages</summary>

![Team profile page](docs/TeamProfile.PNG)
![Match centre page](docs/match_centre.png)
![History page](docs/history.png)

</details>

The Excel workbook (`analytics/pl_stats_hub.xlsx`) has a dashboard built on the same mart.

## How it works

```
football-data.co.uk CSVs ─┐
                          ├─> ingest modules ─> PostgreSQL staging ─> pytest ─> publish to mart ─> Power BI + Excel
Understat xG (soccerdata) ┘   (data/ingest)     (staging schema)      (gate)    (mart schema)     (+ CSV copies in data/marts)
```

## Engineering

- **Config file**: seasons, paths, schemas and HTTP settings live in `config.toml`; credentials in `.env` (see `.env.example`).
- **Logging with run IDs**: every run gets an ID, logged to the console and a rotating `logs/pipeline.log`.
- **HTTP retries**: requests retry with backoff on 429 and 5xx responses.
- **Download validation**: each file must parse as CSV and have the expected columns before it is saved.
- **Test gate before export**: the test suite runs against staging; if any test fails, nothing is published.
- **`pipeline_runs` log**: each run's status, error and row count are written to `ops.pipeline_runs` (and `data/marts/pipeline_runs.csv`).

## How to run

```
pip install -r requirements.txt
python -m data.pipeline
```

Then open `analytics/pl_stats_hub.pbix` and `analytics/pl_stats_hub.xlsx` and refresh both. Run `python -m data.pipeline --help` for options (`--full-refresh`, `--skip-ingest`, `--log-level`).

## P2 — Match prediction and season simulation

A Premier League match model that predicts every 2026/27 fixture before kick-off, simulates the rest of the season into title, top-four and relegation probabilities, and reports honestly how close it gets to the bookmakers — plus a hand-built xG model on StatsBomb event data. The code is in `modelling/`.

### Match model vs the bookmakers

Over 2,660 matches from 2019/20 to 2025/26, the Dixon–Coles model scored a mean RPS of 0.2039, against 0.1968 for market-average closing odds: a gap of 0.0071 (95% bootstrap interval 0.0046 to 0.0095). The model does not beat the closing market. Its probabilities are well calibrated, and it beats both baselines in every season.

| Forecast (walk-forward, refitted weekly) | RPS | Log loss |
|---|---|---|
| Closing odds (benchmark) | 0.1968 | 0.9639 |
| Dixon–Coles | 0.2039 | 0.9935 |
| Plain Poisson (no low-score correction) | 0.2039 | 0.9932 |
| Equal-strength Poisson (baseline) | 0.2338 | 1.0712 |
| League frequencies (baseline) | 0.2339 | 1.0710 |

Lower is better. The low-score correction made no measurable difference here: plain Poisson scores the same.

![Match model calibration](modelling/reports/figures/match_calibration.png)

### Season simulation

10,000 simulated seasons, as of 2 October 2026 (five matches played by each team):

| Team | Expected points | Title | Top four | Relegation |
|---|---|---|---|---|
| Man City | 80.4 | 53.9% | 99.0% | 0.0% |
| Arsenal | 79.0 | 42.8% | 98.5% | 0.0% |
| Liverpool | 64.7 | 2.1% | 58.0% | 0.0% |
| Brighton | 60.2 | 0.6% | 33.8% | 0.1% |
| Hull | 58.9 | 0.2% | 25.3% | 0.2% |
| … | | | | |
| Tottenham | 39.7 | 0.0% | 0.1% | 37.1% |
| Ipswich | 29.4 | 0.0% | 0.0% | 91.5% |
| Coventry | 15.2 | 0.0% | 0.0% | 100.0% |

Two of the promoted teams (Hull, Coventry) are rated on five matches only, so their numbers are overconfident; see Limits.

Each week's predictions are written to the `predictions` schema and committed to `modelling/predictions/` at least a day before kick-off. A ledger then scores each match's first prediction against the result and the closing odds (`modelling/ledger/`).

### xG model

Trained on 38,728 shots from StatsBomb Open Data 2015/16 and scored on a held-out 20% of matches (7,725 non-penalty shots):

| Model | Log loss | AUC |
|---|---|---|
| Constant goal rate (baseline) | 0.3187 | 0.500 |
| Distance only (baseline) | 0.2816 | 0.751 |
| Logistic regression | 0.2646 | 0.793 |
| LightGBM | 0.2631 | 0.798 |
| StatsBomb's own xG (reference, uses player positions) | 0.2494 | 0.824 |

LightGBM's expected calibration error is 0.009, and its total xG on the test set is 750.3 against 750 goals. Distance and angle are the two strongest features (SHAP).

![xG calibration](modelling/reports/figures/xg_calibration.png)

### FPL add-ons

- **Fixture difficulty**: a 1–5 rating for each team's next six fixtures, from the model's win probability (1 = 65% or more, 2 = 50%, 3 = 35%, 4 = 20%, 5 = below 20%).
- **FPL planner**: starting from the squad you own, a mixed-integer linear program picks the transfers (free ones first, then up to two paid ones if they pay for themselves), the starting XI, the captain and the bench for the next gameweek, and checks each chip against a fixed limit. With no team ID set it picks a fresh 15 (the Wildcard plan). Budget, squad, transfer and chip rules are read from the FPL API. The projection counts appearances, goals, assists, clean sheets and goals conceded only. See [FPL suggestion](#fpl-suggestion-updated-daily).

![Fixture difficulty](modelling/reports/figures/fdr_next6.png)

### More detail

- [Match model card](modelling/MODEL_CARD_match.md)
- [xG model card](modelling/MODEL_CARD_xg.md)
- [Results report](modelling/reports/P2_REPORT.md)

### How to run P2

```
python -m data.ingest.statsbomb        # once: 2015/16 shots (slow)
python -m data.pipeline                # weekly: results, odds, fixtures
python -m modelling.xg.train           # xG models
python -m modelling.xg.evaluate        # calibration and SHAP figures
python -m modelling.match.ratings      # current team ratings
python -m modelling.match.backtest     # walk-forward backtest (up to an hour)
python -m modelling.match.benchmark    # compare with closing odds
python -m modelling.match.simulate     # season simulation
python -m modelling.match.predict      # publish next week's predictions; commit them before kick-off
python -m modelling.match.ledger       # score published predictions
python -m modelling.match.fdr          # fixture difficulty
python -m data.ingest.fpl              # FPL players, prices, rules and your team
python -m modelling.match.fpl_squad    # FPL planner: transfers, line-up, chip advice
python -m modelling.match.fpl_pitch    # draw the suggested team (SVG and HTML)
```

## FPL suggestion (updated daily)

![Suggested FPL team](modelling/reports/fpl_pitch.svg)

Made every morning by a GitHub workflow: the Dixon–Coles match model projects each player's
points, and a mixed-integer program picks the transfers, the starting XI and the captain.
Chip advice uses fixed limits. A model suggestion, not advice.
Data: Fantasy Premier League API and football-data.co.uk.

Set `team_id` in `config.toml` to plan transfers for your own team; the three plans and the chip checks are in [`fpl_plan.json`](modelling/reports/fpl_plan.json).

## P3 — Match report entity extraction

A named entity recognition system that finds players, clubs and stadiums in Premier League writing, compares a rule-based baseline, a small pretrained model and a fine-tuned transformer, and links every name it finds to an ID in the warehouse. The code is in `nlp/`.

The plan named Guardian match reports; their terms do not appear to allow storing the text or training on it, so the corpus is English Wikipedia club season articles (2015/16 onwards), which are licensed for reuse.

### Three systems on the same test set

400 paragraphs were labelled by hand in Label Studio and split by article. Scores are exact-match precision, recall and F1 on 75 test paragraphs (547 entities):

| System | Precision | Recall | F1 | Macro F1 |
|---|---|---|---|---|
| Gazetteer (lookup of known names) | 0.817 | 0.671 | 0.737 | 0.726 |
| spaCy `en_core_web_sm` (pretrained) | 0.465 | 0.437 | 0.451 | 0.380 |
| Fine-tuned DistilRoBERTa | 0.860 | 0.947 | 0.902 | 0.850 |

The transformer's F1 by label is 0.911 for players, 0.909 for clubs and 0.731 for stadiums. Of its 113 errors, 72 trace back to the test labels (mostly repeated names left unlabelled), so its precision is understated; the model's own most common error is tagging a manager as a player.

### Linking and the full corpus

Each name is linked to an Understat player ID, a `dim_team` club or a venue, or to NIL when it is not in the knowledge base. On 200 hand-checked test mentions the linker is right 92.5% of the time, and every link it made was correct; its errors are all names it left as NIL.

Run over all 3,030 paragraphs, the model found 28,344 mentions and linked 81% of them. They are stored in `nlp.mention` for P4 to query.

### More detail

- [NER model card](nlp/MODEL_CARD_ner.md)
- [Results report](nlp/reports/P3_REPORT.md)
- [Annotation guidelines](nlp/ANNOTATION_GUIDELINES.md)
- [Text findings](nlp/notebooks/FINDINGS.md)
- [Wikipedia sources](nlp/data/SOURCES.csv) and [dataset licence](nlp/data/labelled/LICENSE.md)

### How to run P3

```
python -m data.ingest.wikipedia        # once: club season articles (about 5 minutes)
python -m nlp.corpus                   # articles -> nlp.document paragraphs
python -m nlp.kb                       # knowledge base: players, clubs, venues
python -m nlp.annotate.make_tasks      # sample 400 paragraphs for Label Studio
python -m nlp.dataset                  # Label Studio export -> train/dev/test files
python -m nlp.annotate.audit           # optional: list labels worth a second look
python -m nlp.models.baselines         # gazetteer and spaCy scores
python -m nlp.models.transformer       # fine-tune and score (about 30 minutes on a CPU)
python -m nlp.linking.review --make    # linking review sheet; fill gold_id by hand
python -m nlp.linking.review --score   # linking accuracy
python -m nlp.run                      # tag and link the whole corpus -> nlp.mention
python -m nlp.attribution              # list every Wikipedia revision used
```

Label Studio runs in its own environment (`.venv-ls`); the labelled export is `nlp/data/labelled/export_main.json`.

## P4 — Tactical analyst assistant

A question-answering assistant for Premier League football that turns a plain-English question into SQL over the warehouse, calls P2's match model for ratings and predictions, searches P3's article paragraphs for written context, and reports a measured accuracy for each of those parts, all on a free local language model. The code is in `assistant/`.

![A warehouse answer with its SQL](docs/assistant_sql.png)
![An article answer with its sources](docs/assistant_articles.png)

The model (`llama3.2:3b` on Ollama, CPU only) chooses one tool and words the answer. It never computes a number: PostgreSQL counts, P2's code computes probabilities, and the search code finds the paragraphs. Model-written SQL runs only as the read-only role `pl_reader`, in a read-only transaction, with a 5-second and 200-row limit.

### Four measurements

| Part | Measure | Result |
|---|---|---|
| Text-to-SQL | Execution accuracy, 40 test questions with gold SQL | 16/40 = 0.400 (baseline prompt: 3/40) |
| Article search | Recall@5 and MRR, 40 questions, hybrid search | 30/40 = 0.750, MRR 0.575 (BM25 alone: 27/40, 0.478) |
| Tool choice | Tool-selection accuracy, 30 labelled questions | 25/30 = 0.833 |
| The whole assistant | Answers correct and grounded, 15 marked by hand | 10/15 correct; 3/4 article answers grounded |

A small model is often wrong, and the numbers are reported as they are. A later prompt scored 21/40 on text-to-SQL, but it was tuned after reading the test failures, so 16/40 is the clean number. Hard questions (window functions, self-joins) are 1/10 in every version. A question takes about 35 seconds.

### Changes to the plan

- The articles are the Wikipedia paragraphs in `nlp.document` from P3, not Guardian articles, for the same licensing reason as in P3.
- A Streamlit chat page was added, so a question can be typed in a browser.
- Everything runs on Ollama on one PC, so the project is free to build and run; no hosted model was used.

### More detail

- [System card](assistant/SYSTEM_CARD.md)
- [Results report](assistant/reports/P4_REPORT.md)
- [Findings and error analysis](assistant/reports/FINDINGS.md)

### How to run P4

```
ollama pull llama3.2:3b                # once: chat model (2.0 GB)
ollama pull all-minilm                 # once: embedding model (46 MB)
python -m assistant.index              # build the search index over nlp.document
streamlit run assistant/app.py         # the chat page
python -m assistant.eval.sql_eval --mode repair   # text-to-SQL accuracy
python -m assistant.eval.rag_eval      # search recall@5 and MRR
python -m assistant.eval.route_eval    # tool-selection accuracy
python -m assistant.eval.e2e --run     # end-to-end answers; mark them, then --score
```

It needs the `pl_reader` login in `.env` (see `.env.example`). Article text from Wikipedia, CC BY-SA 4.0.

## Data credits

- Results, match stats and odds: [football-data.co.uk](https://www.football-data.co.uk/). Match stats (shots, corners, fouls, cards) start in **2000/01**; earlier seasons have results only (checked with `data/checks.py`). Closing odds start in **2019/20**.
- Expected goals and fixtures: [Understat](https://understat.com/) via [soccerdata](https://github.com/probberechts/soccerdata). xG starts in **2014/15**.
- Shot events for the xG model: data provided by [StatsBomb](https://github.com/statsbomb/open-data) (Open Data, 2015/16). Only 34 Bundesliga and 377 Ligue 1 matches were available, so the model has 1,551 matches, not 1,826.
- Players, prices and game rules for the squad optimiser: the [Fantasy Premier League](https://fantasy.premierleague.com/) API.
- Text for the entity model and the assistant's article search: [English Wikipedia](https://en.wikipedia.org/) club season articles, by Wikipedia contributors, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Every revision used is listed in [`nlp/data/SOURCES.csv`](nlp/data/SOURCES.csv); the labelled dataset carries the same licence.

Data problems found and how they are handled are in [`analytics/notebooks/FINDINGS.md`](analytics/notebooks/FINDINGS.md).

## Limits

- P1: no possession, passing or player-level data.
- The match model uses goals only: no injuries, line-ups, transfers or manager changes. It does not beat closing odds.
- A team with only a few matches in the three-year window gets an unregularised rating. After five matches Coventry's relegation probability is 100%, which is too confident.
- The xG model has one season (2015/16) and no player-position features.
- The FPL projection ignores saves, bonus points, cards, penalties and defensive contributions, so goalkeepers and defensive players are underrated. Its shares come from the season so far and are noisy early on.
- The FPL planner does not see transfers made in the app this week until the deadline passes. Selling prices and the free-transfer number are estimates, the plan is chosen once for the five gameweeks, and the chip limits are judgement, not measured.
- The match model ignores injuries; the FPL suggestion reflects them only as of its last download.
- The assistant runs a small local model that is often wrong and slow, remembers nothing between questions, and is measured on small test sets; see its [system card](assistant/SYSTEM_CARD.md).
