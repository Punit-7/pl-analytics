# System description — live match model

## Purpose

A public API that gives home/draw/away probabilities, expected goals and the most likely
score for any Premier League fixture, a simulated final table, and an xG value for a shot.
It is for anyone who wants a transparent, scored football forecast.

It is not betting advice, and it is not a source of record for results or fixtures.

## Architecture

The weekly job produces files. The API serves those files. Git connects them: the job
commits the new files and Render rebuilds the API from the repository.

```
 GitHub Actions, Thursday 06:17 UTC
        │
        v
 1. Download the results CSV files            data/ingest/football_data.py (P1)
 2. Load and check them             ── broken data ──> the job stops, GitHub emails you
        │
        v
 3. Gate: champion config vs challenger config,
    walk-forward on the last 300 matches     ── better ──> challenger becomes champion
        │
        v
 4. Refit Dixon-Coles with the champion config (P2 code)  ──> serving/models/match_model.json
 5. Predict every remaining match; simulate the season
                                              ──> serving/predictions/<season>/<time>_*.csv
                                              ──> serving/predictions/latest.json
 6. Score earlier predictions                 ──> serving/state/ledger/
        │
        v
 7. Tests on the new files; build the Docker image; call /health
        │
        v
 8. git commit + git push  ───────────────────> Render sees the push, rebuilds the image
        │                                       and replaces the running API
        v
 9. Monitoring: PSI drift, Evidently report, metrics to MLflow,
    live performance level, serving/reports/STATUS.md
                                              ──> second commit, marked [skip render]
                                              ──> red level: a GitHub issue, which emails you
```

```
 POST /predict/match {"home_team": "Arsenal", "away_team": "Chelsea"}
        │
        v
 FastAPI checks the body against the MatchRequest schema   ── invalid ──> 422
        │
        v
 ModelService finds the two teams in the fitted model      ── unknown ──> 404 with valid names
        │
        v
 P2's DixonColes.score_matrix  ──> probabilities, expected goals, most likely score
        │
        v
 JSON reply with the model version; one log line with path, status and milliseconds
```

## Models served

- **Dixon–Coles match model (P2)**, refitted every week on the last three years of results.
  The version is `dc-<fit date>-<configuration ID>`, for example `dc-2026-10-09-78dd3802`.
- **Logistic xG model (P2)**, trained once on StatsBomb Open Data 2015/16. It is fixed.

## Data

Results and market-average closing odds from football-data.co.uk: the current season and
the five before it. No personal data.

## Champion/challenger policy

Champion and challenger are configurations; the model is refitted weekly whichever wins.
The challenger replaces the champion only if all of these hold, in a walk-forward test on
the most recent 300 matches:

- at least 150 test matches;
- its mean RPS is lower by at least 0.001;
- the 95% bootstrap interval of the gain is above 0.

Current champion: `78dd3802` (time decay 0.0019, 1,095 days of history, ρ fitted), since
2026-10-09. The challenger with time decay 0.0030 was refused on 2026-10-09: gain 0.0002,
95% interval −0.0010 to 0.0015. Every decision is in `state/registry/history.csv`.

## Monitoring

| Question | Number | Where |
| --- | --- | --- |
| Did the data arrive, and is it recent? | Matches loaded; days since the newest result; stale flag | Status page, MLflow |
| Is the data still shaped as before? | PSI of home goals, away goals, total goals and the result; alert above 0.25 | Status page, MLflow, Evidently report |
| How good are the published predictions? | Live RPS, pick accuracy | Status page, MLflow |
| Are the probabilities honest? | Calibration error | Status page, MLflow |
| How far from the bookmakers? | RPS gap on matches with odds | Status page, MLflow |
| What did the gate decide? | Champion and challenger RPS, the gain, the reason | Status page, MLflow, `history.csv` |

**Alert level**, judged each week on the last 100 scored matches (waiting below 60):

| Check | Green | Amber | Red |
| --- | --- | --- | --- |
| Against the freq baseline | Model RPS lower on average | Model RPS not lower | Model RPS higher, and the 95% interval of the difference is above 0 |
| Against the bookmaker (at least 60 matches with odds) | Gap 0.02 or less | Gap above 0.02 | — |
| Calibration | Error 0.08 or less | Error above 0.08 | — |
| Escalation | — | — | Amber or worse on three runs in a row, each with new matches |

The level is the worst of the checks. Red opens a GitHub issue labelled `model-alert`;
the response plan is in the [runbook](RUNBOOK.md#performance-response-plan).

## Measured so far

As of 2026-10-09:

| Measure | Value |
| --- | --- |
| Matches scored live | 0 (first predictions published 2026-10-09) |
| Live RPS, gap to the bookmaker, calibration error | Not yet: no published prediction has a result |
| Performance level | Waiting |
| Gate test, champion RPS | 0.2088 on 300 matches |
| Docker image size | 691 MB (`docker images pl-api`) |
| Container memory | 144 MiB at rest, against Render's 512 MB |

Current numbers are on the [status page](reports/STATUS.md).

## Limitations

- Predictions have no kick-off dates: they cover every home/away pairing without a result.
- A midweek round is predicted from the previous Thursday.
- Promoted teams start from a prior until they have results.
- Before the first matchweek is complete, nothing is published.
- The free host sleeps after 15 minutes; the first request then takes about a minute.
- The ledger is small early in a season, so live scores are noisy.
- PSI on 150 matches is noisy.
- The weekly history lives in a SQLite file in Git, not on a tracking server.
- The web layer was built from documentation and first tested on this deployment.

## What I would add with a budget

- A hosted database and an MLflow tracking server.
- An always-on instance.
- Alerts to a phone.
