# Weekly status

Last run: 2026-11-26T17:03:29+00:00

| Item | Value |
| --- | --- |
| Live model | `dc-2026-11-26-78dd3802` |
| Trained on results up to | 2026-11-23 |
| Data | 2010 matches, latest 3 days old |
| Predictions published | 270 |
| Gate | gain 0.0000 is below the required 0.001 |
| Matches scored so far | 60 |
| Model RPS | 0.2239 |
| Bookmaker RPS | 0.2099 |
| Gap to bookmaker (lower is better) | 0.014 |
| Headline picks correct | 0.5167 |
| Calibration error | 0.0588 |

## Live performance

Level: **GREEN**

Judged on the last 60 scored matches.

| Check | Level | Detail |
| --- | --- | --- |
| vs_baseline | green | the model beats the freq baseline |
| vs_bookmaker | green | gap 0.0141; the limit is 0.02 |
| calibration | green | error 0.0588; the limit is 0.08 |

## Drift

| Column | PSI | Alert |
| --- | --- | --- |
| home_goals | 0.0067 | ok |
| away_goals | 0.0328 | ok |
| total_goals | 0.0204 | ok |
| result | 0.0235 | ok |

PSI compares the most recent matches with the matches before them.
The full Evidently report is attached to the workflow run as an artifact.
