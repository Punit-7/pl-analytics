# Runbook — live match model

How to operate the weekly job and the API, and what to do when something goes wrong.
The system is described in [SYSTEM.md](SYSTEM.md); the latest run is in [reports/STATUS.md](reports/STATUS.md).

## The weekly run failed

1. Open the run on the repository's **Actions** tab ("Weekly model run") and find the red step.
2. The table says what is and is not published at that step.
3. Fix the cause, then press **Run workflow**.

| Step | If it fails |
| --- | --- |
| Retrain, gate, publish, score | Nothing is committed. The live API keeps last week's model |
| Test the new model files and the API | Nothing is committed |
| Build the image and call it | Nothing is committed |
| Commit the model and predictions | The run is red; rerun it |
| Performance alerts | The predictions are already published; only the status page is old |
| Open or update the alert issue | As above. Read `performance.json` in the run log instead |
| Monitoring - drift, MLflow, status page | As above |
| Commit the monitoring files | As above |
| Keep the Evidently report | Ignored |

What is live changes only after every check before the first commit has passed.

## The data check failed

The error message names the check (missing values, impossible goals, a fixture twice, a
future date, a finished season without 380 matches). Open that season's file on
football-data.co.uk in a browser. If the source is wrong, wait and rerun. Do not weaken the check.

## The data is stale

Normal in an international break. If the newest result is more than 21 days old during the
season (not June or July), check whether the source site is still updating.

## A drift alert

Download `drift-report` from the run page and read the Evidently report. Check the sample
size first: PSI on 150 matches is noisy. Write what you found in `serving/reports/FINDINGS.md`.
Do not change the model because of one alert.

## Roll back a bad model

```
git pull
git revert <the "Weekly model run" commit>
git push
```

Render deploys the previous files.

## Propose a challenger

Edit `[serving.challenger]` in `config.toml` and push. The next run tests it against the
champion. The decision appears in `serving/state/registry/history.csv` and on the status page.
Never promote by hand.

## The API is slow to answer

A cold start on Render's free plan takes about a minute after 15 minutes without traffic.
Later requests are fast.

## Upgrade a package

Install it, run `pytest -q`, run `python -m serving.pin`, and commit both requirements files.

## View the weekly history in MLflow

Opening the committed file can modify it, so view a copy:

```
git pull
copy serving\state\mlflow.db mlflow_view.db
mlflow ui --backend-store-uri sqlite:///mlflow_view.db
```

## Rules

- Only the workflow writes `serving/models/`, `serving/predictions/`, `serving/state/` and
  `serving/reports/`. After a local run, discard those changes before committing:

  ```
  git restore serving/models serving/predictions serving/state serving/reports
  git clean -fd serving/predictions serving/reports
  git pull
  ```

- No secrets in the repository. The job reads public files and needs no password.

## Performance response plan

Amber: write one line in serving/reports/FINDINGS.md with the date, the check and its value.
Change nothing. Look again next week.

Red: work through the steps in order. Do not skip to step 3.
Write what you did and what happened as comments on the alert issue.

Step 1 - Check the data and the code (same day)
  - STATUS.md: is the data stale? Did the data check pass?
  - ledger.csv: are the latest matches there, with the right teams and scores?
    A renamed team in the source file can make matches disappear or mismatch.
  - Promoted teams: are the alerts caused mostly by their matches?
  - git log: did a code or config change go in just before the problem started?
  - If you find a bug: fix it, add a test that would have caught it,
    run the workflow, and close the issue.

Step 2 - Retune (1 to 3 weeks)
  - Only if step 1 found nothing.
  - Put one changed setting in [serving.challenger], for example a faster
    or a slower time decay (dc_xi) than the champion's. Push.
  - The gate decides on the next Thursday. Never promote by hand.
  - If the gate refuses two different challengers, go to step 3.

Step 3 - Redesign (a project of its own)
  - Write the idea in FINDINGS.md first: what you will change and why.
  - Build it in P2 and test it with P2's walk-forward backtest.
  - Possible changes: a better starting rating for promoted teams;
    shots on target from the football-data files as an extra input;
    team strengths that change within the season.
  - It reaches the live system only as a challenger that passes the gate.

Never:
  - change the model because of one matchweek;
  - raise a limit in [serving.alerts] to make an alert go away;
  - judge a new version on the same matches it was tuned on.
