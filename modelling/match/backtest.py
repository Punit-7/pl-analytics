"""Walk-forward backtest: every week, refit on past matches only and predict that week."""
import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start
from modelling.match.data import load_results
from modelling.match.dixon_coles import fit_dixon_coles
from modelling.metrics import multiclass_log_loss, rps

log = logging.getLogger("modelling.match.backtest")
REP = ROOT / "modelling" / "reports"
MODELS = ["dc", "poisson", "freq", "eqpois"]


def outcome_index(home_goals, away_goals):
    """0 = home win, 1 = draw, 2 = away win."""
    return np.select([home_goals > away_goals, home_goals == away_goals], [0, 1], default=2)


def freq_probs(train: pd.DataFrame) -> np.ndarray:
    o = outcome_index(train["home_goals"].to_numpy(), train["away_goals"].to_numpy())
    return np.bincount(o, minlength=3) / len(o)


def eqpois_probs(train: pd.DataFrame, max_goals: int) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, train["home_goals"].mean()),
                 poisson.pmf(goals, train["away_goals"].mean()))
    m /= m.sum()
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--first-season", type=int, help="start year, e.g. 2019")
    parser.add_argument("--last-season", type=int, help="start year, e.g. 2025")
    args = parser.parse_args(argv)
    s = load_settings()
    setup_logging(s.logs)
    m = s.modelling
    first = args.first_season or m["backtest_first_season"]
    last = args.last_season or current_season_start(start_month=s.season_start_month) - 1

    results = load_results(make_engine())
    start_year = results["season"].str[:4].astype(int)
    test = results[(start_year >= first) & (start_year <= last)].copy()
    test["week"] = test["match_date"] - pd.to_timedelta(test["match_date"].dt.weekday, unit="D")

    rows = []
    for week, games in test.groupby("week"):
        window = pd.Timedelta(days=m["dc_history_days"])
        train = results[(results["match_date"] < week) & (results["match_date"] >= week - window)]
        dc = fit_dixon_coles(results, week, m["dc_xi"], m["dc_history_days"], m["dc_max_goals"])
        pois = fit_dixon_coles(results, week, m["dc_xi"], m["dc_history_days"],
                               m["dc_max_goals"], fit_rho=False)
        freq, eq = freq_probs(train), eqpois_probs(train, m["dc_max_goals"])
        for g in games.itertuples():
            row = {"match_id": g.match_id, "season": g.season, "match_date": g.match_date.date(),
                   "train_max_date": train["match_date"].max().date(),
                   "outcome": int(outcome_index(g.home_goals, g.away_goals))}
            for name, probs in (("dc", dc.outcome_probs(g.home_team, g.away_team)),
                                ("poisson", pois.outcome_probs(g.home_team, g.away_team)),
                                ("freq", freq), ("eqpois", eq)):
                row[f"{name}_h"], row[f"{name}_d"], row[f"{name}_a"] = probs
            rows.append(row)
        log.info("week of %s: %d matches", week.date(), len(games))

    out = pd.DataFrame(rows)
    summary = {}
    for name in MODELS:
        p = out[[f"{name}_h", f"{name}_d", f"{name}_a"]].to_numpy()
        out[f"{name}_rps"] = rps(p, out["outcome"].to_numpy())
        summary[name] = {"rps": float(out[f"{name}_rps"].mean()),
                         "log_loss": multiclass_log_loss(p, out["outcome"].to_numpy())}
    by_season = out.groupby("season")[[f"{n}_rps" for n in MODELS]].mean().round(4)
    out.to_csv(REP / "backtest_predictions.csv", index=False)
    (REP / "backtest_summary.json").write_text(json.dumps(
        {"overall": summary, "rps_by_season": by_season.to_dict(orient="index")}, indent=2))
    log.info("%d matches backtested\n%s", len(out), by_season.to_string())
    for name, r in summary.items():
        log.info("%-8s RPS %.4f  log loss %.4f", name, r["rps"], r["log_loss"])


if __name__ == "__main__":
    main()