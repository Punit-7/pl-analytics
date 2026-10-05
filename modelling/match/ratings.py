"""Fit Dixon-Coles as of today and print this season's team ratings."""

from datetime import date

import numpy as np
import pandas as pd

from data.common.config import load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start, season_label
from modelling.match.data import load_fixtures, load_results
from modelling.match.dixon_coles import fit_dixon_coles


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    m = s.modelling
    engine = make_engine()
    season = season_label(current_season_start(start_month=s.season_start_month))
    model = fit_dixon_coles(
        load_results(engine), date.today(), m["dc_xi"], m["dc_history_days"], m["dc_max_goals"]
    )
    fixtures = load_fixtures(engine, season)
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    rows = []
    for t in teams:
        a, d = model.rating(t)
        rows.append(
            {"team": t, "attack": a, "defence": d, "strength": a - d, "in_fit": t in model.teams}
        )
    table = pd.DataFrame(rows).sort_values("strength", ascending=False)
    print(table.round(3).to_string(index=False))
    print(
        f"home advantage {model.home_adv:.3f} (x{np.exp(model.home_adv):.2f}), rho {model.rho:.3f}"
    )


if __name__ == "__main__":
    main()
