"""FPL add-on: fixture difficulty (1 easiest - 5 hardest) for each team's next fixtures."""

import logging
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from data.common.config import ROOT, load_settings  # noqa: E402
from data.common.db import make_engine  # noqa: E402
from data.common.logging_setup import setup_logging  # noqa: E402
from data.common.seasons import current_season_start, season_label  # noqa: E402
from modelling.match.data import load_fixtures, load_results  # noqa: E402
from modelling.match.dixon_coles import fit_dixon_coles  # noqa: E402
from modelling.match.simulate import remaining_fixtures  # noqa: E402

log = logging.getLogger("modelling.match.fdr")
N_NEXT = 6
BANDS = [(0.65, 1), (0.50, 2), (0.35, 3), (0.20, 4)]  # win probability >= threshold -> rating
COLOURS = ["#00FF85", "#04F5FF", "#8C7FA0", "#963CFF", "#E90052"]  # design system, 1..5


def difficulty(p_win: float) -> int:
    for threshold, rating in BANDS:
        if p_win >= threshold:
            return rating
    return 5


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    m = s.modelling
    engine = make_engine()
    season = season_label(current_season_start(start_month=s.season_start_month))
    results = load_results(engine)
    model = fit_dixon_coles(
        results, date.today(), m["dc_xi"], m["dc_history_days"], m["dc_max_goals"]
    )
    remaining = remaining_fixtures(load_fixtures(engine, season), results, season)
    remaining = remaining.sort_values("match_date")

    rows = []
    teams = sorted(set(remaining["home_team"]) | set(remaining["away_team"]))
    for team in teams:
        games = remaining[(remaining["home_team"] == team) | (remaining["away_team"] == team)]
        for n, g in enumerate(games.head(N_NEXT).itertuples(), start=1):
            at_home = g.home_team == team
            p_home, _, p_away = model.outcome_probs(g.home_team, g.away_team)
            p_win = p_home if at_home else p_away
            opponent = g.away_team if at_home else g.home_team
            rows.append(
                {
                    "team": team,
                    "n": n,
                    "match_date": g.match_date.date(),
                    "opponent": f"{opponent} ({'H' if at_home else 'A'})",
                    "p_win": round(p_win, 3),
                    "fdr": difficulty(p_win),
                }
            )
    df = pd.DataFrame(rows)
    grid = df.pivot(index="team", columns="n", values="fdr")
    labels = df.pivot(index="team", columns="n", values="opponent")
    order = grid.mean(axis=1).sort_values().index  # easiest run of fixtures first
    grid, labels = grid.loc[order], labels.loc[order]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(grid.to_numpy() - 1, cmap=ListedColormap(COLOURS), vmin=0, vmax=4, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            text_colour = "#FFFFFF" if grid.iat[i, j] == 5 else "#12001A"
            ax.text(j, i, labels.iat[i, j], ha="center", va="center", fontsize=7, color=text_colour)
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(range(grid.shape[1]), [f"Next {j}" for j in grid.columns])
    ax.set_title("Fixture difficulty from the Dixon-Coles model (1 easiest, 5 hardest)")
    out = ROOT / "modelling" / "reports"
    fig.savefig(out / "figures" / "fdr_next6.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out / "fdr_next6.csv", index=False)
    log.info("Easiest runs: %s", ", ".join(order[:3]))


if __name__ == "__main__":
    main()
