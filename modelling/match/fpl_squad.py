"""FPL add-on (optional): the 15-player squad with the highest projected points."""

import json
import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from data.build_warehouse import name_lookup
from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from modelling.match.data import load_results
from modelling.match.dixon_coles import DixonColes, fit_dixon_coles

log = logging.getLogger("modelling.match.fpl_squad")
N_GAMEWEEKS = 5  # projection horizon
BENCH_WEIGHT = 0.1  # a substitute's points count this much, so the bench is not filled at random


def load_rules(boot: dict) -> dict:
    """Squad rules as the game itself publishes them, so a rule change needs no code change."""
    gs = boot["game_settings"]
    return {
        "squad": gs["squad_squadsize"],
        "starters": gs["squad_squadplay"],
        "club_limit": gs["squad_team_limit"],
        "budget": gs["squad_total_spend"] / 10,  # the API stores prices in tenths of a million
        # position -> (in the squad, fewest starting, most starting)
        "positions": {
            t["singular_name_short"]: (t["squad_select"], t["squad_min_play"], t["squad_max_play"])
            for t in boot["element_types"]
            if t["squad_select"]
        },
    }


def load_players(boot: dict, fixtures: list, lookup: dict) -> pd.DataFrame:
    """One row per selectable player, with season-to-date playing time and chance shares."""
    team_name = {t["id"]: lookup.get(t["name"], t["name"]) for t in boot["teams"]}
    position = {t["id"]: t["singular_name_short"] for t in boot["element_types"]}
    played = pd.Series(
        [f[side] for f in fixtures if f["finished"] for side in ("team_h", "team_a")]
    ).value_counts()
    p = pd.DataFrame(boot["elements"])
    p = p[p["can_select"] & ~p["removed"]].copy()
    team_minutes = 90 * p["team"].map(played).fillna(0)
    chance = p["chance_of_playing_next_round"].astype(float).fillna(100) / 100  # empty = fit
    xg, xa = p["expected_goals"].astype(float), p["expected_assists"].astype(float)
    return pd.DataFrame(
        {
            "name": p["web_name"],
            "team": p["team"].map(team_name),
            "position": p["element_type"].map(position),
            "price": p["now_cost"] / 10,
            "chance": chance,
            # share of the team's minutes played so far, scaled by the injury flag
            "play": (p["minutes"] / team_minutes).fillna(0).clip(0, 1) * chance,
            # share of the team's xG and xA so far
            "xg_share": (xg / xg.groupby(p["team"]).transform("sum")).fillna(0),
            "xa_share": (xa / xa.groupby(p["team"]).transform("sum")).fillna(0),
        }
    ).reset_index(drop=True)


def team_outlook(model: DixonColes, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per team over `fixtures`: games, expected goals for, expected clean sheets and
    expected conceded pairs (the game takes a point off per two goals conceded)."""
    pairs = np.arange(model.max_goals + 1) // 2
    rows = []
    for f in fixtures.itertuples():
        m = model.score_matrix(f.home_team, f.away_team)
        home_goals, away_goals = m.sum(axis=1), m.sum(axis=0)  # each side's goal distribution
        lam, mu = model.expected_goals(f.home_team, f.away_team)
        for team, scored, conceded in (
            (f.home_team, lam, away_goals),
            (f.away_team, mu, home_goals),
        ):
            rows.append(
                {
                    "team": team,
                    "games": 1,
                    "goals_for": scored,
                    "clean_sheets": conceded[0],
                    "conceded_pairs": float((conceded * pairs).sum()),
                }
            )
    return pd.DataFrame(rows).groupby("team").sum()


def project(
    players: pd.DataFrame, outlook: pd.DataFrame, scoring: dict, assists_per_goal: float
) -> pd.Series:
    """Projected points per player over the outlook's fixtures.
    ponytail: appearance, goals, assists, clean sheets and goals conceded only. No saves,
    bonus, cards, penalties or defensive contributions; add them if the squad is ever used
    for real decisions."""
    t = outlook.reindex(players["team"]).fillna(0).reset_index(drop=True)
    pos = players["position"]
    play, chance = players["play"], players["chance"]
    appearance = play * t["games"] * scoring["long_play"]  # assumes 60+ minutes when playing
    goals = players["xg_share"] * chance * t["goals_for"] * pos.map(scoring["goals_scored"])
    assists = players["xa_share"] * chance * t["goals_for"] * assists_per_goal * scoring["assists"]
    clean_sheets = play * t["clean_sheets"] * pos.map(scoring["clean_sheets"])
    conceded = play * t["conceded_pairs"] * pos.map(scoring["goals_conceded"])
    return appearance + goals + assists + clean_sheets + conceded


def optimise(players: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Solve the squad as a MILP. Needs columns team, position, price, points.
    Variables, n of each: x (in the squad), s (starts), c (captain)."""
    n = len(players)
    pts = players["points"].to_numpy(dtype=float)
    zero, one, eye = np.zeros(n), np.ones(n), np.eye(n)
    rows, low, high = [], [], []

    def add(lo: float, hi: float, x=zero, s=zero, c=zero) -> None:
        rows.append(np.concatenate([x, s, c]))
        low.append(lo)
        high.append(hi)

    add(rules["squad"], rules["squad"], x=one)
    add(0, rules["budget"], x=players["price"].to_numpy(dtype=float))
    add(rules["starters"], rules["starters"], s=one)
    add(1, 1, c=one)
    for position, (select, min_play, max_play) in rules["positions"].items():
        in_position = (players["position"] == position).to_numpy(dtype=float)
        add(select, select, x=in_position)
        add(min_play, max_play, s=in_position)  # a valid formation
    for club in players["team"].unique():
        add(0, rules["club_limit"], x=(players["team"] == club).to_numpy(dtype=float))
    constraints = [
        LinearConstraint(np.array(rows), low, high),
        LinearConstraint(np.hstack([-eye, eye, 0 * eye]), -np.inf, 0),  # starters are in the squad
        LinearConstraint(np.hstack([0 * eye, -eye, eye]), -np.inf, 0),  # the captain starts
    ]
    # Starters score once, the captain twice, substitutes a little. milp minimises, so negate.
    objective = -np.concatenate([BENCH_WEIGHT * pts, (1 - BENCH_WEIGHT) * pts, pts])
    res = milp(objective, constraints=constraints, integrality=np.ones(3 * n), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"no valid squad found: {res.message}")
    x, s, c = np.round(res.x).reshape(3, n).astype(bool)
    return players[x].assign(start=s[x], captain=c[x])


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    m = s.modelling
    folder = s.raw / "fpl"
    if not (folder / "bootstrap-static.json").exists():
        raise SystemExit("No FPL data on disk; run python -m data.ingest.fpl first")
    boot = json.loads((folder / "bootstrap-static.json").read_text(encoding="utf-8"))
    fixtures = json.loads((folder / "fixtures.json").read_text(encoding="utf-8"))
    lookup = name_lookup(s, "fpl")

    first = next(e["id"] for e in boot["events"] if e["is_next"])
    gameweeks = range(first, first + N_GAMEWEEKS)
    team_name = {t["id"]: lookup.get(t["name"], t["name"]) for t in boot["teams"]}
    upcoming = pd.DataFrame(
        [
            {"home_team": team_name[f["team_h"]], "away_team": team_name[f["team_a"]]}
            for f in fixtures
            if f["event"] in gameweeks
        ]
    )
    model = fit_dixon_coles(
        load_results(make_engine()),
        date.today(),
        m["dc_xi"],
        m["dc_history_days"],
        m["dc_max_goals"],
    )
    unknown = (set(upcoming["home_team"]) | set(upcoming["away_team"])) - set(model.teams)
    if unknown:
        log.warning("Teams not in the model (add them to team_names.csv): %s", sorted(unknown))

    players = load_players(boot, fixtures, lookup)
    everyone = pd.DataFrame(boot["elements"])
    assists_per_goal = everyone["assists"].sum() / max(everyone["goals_scored"].sum(), 1)
    players["points"] = project(
        players, team_outlook(model, upcoming), boot["game_config"]["scoring"], assists_per_goal
    ).round(2)

    rules = load_rules(boot)
    squad = optimise(players, rules)
    squad["position"] = pd.Categorical(squad["position"], list(rules["positions"]))
    squad = squad.sort_values(["start", "position", "points"], ascending=[False, True, False])
    squad.insert(0, "gameweeks", f"{gameweeks[0]}-{gameweeks[-1]}")
    cols = ["gameweeks", "name", "team", "position", "price", "points", "start", "captain"]
    squad[cols].to_csv(ROOT / "modelling" / "reports" / "fpl_squad.csv", index=False)
    print(squad[cols[1:]].to_string(index=False))
    starters = squad[squad["start"]]
    log.info(
        "Gameweeks %s: squad costs %.1f of %.1f; starting XI projects %.1f points (captain %s)",
        squad["gameweeks"].iloc[0],
        squad["price"].sum(),
        rules["budget"],
        starters["points"].sum() + starters.loc[starters["captain"], "points"].sum(),
        starters.loc[starters["captain"], "name"].iloc[0],
    )


if __name__ == "__main__":
    main()
