import numpy as np
import pandas as pd
import pytest

from data.common.io_utils import DataValidationError
from data.ingest.fpl import validate
from modelling.match.fpl_squad import (
    available_chips,
    chip_advice,
    estimate_free_transfers,
    lineup,
    optimise,
    project,
    selling_price,
)

RULES = {
    "squad": 15,
    "starters": 11,
    "club_limit": 3,
    "budget": 100.0,
    "positions": {"GKP": (2, 1, 1), "DEF": (5, 3, 5), "MID": (5, 2, 5), "FWD": (3, 1, 3)},
}
SCORING = {
    "long_play": 2,
    "assists": 3,
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
}


def fake_players(seed: int = 0) -> pd.DataFrame:
    """8 clubs, each with 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards."""
    rng = np.random.default_rng(seed)
    rows = [
        {"team": f"C{club}", "position": position}
        for club in range(8)
        for position, (count, _, _) in RULES["positions"].items()
        for _ in range(count)
    ]
    df = pd.DataFrame(rows)
    df["price"] = rng.uniform(4.0, 13.0, len(df)).round(1)
    df["points"] = (df["price"] * 2 + rng.normal(0, 3, len(df))).clip(lower=0)  # dearer = better
    return df


# ---------- the squad (unchanged from the first version) ----------


def test_squad_obeys_every_rule():
    squad = optimise(fake_players(), RULES)
    starters = squad[squad["start"]]
    assert len(squad) == 15 and len(starters) == 11
    assert squad["price"].sum() <= RULES["budget"] + 1e-9
    assert squad["team"].value_counts().max() <= RULES["club_limit"]
    for position, (select, min_play, max_play) in RULES["positions"].items():
        assert (squad["position"] == position).sum() == select
        assert min_play <= (starters["position"] == position).sum() <= max_play
    # one captain: the starter with the most projected points
    assert squad["captain"].sum() == 1
    assert squad.loc[squad["captain"], "points"].iloc[0] == starters["points"].max()


def test_squad_is_the_best_one_when_nothing_binds():
    players = fake_players()
    easy = {**RULES, "budget": 1000.0, "club_limit": 15}
    starters = optimise(players, easy).query("start")
    # with no budget or club limit, the XI holds the best goalkeeper and the best forward
    for position in ("GKP", "FWD"):
        best = players.loc[players["position"] == position, "points"].max()
        assert best in starters["points"].to_numpy()


def test_projection_matches_hand_calculation():
    players = pd.DataFrame(
        {
            "team": ["A", "A"],
            "position": ["MID", "DEF"],
            "play": [0.8, 1.0],
            "chance": [1.0, 1.0],
            "xg_share": [0.25, 0.0],
            "xa_share": [0.20, 0.0],
        }
    )
    outlook = pd.DataFrame(
        {"games": [5], "goals_for": [8.0], "clean_sheets": [1.5], "conceded_pairs": [2.0]},
        index=["A"],
    )
    points = project(players, outlook, SCORING, assists_per_goal=0.7)
    assert points[0] == pytest.approx(8.0 + 10.0 + 3.36 + 1.2)  # midfielder: 22.56
    assert points[1] == pytest.approx(10.0 + 6.0 - 2.0)  # defender: 14.0


def test_validate_rejects_wrong_json():
    with pytest.raises(DataValidationError, match="missing keys"):
        validate(b'{"elements": []}', "bootstrap-static")
    with pytest.raises(DataValidationError, match="not readable"):
        validate(b"<html>Not found</html>", "fixtures")
    assert validate(b"[]", "entry_transfers") == 0  # no transfers yet is valid


# ---------- transfers ----------


def owned_squad(players: pd.DataFrame) -> np.ndarray:
    """A legal squad that is clearly not the best one: the cheapest players."""
    weak = {**RULES, "budget": 1000.0}
    cheap = players.assign(points=-players["price"])
    return players.index.isin(optimise(cheap, weak).index)


def test_one_free_transfer_allows_one_new_player():
    players = fake_players()
    owned = owned_squad(players)
    squad = optimise(players, {**RULES, "budget": 1000.0}, owned=owned, free_transfers=1)
    assert (~squad.index.isin(players.index[owned])).sum() == 1
    assert squad.attrs["hits"] == 0


def test_hits_are_taken_only_when_they_pay():
    players = fake_players()
    owned = owned_squad(players)
    rules = {**RULES, "budget": 1000.0}
    # Each extra transfer costs 4 points. The owned players are far worse, so hits pay.
    squad = optimise(players, rules, owned=owned, free_transfers=1, hit_cost=4, max_hits=2)
    assert squad.attrs["hits"] == 2
    assert (~squad.index.isin(players.index[owned])).sum() == 3
    # At 1,000 points per hit, no hit pays.
    squad = optimise(players, rules, owned=owned, free_transfers=1, hit_cost=1000, max_hits=2)
    assert squad.attrs["hits"] == 0


def test_min_gain_keeps_a_transfer_in_the_bank():
    players = fake_players()
    best = optimise(players, {**RULES, "budget": 1000.0})
    owned = players.index.isin(best.index)  # you already own the best squad
    squad = optimise(
        players, {**RULES, "budget": 1000.0}, owned=owned, free_transfers=1, min_gain=1.0
    )
    assert squad.index.isin(players.index[owned]).all()


def test_selling_price_keeps_half_of_a_rise_rounded_down():
    assert selling_price(now=7.8, bought=7.5, fee=0.5) == 7.6  # rise 0.3: keep 0.1
    assert selling_price(now=7.9, bought=7.5, fee=0.5) == 7.7  # rise 0.4: keep 0.2
    assert selling_price(now=7.2, bought=7.5, fee=0.5) == 7.2  # a fall is passed on in full


def test_free_transfers_estimate():
    week = lambda e, n: {"event": e, "event_transfers": n}  # noqa: E731
    history = {"current": [week(1, 15), week(2, 0), week(3, 0), week(4, 1)], "chips": []}
    # after GW1: 1; GW2 none used: 2; GW3 none: 3; GW4 one used: 3
    assert estimate_free_transfers(history, max_free=5) == 3
    history["current"].append(week(5, 4))  # four used with three free: one hit, back to 1
    assert estimate_free_transfers(history, max_free=5) == 1
    history["current"].append(week(6, 12))  # a Wildcard week does not use them
    history["chips"].append({"name": "wildcard", "event": 6})
    assert estimate_free_transfers(history, max_free=5) == 2


# ---------- line-up and chips ----------


def test_lineup_has_a_vice_captain_and_keeper_first_on_the_bench():
    squad = optimise(fake_players(), RULES)
    out = lineup(squad, RULES, "points")
    starters = out[out["start"]].sort_values("points", ascending=False)
    assert out["captain"].sum() == 1 and out["vice"].sum() == 1
    assert out.loc[out["vice"], "points"].iloc[0] == starters["points"].iloc[1]
    bench = out[~out["start"]].sort_values("bench_order")
    assert bench["bench_order"].tolist() == [1, 2, 3, 4]
    assert bench["position"].iloc[0] == "GKP"


def test_chips_available_by_window_and_use():
    boot = {
        "chips": [
            {"name": "wildcard", "start_event": 2, "stop_event": 19},
            {"name": "wildcard", "start_event": 20, "stop_event": 38},
            {"name": "3xc", "start_event": 1, "stop_event": 19},
            {"name": "bboost", "start_event": 1, "stop_event": 19},
        ]
    }
    history = {"chips": [{"name": "3xc", "event": 3}]}
    assert available_chips(boot, history, 6) == ["wildcard", "bboost"]
    assert available_chips(boot, history, 22) == ["wildcard"]  # the second-half Wildcard


def test_chip_advice_picks_the_chip_that_passes():
    plans = {
        "transfers": (
            None,
            {
                "horizon_value": 300.0,
                "next_gw_points": 60.0,
                "bench_points": 8.0,
                "captain_points": 12.0,
            },
        ),
        "wildcard": (None, {"horizon_value": 310.0}),
        "freehit": (None, {"next_gw_points": 63.0}),
    }
    cfg = {
        "wildcard_min_gain": 15,
        "freehit_min_gain": 10,
        "bench_boost_min": 15,
        "triple_captain_min": 10,
    }
    advice = chip_advice(plans, ["wildcard", "freehit", "bboost", "3xc"], cfg)
    assert advice["play"] == "3xc"  # 12 >= 10; the others are below their limits
    assert chip_advice(plans, ["wildcard", "bboost"], cfg)["play"] is None
