"""FPL add-on (optional): transfers for your team, a fresh squad, and chip advice.

python -m modelling.match.fpl_squad                      plan for your team (or a fresh squad)
python -m modelling.match.fpl_squad --mode wildcard      a fresh 15, e.g. when your Wildcard is on
python -m modelling.match.fpl_squad --free-transfers 2   replace the free-transfer estimate
python -m modelling.match.fpl_squad --from-csv           results from the CSV files, no database
"""

import argparse
import json
import logging
import math
from dataclasses import replace
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from data.build_warehouse import load_matches, name_lookup
from data.common.config import ROOT, Settings, load_settings
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start
from modelling.match.dixon_coles import DixonColes, fit_dixon_coles

log = logging.getLogger("modelling.match.fpl_squad")
REPORTS = ROOT / "modelling" / "reports"
BENCH_WEIGHT = 0.1  # a substitute's points count this much, so the bench is not filled at random
CHIP_NAMES = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}


# ---------- Rules, players, projection ----------


def load_rules(boot: dict) -> dict:
    """Squad and transfer rules as the game itself publishes them."""
    gs = boot["game_settings"]
    return {
        "squad": gs["squad_squadsize"],
        "starters": gs["squad_squadplay"],
        "club_limit": gs["squad_team_limit"],
        "budget": gs["squad_total_spend"] / 10,  # the API stores prices in tenths of a million
        "max_free_transfers": 1 + gs["max_extra_free_transfers"],
        "sell_on_fee": gs["transfers_sell_on_fee"],  # share of a price rise you keep when selling
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
    short = {t["id"]: t["short_name"] for t in boot["teams"]}
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
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"].map(team_name),
            "club": p["team"].map(short),
            "team_id": p["team"],
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
    """Per team (and per gameweek, if `fixtures` has an `event` column): games, expected
    goals for, expected clean sheets and expected conceded pairs."""
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
                    "event": getattr(f, "event", 0),
                    "games": 1,
                    "goals_for": scored,
                    "clean_sheets": conceded[0],
                    "conceded_pairs": float((conceded * pairs).sum()),
                }
            )
    keys = ["team", "event"] if "event" in fixtures.columns else ["team"]
    out = pd.DataFrame(rows).groupby(keys).sum()
    return out.drop(columns="event", errors="ignore")


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


def project_by_gameweek(
    players: pd.DataFrame, outlook: pd.DataFrame, scoring: dict, apg: float, gameweeks
) -> pd.DataFrame:
    """One column of projected points per gameweek. A team with no fixture scores 0 that week."""
    columns = {}
    for gw in gameweeks:
        if gw in outlook.index.get_level_values("event"):
            week = outlook.xs(gw, level="event")
        else:
            week = outlook.iloc[0:0].droplevel("event")  # a gameweek with no fixtures at all
        columns[gw] = project(players, week, scoring, apg)
    return pd.DataFrame(columns).round(2)


# ---------- Your team ----------


def selling_price(now: float, bought: float, fee: float) -> float:
    """FPL keeps part of a price rise: you get the purchase price plus `fee` of the rise,
    rounded down to 0.1. A fall is passed on in full."""
    now_t, bought_t = round(now * 10), round(bought * 10)
    if now_t <= bought_t:
        return now_t / 10
    return (bought_t + math.floor((now_t - bought_t) * fee)) / 10


def my_squad(picks: dict, transfers: list, history: dict, players: pd.DataFrame, fee: float):
    """The players you own, each with an estimated selling price, and the money in the bank.

    The purchase price comes from your last transfer in of that player. For a player kept since
    your first gameweek there is no record, so his current price is used. A player who can no
    longer be picked (he left the league) is counted as sold at the price you paid, if known."""
    freehit_weeks = {c["event"] for c in history["chips"] if c["name"] == "freehit"}
    bought = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        if t["event"] not in freehit_weeks:  # Free Hit transfers are undone after one week
            bought[t["element_in"]] = t["element_in_cost"] / 10
    ids = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10
    gone = [i for i in ids if i not in set(players["id"])]
    if gone:
        log.warning("Players you own who can no longer be picked: %s", gone)
        bank += sum(bought.get(i, 0.0) for i in gone)
    owned = players[players["id"].isin(ids)].copy()
    owned["bought"] = owned["id"].map(bought).fillna(owned["price"])
    owned["sell"] = [
        selling_price(n, b, fee) for n, b in zip(owned["price"], owned["bought"], strict=True)
    ]
    return owned, bank


def estimate_free_transfers(history: dict, max_free: int) -> int:
    """Free transfers for the next gameweek, worked out from your transfer history.

    Rules used: no free transfers count in your first gameweek (the squad is new); then one
    more each week, up to `max_free`; transfers used are subtracted; never fewer than 1.
    In a Wildcard or Free Hit week the transfers are free and the saved ones are kept."""
    chips = {c["event"]: c["name"] for c in history["chips"]}
    weeks = sorted(history["current"], key=lambda w: w["event"])
    ft = 1  # for the week after your first gameweek
    for week in weeks[1:]:
        used = 0 if chips.get(week["event"]) in ("wildcard", "freehit") else week["event_transfers"]
        ft = max(1, min(max_free, ft - used + 1))
    return ft


def available_chips(boot: dict, history: dict, next_event: int) -> list[str]:
    """Chips you can play in the next gameweek: its window is open and it has not been used."""
    out = []
    for chip in boot["chips"]:
        if not chip["start_event"] <= next_event <= chip["stop_event"]:
            continue
        used = any(
            c["name"] == chip["name"] and chip["start_event"] <= c["event"] <= chip["stop_event"]
            for c in history["chips"]
        )
        if not used and chip["name"] not in out:
            out.append(chip["name"])
    return out


# ---------- Optimisation ----------


def optimise(
    players: pd.DataFrame,
    rules: dict,
    col: str = "points",
    owned=None,
    free_transfers: int = 0,
    hit_cost: float = 4.0,
    max_hits: int = 0,
    min_gain: float = 0.0,
) -> pd.DataFrame:
    """Solve the squad as a MILP. Needs columns team, position, price and `col`.

    Variables: x, s, c for each of the n players (in the squad, starts, captain), and one
    whole number h (extra transfers that cost points). With `owned` (True for players you
    own), at most free_transfers + h players can be new."""
    n = len(players)
    pts = players[col].to_numpy(dtype=float)
    zero, one, eye = np.zeros(n), np.ones(n), np.eye(n)
    new = zero if owned is None else (~np.asarray(owned, dtype=bool)).astype(float)
    rows, low, high = [], [], []

    def add(lo: float, hi: float, x=zero, s=zero, c=zero, h=0.0) -> None:
        rows.append(np.concatenate([x, s, c, [h]]))
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
    if owned is not None:
        add(-np.inf, free_transfers, x=new, h=-1.0)  # new players <= free transfers + hits
    z = np.zeros((n, 1))
    constraints = [
        LinearConstraint(np.array(rows), low, high),
        LinearConstraint(np.hstack([-eye, eye, 0 * eye, z]), -np.inf, 0),  # starters in squad
        LinearConstraint(np.hstack([0 * eye, -eye, eye, z]), -np.inf, 0),  # the captain starts
    ]
    # Starters score once, the captain twice, substitutes a little; each hit costs points, and
    # each new player must gain at least min_gain. milp minimises, so the gains are negated.
    gain = np.concatenate(
        [BENCH_WEIGHT * pts - min_gain * new, (1 - BENCH_WEIGHT) * pts, pts, [-hit_cost]]
    )
    upper = np.concatenate([np.ones(3 * n), [max_hits]])
    res = milp(
        -gain,
        constraints=constraints,
        integrality=np.ones(3 * n + 1),
        bounds=Bounds(np.zeros(3 * n + 1), upper),
    )
    if not res.success:
        raise RuntimeError(f"no valid squad found: {res.message}")
    x, s, c = np.round(res.x[: 3 * n]).reshape(3, n).astype(bool)
    squad = players[x].assign(start=s[x], captain=c[x])
    squad.attrs["hits"] = int(round(res.x[-1]))
    return squad


def lineup(squad: pd.DataFrame, rules: dict, col: str) -> pd.DataFrame:
    """Best XI, captain, vice-captain and bench order of a fixed squad, for one gameweek."""
    free = {**rules, "budget": np.inf, "club_limit": rules["squad"]}
    out = optimise(squad.drop(columns=["start", "captain"], errors="ignore"), free, col)
    others = out[out["start"] & ~out["captain"]]
    out["vice"] = out.index == others[col].idxmax()  # the best starter after the captain
    bench = out[~out["start"]].copy()
    bench["gk"] = bench["position"] == "GKP"
    order = bench.sort_values(["gk", col], ascending=[False, False]).index  # keeper first
    out["bench_order"] = 0
    out.loc[order, "bench_order"] = range(1, len(order) + 1)
    return out


# ---------- Plans and chip advice ----------


def summarise_plan(squad: pd.DataFrame, horizon: pd.DataFrame, gw: int, hit_cost: float) -> dict:
    """Points for the next gameweek, and the squad's value over the whole horizon."""
    nxt = squad[gw]
    xi = squad["start"]
    total = horizon.loc[squad.index].sum(axis=1)
    hits = squad.attrs.get("hits", 0)
    return {
        "next_gw_points": round(float(nxt[xi].sum() + nxt[squad["captain"]].sum()), 2),
        "bench_points": round(float(nxt[~xi].sum()), 2),
        "captain_points": round(float(nxt[squad["captain"]].sum()), 2),
        "horizon_value": round(
            float(
                total[squad["h_start"]].sum()
                + total[squad["h_captain"]].sum()
                + BENCH_WEIGHT * total[~squad["h_start"]].sum()
                - hit_cost * hits
            ),
            2,
        ),
        "hits": hits,
    }


def make_plan(players, horizon, rules, cfg, mode, owned=None, budget=None, ft=0) -> tuple:
    """One plan: choose the 15, then the XI for the next gameweek."""
    gw = horizon.columns[0]
    pool = players.join(horizon)
    pool["total"] = horizon.sum(axis=1)
    if owned is not None:  # owned players are valued at what you would get for selling them
        pool["price"] = np.where(owned, pool["sell"], pool["price"])
    r = {**rules, "budget": budget if budget is not None else rules["budget"]}
    if mode == "transfers":
        squad = optimise(
            pool,
            r,
            "total",
            owned=owned,
            free_transfers=ft,
            hit_cost=cfg["hit_cost"],
            max_hits=cfg["max_hits"],
            min_gain=cfg["min_gain_per_transfer"],
        )
    elif mode == "wildcard":
        squad = optimise(pool, r, "total")
    else:  # freehit: the best squad for the next gameweek only
        squad = optimise(pool, r, gw)
    hits = squad.attrs["hits"]
    squad = squad.rename(columns={"start": "h_start", "captain": "h_captain"})
    squad = lineup(squad, rules, gw)
    squad.attrs["hits"] = hits
    return squad, summarise_plan(squad, horizon, gw, cfg["hit_cost"])


def chip_advice(plans: dict, available: list, cfg: dict) -> dict:
    """Extra points each available chip is expected to add in the next gameweek (the Wildcard:
    over the whole horizon), and the chip to play, if any passes its threshold."""
    base = plans["transfers"][1]
    gains = {
        "wildcard": plans["wildcard"][1]["horizon_value"] - base["horizon_value"],
        "freehit": plans["freehit"][1]["next_gw_points"] - base["next_gw_points"],
        "bboost": base["bench_points"],
        "3xc": base["captain_points"],
    }
    limits = {
        "wildcard": cfg["wildcard_min_gain"],
        "freehit": cfg["freehit_min_gain"],
        "bboost": cfg["bench_boost_min"],
        "3xc": cfg["triple_captain_min"],
    }
    rows = []
    for chip, gain in gains.items():
        if chip in available:
            rows.append(
                {
                    "chip": chip,
                    "name": CHIP_NAMES[chip],
                    "gain": round(gain, 2),
                    "limit": limits[chip],
                    "passes": bool(gain >= limits[chip]),
                }
            )
    passing = [r for r in rows if r["passes"]]
    best = max(passing, key=lambda r: r["gain"] / limits[r["chip"]]) if passing else None
    return {"available": available, "checks": rows, "play": best["chip"] if best else None}


# ---------- Inputs and output ----------


def results_from_csv(s: Settings) -> pd.DataFrame:
    """Results without the database: P1's own download and loader (for GitHub Actions)."""
    from data.ingest import football_data

    first = current_season_start(start_month=s.season_start_month) - 5
    football_data.run(replace(s, first_season=first))
    m = load_matches(s)
    return m[["match_date", "home_team", "away_team", "home_goals", "away_goals"]].dropna()


def to_records(squad: pd.DataFrame, gw: int, owned_ids: set, opponents: dict) -> list[dict]:
    """The squad as plain records for the JSON file and the pitch picture."""
    out = squad.assign(
        next_points=squad[gw],
        horizon_points=squad["total"],
        transfer_in=~squad["id"].isin(owned_ids) if owned_ids else False,
        opponents=squad["team_id"].map(lambda t: opponents.get(t, [])),
    )
    cols = [
        "id",
        "name",
        "club",
        "position",
        "price",
        "next_points",
        "horizon_points",
        "start",
        "captain",
        "vice",
        "bench_order",
        "transfer_in",
        "opponents",
    ]
    out = out[cols].round({"next_points": 2, "horizon_points": 2})
    return json.loads(out.to_json(orient="records"))


def next_opponents(fixtures: list, boot: dict, gw: int) -> dict:
    """team id -> list like ['LEE (H)'] for the gameweek; two entries in a double gameweek."""
    short = {t["id"]: t["short_name"] for t in boot["teams"]}
    out: dict = {}
    for f in fixtures:
        if f["event"] == gw:
            out.setdefault(f["team_h"], []).append(f"{short[f['team_a']]} (H)")
            out.setdefault(f["team_a"], []).append(f"{short[f['team_h']]} (A)")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="FPL squad, transfers and chip advice")
    parser.add_argument(
        "--mode",
        choices=("auto", "transfers", "wildcard", "freehit"),
        default="auto",
        help="auto: transfers if your team is set up",
    )
    parser.add_argument("--free-transfers", type=int, help="replace the estimate")
    parser.add_argument("--from-csv", action="store_true", help="results without the database")
    args = parser.parse_args(argv)

    s = load_settings()
    setup_logging(s.logs)
    m, cfg = s.modelling, s.modelling["fpl"]
    folder = s.raw / "fpl"
    if not (folder / "bootstrap-static.json").exists():
        raise SystemExit("No FPL data on disk; run python -m data.ingest.fpl first")

    def read(name: str):
        return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))

    boot, fixtures = read("bootstrap-static"), read("fixtures")
    lookup = name_lookup(s, "fpl")
    events = {e["id"]: e for e in boot["events"]}
    first = next(e["id"] for e in boot["events"] if e["is_next"])
    gameweeks = [g for g in range(first, first + cfg["horizon"]) if g in events]
    team_name = {t["id"]: lookup.get(t["name"], t["name"]) for t in boot["teams"]}
    upcoming = pd.DataFrame(
        [
            {
                "event": f["event"],
                "home_team": team_name[f["team_h"]],
                "away_team": team_name[f["team_a"]],
            }
            for f in fixtures
            if f["event"] in gameweeks
        ]
    )

    if args.from_csv:
        results = results_from_csv(s)
    else:
        from data.common.db import make_engine
        from modelling.match.data import load_results

        results = load_results(make_engine())
    model = fit_dixon_coles(
        results, date.today(), m["dc_xi"], m["dc_history_days"], m["dc_max_goals"]
    )
    unknown = (set(upcoming["home_team"]) | set(upcoming["away_team"])) - set(model.teams)
    if unknown:
        log.warning("Teams not in the model (add them to team_names.csv): %s", sorted(unknown))

    rules = load_rules(boot)
    players = load_players(boot, fixtures, lookup)
    everyone = pd.DataFrame(boot["elements"])
    apg = everyone["assists"].sum() / max(everyone["goals_scored"].sum(), 1)
    horizon = project_by_gameweek(
        players, team_outlook(model, upcoming), boot["game_config"]["scoring"], apg, gameweeks
    )
    gw = gameweeks[0]

    has_team = int(cfg["team_id"]) > 0 and (folder / "entry_picks.json").exists()
    plans, out = {}, {"free_transfers": None, "bank": None, "chips": None}
    owned_ids: set = set()
    if has_team:
        history = read("entry_history")
        owned_df, bank = my_squad(
            read("entry_picks"), read("entry_transfers"), history, players, rules["sell_on_fee"]
        )
        owned_ids = set(owned_df["id"])
        players["sell"] = players["id"].map(owned_df.set_index("id")["sell"])
        owned = players["id"].isin(owned_ids).to_numpy()
        budget = bank + owned_df["sell"].sum()
        estimate = estimate_free_transfers(history, rules["max_free_transfers"])
        ft = args.free_transfers if args.free_transfers is not None else estimate
        for mode in ("transfers", "wildcard", "freehit"):
            plans[mode] = make_plan(
                players,
                horizon,
                rules,
                cfg,
                mode,
                owned=owned if mode == "transfers" else None,
                budget=budget,
                ft=ft,
            )
        out.update(
            free_transfers=ft,
            free_transfers_source="you" if args.free_transfers is not None else "estimate",
            bank=round(bank, 1),
            chips=chip_advice(plans, available_chips(boot, history, gw), cfg),
        )
    else:
        plans["wildcard"] = make_plan(players, horizon, rules, cfg, "wildcard")
    shown = args.mode if args.mode != "auto" else ("transfers" if has_team else "wildcard")
    if shown not in plans:
        raise SystemExit(f"--mode {shown} needs your team: set [modelling.fpl] team_id")

    opponents = next_opponents(fixtures, boot, gw)
    report = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "gameweek": gw,
        "deadline": events[gw]["deadline_time"],
        "gameweeks": gameweeks,
        "shown": shown,
        **out,
        "plans": {},
    }
    for mode, (squad, summary) in plans.items():
        squad = squad.assign(total=horizon.loc[squad.index].sum(axis=1))
        ins = squad[~squad["id"].isin(owned_ids)] if owned_ids else squad.iloc[0:0]
        outs = sorted(owned_ids - set(squad["id"]))
        report["plans"][mode] = {
            **summary,
            "transfers_in": ins["name"].tolist() if mode == "transfers" else [],
            "transfers_out": players.set_index("id").loc[outs, "name"].tolist()
            if mode == "transfers"
            else [],
            "players": to_records(
                squad, gw, owned_ids if mode == "transfers" else set(), opponents
            ),
        }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "fpl_plan.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    table = pd.DataFrame(report["plans"][shown]["players"])
    table.insert(0, "gameweek", gw)
    table.to_csv(REPORTS / "fpl_squad.csv", index=False)

    plan = report["plans"][shown]
    print(
        table[
            ["name", "club", "position", "price", "next_points", "start", "captain", "vice"]
        ].to_string(index=False)
    )
    log.info("Gameweek %d (%s plan): next gameweek %.1f points", gw, shown, plan["next_gw_points"])
    if shown == "transfers":
        log.info(
            "Transfers: out %s, in %s, hits %d (free transfers %s: %d)",
            plan["transfers_out"],
            plan["transfers_in"],
            plan["hits"],
            out["free_transfers_source"],
            out["free_transfers"],
        )
    if out["chips"]:
        log.info("Chip advice: %s", CHIP_NAMES.get(out["chips"]["play"], "no chip"))


if __name__ == "__main__":
    main()
