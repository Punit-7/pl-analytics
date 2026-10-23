"""Predictions against results: the scoreboard in the README.

python -m modelling.match.scoreboard              results from the files already on your PC
python -m modelling.match.scoreboard --download   download this season's results first
python -m modelling.match.scoreboard --no-readme  a preview: reports only, README unchanged
"""

import argparse
import json
import logging
import subprocess
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.build_warehouse import load_matches, name_lookup
from data.common.config import ROOT, Settings, load_settings
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start

log = logging.getLogger("modelling.match.scoreboard")
REPORTS = ROOT / "modelling" / "reports"
PREDICTIONS = ROOT / "modelling" / "predictions"
SERVING_PREDICTIONS = ROOT / "serving" / "predictions"  # the weekly job (P5), from 2026-10-09
README = ROOT / "README.md"
START, END = "<!-- scoreboard:start -->", "<!-- scoreboard:end -->"
PROBS = ["p_home", "p_draw", "p_away"]
ODDS = ["avg_close_home_odds", "avg_close_draw_odds", "avg_close_away_odds"]
NAMES = np.array(["Home", "Draw", "Away"])
MIN_FOR_INTERVAL = 30  # fewer matches than this: no interval is shown


# ---------- Scoring (the same formulas as Stages 9 and 10) ----------


def outcome_index(home_goals, away_goals) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    home_goals, away_goals = np.asarray(home_goals), np.asarray(away_goals)
    return np.select([home_goals > away_goals, home_goals == away_goals], [0, 1], default=2)


def rps(p, outcome) -> np.ndarray:
    """Ranked probability score per match. p: (n, 3) as home/draw/away; outcome: 0/1/2."""
    p = np.asarray(p, dtype=float)
    o = np.zeros_like(p)
    o[np.arange(len(p)), outcome] = 1.0
    cum_p, cum_o = p.cumsum(axis=1), o.cumsum(axis=1)
    return ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(axis=1) / 2


def implied_probs(odds: np.ndarray) -> np.ndarray:
    """Proportional method: 1/odds, rescaled to sum to 1."""
    q = 1.0 / np.asarray(odds, dtype=float)
    return q / q.sum(axis=1, keepdims=True)


def freq_probs(results: pd.DataFrame, day: pd.Timestamp, window_days: int) -> np.ndarray:
    """The frequency baseline: home/draw/away shares of the matches in the window before day."""
    past = results[
        (results["match_date"] < day)
        & (results["match_date"] >= day - pd.Timedelta(days=window_days))
    ]
    o = outcome_index(past["home_goals"], past["away_goals"])
    return np.bincount(o, minlength=3) / max(len(o), 1)


def paired_interval(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Mean of a per-match difference and its 95% bootstrap interval (as in Stage 10)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    low, high = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


# ---------- Inputs ----------


def load_predictions(folder: Path) -> pd.DataFrame:
    """Every published matches CSV, with the file it came from. Empty files are skipped."""
    frames = []
    for path in sorted(folder.glob("*/*_matches.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df = df.assign(
            created_at=pd.to_datetime(df["created_at"]), file=path.relative_to(ROOT).as_posix()
        )
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(columns={"match_date": "fixture_date"})


def fixture_dates(s: Settings) -> pd.DataFrame:
    """Home team, away team and kick-off date from the FPL files, with the team names used here.
    Empty if the FPL files are not on disk."""
    folder = s.raw / "fpl"
    if not (folder / "fixtures.json").exists():
        return pd.DataFrame(columns=["home_team", "away_team", "fixture_date"])
    boot = json.loads((folder / "bootstrap-static.json").read_text(encoding="utf-8"))
    fixtures = json.loads((folder / "fixtures.json").read_text(encoding="utf-8"))
    lookup = name_lookup(s, "fpl")
    name = {t["id"]: lookup.get(t["name"], t["name"]) for t in boot["teams"]}
    rows = [
        {
            "home_team": name[f["team_h"]],
            "away_team": name[f["team_a"]],
            "fixture_date": pd.Timestamp(f["kickoff_time"]).tz_localize(None).normalize(),
        }
        for f in fixtures
        if f.get("kickoff_time")
    ]
    return pd.DataFrame(rows)


def add_live_predictions(old: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Add the weekly job's predictions for matches the first folder has no prediction for.
    Matches already in the first folder keep their first prediction, so history does not change."""
    live = load_predictions(SERVING_PREDICTIONS)
    if live.empty:
        return old
    if "fixture_date" not in live.columns:
        live = live.merge(fixture_dates(s), on=["home_team", "away_team"], how="left")
    live = first_predictions(live)
    key = ["season", "home_team", "away_team"]
    seen = old[key].merge(live[key].drop_duplicates(), on=key, how="inner")
    new = live.merge(seen, on=key, how="left", indicator=True)
    new = new[new["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([old, new], ignore_index=True)


def first_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """The first-prediction rule: keep only the earliest prediction of each match."""
    preds = preds.sort_values("created_at", kind="stable")
    return preds.drop_duplicates(["season", "home_team", "away_team"]).reset_index(drop=True)


def commit_times(files: list[str]) -> tuple[dict, bool]:
    """When each file first reached Git (UTC). Returns ({file: time or None}, checked)."""
    try:
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        log.warning("Git is not available here, so commit times are not checked")
        return {f: None for f in files}, False
    if shallow == "true":
        log.warning("Shallow clone: commit times cannot be checked (use fetch-depth: 0)")
        return {f: None for f in files}, False
    out = {}
    for f in files:
        lines = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%cI", "--", f],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.split()
        # git log lists newest first, so the last line is when the file was first added.
        out[f] = pd.Timestamp(lines[-1]).tz_convert("UTC").tz_localize(None) if lines else None
    return out, True


def attach_results(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join each prediction to the first result of that fixture on or after the day it was made.

    Matching on teams and dates, not on the fixture date, keeps a postponed match linked.
    """
    cols = ["home_team", "away_team", "match_date", "home_goals", "away_goals", *ODDS]
    m = preds[["prediction_id", "home_team", "away_team", "created_at"]].merge(
        results[cols], on=["home_team", "away_team"]
    )
    m = m[m["match_date"] >= m["created_at"].dt.normalize()]
    m = m.sort_values("match_date").drop_duplicates("prediction_id")
    keep = ["prediction_id", "match_date", "home_goals", "away_goals", *ODDS]
    out = preds.merge(m[keep], on="prediction_id", how="left")
    return out.rename(columns={"match_date": "result_date"})


def gameweek_by_date(fixtures: list) -> dict:
    """FPL fixtures: kick-off date -> the gameweek most fixtures on that date belong to."""
    rows = [
        (pd.Timestamp(f["kickoff_time"]).date(), f["event"])
        for f in fixtures
        if f.get("event") and f.get("kickoff_time")
    ]
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["day", "event"])
    return df.groupby("day")["event"].agg(lambda e: int(e.mode().iloc[0])).to_dict()


def round_labels(df: pd.DataFrame, gw_map: dict) -> pd.Series:
    """'GW 7' from the FPL fixtures, or the dates of the publication it came from."""
    day = df["result_date"].fillna(df["fixture_date"])
    gw = day.dt.date.map(gw_map)
    first = day.groupby(df["file"]).transform("min")
    last = day.groupby(df["file"]).transform("max")
    by_dates = [
        f"{a.day}–{b.day} {b:%b}" if (a.month, a.day) != (b.month, b.day) else f"{b.day} {b:%b}"
        for a, b in zip(first, last, strict=True)
    ]
    return pd.Series(
        [f"GW {int(g)}" if pd.notna(g) else d for g, d in zip(gw, by_dates, strict=True)],
        index=df.index,
    )


# ---------- Scoring every match ----------


def score(df: pd.DataFrame, results: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Add the outcome, the three forecasts' RPS and whether each pick was right."""
    df = df.copy()
    y = outcome_index(df["home_goals"], df["away_goals"])
    df["result"] = NAMES[y]
    model = df[PROBS].to_numpy()
    freq = np.vstack([freq_probs(results, d, window_days) for d in df["result_date"]])
    df["model_rps"] = rps(model, y)
    df["freq_rps"] = rps(freq, y)
    df["model_pick"] = NAMES[model.argmax(axis=1)]
    df["freq_pick"] = NAMES[freq.argmax(axis=1)]
    has_odds = df[ODDS].notna().all(axis=1).to_numpy()
    df["book_rps"] = np.nan
    df["book_pick"] = None
    if has_odds.any():
        book = implied_probs(df.loc[has_odds, ODDS].to_numpy())
        df.loc[has_odds, "book_rps"] = rps(book, y[has_odds])
        df.loc[has_odds, "book_pick"] = NAMES[book.argmax(axis=1)]
    for who in ("model", "freq", "book"):
        df[f"{who}_right"] = (df[f"{who}_pick"] == df["result"]).where(df[f"{who}_pick"].notna())
    return df


def status(df: pd.DataFrame, commits: dict) -> pd.DataFrame:
    """Mark which predictions count: made, and committed, before the match day."""
    df = df.copy()
    played = df["result_date"].notna()
    day = df["result_date"].fillna(df["fixture_date"])
    df["committed_at"] = pd.to_datetime(df["file"].map(commits))
    df["made_before"] = df["created_at"].dt.normalize() < day
    in_git = df["committed_at"].notna()
    df["committed_before"] = (df["committed_at"].dt.normalize() < day).where(in_git)
    df["counts"] = played & df["made_before"] & df["committed_before"].fillna(True).astype(bool)
    return df


# ---------- Summary ----------


def summarise(scored: pd.DataFrame) -> dict:
    """Season totals, the two comparisons and one row per round (with running means)."""
    n = len(scored)
    odds = scored[scored["book_rps"].notna()]
    out = {
        "matches": n,
        "model_rps": float(scored["model_rps"].mean()),
        "freq_rps": float(scored["freq_rps"].mean()),
        "model_right": int(scored["model_right"].sum()),
        "freq_right": int(scored["freq_right"].sum()),
        "odds_matches": len(odds),
        "book_rps": float(odds["book_rps"].mean()) if len(odds) else None,
        "model_rps_on_odds": float(odds["model_rps"].mean()) if len(odds) else None,
        "book_right": int(odds["book_right"].sum()) if len(odds) else 0,
        "vs_freq": None,
        "vs_book": None,
    }
    diff_freq = (scored["model_rps"] - scored["freq_rps"]).to_numpy()
    out["vs_freq"] = (
        paired_interval(diff_freq)
        if n >= MIN_FOR_INTERVAL
        else (float(diff_freq.mean()), None, None)
    )
    if len(odds):
        diff_book = (odds["model_rps"] - odds["book_rps"]).to_numpy()
        out["vs_book"] = (
            paired_interval(diff_book)
            if len(odds) >= MIN_FOR_INTERVAL
            else (float(diff_book.mean()), None, None)
        )

    rounds = []
    order = scored.groupby("round")["result_date"].min().sort_values().index
    seen = pd.DataFrame()
    for label in order:
        r = scored[scored["round"] == label]
        seen = pd.concat([seen, r])
        r_odds, seen_odds = r["book_rps"].dropna(), seen["book_rps"].dropna()
        rounds.append(
            {
                "round": label,
                "first_date": r["result_date"].min().date().isoformat(),
                "last_date": r["result_date"].max().date().isoformat(),
                "matches": len(r),
                "model_rps": float(r["model_rps"].mean()),
                "freq_rps": float(r["freq_rps"].mean()),
                "book_rps": float(r_odds.mean()) if len(r_odds) else None,
                "model_right": int(r["model_right"].sum()),
                "cum_model_rps": float(seen["model_rps"].mean()),
                "cum_freq_rps": float(seen["freq_rps"].mean()),
                "cum_book_rps": float(seen_odds.mean()) if len(seen_odds) else None,
            }
        )
    out["rounds"] = rounds
    return out


# ---------- The README block ----------


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def day_name(d) -> str:
    d = pd.Timestamp(d)
    return f"{d:%a} {d.day} {d:%b}"


def comparison(name: str, triple, n: int) -> str:
    mean, low, high = triple
    if low is None:
        return (
            f"Model minus {name}: {mean:+.4f} on {n} matches. "
            f"Too few matches for an interval (it is shown from {MIN_FOR_INTERVAL})."
        )
    if high < 0:
        verdict = f"the model is better than the {name}"
    elif low > 0:
        verdict = f"the model is behind the {name}"
    else:
        verdict = "no clear difference yet"
    return (
        f"Model minus {name}: {mean:+.4f} (95% interval {low:+.4f} to {high:+.4f}) "
        f"on {n} matches: {verdict}."
    )


def match_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "| Date | Match | Home / Draw / Away | Pick | Score | Pick right? | RPS model | RPS odds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in df.sort_values(["result_date", "home_team"]).itertuples():
        book = "–" if pd.isna(r.book_rps) else f"{r.book_rps:.3f}"
        lines.append(
            f"| {day_name(r.result_date)} | {r.home_team} v {r.away_team} "
            f"| {pct(r.p_home)} / {pct(r.p_draw)} / {pct(r.p_away)} | {r.model_pick} "
            f"| {int(r.home_goals)}–{int(r.away_goals)} | {'✓' if r.model_right else '✗'} "
            f"| {r.model_rps:.3f} | {book} |"
        )
    return lines


def upcoming_table(df: pd.DataFrame, git_checked: bool) -> list[str]:
    lines = [
        "| Date | Match | Home / Draw / Away | Pick | Most likely score | Expected goals "
        "| Made (UTC) | In Git (UTC) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in df.sort_values(["fixture_date", "home_team"]).itertuples():
        probs = np.array([r.p_home, r.p_draw, r.p_away])
        pick = (
            r.predicted_result.title()
            if isinstance(r.predicted_result, str)
            else (NAMES[probs.argmax()])
        )
        score_text = (
            r.most_likely_score.replace("-", "–") if isinstance(r.most_likely_score, str) else "–"
        )
        xg = (
            f"{r.exp_home_goals:.1f} – {r.exp_away_goals:.1f}"
            if pd.notna(r.exp_home_goals) and pd.notna(r.exp_away_goals)
            else "–"
        )
        if pd.notna(r.committed_at):
            git = f"{r.committed_at:%Y-%m-%d %H:%M}"
        else:
            git = "not yet" if git_checked else "not checked"
        lines.append(
            f"| {day_name(r.fixture_date)} | {r.home_team} v {r.away_team} "
            f"| {pct(r.p_home)} / {pct(r.p_draw)} / {pct(r.p_away)} | {pick} | {score_text} "
            f"| {xg} | {r.created_at:%Y-%m-%d %H:%M} | {git} |"
        )
    return lines


def markdown(
    summary: dict | None,
    scored: pd.DataFrame,
    upcoming: pd.DataFrame,
    excluded: pd.DataFrame,
    waiting: pd.DataFrame,
    today: date,
    git_checked: bool,
    next_round: str = "",
) -> str:
    out = [
        f"_Updated {today.isoformat()}. Only each match's first prediction counts. It must be made "
        "before match day and committed to Git before match day. Results and closing odds: "
        "football-data.co.uk. RPS (ranked probability score): 0 is a perfect forecast; "
        "lower is better._",
        "",
    ]
    if not git_checked:
        out += ["_Commit times were not checked in this run._", ""]
    if summary is None:
        out += ["No predicted match has been played yet.", ""]
    else:
        s = summary
        n, k = s["matches"], s["odds_matches"]
        out += [
            "**Season so far**",
            "",
            "| Forecaster | Matches | Mean RPS | Picks right |",
            "|---|---|---|---|",
            f"| Dixon–Coles (this model) | {n} | {s['model_rps']:.4f} "
            f"| {s['model_right']} of {n} ({pct(s['model_right'] / n)}) |",
            f"| League frequencies (baseline) | {n} | {s['freq_rps']:.4f} "
            f"| {s['freq_right']} of {n} ({pct(s['freq_right'] / n)}) |",
        ]
        if k:
            out.append(
                f"| Bookmakers' closing odds | {k} | {s['book_rps']:.4f} "
                f"| {s['book_right']} of {k} ({pct(s['book_right'] / k)}) |"
            )
        out += ["", comparison("baseline", s["vs_freq"], n)]
        if k:
            out += [
                "",
                comparison("closing odds", s["vs_book"], k)
                + f" The model's RPS on these {k} matches is {s['model_rps_on_odds']:.4f}.",
            ]
        out += [
            "",
            "![Mean RPS after each round](modelling/reports/scoreboard.svg)",
            "",
        ]
        last = s["rounds"][-1]
        latest = scored[scored["round"] == last["round"]]
        book = "no odds yet" if last["book_rps"] is None else f"{last['book_rps']:.4f}"
        out += [
            f"**Latest round: {last['round']}**",
            "",
            *match_table(latest),
            "",
            f"This round: model {last['model_rps']:.4f}, baseline {last['freq_rps']:.4f}, "
            f"closing odds {book}. One round is about 10 matches, too few to judge a model; "
            "the season figures above are the ones to compare.",
            "",
        ]
        if len(s["rounds"]) > 1:
            out += [
                "**Every round**",
                "",
                "| Round | Dates | Matches | Model RPS | Baseline RPS | Odds RPS "
                "| Model picks right |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in reversed(s["rounds"]):
                odds = "–" if r["book_rps"] is None else f"{r['book_rps']:.4f}"
                dates = f"{day_name(r['first_date'])} – {day_name(r['last_date'])}"
                out.append(
                    f"| {r['round']} | {dates} | {r['matches']} | {r['model_rps']:.4f} "
                    f"| {r['freq_rps']:.4f} | {odds} | {r['model_right']} of {r['matches']} |"
                )
            out.append("")
    if len(excluded):
        names = ", ".join(f"{r.home_team} v {r.away_team}" for r in excluded.itertuples())
        out += [
            f"Not counted, because the prediction was made or committed on or after match "
            f"day: {names}.",
            "",
        ]
    if len(waiting):
        names = ", ".join(
            f"{r.home_team} v {r.away_team} ({day_name(r.fixture_date)})"
            for r in waiting.sort_values("fixture_date").itertuples()
        )
        out += [f"Waiting for a result: {names}.", ""]
    if len(upcoming):
        out += [
            f"**Next round: {next_round}**" if next_round else "**Next predictions**",
            "",
            *upcoming_table(upcoming, git_checked),
            "",
        ]
    return "\n".join(out).rstrip() + "\n"


def splice_readme(text: str, block: str) -> str:
    """Replace what is between the two markers. If they are missing, add a section at the end."""
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        return f"{head}{START}\n{block}{END}{tail}"
    section = f"\n## Match predictions against results (updated daily)\n\n{START}\n{block}{END}\n"
    return text.rstrip("\n") + "\n" + section


# ---------- The chart ----------

SERIES = [  # key, label, colour, dash pattern
    ("cum_model_rps", "Dixon–Coles", "#2a78d6", ""),
    ("cum_freq_rps", "Baseline", "#1baf7a", "6 4"),
    ("cum_book_rps", "Closing odds", "#eb6834", "2 3"),
]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FONT = 'font-family="Segoe UI, Arial, sans-serif"'


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_chart(rounds: list[dict]) -> str:
    """Mean RPS so far after each round, one line per forecaster."""
    w, h = 720, 340
    left, right, top, bottom = 64, 132, 70, 56
    pw, ph = w - left - right, h - top - bottom
    values = [r[k] for r in rounds for k, *_ in SERIES if r[k] is not None]
    lo, hi = min(values), max(values)
    step = 0.01 if hi - lo > 0.02 else 0.005
    lo = np.floor((lo - step / 2) / step) * step
    hi = np.ceil((hi + step / 2) / step) * step
    n = len(rounds)

    def x(i: int) -> float:
        return left + (pw / 2 if n == 1 else i * pw / (n - 1))

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="Mean RPS so far after each round">',
        f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
        f'<text x="{left}" y="26" {FONT} font-size="16" '
        f'font-weight="600" fill="{INK}">Mean RPS so far, after each round (lower is better)'
        "</text>",
    ]
    lx = left  # legend row under the title
    for _, label, colour, dash in SERIES:
        parts.append(
            f'<line x1="{lx}" y1="46" x2="{lx + 24}" y2="46" stroke="{colour}" stroke-width="2" '
            f'stroke-dasharray="{dash}"/>'
            f'<text x="{lx + 30}" y="50" {FONT} '
            f'font-size="12" fill="{MUTED}">{esc(label)}</text>'
        )
        lx += 30 + 8 * len(label) + 24
    ticks = np.arange(lo, hi + step / 2, step)
    for t in ticks:
        parts.append(
            f'<line x1="{left}" y1="{y(t):.1f}" x2="{left + pw}" y2="{y(t):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end" '
            f'{FONT} font-size="11" fill="{MUTED}">{t:.3f}</text>'
        )
    every = max(1, int(np.ceil(n / 12)))  # at most about 12 round labels
    for i, r in enumerate(rounds):
        # every few rounds, and always the last; skip one that would touch the last
        if i == n - 1 or (i % every == 0 and n - 1 - i >= every):
            parts.append(
                f'<text x="{x(i):.1f}" y="{top + ph + 20}" text-anchor="middle" '
                f'{FONT} font-size="11" fill="{MUTED}">'
                f"{esc(r['round'])}</text>"
            )
    parts.append(
        f'<text x="{left + pw / 2:.1f}" y="{h - 12}" text-anchor="middle" '
        f'{FONT} font-size="12" fill="{MUTED}">Round</text>'
    )
    ends = []
    for key, label, colour, dash in SERIES:
        pts = [(x(i), y(r[key])) for i, r in enumerate(rounds) if r[key] is not None]
        if not pts:
            continue
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-dasharray="{dash}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for px, py in pts:  # markers with a ring in the surface colour
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colour}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
        last = next(r[key] for r in reversed(rounds) if r[key] is not None)
        ends.append([pts[-1][1], f"{label} {last:.4f}"])
    ends.sort()  # direct labels at the right, pushed apart so they never overlap
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for ly, label in ends:
        parts.append(
            f'<text x="{left + pw + 10}" y="{ly + 4:.1f}" {FONT} '
            f'font-size="12" fill="{INK}">{esc(label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---------- Main ----------


def load_results(s: Settings, download: bool) -> pd.DataFrame:
    """Results and closing odds from P1's files, without the database."""
    if download:
        from data.ingest import football_data

        first = current_season_start(start_month=s.season_start_month) - 5
        football_data.run(replace(s, first_season=first))
    m = load_matches(s)
    m = m.dropna(subset=["home_goals", "away_goals"])
    for c in ODDS:
        if c not in m:  # P2 Stage 1 adds these; without it, there is no odds comparison
            log.warning("Column %s missing from the results files", c)
            m[c] = np.nan
    return m[["match_date", "home_team", "away_team", "home_goals", "away_goals", *ODDS]]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score published predictions for the README")
    parser.add_argument("--download", action="store_true", help="download results first")
    parser.add_argument("--no-readme", action="store_true", help="leave README.md unchanged")
    args = parser.parse_args(argv)
    s = load_settings()
    setup_logging(s.logs)
    today = datetime.now(UTC).date()

    preds = load_predictions(PREDICTIONS)
    if preds.empty and not SERVING_PREDICTIONS.exists():
        log.info("No published predictions yet: nothing to score")
        return
    preds = add_live_predictions(first_predictions(preds), s)
    results = load_results(s, args.download)
    df = attach_results(preds, results)
    commits, git_checked = commit_times(sorted(df["file"].unique()))
    df = status(df, commits)
    fixtures_path = s.raw / "fpl" / "fixtures.json"
    gw_map = (
        gameweek_by_date(json.loads(fixtures_path.read_text(encoding="utf-8")))
        if fixtures_path.exists()
        else {}
    )
    df["round"] = round_labels(df, gw_map)

    played = df[df["result_date"].notna()]
    counted = played[played["counts"]]
    excluded = played[~played["counts"]]
    unplayed = df[df["result_date"].isna()]
    upcoming = unplayed[unplayed["fixture_date"].dt.date >= today]
    next_round = ""
    if len(upcoming):
        labels = round_labels(upcoming, gw_map)
        next_round = labels.loc[upcoming["fixture_date"].idxmin()]
        upcoming = upcoming[labels == next_round]
    waiting = unplayed[unplayed["fixture_date"].dt.date < today]  # postponed, or no result yet
    scored = score(counted, results, s.modelling["dc_history_days"]) if len(counted) else counted
    summary = summarise(scored) if len(scored) else None

    REPORTS.mkdir(parents=True, exist_ok=True)
    block = markdown(summary, scored, upcoming, excluded, waiting, today, git_checked, next_round)
    (REPORTS / "scoreboard.md").write_text(block, encoding="utf-8")
    (REPORTS / "scoreboard.json").write_text(
        json.dumps(
            {"updated": today.isoformat(), "git_checked": git_checked, "summary": summary}, indent=2
        ),
        encoding="utf-8",
    )
    if len(scored):
        scored.to_csv(REPORTS / "scoreboard.csv", index=False)
        (REPORTS / "scoreboard.svg").write_text(svg_chart(summary["rounds"]), encoding="utf-8")
    if not args.no_readme:
        readme = README.read_text(encoding="utf-8") if README.exists() else "# pl-analytics\n"
        README.write_text(splice_readme(readme, block), encoding="utf-8")

    if summary:
        log.info(
            "%d matches scored | model %.4f | baseline %.4f | odds %s on %d",
            summary["matches"],
            summary["model_rps"],
            summary["freq_rps"],
            "–" if summary["book_rps"] is None else f"{summary['book_rps']:.4f}",
            summary["odds_matches"],
        )
    for row in waiting.itertuples():
        log.warning(
            "No result yet for %s v %s (%s)", row.home_team, row.away_team, row.fixture_date.date()
        )
    log.info(
        "%d not counted, %d upcoming, %d waiting. Wrote %s",
        len(excluded),
        len(upcoming),
        len(waiting),
        "modelling/reports/scoreboard.md" if args.no_readme else "README.md and reports",
    )


if __name__ == "__main__":
    main()
