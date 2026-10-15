import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from modelling.match import scoreboard as sb

T = pd.Timestamp


def test_rps_matches_the_hand_calculation_and_stage_6():
    # Stage 12's example: 0.50 / 0.27 / 0.23, result a draw -> 0.1515
    assert sb.rps([[0.50, 0.27, 0.23]], [1])[0] == pytest.approx(0.1515, abs=5e-5)
    metrics = pytest.importorskip("modelling.metrics")  # needs scikit-learn
    p = np.random.default_rng(0).dirichlet([3, 2, 2], size=50)
    y = np.random.default_rng(1).integers(0, 3, size=50)
    assert np.allclose(sb.rps(p, y), metrics.rps(p, y))


def test_first_prediction_rule_keeps_the_earliest():
    preds = pd.DataFrame(
        {
            "season": ["2026/27"] * 2,
            "home_team": ["Arsenal"] * 2,
            "away_team": ["Chelsea"] * 2,
            "created_at": [T("2026-10-09 09:00"), T("2026-10-08 09:00")],
            "p_home": [0.46, 0.50],
        }
    )
    kept = sb.first_predictions(preds)
    assert len(kept) == 1 and kept["p_home"].iloc[0] == 0.50


def test_results_link_to_a_postponed_match_and_skip_last_season():
    preds = pd.DataFrame(
        {
            "prediction_id": ["a"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "created_at": [T("2026-10-08 09:00")],
            "fixture_date": [T("2026-10-10")],
        }
    )
    results = pd.DataFrame(
        {
            "match_date": [T("2026-03-01"), T("2026-10-21")],
            "home_team": ["Arsenal"] * 2,
            "away_team": ["Chelsea"] * 2,
            "home_goals": [3, 1],
            "away_goals": [0, 1],
            **{c: [np.nan, np.nan] for c in sb.ODDS},
        }
    )
    out = sb.attach_results(preds, results)
    assert out["result_date"].iloc[0] == T("2026-10-21")  # postponed to 21 October
    assert out["home_goals"].iloc[0] == 1


def test_status_counts_only_predictions_made_and_committed_before_match_day():
    df = pd.DataFrame(
        {
            "file": ["early.csv", "late_commit.csv", "made_on_day.csv", "local.csv"],
            "created_at": [T("2026-10-08 09:00")] * 2
            + [T("2026-10-10 08:00"), T("2026-10-08 09:00")],
            "result_date": [T("2026-10-10")] * 4,
            "fixture_date": [T("2026-10-10")] * 4,
        }
    )
    commits = {
        "early.csv": T("2026-10-08 10:00"),
        "late_commit.csv": T("2026-10-10 06:00"),
        "made_on_day.csv": T("2026-10-10 09:00"),
        "local.csv": None,
    }
    out = sb.status(df, commits)
    assert out["counts"].tolist() == [True, False, False, True]


def test_frequency_baseline_uses_only_the_window_before_the_day():
    results = pd.DataFrame(
        {
            "match_date": [T("2020-01-01"), T("2026-10-01"), T("2026-10-02"), T("2026-10-10")],
            "home_goals": [0, 2, 1, 0],
            "away_goals": [5, 0, 1, 3],
        }
    )
    p = sb.freq_probs(results, T("2026-10-10"), 1095)
    assert p.tolist() == [0.5, 0.5, 0.0]  # 2020 is too old; 10 October is the day itself


def test_readme_block_is_replaced_in_place_or_added_once():
    text = f"# Repo\n\n{sb.START}\nold\n{sb.END}\n\n## Next\n"
    new = sb.splice_readme(text, "new\n")
    assert "old" not in new and "## Next" in new and new.count(sb.START) == 1
    assert sb.splice_readme(new, "new\n") == new
    added = sb.splice_readme("# Repo\n", "first\n")
    assert added.count(sb.START) == 1 and "first" in added


def rounds(n: int) -> list[dict]:
    rng = np.random.default_rng(n)
    return [
        {
            "round": f"GW {i + 7}",
            "cum_model_rps": 0.20 + rng.random() / 100,
            "cum_freq_rps": 0.2005 + rng.random() / 100,
            "cum_book_rps": None if i == 0 else 0.20 + rng.random() / 100,
        }
        for i in range(n)
    ]


@pytest.mark.parametrize("n", [1, 2, 32])
def test_chart_is_valid_svg_and_end_labels_never_overlap(n):
    svg = sb.svg_chart(rounds(n))
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    end_labels = [t for t in root.iter(f"{ns}text") if t.get("x") == str(64 + 524 + 10)]
    ys = sorted(float(t.get("y")) for t in end_labels)
    assert len(ys) == (2 if n == 1 else 3)  # no odds yet in the first round
    assert all(b - a >= 15 - 1e-9 for a, b in zip(ys, ys[1:], strict=False))


def test_gameweeks_come_from_fpl_dates_or_fall_back_to_dates():
    fixtures = [
        {"event": 7, "kickoff_time": "2026-10-10T14:00:00Z"},
        {"event": 7, "kickoff_time": "2026-10-11T14:00:00Z"},
        {"event": None, "kickoff_time": None},
    ]
    gw = sb.gameweek_by_date(fixtures)
    df = pd.DataFrame(
        {
            "file": ["a", "a", "b"],
            "result_date": [T("2026-10-10"), T("2026-10-11"), pd.NaT],
            "fixture_date": [T("2026-10-10"), T("2026-10-11"), T("2026-10-17")],
        }
    )
    assert sb.round_labels(df, gw).tolist() == ["GW 7", "GW 7", "17 Oct"]


def test_commit_time_is_when_the_file_was_first_added(tmp_path, monkeypatch):
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@e.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@e.com",
    )

    def git(*args, when=None):
        e = dict(env, GIT_COMMITTER_DATE=when, GIT_AUTHOR_DATE=when) if when else env
        subprocess.run(["git", *args], cwd=tmp_path, check=True, env=e, capture_output=True)

    git("init", "-q")
    (tmp_path / "p.csv").write_text("a\n")
    git("add", "p.csv")
    git("commit", "-qm", "add", when="2026-10-08T15:00:00+05:30")
    (tmp_path / "p.csv").write_text("b\n")  # a later edit must not move the time
    git("commit", "-qam", "edit", when="2026-10-11T15:00:00+05:30")
    monkeypatch.setattr(sb, "ROOT", tmp_path)
    times, checked = sb.commit_times(["p.csv", "missing.csv"])
    assert checked
    assert times["p.csv"] == T("2026-10-08 09:30")  # 15:00 in India is 09:30 UTC
    assert times["missing.csv"] is None


def test_a_match_with_no_result_after_its_date_is_listed_as_waiting():
    empty = pd.DataFrame()
    waiting = pd.DataFrame(
        {"home_team": ["Arsenal"], "away_team": ["Chelsea"], "fixture_date": [T("2026-10-10")]}
    )
    text = sb.markdown(None, empty, empty, empty, waiting, pd.Timestamp("2026-10-12").date(), True)
    assert "No predicted match has been played yet." in text
    assert "Waiting for a result: Arsenal v Chelsea (Sat 10 Oct)." in text
