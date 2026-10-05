import logging
import warnings

import pandas as pd
from statsbombpy import sb

from data.common.config import Settings
from data.common.io_utils import write_atomic

log = logging.getLogger(__name__)

SHOT_COLS = [
    "match_id",
    "id",
    "period",
    "minute",
    "second",
    "team",
    "player",
    "location",
    "under_pressure",
    "play_pattern",
    "shot_type",
    "shot_body_part",
    "shot_technique",
    "shot_first_time",
    "shot_outcome",
    "shot_statsbomb_xg",
]


def shots_for_match(match_id: int) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsbombpy warns that open data needs no login
        events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"].reindex(columns=SHOT_COLS)
    loc = shots["location"].apply(lambda v: v if isinstance(v, list) else [None, None])
    shots["x"] = loc.str[0]
    shots["y"] = loc.str[1]
    return shots.drop(columns=["location"])


def run(s: Settings) -> dict:
    folder = s.raw / "statsbomb"
    shot_dir = folder / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for comp_id, season_id in s.modelling["xg_competitions"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        log.info("Competition %s season %s: %d matches", comp_id, season_id, len(matches))
        matches["competition_id"] = comp_id
        metas.append(
            matches[["match_id", "competition_id", "match_date", "home_team", "away_team"]]
        )
        for match_id in matches["match_id"]:
            path = shot_dir / f"{match_id}.csv"
            if path.exists():
                continue  # resumable: already downloaded
            shots = shots_for_match(int(match_id))
            write_atomic(path, shots.to_csv(index=False).encode("utf-8"))
    meta = pd.concat(metas, ignore_index=True)
    write_atomic(folder / "matches.csv", meta.to_csv(index=False).encode("utf-8"))
    n_files = len(list(shot_dir.glob("*.csv")))
    log.info("StatsBomb: %d matches listed, %d shot files on disk", len(meta), n_files)
    return {"matches": len(meta), "files": n_files}


if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings.logs)
    run(settings)
