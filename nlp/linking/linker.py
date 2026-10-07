"""Link entity mentions to knowledge-base IDs, or NIL."""

import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process


def normalise(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


class Linker:
    def __init__(self, kb: pd.DataFrame, threshold: float):
        kb = kb.copy()
        kb["norm"] = kb["alias"].map(normalise)
        self.threshold = threshold
        self.teams = kb[kb["entity_type"] == "TEAM"]
        self.venues = kb[kb["entity_type"] == "VENUE"]
        self.players = kb[kb["entity_type"] == "PLAYER"]

    def _fuzzy(self, m: str, candidates: pd.DataFrame) -> tuple[str | None, float]:
        if candidates.empty:
            return None, 0.0
        exact = candidates[candidates["norm"] == m]
        if not exact.empty:
            return exact.iloc[0]["entity_id"], 100.0
        best = process.extractOne(m, candidates["norm"].tolist(), scorer=fuzz.ratio)
        return candidates.iloc[best[2]]["entity_id"], float(best[1])

    def link(self, mention: str, label: str, season: str, team: str) -> tuple[str | None, float]:
        """Return (entity_id, score). The caller treats a score below the threshold as NIL."""
        m = normalise(mention)
        if label == "TEAM":
            if " " not in m:  # a one-word name such as 'united' may mean the article's own club
                own = self.teams[self.teams["team"] == team]
                if own["norm"].str.split().apply(lambda words: m in words).any():
                    return f"team:{team}", 90.0
            return self._fuzzy(m, self.teams)
        if label == "VENUE":
            return self._fuzzy(m, self.venues)
        season_players = self.players[self.players["season"] == season]
        exact = season_players[season_players["norm"] == m]
        if exact["entity_id"].nunique() > 1:  # a surname shared by several players
            same_club = exact[exact["team"] == team]
            if same_club["entity_id"].nunique() == 1:
                return same_club.iloc[0]["entity_id"], 95.0
            best = exact.sort_values("minutes", ascending=False).iloc[0]
            return best["entity_id"], 60.0  # a guess: below the threshold, so NIL
        if not exact.empty:
            return exact.iloc[0]["entity_id"], 100.0
        return self._fuzzy(m, season_players[season_players["alias_kind"] == "full"])

    def resolve(self, mention: str, label: str, season: str, team: str) -> tuple[str | None, float]:
        entity_id, score = self.link(mention, label, season, team)
        return (entity_id if score >= self.threshold else None), score
