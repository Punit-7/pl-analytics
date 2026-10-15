# Data sources: terms checks

Each check is dated. Repeat a check before relying on a source again, and add a new dated row rather than editing an old one.

| Date | Source | What was checked | Finding | Action |
|---|---|---|---|---|
| 2026-10-15 | football-data.co.uk | `https://www.football-data.co.uk/robots.txt` | Empty: no crawl rules | None; the ingest downloads one CSV per season, finished seasons once |
| 2026-10-15 | Understat (through soccerdata) | `https://understat.com/robots.txt`; home page for a terms-of-use link | `User-agent: *` / `Disallow: /`: every page is disallowed for every crawler. No terms-of-use link found on the home page | **Open decision.** The P1 guide says to set `understat_enabled = false` when scraping is not allowed. It is still `true` in `config.toml`. Turning it off blanks the xG visuals and stops new Understat downloads; P3's player knowledge base also comes from Understat |
| 2026-10-15 | Fantasy Premier League API | `https://fantasy.premierleague.com/robots.txt` | No robots file is served (the site returns its HTML page) | None; the API is undocumented, so the ingest keeps to two requests per run (P2 FPL appendix) |
| 2026-10-02 | StatsBomb Open Data | Licence in StatsBomb's open-data repository (P2 guide) | Free use with credit to StatsBomb as the data source | Credited in the README's Data credits section |
| 2026-10-02 | English Wikipedia | Licence of article text (P3 guide) | CC BY-SA 4.0: credit and share-alike | Every revision listed in `nlp/data/SOURCES.csv`; labelled data under the same licence |
