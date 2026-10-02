# Model card — Premier League match model (Dixon–Coles, version dc-1.0)

## Intended use
Pre-match home/draw/away probabilities and season-outcome probabilities for the
Premier League, for analysis and portfolio purposes. Not betting advice.

## Model
Dixon–Coles (1997), written from scratch: Poisson goals with attack and defence
ratings per team, home advantage, low-score correction (rho), time decay
(xi = 0.0019 per day, half-life about one year), three-year training window.
Fitted by maximum likelihood with SciPy L-BFGS-B.

## Data
Results: football-data.co.uk, 1993/94 to date. Fixtures: Understat.
Market-average closing odds (2019/20 onwards): evaluation only, never training.

## Evaluation (walk-forward, weekly refits, 2019/20–2025/26, N = 2,660)
| Model | RPS | Log loss |
|---|---|---|
| Dixon–Coles | 0.2039 | 0.9935 |
| Plain Poisson (rho = 0) | 0.2039 | 0.9932 |
| Equal-strength Poisson | 0.2338 | 1.0712 |
| League frequencies | 0.2339 | 1.0710 |
| Closing odds (benchmark) | 0.1968 | 0.9639 |

Gap to closing odds: 0.0071 (95% bootstrap interval 0.0046 to 0.0095). The model does
not beat the closing market. Calibration: [reports/figures/match_calibration.png](reports/figures/match_calibration.png)

RPS by season:

| Season | Dixon–Coles | Closing odds | Gap | Equal-strength Poisson | League frequencies |
|---|---|---|---|---|---|
| 2019/20 | 0.2014 | 0.1985 | 0.0028 | 0.2300 | 0.2303 |
| 2020/21 | 0.2165 | 0.2109 | 0.0055 | 0.2441 | 0.2435 |
| 2021/22 | 0.1970 | 0.1889 | 0.0080 | 0.2352 | 0.2351 |
| 2022/23 | 0.2116 | 0.1975 | 0.0141 | 0.2303 | 0.2308 |
| 2023/24 | 0.1890 | 0.1808 | 0.0082 | 0.2342 | 0.2341 |
| 2024/25 | 0.2018 | 0.1961 | 0.0057 | 0.2352 | 0.2358 |
| 2025/26 | 0.2100 | 0.2045 | 0.0055 | 0.2277 | 0.2279 |

Dixon–Coles beats both baselines in every season and is behind the closing odds in
every season. The low-score correction made no measurable difference: plain Poisson
scores the same RPS and a slightly better log loss.

## Live record
Predictions published at least a day before each match and committed to Git.
Only the first prediction per match is scored. Ledger: ledger/ledger_2026-27.csv

As of 2 October 2026 no match has been scored. The first run fell in an international
break, with no fixture inside the 7-day prediction window, so the ledger starts with the
10 October matchweek.

## Limitations
- Uses goals only: no injuries, line-ups, transfers or manager changes.
- A promoted team with no recent data starts at the average of the three weakest teams.
- That prior applies only before a team's first match. After a few matches the rating is
  fitted without any regularisation, so it is overconfident: on 2 October 2026, five
  matches into the season, Coventry's relegation probability was 100% and Hull were
  fifth on expected points.
- The season simulation breaks ties on points, goal difference and goals, then at random;
  the real rules use head-to-head results.
- Works in whole days; kick-off times are not used.
- Does not beat closing odds.
