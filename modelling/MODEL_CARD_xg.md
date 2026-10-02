# Model card — Expected goals (xG) model

## Intended use
Estimating the probability that a non-penalty shot is scored, for analysis and portfolio
purposes. Not for current-season player or team evaluation: it has not been validated
on current data.

## Model
Two models on the same features, with two baselines for comparison:

- Logistic regression on standardised features.
- LightGBM (gradient-boosted trees), with early stopping on a validation split of the
  training matches.
- Baselines: the training goal rate for every shot, and logistic regression on distance only.

## Data
StatsBomb Open Data 2015/16: Premier League, La Liga, Bundesliga, Serie A and Ligue 1.
Data provided by StatsBomb.

1,551 matches and 38,728 shots (38,318 without penalties). The open data held only 34
Bundesliga matches and 377 Ligue 1 matches, not the 306 and 380 played, so the
Bundesliga is barely represented.

## Features
Distance, angle, body part, technique, shot type, play pattern, under pressure, first
time. Penalties are excluded. The shot outcome and StatsBomb's own xG are banned as
features; StatsBomb's xG is used only as a reference score.

## Evaluation (test set: 20% of matches, 311 matches, 7,725 shots, 750 goals)
| Model | Log loss | Brier | AUC | Total xG | ECE |
|---|---|---|---|---|---|
| Constant goal rate (baseline) | 0.3187 | 0.0877 | 0.500 | 732.5 | |
| Distance only (baseline) | 0.2816 | 0.0805 | 0.751 | 736.5 | |
| Logistic regression | 0.2646 | 0.0755 | 0.793 | 747.5 | 0.0108 |
| LightGBM | 0.2631 | 0.0753 | 0.798 | 750.3 | 0.0086 |
| StatsBomb xG (reference) | 0.2494 | 0.0712 | 0.824 | 718.0 | 0.0104 |

Matches, not shots, were split, so no match is in both the training and the test set.
StatsBomb's model is better because it uses player positions (freeze frames), which
this model does not.

Calibration: [reports/figures/xg_calibration.png](reports/figures/xg_calibration.png)

LightGBM log loss by league:

| League | Test shots | Log loss |
|---|---|---|
| Premier League | 1,999 | 0.2913 |
| La Liga | 1,778 | 0.2645 |
| Bundesliga | 150 | 0.3024 |
| Serie A | 1,948 | 0.2357 |
| Ligue 1 | 1,850 | 0.2569 |

The Bundesliga figure rests on 6 test matches.

Top features by mean absolute SHAP value (LightGBM): distance 0.554, angle 0.444,
header 0.194, normal technique 0.110, from a corner 0.074.
Plots: [reports/figures/xg_shap_summary.png](reports/figures/xg_shap_summary.png),
[reports/figures/xg_shap_distance.png](reports/figures/xg_shap_distance.png)

## Limitations
- One season only (2015/16), so playing styles may have changed.
- No freeze-frame (player position) features.
- Not validated on current-season data, because free current-season event data does not exist.
- Almost no Bundesliga data.
- A penalty is not modelled; use the historical conversion rate for it.
