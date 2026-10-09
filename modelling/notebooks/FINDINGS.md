# Shot findings

What `01_explore_shots.ipynb` shows about the StatsBomb 2015/16 shots, and what the xG model must do about it.

## Data coverage

`python -m data.ingest.statsbomb` found 1,551 matches, not the 1,826 of five full seasons: the
Bundesliga file holds only 34 matches and Ligue 1 has 377. That leaves 38,728 shots, not about 45,000.
The model card and the README state this.

## Findings

| # | What I saw | Cell | What the model must do |
| --- | --- | --- | --- |
| 1 | 38,728 shots, of which 10.2% are goals. | Load | The classes are unbalanced, so score with log loss and calibration, not accuracy. A constant 10% is the first baseline. |
| 2 | Penalties score 75.1% of the time (410 shots); open play 9.7%, free kicks 6.4%. | Shot types | Penalties are a different event. Leave them out of training and give them a fixed value. |
| 3 | Only 9 shots direct from a corner, and 89 with a body part of "Other". | Shot types, Body part | Too few to learn from. Their goal rates (33%, 35%) are noise; do not read anything into them. |
| 4 | Headers score 10.7%, right foot 9.5%, left foot 8.9%, before distance is taken into account. | Body part | Body part is a feature, but headers are taken from closer in, so it only means something next to distance. |
| 5 | The goal rate falls steeply with distance: 44.6% inside 6 yards, 17.8% at 6–12, 5.5% at 18–24, 1.6% at 30–40. | Distance | Distance is the main feature, and the fall is not a straight line, so the model works on log-odds. |
| 6 | Beyond 40 yards the rate rises again to 2.2%, on 360 shots. | Distance | Likely shots at an empty net. Without player positions the model cannot know; listed as a limit. |
| 7 | Goals cluster in a wedge in front of goal; shots from wide positions at the same distance rarely score. | Shot map | Distance alone is not enough. Add the angle between the two posts. |
| 8 | Non-penalty shots: 3,651 goals against 3,515.9 StatsBomb xG. | StatsBomb's xG | StatsBomb's own model is close to calibrated on this data (about 4% low), so it is a fair reference to compare with. |
