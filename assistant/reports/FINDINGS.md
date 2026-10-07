# P4 findings

## Stage 1 — model speed

Tokens per second from `python -m assistant.llm`: _not recorded yet — run it and write the number here._

## Stage 5 — text-to-SQL execution accuracy

Model `llama3.2:3b` on Ollama, temperature 0, seed 2026. 40 test questions (14 easy, 16 medium, 10 hard). Run on 7 October 2026 (UTC).

| Mode     | Accuracy      | 95% interval  | Easy | Medium | Hard | Mean seconds |
|----------|---------------|---------------|------|--------|------|--------------|
| baseline | 3/40 = 0.075  | 0.000 – 0.157 | 0/14 | 2/16   | 1/10 | 17.9         |
| full     | 14/40 = 0.350 | 0.202 – 0.498 | 6/14 | 7/16   | 1/10 | 20.6         |
| repair   | 16/40 = 0.400 | 0.248 – 0.552 | 7/14 | 8/16   | 1/10 | 30.1         |

Outcomes:

| Mode     | correct | wrong_result | sql_error | model_error |
|----------|---------|--------------|-----------|-------------|
| baseline | 3       | 18           | 19        | 0           |
| full     | 14      | 14           | 12        | 0           |
| repair   | 16      | 15           | 9         | 0           |

What the numbers say:

- The schema card and six examples help: baseline and full intervals do not overlap.
- The retry is not shown to help. It fixed 3 of 12 SQL errors (t13 and t23 became correct, t25 became a wrong result). Two extra correct answers is well inside the interval, and the retry costs about 10 more seconds per question.
- Hard questions are 1/10 in every mode. The model cannot write window functions, self-joins or subqueries reliably.
- The interval formula breaks down near 0: baseline easy is 0/14, so the count is reported, not an interval.

### Error analysis — repair mode, 24 questions not correct

| Category                    | Count | Questions                              |
|-----------------------------|-------|----------------------------------------|
| Wrong table or column       | 8     | t03, t08, t26, t27, t29, t35, t37, t39 |
| Wrong team name             | 6     | t04, t06, t09, t10, t12, t28           |
| Logic too hard              | 6     | t25, t33, t34, t36, t38, t40           |
| Wrong aggregation or filter | 4     | t16, t17, t24, t32                     |
| Missing filter              | 0     |                                        |
| Extra or missing column     | 0     |                                        |

- **Wrong table or column.** The model mixes the two fact tables: it uses `home_goals`, `home_xg` or `home_ht_goals` on `mart.fact_team_match` (t26, t37, t39), or counts matches and goals from `mart.fact_team_match` and gets double (t08, t29). In t27 it invented columns such as `home_match_date`.
- **Wrong team name.** The model copies the long name from the question ('Leicester City', 'Tottenham Hotspur', 'Leeds United') although the exact names are in the prompt. The query runs and returns nothing, so the retry never sees it. Five of the six are easy questions.
- **Logic too hard.** Consecutive wins (t33), first half against second half of a season (t34), season-to-season change (t36), running total (t38), home and away points per game (t40), home-win percentage (t25).
- **Wrong aggregation or filter.** `points > 80` in WHERE where HAVING SUM(points) was needed (t16); xG for and against added together (t17); total goals > 0 where both teams had to score (t24); no GROUP BY and the wrong OFFSET (t32).

### What to fix first

Wrong table or column is the largest category, but wrong team name is the cheapest to fix and would recover the most easy questions. Any prompt change is tuned on the 10 dev questions only; the 40 test questions are then run once more.

## Stage 5 — second prompt version (repair_v2)

Three rules were added to `RULES` in `assistant/schema.py`: use the short team name from the team list; `mart.fact_team_match` has no `home_`/`away_` columns and has two rows per match; a team's xG is `SUM(xg_for)`. Same model, settings and 40 test questions. Run on 8 October 2026.

| Mode      | Accuracy      | 95% interval  | Easy  | Medium | Hard | Mean seconds |
|-----------|---------------|---------------|-------|--------|------|--------------|
| repair    | 16/40 = 0.400 | 0.248 – 0.552 | 7/14  | 8/16   | 1/10 | 30.1         |
| repair_v2 | 21/40 = 0.525 | 0.370 – 0.680 | 10/14 | 10/16  | 1/10 | 23.5         |

- Seven questions became correct (t03, t06, t10, t17, t25, t28, t29) and two became wrong (t21, t23): a net gain of 5.
- This is not proof that the rules help. The intervals overlap, and 7 gains against 2 losses could happen by chance about 1 time in 5 (exact sign test, p = 0.18).
- **The second number is optimistic.** The team-name and table rules were written after reading the test failures, so the test set is no longer unseen for this prompt. Only the xG rule came from a dev question (d03). The first run, 16/40, is the clean measurement.
- The dev set could not check the rules: it stayed at 8/10 for all three prompt versions (`sql_eval_repair_dev_v1` to `_v3`), with d03 fixed and d08 broken by the last one.
- Hard questions are still 1/10. Rules do not fix them.

### Error analysis — repair_v2, 19 questions not correct

| Category                    | Before | After | Questions after              |
|-----------------------------|--------|-------|------------------------------|
| Wrong table or column       | 8      | 6     | t08, t12, t26, t27, t37, t40 |
| Wrong team name             | 6      | 2     | t04, t09                     |
| Logic too hard              | 6      | 5     | t33, t34, t36, t38, t39      |
| Wrong aggregation or filter | 4      | 5     | t16, t23, t24, t32, t35      |
| Missing filter              | 0      | 1     | t21                          |

- Team names: four of six fixed. 'Tottenham Hotspur' and 'Newcastle United' are still copied from the question.
- Table confusion moved, not vanished: t08 and t12 now use `mart.fact_match` but with columns that belong to `mart.fact_team_match` (`result`, `game_no`).
- The two losses are side effects: t21 dropped its season filter and t23 changed its xG arithmetic after the xG rule.

No more prompt changes are made against these 40 questions. A further fix needs new test questions.
