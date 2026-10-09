# P4 report — Tactical analyst assistant

## The question
How far does a free 3-billion-parameter model get as a football analyst, when the
database and this project's own models do the computing?

## The answer in four numbers
| | |
|---|---|
| Text-to-SQL execution accuracy | 16/40 = 0.400 (clean run) |
| Article search, hybrid recall@5 | 30/40 = 0.750 |
| Tool-selection accuracy | 25/30 = 0.833 |
| End-to-end answers correct | 10/15 = 0.667 |

Part of the way. `llama3.2:3b` on a CPU picks the right tool five times in six and
answers two questions in three correctly, at about 35 seconds each. Writing SQL is the
weak part: under half of the test queries return the right rows, and almost none of the
hard ones. Nothing it writes can change the database.

## Text-to-SQL
Execution accuracy on 40 test questions with gold SQL (14 easy, 16 medium, 10 hard).

| Mode | Accuracy | 95% interval | Easy | Medium | Hard | Mean seconds |
|---|---|---|---|---|---|---|
| baseline (table names only) | 3/40 = 0.075 | 0.000 – 0.157 | 0/14 | 2/16 | 1/10 | 17.9 |
| full (schema card + 6 examples) | 14/40 = 0.350 | 0.202 – 0.498 | 6/14 | 7/16 | 1/10 | 20.6 |
| repair (full + one retry on an error) | 16/40 = 0.400 | 0.248 – 0.552 | 7/14 | 8/16 | 1/10 | 30.1 |

- The schema card and examples are what matter: the baseline and full intervals do not
  overlap.
- The retry fixed 3 of 12 SQL errors and made 2 more answers correct, which is inside
  the interval, for about 10 more seconds per question.
- A second prompt with three more rules scored 21/40 = 0.525. Two of the rules were
  written after reading the test failures, so that number is optimistic and 16/40 stays
  the clean measurement.
- The largest error categories in repair mode are a wrong table or column (8 of 24), a
  wrong team name (6) and logic too hard for the model (6).

## Article search
40 questions written by the model from sampled paragraphs, reviewed by hand, searched
over 3,030 paragraphs.

| Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 (keyword baseline) | 13/40 = 0.325 | 27/40 = 0.675 | 0.478 |
| Embeddings (`all-minilm`) | 5/40 = 0.125 | 22/40 = 0.550 | 0.284 |
| Hybrid (reciprocal rank fusion) | 18/40 = 0.450 | 30/40 = 0.750 | 0.575 |

Hybrid is best on every measure, but by 3 questions in 40 at recall@5. Embeddings alone
lose to the keyword baseline. The test favours BM25, because the model wrote each
question while reading the paragraph and reused its words.

## Tool choice
25/30 = 0.833 (95% interval 0.700 – 0.967). All 17 questions for the three P2 tools and
all 3 off-topic questions were routed correctly. The 5 mistakes all involve the warehouse
tool: `query_warehouse` was expected 9 times and chosen 5 times.

## End to end
15 answers marked by hand: 10/15 correct, and 3/4 article answers grounded in their
paragraphs. By type: warehouse 3/5, match model 3/5, articles 3/4, off-topic 1/1. Mean
34.8 seconds. With 15 questions the interval is 0.428 – 0.905, so read the counts.

## Three example answers
**Good.** *What was Manchester City's goal difference in 2017/18?*

```sql
SELECT ROUND(SUM(goals_for) - SUM(goals_against)) AS goal_diff
FROM mart.fact_team_match WHERE team = 'Man City' AND season = '2017/18'
```

> Manchester City's goal difference in the 2017/18 season was 79.

The model used the warehouse's short team name, the database did the arithmetic, and the
sentence copies the number.

**Wrong: the wrong tool.** *How many points did Arsenal win in 2023/24?*

> I can only help with Premier League questions. I was unable to find the information
> you requested.

The model chose `search_articles`, not `query_warehouse`, where one query returns 89. The
third paragraph retrieved even says "They ended this season on 89 points", and the model
still refused, using the sentence meant for questions that are not about football.

**Wrong: the tool result misread.** *Who is most likely to be relegated this season?*

> Based on the simulated season outlook, Hull City is the most likely to be relegated
> this season with an expected rank of 5 and an expected points total of 58.889.

The right tool ran and returned Coventry with a relegation probability of 1.000; Hull's
is 0.002. The model picked a row from the middle of a 20-row result. The same thing
happened with defence ratings, where a lower number is better and it named Liverpool
(−0.012), not Arsenal (−0.474). A tool that returned only the top rows for the question
would remove this error.

## Why there is no vector database
A vector database is built to search millions of embeddings quickly. This corpus has
3,030 paragraphs of 384 numbers each. Multiplying that matrix by one query vector in
NumPy takes a few milliseconds and is exact. A vector database would add an installation
and no benefit.

## Limitations
- A small model that is often wrong: about half of warehouse questions get wrong SQL.
- About 35 seconds per question on a CPU.
- Each question stands alone; there is no conversation memory.
- No player-level statistics, and points ignore official deductions.
- Articles are Wikipedia club season pages from 2015/16 only.
- Small test sets (40, 40, 30 and 15 questions) with wide intervals, and one annotator.
- None of the four reviewed article answers wrote a citation marker, although the prompt
  asks for one. Citations show where to check; they do not prove a statement.
- Prompt injection through article text is reduced, not removed.
- No hosted model was run, so how much of the weakness is the small model is not
  measured.

The full list is in the [system card](../SYSTEM_CARD.md), and the per-question error
analysis in [FINDINGS.md](FINDINGS.md).

## What P5 adds
P4 runs from the command line and from Streamlit on one PC. P5 puts the project behind a
FastAPI service in Docker, schedules the pipeline and the weekly predictions, and
monitors them.
