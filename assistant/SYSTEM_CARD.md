# System card — Tactical analyst assistant (version assistant-1.0)

## Intended use
Answering questions about Premier League results, this project's match predictions, and
club seasons described on Wikipedia. A portfolio demonstration, not a betting tool and
not a source of record.

## How it works
The language model chooses one tool, the code runs the tool, and the model words the
answer from the tool's result. The model never computes a number and is told never to
answer a football question from memory: PostgreSQL does the counting, P2's code computes
the probabilities, and the search code finds the paragraphs.

```text
                  question typed in the Streamlit page
                                     │
                                     v
                       agent: the model picks ONE tool
        ┌────────────────┬───────────┴─────┬────────────────────┐
        v                v                 v                    v
 query_warehouse    predict_match     season_outlook      search_articles
 text-to-SQL        team_ratings      reads the stored    BM25 and
        │           fits P2's         simulation          embeddings over
        v           Dixon-Coles       predictions.        nlp.document
 safe runner:       model             season_sim               │
 pl_reader only,         │                 │                   │
 read-only, 5 s,         │                 │                   │
 200 rows                │                 │                   │
        │                │                 │                   │
        v                v                 v                   v
      rows          probabilities     title, top-four,    numbered
                                      relegation chances  paragraphs
        └────────────────┴────────┬────────┴───────────────────┘
                                  v
             the model words the answer from the tool result only
                                  │
                                  v
     answer, SQL, rows and sources shown; one line added to traces.jsonl
```

## Models
| | |
|---|---|
| Chat model | `llama3.2:3b` (Q4_K_M) on Ollama 0.40.0, CPU only |
| Embedding model | `all-minilm` on Ollama, 384 numbers per vector |
| Temperature and seed | 0.0 and 2026 |
| Context window | 4,096 tokens |
| Step limit | 3 model calls per question |
| Version | `assistant-1.0` |

Measured on an Intel Core i5-3450 with 12 GB of RAM: 3.3 to 4.6 output tokens per second.

## Data
- **Warehouse**: `mart.fact_team_match`, `mart.fact_match` and `mart.fixture`, the only
  tables the model is told about. Last pipeline run: 5 October 2026.
- **Match model**: P2's Dixon–Coles model (`dc-1.0`), refitted when the assistant starts,
  and the stored season simulation in `predictions.season_sim` (simulated 5 October 2026).
- **Articles**: 3,030 paragraphs from 238 English Wikipedia club season articles,
  2015/16 to 2026/27, in `nlp.document` from P3. The exact revision of every article is
  in [`nlp/data/SOURCES.csv`](../nlp/data/SOURCES.csv).

The assistant reads all three and writes to none of them.

## Safety controls
Model-written SQL passes four layers:

| Layer | What it stops | Enforced by |
|---|---|---|
| `check_sql`: one statement, starting with `SELECT` or `WITH` | Obvious non-queries; two statements in one string | Python, before the database is contacted |
| Role `pl_reader` | Writing to any table; reading schemas it was not granted | PostgreSQL permissions |
| Read-only transaction | Any write, including one hidden inside a `WITH` clause | PostgreSQL |
| Statement timeout (5 s) and row limit (200) | Slow queries and huge results | PostgreSQL and `fetchmany` |

The first layer gives a clear error early. The other three are the real protection and
hold even if the first is wrong. Tests prove that a write is refused.

- **Read-only login**: the assistant connects only as `pl_reader`, and every transaction
  on that connection starts read-only. It never uses the `pl_app` password.
- **Step limit**: at most 3 model calls per question: one tool call, one retry after an
  error, and the answer.
- **Sources as quoted text**: the prompt tells the model that article paragraphs are
  quoted text and that instructions inside them must not be followed.
- **Traces**: one line per question in `assistant/logs/traces.jsonl` (git-ignored).

## Evaluation
All runs: `llama3.2:3b`, temperature 0, seed 2026, 7–8 October 2026. Intervals are
approximate 95% intervals, p ± 1.96 × standard error.

### 1. Text-to-SQL — execution accuracy
40 test questions with gold SQL (14 easy, 16 medium, 10 hard), kept back from the 10
development questions.

| Mode | Accuracy | 95% interval | Easy | Medium | Hard | Mean seconds |
|---|---|---|---|---|---|---|
| baseline (table names only) | 3/40 = 0.075 | 0.000 – 0.157 | 0/14 | 2/16 | 1/10 | 17.9 |
| full (schema card + 6 examples) | 14/40 = 0.350 | 0.202 – 0.498 | 6/14 | 7/16 | 1/10 | 20.6 |
| repair (full + one retry on an error) | 16/40 = 0.400 | 0.248 – 0.552 | 7/14 | 8/16 | 1/10 | 30.1 |
| repair_v2 (three more rules) | 21/40 = 0.525 | 0.370 – 0.680 | 10/14 | 10/16 | 1/10 | 23.5 |

- The schema card and examples help: the baseline and full intervals do not overlap.
- The retry is not shown to help: 2 more correct answers is well inside the interval.
- **repair_v2 is the prompt the assistant uses, and its number is optimistic.** Two of
  its three rules were written after reading the test failures, so the test set is no
  longer unseen for it. 16/40 is the clean measurement.
- Hard questions are 1/10 in every mode.

### 2. Article search — recall@5 and MRR
40 questions written by the model from sampled paragraphs and reviewed by hand. A hit
means the paragraph the question was written from is in the results.

| Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 (keyword baseline) | 13/40 = 0.325 | 27/40 = 0.675 | 0.478 |
| Embeddings (`all-minilm`) | 5/40 = 0.125 | 22/40 = 0.550 | 0.284 |
| Hybrid (reciprocal rank fusion) | 18/40 = 0.450 | 30/40 = 0.750 | 0.575 |

The assistant uses hybrid search. Its lead over BM25 is 3 questions in 40, which is
inside the noise for a test this small.

### 3. Tool choice — tool-selection accuracy
30 labelled questions: 25/30 = 0.833 (0.700 – 0.967).

| Expected tool | Correct |
|---|---|
| `query_warehouse` | 5/9 |
| `search_articles` | 4/5 |
| `predict_match` | 5/5 |
| `team_ratings` | 4/4 |
| `season_outlook` | 4/4 |
| none (not about football) | 3/3 |

### 4. The whole assistant — end-to-end review
15 answers marked by hand: 10/15 correct (0.667, interval 0.428 – 0.905). Of the 4
article answers, 3/4 were grounded in the paragraphs retrieved. Mean 34.8 seconds per
question.

| Question type | Correct |
|---|---|
| Warehouse (SQL) | 3/5 |
| Match model | 3/5 |
| Articles | 3/4 |
| Not about football | 1/1 |

No hosted model was run, so there is no hosted comparison.

## Error analysis
### Text-to-SQL (repair mode, 24 of 40 not correct; repair_v2, 19 of 40)
| Category | repair | repair_v2 |
|---|---|---|
| Wrong table or column | 8 | 6 |
| Wrong team name | 6 | 2 |
| Logic too hard | 6 | 5 |
| Wrong aggregation or filter | 4 | 5 |
| Missing filter | 0 | 1 |

- **Wrong table or column**: the model mixes the two fact tables, such as `home_goals` on
  `mart.fact_team_match`, or counts matches from the two-rows-per-match table and gets
  double.
- **Wrong team name**: it copies the long name from the question ('Tottenham Hotspur')
  although the exact names are in the prompt. The query runs and returns nothing, so the
  retry never sees it.
- **Logic too hard**: consecutive wins, running totals, season-to-season change. Window
  functions, self-joins and subqueries are beyond this model.

Details and question IDs are in [`reports/FINDINGS.md`](reports/FINDINGS.md).

### Tool choice (5 mistakes in 30)
| Expected → chosen | Count | Questions |
|---|---|---|
| `query_warehouse` → none | 2 | Arsenal's home goals in 2022/23; Leicester's points in 2015/16 |
| `query_warehouse` → `search_articles` | 2 | Liverpool's record against Everton; Newcastle's total xG in 2023/24 |
| `search_articles` → `query_warehouse` | 1 | "Tell me about Luton Town's season" |

All five involve the warehouse tool. The four prediction and rating tools were chosen
correctly every time.

### End-to-end (5 wrong in 15)
- **Wrong tool (2)**: two warehouse questions went to `search_articles` and got a refusal.
- **Tool result misread (2)**: the tool returned the right data and the model worded it
  wrongly: Liverpool named as the best defence (it is Arsenal; a lower rating is better),
  and Hull named as most likely to be relegated (it is Coventry).
- **Refusal despite a good source (1)**: the paragraph with the answer was retrieved and
  the model still said it could only help with Premier League questions.
- None of the four article answers wrote a `[1]`-style citation marker, although the
  prompt asks for one. The page still lists the sources under the answer.

## Limitations
- A small model that is often wrong. About half of warehouse questions get wrong SQL,
  and a right tool result can still be worded wrongly.
- Slow on a CPU: about 35 seconds per question, and 85 to 90 seconds for the first one.
- Each question stands alone. The agent does not remember earlier questions.
- No player-level statistics in the warehouse, so "who scored the most goals" cannot be
  answered with SQL.
- Points are computed from results and ignore official deductions: Everton show 48 points
  for 2023/24 here, more than in the official table.
- Article coverage is limited to Wikipedia club season pages from 2015/16.
- The search test favours keyword search: the model wrote each question while reading
  the paragraph, so questions reuse the paragraph's words.
- Small test sets with wide intervals: 40, 40, 30 and 15 questions. One annotator marked
  the end-to-end answers.
- Citations show where to check. They do not prove a statement.
- Prompt injection through article text is reduced by the prompt, not removed. The
  database controls do not depend on the model.

## Licence and attribution
- Article text: [English Wikipedia](https://en.wikipedia.org/), by Wikipedia
  contributors, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Sources are listed in [`nlp/data/SOURCES.csv`](../nlp/data/SOURCES.csv). The chat page
  shows the article title, its URL and the licence under every article answer.
- Results and match statistics: [football-data.co.uk](https://www.football-data.co.uk/).
  Expected goals and fixtures: [Understat](https://understat.com/).
