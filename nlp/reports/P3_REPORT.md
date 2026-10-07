# P3 report — Match report entity extraction

## The question
Can a small, self-labelled dataset train a useful football entity recognition model?

## The answer in three numbers
| | |
|---|---|
| Fine-tuned DistilRoBERTa, test F1 | 0.902 |
| Best baseline (gazetteer), test F1 | 0.737 |
| Entity linking accuracy | 0.925 (200 hand-checked mentions) |

Yes. 275 labelled training paragraphs were enough for the transformer to beat a
lookup of every known name by 0.165 F1, on 75 test paragraphs from articles it never
saw. The test labels are incomplete, so the true precision is higher than reported; see
Typical errors.

## Comparison
Exact match on start, end and label, 547 gold spans.

| System | Precision | Recall | F1 | Macro F1 | PLAYER F1 | TEAM F1 | VENUE F1 |
|---|---|---|---|---|---|---|---|
| Gazetteer | 0.817 | 0.671 | 0.737 | 0.726 | 0.743 | 0.735 | 0.700 |
| spaCy `en_core_web_sm` | 0.465 | 0.437 | 0.451 | 0.380 | 0.601 | 0.350 | 0.189 |
| Fine-tuned DistilRoBERTa | 0.860 | 0.947 | 0.902 | 0.850 | 0.911 | 0.909 | 0.731 |

- The gazetteer is precise but misses a third of the names: clubs outside the Premier
  League, shared surnames and spellings not in the knowledge base.
- spaCy's general model was not trained on football. It produces 147 false TEAMs
  against 96 correct ones, and finds 5 of the 22 stadiums.
- The transformer's gain is recall: 518 of 547 spans found, against 367 for the
  gazetteer.

## Typical errors
All 113 transformer errors were read. 72 come from the test labels and 41 from the model.

**The label missed a repeat (47 errors).** The model is right; the paragraph's later
mentions were not labelled.

> ...goals from Gabriel Jesus, Raheem Sterling and David Silva. **Stoke** briefly made a
> comeback with a goal from Diouf...

**A manager tagged PLAYER (19 errors).** Eight are `Arteta`, who is in the knowledge base
as a 2015/16 player and was left labelled PLAYER in 10 training and dev paragraphs.

> In the 64th minute **Arteta** handed a belated debut to summer signing Merino...

**A wrong edge (8 errors).**

> ...United travelled to Dean Court to face AFC **Bournemouth**, who were new to the
> Premier League.

The full table of categories is in the [model card](../MODEL_CARD_ner.md).

## Entity linking
On 200 gold test mentions checked by hand, accuracy is 0.925: PLAYER 0.854 (82
mentions), TEAM 0.973 (111), VENUE 1.000 (7). Every link made was correct. All 15 errors
are mentions left as NIL although the entity is in the knowledge base, such as
`Ødegaard` and `Gabriel Magalhães`, whose spellings differ from Understat's, and
`Glenn Whelan`, named in a Stoke article for the season after he left the league.

## The whole corpus
28,344 mentions in 3,030 paragraphs, 81% linked, stored in `nlp.mention`.

Most-mentioned players in the 2024/25 articles:

| Player | Mentions |
|---|---|
| Kai Havertz | 37 |
| David Raya | 32 |
| Bukayo Saka | 30 |
| Leandro Trossard | 29 |
| Declan Rice | 27 |
| Gabriel Martinelli | 27 |
| Mikel Merino | 26 |
| William Saliba | 21 |
| Jamie Vardy | 17 |
| Myles Lewis-Skelly | 16 |

Nine of the ten are Arsenal players. The 2024/25 Arsenal article has 114 paragraphs,
against 46 for the next longest, so this ranks Wikipedia's coverage, not the players.

Most-mentioned stadiums, all seasons:

| Stadium | Mentions |
|---|---|
| Emirates Stadium | 129 |
| Anfield | 109 |
| Old Trafford | 101 |
| Stamford Bridge | 91 |
| London Stadium | 73 |
| Wembley Stadium | 67 |
| Etihad Stadium | 59 |
| St Mary's Stadium | 42 |
| Elland Road | 37 |
| Selhurst Park | 29 |

## Limitations
- One annotator, consistency not measured, and incomplete test labels.
- Managers are sometimes tagged PLAYER: `Arteta` 192 times across the corpus.
- 19% of mentions are NIL. The most common are `Blues` (204) and `Reds` (156), which
  several clubs share, and foreign clubs such as `Real Madrid` (80) and `Barcelona` (69).
- VENUE scores rest on 22 test spans.
- 85 long paragraphs were cut at 384 subword tokens.
- Wikipedia season articles are not match reports.

## Next steps
- Correct the labels found in the error analysis, then rerun the dataset, baseline and
  transformer stages so the scores are measured against clean labels.
- Add foreign clubs and a manager list to the knowledge base.
- P4 will query `nlp.mention` together with the match and prediction tables.
