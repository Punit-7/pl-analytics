# Model card — Football entity recognition and linking (version ner-1.0)

## Intended use
Finding players, clubs and stadiums in English football writing and linking each one to
an ID in this project's warehouse, for analysis and portfolio purposes. Not for
identifying people in any other context.

## Data
400 paragraphs from 238 English Wikipedia articles on Premier League club seasons,
2015/16 to 2026/27, labelled by one annotator in Label Studio with written guidelines
([ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md)). Consistency was not measured: no
self-agreement recheck was done.

Articles, not paragraphs, were assigned to splits, so no article is in two splits.
Train and dev paragraphs were pre-annotated by the gazetteer and corrected; test
paragraphs were labelled from scratch.

| Split | Paragraphs | Articles | PLAYER | TEAM | VENUE |
|---|---|---|---|---|---|
| Train | 275 | 102 | 1,205 | 1,482 | 130 |
| Dev | 50 | 16 | 133 | 226 | 12 |
| Test | 75 | 33 | 219 | 306 | 22 |

76 labels had their edges cleaned automatically (possessive `'s`, punctuation, leading
`the`). 4 spans do not line up with token edges and 18 spans overlap another span.

## Models compared
- **Gazetteer**: exact lookup of knowledge-base names with spaCy's `PhraseMatcher`.
- **spaCy `en_core_web_sm`**: pretrained, not fine-tuned; PERSON → PLAYER, ORG → TEAM,
  FAC → VENUE.
- **Fine-tuned `distilroberta-base`**: token classification over 7 BIO tags. Learning
  rate 5e-5, batch size 8, weight decay 0.01, up to 4 epochs, maximum 384 subword tokens,
  seed 2026, trained on a CPU. The best epoch by dev F1 was kept: epoch 3, dev F1 0.912
  (0.720 after epoch 1, 0.904 after epoch 2).

## Results (75 test paragraphs, 547 gold spans, exact match on start, end and label)
| System | Precision | Recall | F1 | Macro F1 |
|---|---|---|---|---|
| Gazetteer | 0.817 | 0.671 | 0.737 | 0.726 |
| spaCy `en_core_web_sm` | 0.465 | 0.437 | 0.451 | 0.380 |
| Fine-tuned DistilRoBERTa | 0.860 | 0.947 | 0.902 | 0.850 |

F1 by label:

| System | PLAYER | TEAM | VENUE |
|---|---|---|---|
| Gazetteer | 0.743 | 0.735 | 0.700 |
| spaCy `en_core_web_sm` | 0.601 | 0.350 | 0.189 |
| Fine-tuned DistilRoBERTa | 0.911 | 0.909 | 0.731 |

The transformer beats both baselines on every label. Its gain is mostly recall: it
finds 518 of 547 gold spans, against 367 for the gazetteer. VENUE is its weakest label
(precision 0.633, recall 0.864) and rests on 22 test spans.

### Entity linking
Measured on 200 of the 547 gold test mentions, each checked by hand against the
knowledge base, so NER mistakes do not affect the score.

| | All | PLAYER | TEAM | VENUE |
|---|---|---|---|---|
| Mentions | 200 | 82 | 111 | 7 |
| Accuracy (a correct NIL counts) | 0.925 | 0.854 | 0.973 | 1.000 |

Every link the linker made was correct (precision of links 1.000). It answered NIL for
30% of mentions; all 15 errors are mentions it left as NIL although the entity is in the
knowledge base. The gold IDs were filled in with the linker's suggestion visible.

Over the whole corpus the model found 28,344 mentions in 3,030 paragraphs and linked 81%
of them (PLAYER 80.7%, TEAM 80.5%, VENUE 82.2%).

## Error analysis
All 113 test errors (84 false positives, 29 false negatives) were read and sorted:

| Category | Errors | Example |
|---|---|---|
| Test label missed a real entity; the model found it | 47 | `Stoke`, `Arsenal`, `PSV` repeated later in a paragraph |
| Test label has the wrong edge, type or a nested span | 25 | `Norwich` labelled inside `Norwich City`; `Trossard` labelled TEAM |
| Model tagged a manager, referee or presenter as PLAYER | 19 | `Arteta` (8), `Mark Hughes`, `Skomina` |
| Model missed an entity | 8 | `AFC Wimbledon`, `BSC Young Boys`, `Havertz` |
| Model got the edge wrong | 8 | `Bournemouth` for `AFC Bournemouth`; `Yokohama F` + `Marinos` |
| Model tagged something that is not an entity | 4 | `England`, `Albania`, `Emirates` in `Emirates Cup` |
| Model gave the wrong type | 2 | `Gyökeres` as TEAM |

72 of the 113 errors come from the test labels, not the model, so the reported precision
of 0.860 understates it. The scores above are against the labels as they stand; they have
not been recomputed on corrected labels.

The manager errors have a cause in the training labels. The knowledge base holds Mikel
Arteta as a 2015/16 player, the gazetteer suggested him as PLAYER in later Arsenal
articles, and 10 of those suggestions were accepted in train and dev. The model learned
it: in the full corpus `Arteta` is tagged PLAYER 192 times.

## Limitations
- One annotator; consistency not measured. The test labels are incomplete (see above).
- Wikipedia season articles are not match reports: the style is drier, with more squad
  lists, cup draws and pre-season fixtures.
- VENUE is rare: 130 training spans and 22 test spans.
- Managers, referees and presenters are sometimes tagged PLAYER.
- A surname shared by two players in a season is linked only when the article's club
  settles it; otherwise the answer is NIL.
- The knowledge base holds Premier League players since 2015/16, the clubs in the
  warehouse and 39 stadiums. Foreign and lower-league clubs become NIL (`Real Madrid`,
  `Barcelona`), as do the nicknames `Blues` and `Reds`, which several clubs share.
- Player candidates come from the article's season only. A player named in a season when
  he was not in a Premier League squad becomes NIL, and so does a name spelt differently
  from Understat (`Ødegaard`, `Gabriel Magalhães`).
- Paragraphs longer than 384 subword tokens are cut: 85 of 3,030 in the corpus had their
  ends left untagged.

## Licence and attribution
Text from English Wikipedia, by Wikipedia contributors, licensed CC BY-SA 4.0. Every
article revision used is listed in [data/SOURCES.csv](data/SOURCES.csv). The labelled
dataset is released under the same licence: [data/labelled/LICENSE.md](data/labelled/LICENSE.md).
Model weights are not published. Player names and IDs: Understat via soccerdata.
