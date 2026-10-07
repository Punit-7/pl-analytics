# Licence — labelled NER dataset

The paragraphs in this folder (`tasks.json`, `export_main.json`) and in `../splits/` are adapted
from English Wikipedia articles about Premier League club seasons, written by Wikipedia
contributors. Every article used is listed in [`../SOURCES.csv`](../SOURCES.csv) with its title,
revision ID and a link to that exact revision.

The source text is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0).

## What was changed

- Each article was split into paragraphs; headings, reference sections and paragraphs shorter than
  200 characters were dropped.
- 400 paragraphs were sampled and labelled by hand with PLAYER, TEAM and VENUE spans, following
  [`../../ANNOTATION_GUIDELINES.md`](../../ANNOTATION_GUIDELINES.md).
- The split files add tokens and BIO tags. The text itself is unchanged.

## Licence of this dataset

The labelled dataset, including the labels, is released under the same licence, CC BY-SA 4.0. If
you reuse it, credit the Wikipedia contributors through `SOURCES.csv`, credit this project for the
labels, and share your adaptation under the same licence.

## Model weights

The trained model files are not published; `nlp/artifacts/` is kept out of Git. Whether CC BY-SA
applies to model weights trained on CC BY-SA text is legally unsettled.
