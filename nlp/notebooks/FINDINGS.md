# Text findings

What `01_explore_text.ipynb` showed about the corpus and the knowledge base, and what each finding
means for labelling and modelling.

## Corpus size and spread

3,030 paragraphs from 238 articles. 240 articles were downloaded; two have no paragraph of 200
characters or more. Paragraphs per season range from 98 (2026/27, still being written) and 116
(2019/20) to 378 (2022/23). Sampling for labelling is therefore spread evenly across seasons, not
drawn in proportion, so thin seasons are not under-represented.

## Paragraph length

The median paragraph is 462 characters; the middle half is 305 to 709. The longest is 4,048
characters, which is longer than the transformer's 384-subword limit, so the longest paragraphs will
be truncated in Stage 8.

## Sections

`Introduction` is the largest section (414 paragraphs). Month sections (`December` 149, `April` 125,
`January` 124 and so on) and cup sections (`EFL Cup` 136, `FA Cup` 114) follow, then `Pre-season`
(119) and `Pre-season and friendlies` (107). Pre-season and cup paragraphs name many clubs outside
the Premier League, which the knowledge base does not hold.

## Not every paragraph is match prose

Some paragraphs that pass the length filter are table notes or keys (`Players and squad numbers last
updated on...`, `Includes all competitive matches. The list is sorted by squad number...`), kit
descriptions or fixture-list announcements. They contain no entities and are labelled as empty.

## How often known names appear

In a 500-paragraph sample, 94% of paragraphs contain at least one knowledge-base club name and 59%
contain at least one player's full name. Clubs are the most common entity; venues are the rarest.

## Names outside the knowledge base

The sampled paragraphs mention academy players (`Louie Bradbury`, `Zac Watson`), lower-league clubs
(`Shrewsbury Town`, `Doncaster Rovers`) and foreign clubs (`Paris Saint-Germain`, `Roma`,
`Juventus`). None are in the knowledge base, so the gazetteer cannot find them and the linker must
answer NIL.

## Shared surnames

254 season-surname pairs are shared by two or more players in the same season, about 21 per season.
The worst cases have three or four owners: `Silva` (4 in 2018/19, 3 in 2021/22 and 2023/24), `Mendy`
(3 in 2020/21), `Jones` and `Davies` (3 in 2016/17), `Ward` and `Sánchez` (3 in 2018/19). A surname
alone cannot identify these players; linking needs the article's club and season.

## Players who became managers

The knowledge base holds every player since 2015/16, so a former player who later manages a club
still matches by name. `Arteta` (a player in 2015/16) is suggested as PLAYER in Arsenal articles from
2019/20 onwards, where he is the manager. The guidelines say managers are not labelled, so these
suggestions must be deleted.
