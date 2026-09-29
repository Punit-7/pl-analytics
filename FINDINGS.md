# Data findings

Problems found in the source data, and how the pipeline handles them.

## Shots on target greater than total shots

Found by `test_values_in_valid_ranges` (Stage 6). football-data.co.uk has three matches where a side's
shots on target (`HST`/`AST`) exceed its total shots (`HS`/`AS`). The true values cannot be recovered
from the file.

| Season | Date | Match | Home shots / on target | Away shots / on target |
| --- | --- | --- | --- | --- |
| 2000/01 | 14/10/2000 | Coventry 2-1 Tottenham | 0 / 5 (bad) | 4 / 5 (bad) |
| 2000/01 | 13/04/2001 | Bradford 2-0 Charlton | 4 / 8 (bad) | 3 / 8 (bad) |
| 2021/22 | 15/08/2021 | Newcastle 2-4 West Ham | 17 / 3 | 8 / 9 (bad) |

**Handling:** `load_matches` in `data/build_warehouse.py` sets that side's shots and shots on target
to NULL and logs a warning with the count (2 home, 3 away). Goals and every other stat are kept.

## Extra columns in 2003/04 and 2004/05

Late in both seasons the source files add betting-odds columns, so 45 rows per season have more
fields than the header. Reading with `on_bad_lines="skip"` silently dropped them (90 matches).

**Handling:** `load_matches` cuts each long row to the header's width, so all 380 matches per season
load. `test_finished_seasons_are_complete` guards against this coming back.
