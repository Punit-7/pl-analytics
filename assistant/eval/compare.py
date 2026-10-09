"""Compare the latest P4 evaluation reports with a saved reference copy.

python -m assistant.eval.compare --save   keep today's reports as the reference (do this once)
python -m assistant.eval.compare          compare the latest reports with the reference
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from assistant.eval.common import REPORTS

REFERENCE = REPORTS / "reference"
TOLERANCE = 0.05  # a drop of this much or less is never flagged
WATCHED = {  # report file -> where the headline number is in it
    "sql_eval_repair.json": ("execution_accuracy",),
    "route_eval.json": ("tool_selection_accuracy",),
    "rag_eval.json": ("hybrid", "recall_at_5"),
}


def headline(report: dict, keys: tuple) -> tuple[float, float | None]:
    """The number, and the lower end of its 95% interval when the report has one."""
    value = report
    for key in keys:
        value = value[key]
    if isinstance(value, dict):
        return value["value"], value.get("low_95")
    return value, None


def compare(old: dict, new: dict, keys: tuple) -> dict:
    """Flag a drop below the old 95% interval and more than 0.05 below the old value."""
    old_value, old_low = headline(old, keys)
    new_value, _ = headline(new, keys)
    limit = old_value - TOLERANCE if old_low is None else min(old_low, old_value - TOLERANCE)
    return {
        "metric": ".".join(keys),
        "reference": old_value,
        "latest": new_value,
        "limit": round(limit, 3),
        "dropped": new_value < limit,
        "model_changed": old.get("model") != new.get("model"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare P4 evaluations with the reference")
    parser.add_argument("--save", action="store_true", help="save the latest reports as reference")
    args = parser.parse_args(argv)
    if args.save:
        REFERENCE.mkdir(parents=True, exist_ok=True)
        for name in WATCHED:
            shutil.copy(REPORTS / name, REFERENCE / name)
        print(f"Saved {len(WATCHED)} reports as the reference")
        return 0
    rows = []
    for name, keys in WATCHED.items():
        old = json.loads((REFERENCE / name).read_text(encoding="utf-8"))
        new = json.loads((REPORTS / name).read_text(encoding="utf-8"))
        rows.append(compare(old, new, keys))
    for r in rows:
        flag = "DROPPED" if r["dropped"] else "ok"
        note = "  (the model is different)" if r["model_changed"] else ""
        print(f"{r['metric']:<24} {r['reference']:.3f} -> {r['latest']:.3f}  "
              f"limit {r['limit']:.3f}  {flag}{note}")  # fmt: skip
    return 1 if any(r["dropped"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
