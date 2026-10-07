"""Optional label audit: list likely misses and wrong label types for a quick review."""
import json
import re
from collections import Counter

import pandas as pd

from data.common.config import ROOT
from data.common.db import make_engine
from nlp.dataset import LABELLED
from nlp.models.gazetteer import Gazetteer, gazetteer_terms, load_kb


def overlaps(a: int, b: int, spans) -> bool:
    return any(a < e and s < b for s, e, _ in spans)


def main() -> None:
    kb = load_kb(make_engine())
    gaz = Gazetteer(gazetteer_terms(kb))
    kb_types = kb.groupby("alias")["entity_type"].agg(set).to_dict()
    tasks = json.loads((LABELLED / "export_main.json").read_text(encoding="utf-8"))
    rows, caps = [], Counter()
    for t in tasks:
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        text = t["data"]["text"]
        gold = [(r["value"]["start"], r["value"]["end"], r["value"]["labels"][0])
                for r in anns[-1]["result"] if r.get("type") == "labels"]
        for a, b, lab in gaz.predict(text):  # known names you did not label
            if not overlaps(a, b, gold):
                rows.append({"task_id": t["id"], "issue": "possible_miss", "text": text[a:b],
                             "suggested": lab, "context": text[max(0, a - 60):b + 60]})
        for a, b, lab in gold:  # labels whose type disagrees with the knowledge base
            types = kb_types.get(text[a:b], set())
            if types and lab not in types:
                rows.append({"task_id": t["id"], "issue": "possible_wrong_type",
                             "text": text[a:b], "suggested": "/".join(sorted(types)),
                             "context": text[max(0, a - 60):b + 60]})
        for m in re.finditer(r"(?<=[a-z,;] )[A-Z][\w'-]+", text):  # mid-sentence capitals
            if not overlaps(m.start(), m.end(), gold):
                caps[m.group()] += 1
    out = ROOT / "nlp" / "reports" / "label_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(rows)} flagged cases written to {out}")
    print("Frequent unlabelled capitalised words:",
          [w for w, n in caps.most_common(40) if n >= 5])


if __name__ == "__main__":
    main()