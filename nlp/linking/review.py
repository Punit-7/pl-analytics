"""Make a linking review sheet from gold test mentions, then score the linker against it."""

import argparse
import json

import pandas as pd
from sqlalchemy import text

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from nlp.linking.linker import Linker
from nlp.models.baselines import read_split
from nlp.models.gazetteer import load_kb

REVIEW = ROOT / "nlp" / "data" / "linking_review.csv"


def make(s) -> None:
    engine = make_engine()
    kb = load_kb(engine)
    names = kb.drop_duplicates("entity_id").set_index("entity_id")["name"]
    meta = pd.read_sql(text("SELECT doc_id, season, team FROM nlp.document"), engine)
    meta = meta.set_index("doc_id")
    linker = Linker(kb, s.nlp["link_threshold"])
    rows = []
    for r in read_split("test"):
        season, team = meta.loc[r["doc_id"], ["season", "team"]]
        for a, b, label in r["spans"]:
            eid, score = linker.link(r["text"][a:b], label, season, team)
            rows.append(
                {
                    "doc_id": r["doc_id"],
                    "mention": r["text"][a:b],
                    "label": label,
                    "season": season,
                    "team": team,
                    "suggested_id": eid,
                    "suggested_name": names.get(eid, ""),
                    "score": round(score, 1),
                    "gold_id": "",
                    "context": r["text"][max(0, a - 60) : b + 60],
                }
            )
    pd.DataFrame(rows).to_csv(REVIEW, index=False, encoding="utf-8-sig")
    print(f"{len(rows)} mentions written to {REVIEW}")


def score(s) -> None:
    df = pd.read_csv(REVIEW, encoding="utf-8-sig").fillna("")
    if (df["gold_id"] == "").any():
        raise SystemExit("Fill gold_id for every row first (an entity_id or NIL).")
    threshold = s.nlp["link_threshold"]
    df["pred_id"] = [
        sid if sc >= threshold else "NIL"
        for sid, sc in zip(df["suggested_id"], df["score"], strict=True)
    ]
    df["pred_id"] = df["pred_id"].replace("", "NIL")
    correct = df["pred_id"] == df["gold_id"]
    linked = df["pred_id"] != "NIL"
    result = {
        "mentions": int(len(df)),
        "accuracy": float(correct.mean()),
        "precision_of_links": float(correct[linked].mean()) if linked.any() else 0.0,
        "nil_rate": float((~linked).mean()),
        "accuracy_by_label": df.assign(ok=correct).groupby("label")["ok"].mean().round(3).to_dict(),
    }
    out = ROOT / "nlp" / "reports" / "linking_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    if args.make:
        make(settings)
    if args.score:
        score(settings)
