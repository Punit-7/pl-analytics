"""Exact-match span precision, recall and F1; also the self-agreement check."""

import argparse

from data.common.config import ROOT

LABELS = ["PLAYER", "TEAM", "VENUE"]


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def span_scores(gold: list, pred: list, labels=LABELS) -> dict:
    counts = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in labels}
    for g, p in zip(gold, pred, strict=True):
        g = {tuple(x) for x in g if x[2] in counts}
        p = {tuple(x) for x in p if x[2] in counts}
        for _, _, lab in g & p:
            counts[lab]["tp"] += 1
        for _, _, lab in p - g:
            counts[lab]["fp"] += 1
        for _, _, lab in g - p:
            counts[lab]["fn"] += 1
    per_label = {lab: prf(**c) for lab, c in counts.items()}
    totals = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")}
    return {
        "micro": prf(**totals),
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(per_label),
        "per_label": per_label,
    }


def main(argv=None) -> None:
    from nlp.dataset import load_export

    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", action="store_true")
    args = parser.parse_args(argv)
    if args.agreement:
        folder = ROOT / "nlp" / "data" / "labelled"
        first = load_export(folder / "export_main.json")
        second = load_export(folder / "export_recheck.json")
        both = sorted(set(first) & set(second))
        scores = span_scores([first[d]["spans"] for d in both], [second[d]["spans"] for d in both])
        print(f"{len(both)} paragraphs | self-agreement F1 {scores['micro']['f1']:.3f}")
        for lab, v in scores["per_label"].items():
            print(f"  {lab:<7} F1 {v['f1']:.3f}  (agree {v['tp']}, differ {v['fp'] + v['fn']})")


if __name__ == "__main__":
    main()
