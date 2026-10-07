"""Convert a Label Studio export into token/BIO JSONL files, split by article."""

import json
import logging
import re
from collections import Counter
from pathlib import Path

from data.common.config import ROOT, load_settings
from data.common.logging_setup import setup_logging

log = logging.getLogger("nlp.dataset")
LABELLED = ROOT / "nlp" / "data" / "labelled"
SPLITS = ROOT / "nlp" / "data" / "splits"
TOKEN = re.compile(r"\w+(?:-\w+)*|[^\w\s]")  # words (with hyphens) or single punctuation marks


def load_export(path: Path) -> dict[str, dict]:
    out = {}
    for task in json.loads(path.read_text(encoding="utf-8")):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        spans = sorted(
            {
                (r["value"]["start"], r["value"]["end"], r["value"]["labels"][0])
                for r in anns[-1]["result"]
                if r.get("type") == "labels"
            }
        )
        d = task["data"]
        out[d["doc_id"]] = {
            "text": d["text"],
            "split": d.get("split"),
            "article": d.get("article"),
            "spans": spans,
        }
    return out


STRIP_CHARS = set(" \t\n.,;:!?\"'“”‘’()[]")


def clean_span(text: str, start: int, end: int, label: str) -> tuple[int, int]:
    """Apply the guidelines' boundary rules: no leading 'the', no possessive 's
    (players and clubs only), no surrounding spaces or punctuation."""
    for article in ("the ", "The "):
        if text.startswith(article, start) and end - start > len(article):
            start += len(article)
    if label in ("PLAYER", "TEAM"):  # venues such as St Mary's keep their 's
        for suffix in ("'s", "’s"):
            if text.endswith(suffix, start, end) and end - start > len(suffix):
                end -= len(suffix)
    while start < end and text[start] in STRIP_CHARS:
        start += 1
    while end > start and text[end - 1] in STRIP_CHARS:
        end -= 1
    return start, end


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def to_bio(tokens, spans) -> tuple[list[str], int]:
    tags, misaligned = ["O"] * len(tokens), 0
    for start, end, label in spans:
        inside = [i for i, (_, a, b) in enumerate(tokens) if a >= start and b <= end]
        if not inside or tokens[inside[0]][1] != start or tokens[inside[-1]][2] != end:
            misaligned += 1
        for k, i in enumerate(inside):
            tags[i] = ("B-" if k == 0 else "I-") + label
    return tags, misaligned


def bio_to_spans(tokens, tags) -> list[tuple[int, int, str]]:
    spans, current = [], None
    for (_, a, b), tag in zip(tokens, tags, strict=True):
        starts_new = tag.startswith("B-") or (
            tag.startswith("I-") and (current is None or current[2] != tag[2:])
        )
        if starts_new:
            if current:
                spans.append(tuple(current))
            current = [a, b, tag[2:]]
        elif tag.startswith("I-"):
            current[1] = b
        else:
            if current:
                spans.append(tuple(current))
            current = None
    if current:
        spans.append(tuple(current))
    return spans


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    data = load_export(LABELLED / "export_main.json")
    files = {
        name: open(SPLITS / f"{name}.jsonl", "w", encoding="utf-8")
        for name in ("train", "dev", "test")
    }
    label_counts, total_misaligned, cleaned = Counter(), 0, 0
    for doc_id, d in data.items():
        spans = set()
        for a, b, lab in d["spans"]:
            na, nb = clean_span(d["text"], a, b, lab)
            cleaned += (na, nb) != (a, b)
            if na < nb:
                spans.add((na, nb, lab))
        d["spans"] = sorted(spans)  # also removes exact duplicates
        tokens = tokenize(d["text"])
        tags, misaligned = to_bio(tokens, d["spans"])
        total_misaligned += misaligned
        label_counts.update(f"{d['split']}:{lab}" for _, _, lab in d["spans"])
        row = {
            "doc_id": doc_id,
            "article": d["article"],
            "text": d["text"],
            "tokens": [t for t, _, _ in tokens],
            "offsets": [[a, b] for _, a, b in tokens],
            "tags": tags,
            "spans": [list(sp) for sp in d["spans"]],
        }
        files[d["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    log.info(
        "%d labelled paragraphs, %d labels cleaned, %d misaligned spans",
        len(data),
        cleaned,
        total_misaligned,
    )
    log.info("entities by split and label: %s", dict(sorted(label_counts.items())))


if __name__ == "__main__":
    main()
