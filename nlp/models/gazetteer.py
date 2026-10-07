"""Rule-based NER: exact lookup of knowledge-base names with spaCy's PhraseMatcher."""
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans
from sqlalchemy import text

# One-word aliases that are ordinary English words or shared by several clubs
COMMON_WORDS = {"City", "United", "Forest", "Villa", "Palace", "Town", "Park", "Albion"}


def load_kb(engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM nlp.kb_entity"), engine)


def gazetteer_terms(kb: pd.DataFrame) -> dict[str, set[str]]:
    terms = {"PLAYER": set(), "TEAM": set(), "VENUE": set()}
    for r in kb[kb["alias_kind"] != "surname"].itertuples():
        if r.alias not in COMMON_WORDS:
            terms[r.entity_type].add(r.alias)
    surnames = kb[kb["alias_kind"] == "surname"]
    owners = surnames.groupby("alias")["entity_id"].nunique()
    unique = {a for a in owners[owners == 1].index if len(a) > 2 and a not in COMMON_WORDS}
    terms["PLAYER"] |= unique - terms["TEAM"] - terms["VENUE"]
    return terms


class Gazetteer:
    def __init__(self, terms: dict[str, set[str]]):
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        for label, names in terms.items():
            self.matcher.add(label, list(self.nlp.tokenizer.pipe(sorted(names))))

    def predict(self, text: str) -> list[tuple[int, int, str]]:
        doc = self.nlp.make_doc(text)
        spans = [Span(doc, start, end, label=match_id)
                 for match_id, start, end in self.matcher(doc)]
        return [(s.start_char, s.end_char, s.label_) for s in filter_spans(spans)]