"""係り受け解析。"""
from __future__ import annotations

from typing import Optional

from .utils import load_spacy_model, optional_import


class DependencyParser:
    """GiNZA / spaCy 係り受け解析。"""

    def __init__(self, engine: str = "ginza"):
        self.engine = engine
        self.nlp = None

    def _load(self):
        if self.nlp is not None:
            return
        optional_import("spacy", "係り受け")
        if self.engine == "ginza":
            self.nlp = load_spacy_model("ja_ginza")
        elif self.engine == "spacy_en":
            try:
                self.nlp = load_spacy_model("en_core_web_sm")
            except OSError:
                raise RuntimeError("`python -m spacy download en_core_web_sm` が必要です")
        else:
            raise ValueError(f"unknown engine: {self.engine}")

    def parse(self, text: str) -> list[dict]:
        self._load()
        doc = self.nlp(text)
        return [
            {
                "i": t.i,
                "token": t.text,
                "lemma": t.lemma_,
                "pos": t.pos_,
                "head": t.head.text,
                "head_i": t.head.i,
                "dep": t.dep_,
            }
            for t in doc if not t.is_space
        ]

    def extract_pairs(
        self,
        text: str,
        relation: str = "verb_obj",
    ) -> list[tuple[str, str]]:
        """relation: 'verb_obj' | 'adj_noun' | 'subject_verb'"""
        self._load()
        doc = self.nlp(text)
        pairs = []
        for t in doc:
            if relation == "verb_obj":
                if t.dep_ in ("obj", "dobj") and t.head.pos_ == "VERB":
                    pairs.append((t.head.lemma_, t.lemma_))
            elif relation == "subject_verb":
                if t.dep_ in ("nsubj", "nsubj:pass") and t.head.pos_ == "VERB":
                    pairs.append((t.lemma_, t.head.lemma_))
            elif relation == "adj_noun":
                if t.pos_ == "ADJ" and t.head.pos_ == "NOUN":
                    pairs.append((t.lemma_, t.head.lemma_))
            else:
                raise ValueError(f"unknown relation: {relation}")
        return pairs

    def visualize(self, text: str, style: str = "dep") -> str:
        """displaCy で SVG を生成して返す。"""
        self._load()
        from spacy import displacy
        doc = self.nlp(text)
        return displacy.render(doc, style=style, jupyter=False)
