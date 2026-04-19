"""固有表現抽出 (GiNZA / spaCy / ルールベース)。"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import pandas as pd

from .utils import load_spacy_model, optional_import


class NERExtractor:
    """GiNZA (日本語) / spaCy (英語) / ルールベースで NER。

    model:
        'ginza'       — 日本語 (ja_ginza)
        'spacy_en'    — 英語 (en_core_web_sm)
        'rule'        — ルールベース
        'mednerj'     — MedNER-J (別途手動インストール)
    """

    def __init__(self, model: str = "ginza", rules: Optional[dict[str, list[str]]] = None):
        self.model_name = model
        self.nlp = None
        self.rules = rules or {}

    def _load(self):
        if self.nlp is not None or self.model_name == "rule":
            return
        optional_import("spacy", "NER")  # 存在確認
        if self.model_name == "ginza":
            self.nlp = load_spacy_model("ja_ginza")
        elif self.model_name == "spacy_en":
            try:
                self.nlp = load_spacy_model("en_core_web_sm")
            except OSError:
                raise RuntimeError("`python -m spacy download en_core_web_sm` が必要です")
        elif self.model_name == "mednerj":
            try:
                self.nlp = load_spacy_model("ja_ginza_electra")
            except OSError:
                raise RuntimeError("MedNER-J モデルが見つかりません。手動インストールが必要です")
        else:
            raise ValueError(f"unknown model: {self.model_name}")

    def extract(self, text: str) -> list[dict]:
        if self.model_name == "rule":
            return self._rule_extract(text)
        self._load()
        doc = self.nlp(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    def _rule_extract(self, text: str) -> list[dict]:
        out = []
        for label, words in self.rules.items():
            for w in words:
                for m in re.finditer(re.escape(w), text):
                    out.append({"text": m.group(), "label": label, "start": m.start(), "end": m.end()})
        return sorted(out, key=lambda x: x["start"])

    def extract_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        rows = []
        for i, text in enumerate(df[text_col].fillna("")):
            for ent in self.extract(text):
                rows.append({"doc_id": i, **ent})
        return pd.DataFrame(rows)

    def aggregate(self, docs: list[str], label: Optional[str] = None) -> pd.DataFrame:
        """指定ラベルの出現頻度を集計。label=None なら全ラベル。"""
        counter: Counter = Counter()
        for d in docs:
            for ent in self.extract(d):
                if label is None or ent["label"] == label:
                    counter[(ent["label"], ent["text"])] += 1
        rows = [{"label": l, "entity": t, "count": c} for (l, t), c in counter.items()]
        return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


class MedicalNER(NERExtractor):
    """医療向け NER。MedNER-J が無い場合はルールベースにフォールバック。"""

    DEFAULT_MEDICAL_RULES = {
        "Disease": ["糖尿病", "高血圧", "心不全", "心房細動", "脳梗塞", "肺炎", "敗血症", "腎不全"],
        "Drug": ["アスピリン", "ワーファリン", "ヘパリン", "インスリン", "アセトアミノフェン"],
        "Symptom": ["発熱", "呼吸困難", "浮腫", "動悸", "倦怠感", "頭痛", "腹痛"],
    }

    def __init__(self, rules: Optional[dict[str, list[str]]] = None, fallback: str = "rule"):
        super().__init__(model="mednerj", rules=rules or self.DEFAULT_MEDICAL_RULES)
        self.fallback = fallback
        try:
            self._load()
        except (RuntimeError, ImportError):
            self.model_name = self.fallback
