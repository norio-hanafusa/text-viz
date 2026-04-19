"""感情分析 (oseti / VADER / カスタム辞書)。"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .utils import detect_language, optional_import


class SentimentAnalyzer:
    """method: 'auto' | 'oseti' (日) | 'vader' (英) | 'custom_dict'"""

    def __init__(
        self,
        method: str = "auto",
        custom_dict: Optional[dict[str, int]] = None,
    ):
        self.method = method
        self.custom_dict = custom_dict or {}
        self._oseti = None
        self._vader = None

    def _get_oseti(self):
        if self._oseti is None:
            oseti = optional_import("oseti", "感情分析 (日本語)")
            # oseti は mecab-python3 を使うが辞書を同梱しないため、ipadic の辞書パスを渡す
            mecab_args = ""
            try:
                import ipadic
                mecab_args = f"-r /dev/null -d {ipadic.DICDIR}"
            except ImportError:
                pass
            try:
                self._oseti = oseti.Analyzer(mecab_args=mecab_args)
            except TypeError:
                # 旧版 oseti で mecab_args 引数を持たない場合のフォールバック
                self._oseti = oseti.Analyzer()
        return self._oseti

    def _get_vader(self):
        if self._vader is None:
            nltk = optional_import("nltk", "VADER")
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                from nltk.sentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
        return self._vader

    def analyze(self, text: str) -> dict:
        """return: {positive, negative, score}"""
        if not text:
            return {"positive": 0, "negative": 0, "score": 0.0}

        method = self.method
        if method == "auto":
            method = "oseti" if detect_language(text) == "ja" else "vader"

        if method == "oseti":
            scores = self._get_oseti().count_polarity(text)
            pos = sum(s.get("positive", 0) for s in scores) if scores else 0
            neg = sum(s.get("negative", 0) for s in scores) if scores else 0
            total = pos + neg
            return {
                "positive": pos,
                "negative": neg,
                "score": (pos - neg) / total if total else 0.0,
            }
        if method == "vader":
            s = self._get_vader().polarity_scores(text)
            return {
                "positive": s["pos"],
                "negative": s["neg"],
                "score": s["compound"],
            }
        if method == "custom_dict":
            return self._custom(text)
        raise ValueError(f"unknown method: {method}")

    def _custom(self, text: str) -> dict:
        pos = neg = 0
        for w, sc in self.custom_dict.items():
            hits = len(re.findall(re.escape(w), text))
            if sc > 0:
                pos += hits * sc
            else:
                neg += hits * (-sc)
        total = pos + neg
        return {
            "positive": pos,
            "negative": neg,
            "score": (pos - neg) / total if total else 0.0,
        }

    def analyze_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        out = df.copy()
        res = [self.analyze(t) for t in df[text_col].fillna("")]
        out["sentiment_positive"] = [r["positive"] for r in res]
        out["sentiment_negative"] = [r["negative"] for r in res]
        out["sentiment_score"] = [r["score"] for r in res]
        return out


class EvaluationExtractor:
    """評価表現 (良い/悪い・効果的/副作用など) を辞書から抽出。"""

    DEFAULT_MEDICAL_DICT = {
        "positive": ["改善", "軽快", "安定", "寛解", "治癒", "有効", "効果的", "回復"],
        "negative": ["悪化", "増悪", "副作用", "有害事象", "再発", "無効", "合併症", "死亡"],
    }

    def __init__(self, evaluation_dict: Optional[dict[str, list[str]]] = None):
        self.dict = evaluation_dict or self.DEFAULT_MEDICAL_DICT

    def extract(self, text: str) -> list[dict]:
        out = []
        for polarity, words in self.dict.items():
            for w in words:
                for m in re.finditer(re.escape(w), text):
                    out.append({
                        "text": m.group(),
                        "polarity": polarity,
                        "start": m.start(),
                        "end": m.end(),
                    })
        return sorted(out, key=lambda x: x["start"])

    def extract_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        rows = []
        for i, text in enumerate(df[text_col].fillna("")):
            for e in self.extract(text):
                rows.append({"doc_id": i, **e})
        return pd.DataFrame(rows)
