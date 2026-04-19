"""前処理・形態素解析。

日本語(fugashi / GiNZA / SudachiPy)と英語(spaCy / 空白分割)を統一APIで扱う。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from .utils import detect_language, get_default_stopwords, load_spacy_model, optional_import


@dataclass
class Token:
    surface: str
    lemma: str
    pos: str
    reading: str = ""


class Tokenizer:
    """多エンジン・多言語対応の形態素解析器。

    Args:
        engine: 'fugashi' | 'ginza' | 'sudachi' | 'spacy_en' | 'simple'
        language: 'ja' | 'en' | 'auto'
        user_dict: ユーザー辞書パス (sudachi のみ対応)
    """

    def __init__(
        self,
        engine: str = "fugashi",
        language: str = "auto",
        dict_path: Optional[str] = None,
        user_dict: Optional[str] = None,
    ):
        self.engine = engine
        self.language = language
        self._dict_path = dict_path
        self._user_dict = user_dict
        self._impl = None  # 遅延初期化

    def _init_ja_fugashi(self):
        fugashi = optional_import("fugashi", "fugashi tokenizer")
        return fugashi.Tagger()

    def _init_ja_ginza(self):
        return load_spacy_model("ja_ginza")

    def _init_ja_sudachi(self):
        sudachipy = optional_import("sudachipy", "SudachiPy tokenizer")
        from sudachipy import dictionary
        return dictionary.Dictionary().create()

    def _init_en_spacy(self):
        spacy = optional_import("spacy", "spaCy English")
        try:
            return load_spacy_model("en_core_web_sm")
        except OSError:
            return spacy.blank("en")

    def _ensure(self):
        if self._impl is not None:
            return
        if self.engine == "fugashi":
            self._impl = self._init_ja_fugashi()
        elif self.engine == "ginza":
            self._impl = self._init_ja_ginza()
        elif self.engine == "sudachi":
            self._impl = self._init_ja_sudachi()
        elif self.engine == "spacy_en":
            self._impl = self._init_en_spacy()
        elif self.engine == "simple":
            self._impl = "simple"
        else:
            raise ValueError(f"unknown engine: {self.engine}")

    def _resolve_language(self, text: str) -> str:
        if self.language != "auto":
            return self.language
        return detect_language(text)

    def tokenize(self, text: str) -> list[Token]:
        """1文書を Token のリストに分割。"""
        if not text:
            return []
        lang = self._resolve_language(text)
        # 言語auto時のエンジン自動切替
        engine = self.engine
        if self.language == "auto":
            if lang == "ja" and engine in ("spacy_en", "simple"):
                engine = "fugashi"
            if lang == "en" and engine in ("fugashi", "ginza", "sudachi"):
                engine = "spacy_en"
            if engine != self.engine:
                self.engine = engine
                self._impl = None
        self._ensure()

        if self.engine == "fugashi":
            return self._tokens_fugashi(text)
        if self.engine == "ginza":
            return self._tokens_spacy(text)
        if self.engine == "sudachi":
            return self._tokens_sudachi(text)
        if self.engine == "spacy_en":
            return self._tokens_spacy(text)
        # simple
        return [Token(surface=w, lemma=w.lower(), pos="X") for w in text.split()]

    def _tokens_fugashi(self, text: str) -> list[Token]:
        out = []
        for w in self._impl(text):
            feat = w.feature
            pos = getattr(feat, "pos1", "") or getattr(feat, "pos", "")
            lemma = getattr(feat, "lemma", None) or w.surface
            reading = getattr(feat, "kana", "") or getattr(feat, "pron", "") or ""
            out.append(Token(surface=w.surface, lemma=lemma, pos=pos, reading=reading))
        return out

    def _tokens_spacy(self, text: str) -> list[Token]:
        doc = self._impl(text)
        return [
            Token(
                surface=t.text,
                lemma=t.lemma_ or t.text,
                pos=t.pos_,
                reading=getattr(t, "norm_", ""),
            )
            for t in doc
            if not t.is_space
        ]

    def _tokens_sudachi(self, text: str) -> list[Token]:
        from sudachipy import tokenizer as stok
        mode = stok.Tokenizer.SplitMode.C
        out = []
        for m in self._impl.tokenize(text, mode):
            pos = m.part_of_speech()[0] if m.part_of_speech() else ""
            out.append(Token(
                surface=m.surface(),
                lemma=m.dictionary_form() or m.surface(),
                pos=pos,
                reading=m.reading_form(),
            ))
        return out

    def tokenize_to_surface(self, text: str) -> list[str]:
        return [t.surface for t in self.tokenize(text)]

    def tokenize_to_lemma(self, text: str) -> list[str]:
        return [t.lemma for t in self.tokenize(text)]

    def tokenize_batch(
        self,
        texts: list[str],
        n_jobs: int = 1,
        batch_size: int = 50,
        as_lemma: bool = False,
    ) -> list[list[str]]:
        """複数文書を一括トークン化し、surface (or lemma) のリストを返す。

        n_jobs > 1 で spaCy 系エンジン (spacy_en / ginza) は nlp.pipe の
        multiprocessing 並列を利用。fugashi/sudachi/simple は C 拡張で
        GIL を解放するがスレッド並列は効果が薄いため、逐次処理。
        """
        self._ensure()
        # spaCy 系: nlp.pipe で並列
        if n_jobs != 1 and self.engine in ("spacy_en", "ginza"):
            try:
                from spacy.language import Language
                if isinstance(self._impl, Language):
                    if as_lemma:
                        return [
                            [t.lemma_ or t.text for t in doc if not t.is_space]
                            for doc in self._impl.pipe(
                                texts, n_process=n_jobs, batch_size=batch_size,
                            )
                        ]
                    return [
                        [t.text for t in doc if not t.is_space]
                        for doc in self._impl.pipe(
                            texts, n_process=n_jobs, batch_size=batch_size,
                        )
                    ]
            except Exception:
                pass  # 並列失敗時は逐次にフォールバック
        fn = self.tokenize_to_lemma if as_lemma else self.tokenize_to_surface
        return [fn(t) for t in texts]

    def tokenize_df(
        self,
        df: pd.DataFrame,
        text_col: str,
        out_col: str = "tokens",
        as_lemma: bool = False,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        texts = df[text_col].fillna("").astype(str).tolist()
        tokens = self.tokenize_batch(texts, n_jobs=n_jobs, as_lemma=as_lemma)
        out = df.copy()
        out[out_col] = pd.Series(tokens, index=df.index, dtype="object")
        return out


class Normalizer:
    """テキスト正規化。"""

    _URL = re.compile(r"https?://\S+")
    _EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    _NUM = re.compile(r"\d+")

    def __init__(self, language: str = "auto"):
        self.language = language

    def normalize(
        self,
        text: str,
        half_to_full: bool = False,
        lowercase: bool = True,
        remove_url: bool = True,
        remove_email: bool = True,
        remove_number: bool = False,
    ) -> str:
        if not text:
            return ""
        lang = detect_language(text) if self.language == "auto" else self.language

        if remove_url:
            text = self._URL.sub(" ", text)
        if remove_email:
            text = self._EMAIL.sub(" ", text)

        if lang == "ja":
            try:
                import neologdn
                text = neologdn.normalize(text)
            except ImportError:
                pass
            try:
                import jaconv
                if half_to_full:
                    text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
            except ImportError:
                pass
        else:
            if lowercase:
                text = text.lower()

        if remove_number:
            text = self._NUM.sub(" ", text)

        return re.sub(r"\s+", " ", text).strip()


class SynonymExpander:
    """略語・同義語の正規化。"""

    def __init__(self, synonym_dict: Union[dict[str, str], str, Path]):
        if isinstance(synonym_dict, (str, Path)):
            path = Path(synonym_dict)
            with path.open(encoding="utf-8") as f:
                self.synonyms = json.load(f)
        else:
            self.synonyms = dict(synonym_dict)

    def expand(self, tokens: Iterable[str]) -> list[str]:
        return [self.synonyms.get(t, t) for t in tokens]

    def expand_text(self, text: str) -> str:
        out = text
        for src, dst in self.synonyms.items():
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
        return out


def filter_pos(
    tokens: list[Token],
    allowed_pos: Optional[list[str]] = None,
) -> list[Token]:
    """品詞フィルタ。日本語の pos1 ('名詞', '動詞' 等) と spaCy POS ('NOUN' 等) の両方に対応。"""
    if not allowed_pos:
        allowed_pos = ["名詞", "動詞", "形容詞", "NOUN", "VERB", "ADJ", "PROPN"]
    allowed = set(allowed_pos)
    return [t for t in tokens if t.pos in allowed]


def remove_stopwords(
    tokens: list[str],
    stopwords: Optional[set[str]] = None,
    language: str = "auto",
) -> list[str]:
    if stopwords is None:
        if language == "auto":
            # 推測不能なら両方マージ
            sw = get_default_stopwords("ja") | get_default_stopwords("en")
        else:
            sw = get_default_stopwords(language)
    else:
        sw = set(stopwords)
    return [t for t in tokens if t and t not in sw]
