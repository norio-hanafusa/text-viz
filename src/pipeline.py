"""トークナイズ・フィルタ・前処理のラッパー (text-viz 用)。

キャッシュしやすいよう、純関数として副作用を分離。
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

from jp_nlp_toolkit import (
    Normalizer,
    Tokenizer,
    detect_language,
    remove_stopwords,
)
from jp_nlp_toolkit.utils import get_default_stopwords


def parse_user_stopwords(text: str) -> list[str]:
    if not text:
        return []
    return [w for w in re.split(r"[,\s]+", text.strip()) if w]


def _expand_case_variants(words: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for w in words:
        if not w:
            continue
        out |= {w, w.lower(), w.upper(), w.capitalize()}
        out |= {f"{x}:" for x in {w, w.lower(), w.upper(), w.capitalize()}}
    return out


def sample_language(texts: list[str]) -> str:
    sample = " ".join(str(t) for t in texts[:20] if t)
    return detect_language(sample)


def normalize_texts(
    texts: list[str],
    language: str = "auto",
    lowercase_en: bool = True,
    remove_url: bool = True,
    remove_email: bool = True,
    remove_number: bool = False,
) -> list[str]:
    norm = Normalizer(language=language)
    return [
        norm.normalize(
            t or "",
            lowercase=lowercase_en,
            remove_url=remove_url,
            remove_email=remove_email,
            remove_number=remove_number,
        )
        for t in texts
    ]


def tokenize_texts(
    texts: list[str],
    engine: str = "auto",
    language: str = "auto",
    as_lemma: bool = False,
    n_jobs: int = 1,
) -> list[list[str]]:
    """テキストをトークン化。n_jobs > 1 で spaCy 系は並列。"""
    if language == "auto":
        language = sample_language(texts)
    if engine == "auto":
        engine = "fugashi" if language == "ja" else "spacy_en"
    try:
        tok = Tokenizer(engine=engine, language=language)
        return tok.tokenize_batch(texts, n_jobs=n_jobs, as_lemma=as_lemma)
    except (ImportError, RuntimeError):
        tok = Tokenizer(engine="simple", language=language)
        fn = tok.tokenize_to_lemma if as_lemma else tok.tokenize_to_surface
        return [fn(t) for t in texts]


def filter_tokens(
    tokens_list: list[list[str]],
    language: str = "auto",
    extra_stopwords: Optional[Iterable[str]] = None,
    min_token_len: int = 2,
) -> list[list[str]]:
    sw = get_default_stopwords(language if language != "auto" else "en")
    if extra_stopwords:
        sw = sw | _expand_case_variants(extra_stopwords)
    out = []
    for toks in tokens_list:
        filtered = [t for t in remove_stopwords(toks, stopwords=sw) if len(t) >= min_token_len]
        out.append(filtered)
    return out


def subset_tokens(tokens_full: list[list[str]], df_subset: pd.DataFrame) -> list[list[str]]:
    """df_subset.index (元 df での位置) に対応するトークンを抽出 (RangeIndex 前提)。"""
    return [tokens_full[i] for i in df_subset.index if 0 <= i < len(tokens_full)]
