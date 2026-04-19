"""共通ユーティリティ。"""
from __future__ import annotations

import importlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

_JA_CHARS = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")

DATA_DIR = Path(__file__).parent / "data"


def detect_language(text: str, threshold: float = 0.1) -> str:
    """日本語文字の比率から言語を判定。

    Returns:
        'ja' または 'en'
    """
    if not text:
        return "en"
    ja = len(_JA_CHARS.findall(text))
    return "ja" if ja / max(len(text), 1) >= threshold else "en"


def optional_import(name: str, feature: str = ""):
    """オプション依存を必要時にのみインポート。失敗時は明瞭なエラー。"""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        extra = f" ({feature} で必要)" if feature else ""
        # 元エラーも表示 — 単なる未インストールか、依存する外部ライブラリの問題か判別するため
        raise ImportError(
            f"'{name}' の読み込みに失敗しました{extra}。"
            f"[原因] {type(e).__name__}: {e}。"
            f"未インストールなら `pip install {name}`、"
            f"他のエラーならそのメッセージの依存を追加してください。"
        ) from e


@lru_cache(maxsize=1)
def load_japanese_stopwords() -> set[str]:
    """SlothLib 相当の日本語ストップワードを同梱ファイルから読み込む。"""
    path = DATA_DIR / "stopwords_ja.txt"
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


@lru_cache(maxsize=1)
def load_english_stopwords() -> set[str]:
    """NLTK の英語ストップワードを取得 (初回はダウンロード)。"""
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
    except ImportError:
        return set()
    return sw | {w.capitalize() for w in sw}


def get_default_stopwords(language: str) -> set[str]:
    """言語に応じたデフォルトのストップワード集合。"""
    if language == "ja":
        return load_japanese_stopwords()
    return load_english_stopwords()


def load_spacy_model(name: str, **kwargs):
    """spaCy モデルをロード。GiNZA 5.2.0 + 新しい confection/spaCy との
    互換性問題 (compound_splitter.split_mode=None で config 検証エラー) を回避。
    """
    spacy = optional_import("spacy", "spaCy モデル")
    if name.startswith("ja_ginza"):
        # compound_splitter を除外 (split_mode=None で validate エラーになる)
        # NER / 係り受け / トークン化には不要
        exclude = kwargs.pop("exclude", None) or []
        if "compound_splitter" not in exclude:
            exclude = list(exclude) + ["compound_splitter"]
        try:
            return spacy.load(name, exclude=exclude, **kwargs)
        except Exception:
            # フォールバック: split_mode を明示的に指定して再試行
            cfg = kwargs.pop("config", {}) or {}
            cfg.setdefault("components", {}).setdefault(
                "compound_splitter", {"split_mode": "C"}
            )
            return spacy.load(name, config=cfg, **kwargs)
    return spacy.load(name, **kwargs)


def ensure_font(font_path: Optional[str] = None) -> Optional[str]:
    """日本語フォント (matplotlib 向け) の解決。japanize-matplotlib が使える場合は適用。"""
    if font_path:
        return font_path
    try:
        import japanize_matplotlib  # noqa: F401
        return None
    except ImportError:
        pass
    # Windows フォールバック
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None
