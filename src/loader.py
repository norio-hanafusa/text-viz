"""ファイル読み込み (CSV/Excel/TSV/TXT/JSON)。"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def load_file(uploaded) -> pd.DataFrame:
    """Streamlit UploadedFile を受け取り DataFrame を返す。"""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data), engine="xlrd")
    if name.endswith(".tsv"):
        return pd.read_csv(io.BytesIO(data), sep="\t")
    if name.endswith(".json"):
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame([obj])
        raise ValueError("JSON は object の配列である必要があります")
    if name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
        lines = [l for l in text.split("\n") if l.strip()]
        return pd.DataFrame({"text": lines})
    raise ValueError(f"サポート外のファイル形式: {name}")


def load_sample(name: str) -> Optional[pd.DataFrame]:
    """サンプルデータを読み込む。見つからなければ None。"""
    candidates = [
        Path(__file__).parent.parent / "data" / name,
        Path("/opt/jp-nlp-toolkit/examples/data") / name,
        Path(__file__).parent.parent.parent / "jp-nlp-toolkit" / "examples" / "data" / name,
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None
