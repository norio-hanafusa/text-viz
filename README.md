# text-viz

**jp_nlp_toolkit 全機能** を内包する汎用テキスト解析 GUI(自己完結型・LLM非依存)。
日本語/英語の両方に対応し、KH Coder + nlplot 以上の機能をブラウザで利用可能。

コアエンジン `jp_nlp_toolkit/` はリポジトリ内に同梱されており、**別途 pip install は不要**です。

## 搭載機能 (10 タブ)

| # | タブ | 機能 |
|---|---|---|
| 1 | 📂 データ/前処理 | アップロード (CSV/Excel/TSV/TXT/JSON)、正規化・形態素解析プレビュー |
| 2 | 📊 頻度・KWIC | 単語頻度 / N-gram / TF-IDF / KWIC / 共起統計 (Jaccard/Dice/PMI/t-score/対数尤度) |
| 3 | 🕸️ 共起NW | pyvis ネットワーク + Louvain/Greedy/Girvan-Newman + 5 種の中心性 |
| 4 | 🔬 多変量解析 | 対応分析 (CA) / クラスタリング + UMAP/t-SNE/PCA/MDS / SOM / 特徴語 (χ²・対数尤度・Jaccard) |
| 5 | 📑 トピックモデル | LDA + pyLDAvis + 最適トピック数探索 / NMF |
| 6 | 🧬 埋め込み&類似検索 | Word2Vec (類似語・アナロジー) / SBERT + FAISS (flat/ivf/hnsw) |
| 7 | 🧭 NER/係り受け | GiNZA/spaCy/ルール/MedNER-J / displaCy 可視化 / 述語項構造抽出 |
| 8 | 💬 感情分析 | oseti (日) / VADER (英) / カスタム辞書 / 評価表現抽出 |
| 9 | 📆 時系列 | 単語トレンド / 急増語検出 / 期間別ネットワーク |
| 10 | 📋 コーディング | YAML ルール → コード別頻度・クロス集計・コード間共起 |

## 起動

```bash
docker compose up --build
```

ブラウザで http://localhost:**8502** を開く(`pubmed-viz` の 8501 と衝突しないよう 8502)。

初回ビルドは **10〜20 分** かかります(torch / spaCy モデル / NLTK / fugashi / GiNZA / SBERT)。

**自己完結**なので、他ディレクトリは参照しません。このフォルダだけをクローンしても動作します。

停止:
```bash
docker compose down
```

## 入力データ

- **CSV / Excel / TSV**: テキスト列必須、日付列・カテゴリ列は任意(時系列・対応分析・特徴語で使用)
- **JSON**: オブジェクトの配列
- **TXT**: 1 行 1 文書

サンプルデータ(医療安全インシデント 50 件、`data/sample_incidents.csv`)が組み込みで利用可能。

## プロジェクト構成

```
text-viz/
├── app.py                # Streamlit エントリ (10 タブ)
├── src/
│   ├── loader.py         # ファイル読み込み
│   └── pipeline.py       # 正規化・トークン化・フィルタ (純関数)
├── jp_nlp_toolkit/       # ← コアライブラリを同梱 (自己完結化)
│   ├── preprocess.py
│   ├── frequency.py
│   ├── cooccurrence.py
│   ├── correspondence.py
│   ├── clustering.py
│   ├── topic_model.py
│   ├── sentiment.py
│   ├── ner.py
│   ├── dependency.py
│   ├── embedding.py
│   ├── similarity.py
│   ├── feature_words.py
│   ├── timeseries.py
│   ├── coding.py
│   ├── visualize.py
│   └── data/             # ストップワード / 医療同義語辞書
├── data/
│   └── sample_incidents.csv
├── output/               # ネットワーク HTML / pyLDAvis 生成物
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## ローカル実行(Docker なし)

```bash
pip install -r requirements.txt
python -m spacy download ja_ginza en_core_web_sm
python -m unidic download
python -m nltk.downloader stopwords vader_lexicon
streamlit run app.py
```

## ライブラリの更新

`jp_nlp_toolkit/` 配下を直接編集するか、上流の
[jp-nlp-toolkit](../jp-nlp-toolkit/) から同名ディレクトリを上書きコピーしてください:

```bash
cp -r ../jp-nlp-toolkit/jp_nlp_toolkit ./
```

## pubmed-viz との違い

| | text-viz | pubmed-viz |
|---|---|---|
| 入力 | CSV/Excel/TSV/TXT/JSON (汎用) | PubMed MEDLINE 特化 |
| 対応言語 | 日本語 / 英語 | 英語 (PubMed 前提) |
| タブ数 | 10 | 5 |
| 主眼 | **網羅性** (全機能) | **特化性** (PubMed 分析) |
| 起動ポート | 8502 | 8501 |
