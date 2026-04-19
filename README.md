# text-viz

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](./docker-compose.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Guide](https://img.shields.io/badge/📖_解析ガイド-GitHub_Pages-2c5282)](https://norio-hanafusa.github.io/text-viz/analysis_guide.html)

**jp_nlp_toolkit 全機能** を内包する汎用テキスト解析 GUI(自己完結型・LLM非依存)。
日本語/英語の両方に対応し、**計量テキスト分析**(頻度・共起・対応分析・コーディング等、KH Coder で知られる領域)に加え、**現代的 NLP**(分散表現・NER・感情分析・トピックモデル)をブラウザで利用できます。

## 📖 解析ガイド (GitHub Pages)

各分析手法の **意味・指標・解釈・使い方** を図解したブラウザ向けガイドを公開しています。

### 👉 [**解析ガイドを開く**](https://norio-hanafusa.github.io/text-viz/analysis_guide.html)

17 セクション(頻度 / 共起 / 対応分析 / クラスタリング / **SOM** / トピックモデル / 分散表現 / NER / 係り受け / 感情分析 / 時系列 / コーディング等)を、SVG 図解と指標の解釈表入りで網羅。

> **初回のみ** GitHub 側で設定が必要です:
> [Settings → Pages](https://github.com/norio-hanafusa/text-viz/settings/pages) → **Source: Deploy from a branch** → **Branch: `main` / `/ (root)`** → Save
> 数分後に上記 URL で公開されます。ローカルで見る場合は [`analysis_guide.html`](./analysis_guide.html) を直接開いてください。

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

## 謝辞 (Acknowledgements)

本プロジェクトは以下の先行ソフトウェアから**機能設計の着想**を得ています。
コードの流用はなく、いずれも独立した別プロジェクトです。

- **[KH Coder](https://khcoder.net/)** (樋口耕一氏) — 計量テキスト分析の代表ツール。
  対応分析・コーディングルール・特徴語抽出・KWIC 等の機能設計を参考にしました
- **[nlplot](https://github.com/takapy0210/nlplot)** (takapy0210 氏) — Python の
  テキスト可視化ライブラリ。共起ネットワーク / ワードクラウド / Treemap 等で参考にしました
- **[GiNZA](https://megagonlabs.github.io/ginza/)** / **[fugashi](https://github.com/polm/fugashi)** /
  **[SudachiPy](https://github.com/WorksApplications/SudachiPy)**
- **[oseti](https://github.com/ikegami-yukino/oseti)** / **[neologdn](https://github.com/ikegami-yukino/neologdn)** (池上雄一郎氏)
- **[gensim](https://radimrehurek.com/gensim/)** / **[scikit-learn](https://scikit-learn.org/)** /
  **[sentence-transformers](https://www.sbert.net/)** / **[FAISS](https://github.com/facebookresearch/faiss)**

これらの優れたプロジェクトと学術研究の成果の上に成立しています。

## ライセンス

本プロジェクトは **[Apache License 2.0](./LICENSE)** で公開されています。

- ✅ 商用利用・改変・再配布・社内/製品への組み込み — すべて可能(無償)
- 📄 再配布時は [LICENSE](./LICENSE) と [NOTICE](./NOTICE) を含めてください
- 🔒 特許権の明示的付与条項あり(企業利用時の法務リスク軽減)
- 🚫 改変ファイルには変更の旨を明示(Apache 2.0 §4b)

同梱の `jp_nlp_toolkit/` パッケージも同じ著作者による Apache-2.0 ライセンスです。
各依存ライブラリは別ライセンスで配布されています — 詳細は [NOTICE](./NOTICE) を参照。
