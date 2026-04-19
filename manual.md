# text-viz - マニュアル

[jp-nlp-toolkit](https://github.com/norio-hanafusa/jp-nlp-toolkit) 全機能を内包する**汎用テキスト解析 GUI**。日本語/英語両対応、LLM 非依存。**計量テキスト分析**(頻度・共起・対応分析・コーディング等、KH Coder で知られる領域)と**現代的 NLP**(分散表現・NER・感情分析・トピックモデル)をブラウザ上で提供します。

---

## 目次

1. [前提条件](#1-前提条件)
2. [ディレクトリ構成](#2-ディレクトリ構成)
3. [Docker での起動](#3-docker-での起動)
4. [入力データの準備](#4-入力データの準備)
5. [サイドバー設定](#5-サイドバー設定)
6. [各タブの使い方](#6-各タブの使い方)
    - [6.1 データ/前処理](#61-データ前処理)
    - [6.2 頻度・KWIC](#62-頻度kwic)
    - [6.3 共起ネットワーク](#63-共起ネットワーク)
    - [6.4 多変量解析](#64-多変量解析)
    - [6.5 トピックモデル](#65-トピックモデル)
    - [6.6 埋め込み & 類似検索](#66-埋め込み--類似検索)
    - [6.7 NER/係り受け](#67-ner係り受け)
    - [6.8 感情分析](#68-感情分析)
    - [6.9 時系列](#69-時系列)
    - [6.10 コーディング](#610-コーディング)
7. [典型的な分析ワークフロー](#7-典型的な分析ワークフロー)
8. [出力・ダウンロード](#8-出力ダウンロード)
9. [パフォーマンス](#9-パフォーマンス)
10. [トラブルシューティング](#10-トラブルシューティング)
11. [ローカル実行(Docker なし)](#11-ローカル実行docker-なし)
12. [jp_nlp_toolkit の更新](#12-jp_nlp_toolkit-の更新)

---

## 1. 前提条件

- **Docker Engine** 20.10 以上
- **Docker Compose** v2 以上(`docker compose` コマンド)
- ディスク空き容量: **約 8 GB**(初回ビルド時に torch / sentence-transformers / GiNZA / SudachiPy 等を含む)
- メモリ: **8 GB 以上推奨**(SBERT / LDA / 大規模ネットワーク構築時)

GPU は不要です(CPU 推論)。

## 2. ディレクトリ構成

```
text-viz/
├── app.py                # Streamlit エントリポイント (10 タブ、~800 行)
├── src/
│   ├── loader.py         # ファイル読み込み (CSV/Excel/TSV/TXT/JSON)
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
│   ├── utils.py
│   └── data/             # ストップワード、医療同義語辞書
├── data/
│   └── sample_incidents.csv  # サンプル (医療安全インシデント 50 件)
├── output/               # ネットワーク HTML / pyLDAvis 生成物 (ビルド時空)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── manual.md             # 本ファイル
└── README.md
```

## 3. Docker での起動

```bash
cd text-viz
docker compose up --build
```

**初回ビルドは 10〜20 分**かかります(torch / sentence-transformers / fugashi / ja-ginza / SudachiPy などの大量のパッケージを導入するため)。二回目以降はキャッシュが効き数秒で起動します。

ビルド完了後、ブラウザで以下を開きます:
```
http://localhost:8502
```

停止:
```bash
docker compose down
```

> ポートは `pubmed-viz` (8501) と衝突しないようホスト側 **8502** を使用しています。変更したい場合は `docker-compose.yml` の `ports:` を編集してください。

## 4. 入力データの準備

以下の 5 形式に対応しています。

| 形式 | 条件 |
|---|---|
| **CSV** | テキスト列を含む。日付列・カテゴリ列があると時系列・対応分析・特徴語タブが使える |
| **Excel** (.xlsx / .xls) | 同上 |
| **TSV** | 同上(タブ区切り) |
| **JSON** | オブジェクトの配列 `[{"text": "...", "date": "2024-01-01"}, ...]` |
| **TXT** | 1 行 1 文書(`text` 列として取り込まれる) |

### サンプルデータ

`data/sample_incidents.csv` に架空の医療安全インシデントレポート 50 件が同梱されています。サイドバー「または サンプル」から選択可能。
- 列: `id`, `date`, `department`, `severity`, `incident_text`
- テキスト列: `incident_text`
- 日付列: `date`(時系列解析で使用)
- カテゴリ列: `department` や `severity`(対応分析・特徴語で使用)

## 5. サイドバー設定

サイドバーで全タブに共通する設定を行います。

### 5.1 データ

- **ファイルアップロード** または **サンプル選択**

### 5.2 列設定

- **テキスト列** (必須)
- **日付列** (任意、時系列タブで使用)
- **カテゴリ列** (任意、対応分析・特徴語・コーディングで使用)

### 5.3 前処理

| 項目 | 説明 |
|---|---|
| **言語** | `auto`(自動判定) / `ja` / `en` |
| **形態素解析エンジン** | `auto` / `fugashi` / `ginza` / `sudachi` / `spacy_en` / `simple` |
| **原形 (lemma) で抽出** | `走った` → `走る` のように辞書形に統一 |
| **英語を小文字化** | `Patient` と `patient` を同一視 |
| **最小トークン長** | N 文字未満のトークンを除外(既定 2) |

### 5.4 追加ストップワード 🚫

「意味が薄いが頻出する語」を自由に追加できます。

- 1 行 1 語または **カンマ区切り** で入力
- **大小文字・末尾コロン**(`RESULTS:` 等)を**自動展開**
  - 例: `patient` を入力 → `patient` / `Patient` / `PATIENT` / `patient:` / `Patient:` / `PATIENT:` すべてを除外
- 入力するとすぐに全タブに反映(トークン化結果はキャッシュ、フィルタのみ再実行)

### 5.5 並列処理 ⚡

- **n_jobs**: 形態素解析の並列度(spaCy 系エンジンでのみ有効)
- 数百件なら 1 で十分、数千件以上で効果大
- `(file_key, n_jobs)` でキャッシュされるため、同じ条件で再実行は瞬時

---

## 6. 各タブの使い方

### 6.1 データ/前処理

- **データプレビュー**: アップロードした表の先頭 30 行を確認
- **前処理プレビュー**: 任意の行について **原文 / 正規化後 / トークン(フィルタ後)** を三並び表示
- 前処理の設定がうまく機能しているか目視確認する用途

### 6.2 頻度・KWIC

テキスト解析の基礎となる頻度系情報をまとめて表示します。

| 機能 | 説明 |
|---|---|
| **単語頻度** | 上位 N 単語の棒グラフ + 表 + CSV ダウンロード |
| **N-gram** | 2〜5-gram の頻度表 |
| **TF-IDF** | scikit-learn で計算した行列と上位特徴語 |
| **KWIC** | 指定キーワード前後 window 語を並べたコンコーダンス表 |
| **共起統計量** | 対象語と共起する語を 5 指標で評価 |

**共起指標の使い分け:**
- `jaccard` / `dice`: 頻度を揃えて比較したい
- `pmi`: 珍しい組み合わせを強調
- `t_score`: 統計的に有意な共起
- `log_likelihood`: Dunning の対数尤度比(科学論文で標準)

### 6.3 共起ネットワーク

単語間の共起関係をネットワークとして可視化します。

**パラメータ:**
| 項目 | 選択肢 | 用途 |
|---|---|---|
| **共起範囲** | `document` / `sentence` / `window` | 1 文書内 / 1 文内 / 窓内の共起 |
| **最小エッジ重み** | 1〜50 | 疎なエッジを除外して見やすく |
| **ノード数上限** | 20〜500 | 上位頻度語だけを対象 |
| **エッジ指標** | `jaccard` / `dice` / `pmi` | エッジの太さの計算方式 |
| **コミュニティ検出** | `louvain` / `greedy` / `girvan_newman` / なし | ノード色分け |

**出力:**
- pyvis 製のインタラクティブ HTML(ドラッグ・ズーム・ハイライト可能)
- ノード上位 30 の頻度表
- 5 種の中心性(`betweenness` / `closeness` / `eigenvector` / `pagerank` / `degree`)

### 6.4 多変量解析

サブタブから 4 つの手法を選択:

#### 対応分析 (CA) — カテゴリと語を同じ平面に布置

- **カテゴリ列 × 頻出語** のクロス表を主成分分析
- カテゴリと語を**同じ 2 次元平面にプロット**
- どの語がどのカテゴリに特徴的かを視覚的に把握
- 使用例: `department × 頻出語` で「ICU では `人工呼吸器` が特徴的」等

#### クラスタリング + 次元削減

- **手法**: `kmeans` / `hierarchical` / `dbscan`
- **k**: クラスタ数
- **次元削減**: `umap` / `pca` / `tsne` / `mds`
- silhouette スコアを表示
- クラスタ毎の代表文書を expand で確認可能
- `hierarchical` 選択時はデンドログラムも表示

#### SOM (自己組織化マップ)

- X × Y 格子で文書を配置
- U-Matrix(近傍距離マップ)をヒートマップで表示
- 類似文書が近くに配置される

#### 特徴語抽出

- **指標**: `chi2` / `log_likelihood` / `jaccard`
- **カテゴリ列**毎に特徴的な語を上位 N 抽出
- 「2023 年度 vs 2024 年度の特徴語」「病棟別の特徴語」等に有効

### 6.5 トピックモデル

文書集合に潜むトピックを自動抽出します。

#### LDA (Latent Dirichlet Allocation)

| パラメータ | 意味 |
|---|---|
| **トピック数** | 通常 5〜20 |
| **passes** | 反復回数(既定 5、増やすと安定性向上) |
| **各トピックで表示する語数** | 上位 N |
| **最適トピック数を探索** | コヒーレンス c_v で自動探索 |

**出力:**
- トピック × 語の確率表
- coherence c_v スコア
- **pyLDAvis** インタラクティブ可視化(各トピックの分離度・代表語を直感的に確認)
- 最適トピック数の推移グラフ

#### NMF (非負値行列因子分解)

- LDA より高速
- トピック × 語のランキング表

### 6.6 埋め込み & 類似検索

#### Word2Vec

| パラメータ | 意味 |
|---|---|
| **vector_size** | 語ベクトル次元(200 推奨) |
| **window** | 文脈窓幅 |
| **min_count** | 対象語の最小出現回数 |

学習後の操作:
- **類似語検索**: 指定語に近い上位 15 語
- **アナロジー**: `king - man + woman = queen` のようなベクトル演算

#### SBERT + FAISS 類似検索

| 項目 | 選択肢 |
|---|---|
| **SBERT モデル** | `multi` (多言語) / `ja` (日本語) / `en` (英語) |
| **FAISS インデックス** | `flat` / `ivf` / `hnsw` |

- 初回は SBERT モデルを Hugging Face からダウンロード(数分)
- インデックス構築後、自由文クエリで類似文書を検索

### 6.7 NER/係り受け

#### NER (固有表現抽出)

| モデル | 対応言語 | 抽出可能な表現 |
|---|---|---|
| `ginza` | 日本語 | PERSON / ORG / LOC / DATE / PRODUCT / MONEY 等 |
| `spacy_en` | 英語 | 同上 |
| `rule` | 任意 | ユーザー定義ルール |
| `mednerj (医療)` | 日本語 | 病名・薬剤・症状(MedNER-J が導入されていれば) |

- 対象文書数を絞ることでネットワーク負荷を調整
- ラベル毎の棒グラフで「どの種類の実体が多いか」を把握

#### 係り受け

| 関係 | 例 |
|---|---|
| `verb_obj` | 「薬を投与する」→ `(投与, 薬)` |
| `subject_verb` | 「患者が改善した」→ `(患者, 改善)` |
| `adj_noun` | 「重い症状」→ `(重い, 症状)` |

- 1 文書を **displaCy** で SVG 可視化
- 最大 200 文書からペアを集計

### 6.8 感情分析

文書ごとのポジティブ/ネガティブ傾向を評価します。

| 手法 | 対象言語 | 辞書 |
|---|---|---|
| `auto` | 自動判定 | 日本語 → oseti、英語 → VADER |
| `oseti (日)` | 日本語 | 日本語評価極性辞書 |
| `vader (英)` | 英語 | VADER lexicon |
| `custom_dict` | 任意 | ユーザー定義の語×スコア |

カスタム辞書の書式(`custom_dict` 選択時):
```
改善:1
悪化:-1
効果:1
副作用:-1
```

**出力:**
- 文書 × (positive / negative / score) 表
- スコアの時系列折れ線
- スコア分布ヒストグラム
- CSV ダウンロード

**評価表現抽出** では、医療向けのポジティブ/ネガティブ語を文書から列挙します(例: 「改善」「悪化」「副作用」等)。

### 6.9 時系列

**日付列必須**。以下の 3 機能を提供:

#### 単語トレンド
- 追跡したい単語をカンマ区切りで入力
- **集計頻度**: `D`(日) / `W`(週) / `M`(月) / `Q`(四半期) / `Y`(年)
- 指標: `count`(出現回数) / `freq`(正規化頻度)
- 折れ線グラフで時系列推移を描画

#### 急増語検出
- **基準期間**と**比較期間**を入力(例: `2023`, `2024`)
- 比較期間で基準期間より**急増した語**を ratio 降順で表示
- Laplace 平滑化で 0 除算回避

#### 期間別ネットワーク
- 指定頻度(Y/Q/M 等)ごとに別々の共起ネットワークを構築
- 期間ごとの構造変化を確認

### 6.10 コーディング

計量テキスト分析で広く用いられるコーディングルール機能(KH Coder で知られる仕組みと同種)を独自実装。

YAML でルールを定義:
```yaml
透析関連:
  - 透析
  - HD
  - シャント
合併症:
  - 心不全
  - 糖尿病
  - 感染症
薬剤:
  - 抗菌薬
  - インスリン
```

**出力:**
- コード別の総ヒット数(棒グラフ)
- 文書 × コードのヒット行列
- コード × カテゴリ列のクロス集計
- コード間共起ネットワーク

---

## 7. 典型的な分析ワークフロー

### 7.1 まずは全体像を把握したい

1. **タブ 1** でデータとトークン化結果を確認
2. **タブ 2** で頻出語・N-gram を見て基本的な語彙分布を把握
3. **タブ 3** で共起ネットワークを構築し、話題のクラスタを俯瞰
4. 気になる語があれば **タブ 2 の KWIC** で実際の文脈を確認

### 7.2 グループ間の違いを調べたい

1. サイドバーで**カテゴリ列を選択**
2. **タブ 4 → 対応分析**: カテゴリと語の関係を 2D プロット
3. **タブ 4 → 特徴語**: 各カテゴリの特徴語を χ² で統計的に抽出

### 7.3 時系列でトレンドを追いたい

1. サイドバーで**日付列を選択**
2. **タブ 9 → 単語トレンド**: 注目語の年次推移
3. **タブ 9 → 急増語検出**: 最近になって急増した語を発見

### 7.4 大量文書から類似事例を検索したい

1. **タブ 6 → SBERT + FAISS**でインデックスを構築
2. 自由文クエリ(例:「術後の痛みに対する対応が遅れた事例」)で検索
3. スコア順に類似事例を確認

### 7.5 手動コーディングを自動化したい

1. **タブ 10 → コーディング**で YAML ルールを入力
2. 各コードの頻度・クロス集計を自動算出
3. コード間の共起関係から上位概念を発見

---

## 8. 出力・ダウンロード

| 内容 | 取得方法 |
|---|---|
| 単語頻度 CSV | タブ 2 の「CSV ダウンロード」 |
| 感情分析結果 CSV | タブ 8 の「CSV ダウンロード」 |
| 共起ネットワーク HTML | `output/network.html` (pyvis) |
| pyLDAvis HTML | `output/lda.html` |

コンテナ内の `output/` は `docker-compose.yml` でホスト側の `./output/` にマウントされているため、ホスト側から直接ファイルを確認できます。

---

## 9. パフォーマンス

### キャッシング戦略

1. **正規化** (`@st.cache_data`): 入力テキスト + 言語 + 設定でキャッシュ
2. **形態素解析** (`@st.cache_data`): 正規化結果 + エンジン + n_jobs でキャッシュ
3. **ストップワード除去**: **キャッシュせず毎回実行**(軽量、即時反映)

スライダー操作・ストップワード追加時はトークン化をスキップするので**瞬時に再描画**されます。

### 並列トークン化

| データ量 | n_jobs=1 | n_jobs=4 |
|---|---|---|
| 134 件 | < 1 秒 | < 1 秒(並列オーバーヘッド優勢) |
| 1,000 件 | 5〜10 秒 | 2〜4 秒 |
| 10,000 件 | 60〜120 秒 | 20〜40 秒 |

### 重量級処理はボタン実行

LDA / Word2Vec / SBERT インデックス構築 / 共起ネットワーク構築はサイドバー操作時に自動走行しません。各タブの「実行」ボタンで明示的に起動してください。

---

## 10. トラブルシューティング

### ビルド/起動関連

| 症状 | 原因 | 対処 |
|---|---|---|
| `streamlit: command not found` | pip install が途中で失敗 | `docker compose build --no-cache` |
| `oseti を導入してください` | oseti 依存の `bunkai` が `transformers 4.40+` と非互換 | `requirements.txt` の `transformers>=4.30,<4.40` pin を確認 |
| `mecabrc を指定してください` | MeCab 辞書パスが未設定 | `ipadic>=1.0.0` が requirements に入っているか確認、`sentiment.py` の `mecab_args` で自動指定 |
| `ConfigValidationError (compound_splitter)` | GiNZA 5.2.0 と新しい confection の非互換 | `utils.py` の `load_spacy_model` で `exclude=["compound_splitter"]` を指定(対応済) |

### 実行時エラー

| 症状 | 対処 |
|---|---|
| "Invalid value for dtype 'str'" | nlplot + pandas 2.x の互換性問題。`build_word_df` が list のまま DataFrame を作るか確認 |
| SBERT モデルのダウンロードが遅い | 初回のみ数分。以後はコンテナ内にキャッシュ |
| `en_core_web_sm` が無い | `docker compose exec app python -m spacy download en_core_web_sm` |

### メモリ不足

- データ件数が多い場合、**タブ 3 のノード数上限**を減らす
- **タブ 6 の SBERT インデックス**は文書数 × 768 float の配列を保持(10,000 件で ~30 MB)
- Docker Desktop の割当メモリを 8 GB 以上に

### ポート衝突

`docker-compose.yml` の `ports: "8502:8501"` を空いているポートに変更:
```yaml
ports:
  - "9000:8501"
```
→ http://localhost:9000 で開く

---

## 11. ローカル実行(Docker なし)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download ja_ginza en_core_web_sm
python -m unidic download
python -m nltk.downloader stopwords vader_lexicon
streamlit run app.py
```

Python **3.11** を推奨します(3.12 は一部依存が未対応の可能性)。

---

## 12. jp_nlp_toolkit の更新

`jp_nlp_toolkit/` はリポジトリ内に同梱されているため、独立更新が可能です。

### 直接編集
`jp_nlp_toolkit/*.py` を編集後、`docker compose up --build` で再ビルド(COPY 層のみ再走、数十秒)。

### 上流 (jp-nlp-toolkit) からの同期

```bash
cp -r ../jp-nlp-toolkit/jp_nlp_toolkit ./
```

上書き後、同じく `docker compose up --build`。

---

## 参考資料

- [jp-nlp-toolkit](https://github.com/norio-hanafusa/jp-nlp-toolkit) — コアライブラリ
- [pubmed-viz](https://github.com/norio-hanafusa/pubmed-viz) — PubMed 論文特化の姉妹プロジェクト

---

## 謝辞 (Acknowledgements)

本プロジェクトは以下の先行ソフトウェアから**機能設計の着想**を得ています。
コードの流用はなく、いずれも独立した別プロジェクトです。

- **[KH Coder](https://khcoder.net/)** (樋口耕一氏) — 計量テキスト分析の代表ツール。
  対応分析・コーディングルール・特徴語抽出・KWIC 等の機能設計を参考にしました
- **[nlplot](https://github.com/takapy0210/nlplot)** (takapy0210 氏) — Python の
  テキスト可視化ライブラリ。共起ネットワーク / ワードクラウド / Treemap の実装で参考にしました
- **[GiNZA](https://megagonlabs.github.io/ginza/)** / **[fugashi](https://github.com/polm/fugashi)** /
  **[SudachiPy](https://github.com/WorksApplications/SudachiPy)** — 日本語形態素解析・係り受け解析
- **[oseti](https://github.com/ikegami-yukino/oseti)** / **[neologdn](https://github.com/ikegami-yukino/neologdn)** (池上雄一郎氏)
- **[gensim](https://radimrehurek.com/gensim/)** / **[scikit-learn](https://scikit-learn.org/)** /
  **[sentence-transformers](https://www.sbert.net/)** / **[FAISS](https://github.com/facebookresearch/faiss)** /
  **[prince](https://github.com/MaxHalford/prince)** / **[NetworkX](https://networkx.org/)** /
  **[pyvis](https://github.com/WestHealth/pyvis)** / **[pyLDAvis](https://github.com/bmabey/pyLDAvis)**

これらの優れたプロジェクトと、Benzécri (1973) / Dunning (1993) / Blei (2003) /
Mikolov (2013) / Reimers & Gurevych (2019) 等の研究成果の上に成立しています。
