"""text-viz — jp-nlp-toolkit 全機能を内包する汎用テキスト解析 GUI。

10 タブ構成:
  1. 📂 データ & 前処理
  2. 📊 頻度・KWIC・TF-IDF
  3. 🕸️ 共起ネットワーク
  4. 🔬 多変量解析 (CA / クラスタリング / SOM / 特徴語)
  5. 📑 トピックモデル (LDA / NMF)
  6. 🧬 埋め込み & 類似検索 (Word2Vec / SBERT / FAISS)
  7. 🧭 NER & 係り受け
  8. 💬 感情分析
  9. 📆 時系列
 10. 📋 コーディング
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.loader import load_file, load_sample
from src.pipeline import (
    filter_tokens,
    normalize_texts,
    parse_user_stopwords,
    sample_language,
    subset_tokens,
    tokenize_texts,
)

st.set_page_config(page_title="text-viz", layout="wide", page_icon="🧪")
st.title("🧪 text-viz")
st.caption("jp-nlp-toolkit 全機能を搭載した汎用テキスト解析 GUI (LLM非依存)")

OUTPUT_DIR = Path(os.environ.get("TEXT_VIZ_OUTPUT", "output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# サイドバー 1: データ入力
# =============================================================================
with st.sidebar:
    st.header("📂 1. データ")
    uploaded = st.file_uploader(
        "CSV / Excel / TSV / TXT / JSON",
        type=["csv", "xlsx", "xls", "tsv", "txt", "json"],
    )
    sample_choice = st.selectbox(
        "またはサンプル",
        ["(未使用)", "sample_incidents.csv (医療安全 50件)"],
        index=0,
    )

# データ取得
df_raw: "pd.DataFrame | None" = None
data_source_key = None
if uploaded is not None:
    df_raw = load_file(uploaded)
    data_source_key = f"upload:{uploaded.name}:{len(uploaded.getvalue())}"
elif sample_choice.startswith("sample_incidents"):
    df_raw = load_sample("sample_incidents.csv")
    data_source_key = "sample:incidents"

if df_raw is None:
    st.info("← サイドバーからデータをアップロードするか、サンプルを選択してください。")
    st.markdown(
        """
        ### 機能一覧

        - **前処理** — fugashi/GiNZA/SudachiPy/spaCy 切替、正規化、同義語展開
        - **頻度・N-gram・KWIC・TF-IDF** — 5 種の共起指標 (Jaccard/Dice/PMI/t-score/対数尤度)
        - **共起ネットワーク** — Louvain/Leiden/Girvan-Newman、中心性、pyvis
        - **対応分析 (CA/MCA)** — カテゴリ変数 × 頻出語の布置
        - **クラスタリング** — KMeans/階層/DBSCAN + PCA/t-SNE/UMAP/MDS + SOM
        - **特徴語抽出** — χ²/対数尤度比/Jaccard
        - **トピックモデル** — LDA + pyLDAvis + 最適トピック数探索 / NMF
        - **分散表現** — Word2Vec (類似語・アナロジー) + SBERT
        - **類似検索** — SBERT + FAISS
        - **NER** — GiNZA/spaCy/ルールベース + 医療 NER
        - **係り受け** — verb-obj / subject-verb / adj-noun の抽出
        - **感情分析** — oseti (日) + VADER (英) + カスタム辞書 + 評価表現抽出
        - **時系列** — 単語トレンド + 急増語 + 期間別ネットワーク
        - **コーディング** — YAML ルール → コード別頻度・クロス集計
        """
    )
    st.stop()

df_raw = df_raw.reset_index(drop=True)
st.success(f"読み込み完了: {len(df_raw)} 件 × {len(df_raw.columns)} 列")

# =============================================================================
# サイドバー 2: 列設定 + 前処理パラメータ
# =============================================================================
cols = list(df_raw.columns)
with st.sidebar:
    st.header("🔧 2. 列設定")
    text_col = st.selectbox("テキスト列", cols, index=0)
    date_col = st.selectbox("日付列 (任意)", ["(なし)"] + cols, index=0)
    label_col = st.selectbox("カテゴリ列 (任意)", ["(なし)"] + cols, index=0)

raw_texts = df_raw[text_col].fillna("").astype(str).tolist()
auto_lang = sample_language(raw_texts)

with st.sidebar:
    st.header("⚙️ 3. 前処理")
    language = st.radio(
        "言語", ["auto", "ja", "en"], index=0, horizontal=True,
        help=f"自動判定: {auto_lang}",
    )
    resolved_lang = auto_lang if language == "auto" else language
    engine = st.selectbox(
        "形態素解析エンジン",
        ["auto", "fugashi", "ginza", "sudachi", "spacy_en", "simple"],
        index=0,
    )
    as_lemma = st.checkbox("原形 (lemma) で抽出", value=True)
    lowercase_en = st.checkbox("英語を小文字化", value=True)
    min_token_len = st.number_input("最小トークン長", 1, 10, 2)

    st.header("🚫 4. 追加ストップワード")
    extra_sw_text = st.text_area(
        "解析から除外する語",
        value="",
        height=100,
        help="1 行 1 語 or カンマ区切り。大小文字・末尾コロンを自動展開。",
        placeholder="する\nある, いる",
    )
    extra_stopwords = parse_user_stopwords(extra_sw_text)
    if extra_stopwords:
        st.caption(f"追加除外: {len(extra_stopwords)} 語")

    st.header("⚡ 5. 並列処理")
    cpu_count = os.cpu_count() or 2
    n_jobs = st.slider(
        "n_jobs (spaCy のみ有効)",
        1, max(cpu_count, 2), min(cpu_count, 4),
    )

# =============================================================================
# 計算 (キャッシュ)
# =============================================================================
data_params_key = (
    data_source_key,
    text_col,
    language,
    engine,
    as_lemma,
    lowercase_en,
    n_jobs,
)
params_hash = hashlib.md5(str(data_params_key).encode()).hexdigest()


@st.cache_data(show_spinner="正規化中…", max_entries=3)
def _normalize_cached(key: str, _texts: tuple, language: str, lowercase_en: bool) -> list[str]:
    return normalize_texts(list(_texts), language=language, lowercase_en=lowercase_en)


@st.cache_data(show_spinner="形態素解析中…", max_entries=3)
def _tokenize_cached(
    key: str, _texts: tuple, engine: str, language: str, as_lemma: bool, n_jobs: int,
) -> list[list[str]]:
    return tokenize_texts(
        list(_texts), engine=engine, language=language,
        as_lemma=as_lemma, n_jobs=n_jobs,
    )


norm_texts = _normalize_cached(params_hash, tuple(raw_texts), language, lowercase_en)
tokens_full = _tokenize_cached(
    params_hash, tuple(norm_texts), engine, language, as_lemma, n_jobs,
)

# ストップワード除去 (軽量・非キャッシュ)
tokens_filtered = filter_tokens(
    tokens_full, language=resolved_lang,
    extra_stopwords=extra_stopwords, min_token_len=int(min_token_len),
)


# =============================================================================
# 概要メトリクス
# =============================================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("文書数", len(df_raw))
m2.metric("総トークン数", sum(len(t) for t in tokens_filtered))
m3.metric("語彙サイズ", len({w for t in tokens_filtered for w in t}))
m4.metric("判定言語", resolved_lang)

# =============================================================================
# タブ
# =============================================================================
TABS = st.tabs([
    "📂 データ/前処理",  # 1
    "📊 頻度・KWIC",       # 2
    "🕸️ 共起NW",          # 3
    "🔬 多変量解析",        # 4
    "📑 トピック",          # 5
    "🧬 埋め込み&類似検索",  # 6
    "🧭 NER/係り受け",      # 7
    "💬 感情分析",          # 8
    "📆 時系列",            # 9
    "📋 コーディング",       # 10
])


# -----------------------------------------------------------------------------
# Tab 1: データ & 前処理プレビュー
# -----------------------------------------------------------------------------
with TABS[0]:
    st.subheader("データプレビュー")
    st.dataframe(df_raw.head(30), width="stretch")

    st.subheader("前処理プレビュー")
    idx = st.number_input("行番号", 0, max(len(df_raw) - 1, 0), 0, key="preview_row")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**原文**")
        st.text_area("", raw_texts[idx], height=150, key="prev_raw", label_visibility="collapsed")
    with c2:
        st.markdown("**正規化後**")
        st.text_area("", norm_texts[idx], height=150, key="prev_norm", label_visibility="collapsed")
    with c3:
        st.markdown("**トークン (フィルタ後)**")
        st.write(tokens_filtered[idx])


# -----------------------------------------------------------------------------
# Tab 2: 頻度・N-gram・TF-IDF・KWIC・共起統計
# -----------------------------------------------------------------------------
with TABS[1]:
    from jp_nlp_toolkit import (
        KWIC, cooccurrence_stats, ngram_frequency, tfidf, visualize,
        word_frequency,
    )

    top_n = st.slider("上位 N", 10, 300, 50, key="freq_top_n")

    st.subheader("単語頻度")
    freq_df = word_frequency(tokens_filtered, top_n=top_n)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(visualize.bar_frequency(freq_df, top_n=top_n), width="stretch")
    with c2:
        st.dataframe(freq_df, width="stretch", height=500)
        st.download_button(
            "CSV ダウンロード", freq_df.to_csv(index=False).encode("utf-8"),
            file_name="word_frequency.csv", mime="text/csv",
        )

    st.divider()
    st.subheader("N-gram")
    n = st.selectbox("N", [2, 3, 4, 5], index=0, key="ngram_n")
    st.dataframe(ngram_frequency(tokens_filtered, n=n, top_n=top_n), width="stretch")

    st.divider()
    st.subheader("TF-IDF")
    if st.button("計算", key="tfidf_btn"):
        X, feats = tfidf(tokens_filtered)
        st.write(f"行列: {X.shape}, 特徴語: {len(feats)}")
        scores = X.mean(axis=0)
        df_tf = (
            pd.DataFrame({"word": feats, "mean_tfidf": scores})
            .sort_values("mean_tfidf", ascending=False)
            .head(top_n)
        )
        st.dataframe(df_tf, width="stretch")

    st.divider()
    st.subheader("KWIC コンコーダンス")
    kwic_keyword = st.text_input("キーワード", key="kwic_kw")
    kwic_window = st.slider("窓幅", 1, 20, 5, key="kwic_win")
    if kwic_keyword:
        kwic_corpus = [" ".join(t) for t in tokens_filtered]
        kwic = KWIC(kwic_corpus)
        res_kwic = kwic.search(kwic_keyword, window=kwic_window)
        st.dataframe(res_kwic, width="stretch")

    st.divider()
    st.subheader("共起統計量")
    co_target = st.text_input("対象語", key="co_target")
    co_measure = st.selectbox(
        "指標", ["jaccard", "dice", "pmi", "t_score", "log_likelihood"],
        key="co_measure",
    )
    if co_target:
        st.dataframe(
            cooccurrence_stats(tokens_filtered, co_target, measure=co_measure, top_n=30),
            width="stretch",
        )


# -----------------------------------------------------------------------------
# Tab 3: 共起ネットワーク
# -----------------------------------------------------------------------------
with TABS[2]:
    from jp_nlp_toolkit import CooccurrenceNetwork

    c1, c2, c3 = st.columns(3)
    scope = c1.selectbox("共起範囲", ["document", "sentence", "window"], key="net_scope")
    min_edge_weight = c2.number_input("最小エッジ重み", 1, 50, 3, key="net_min_ew")
    top_nodes = c3.number_input("ノード数上限", 20, 500, 100, step=10, key="net_top_nodes")

    c4, c5 = st.columns(2)
    edge_measure = c4.selectbox("エッジ指標", ["jaccard", "dice", "pmi"], key="net_edge_m")
    community_alg = c5.selectbox(
        "コミュニティ検出", ["louvain", "greedy", "girvan_newman", "(なし)"],
        key="net_comm_alg",
    )

    if st.button("構築", key="net_build"):
        with st.spinner("ネットワーク構築中…"):
            net = CooccurrenceNetwork(tokens_filtered, scope=scope, window_size=5)
            G = net.build(
                min_edge_weight=int(min_edge_weight),
                measure=edge_measure, top_n_nodes=int(top_nodes),
            )
            st.write(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

            communities = None
            if community_alg != "(なし)":
                try:
                    communities = net.detect_communities(community_alg)
                    st.caption(f"コミュニティ数: {len(set(communities.values()))}")
                except ImportError as e:
                    st.warning(str(e))

            # pyvis インタラクティブ可視化
            out_html = str(OUTPUT_DIR / "network.html")
            try:
                net.visualize(
                    backend="pyvis", output=out_html, communities=communities,
                )
                st.components.v1.html(
                    Path(out_html).read_text(encoding="utf-8"),
                    height=720, scrolling=True,
                )
            except ImportError as e:
                st.warning(str(e))

            # ノード/エッジ表 + 中心性
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**ノード (上位)**")
                node_df = net.node_dataframe().sort_values("frequency", ascending=False).head(30)
                st.dataframe(node_df, width="stretch")
            with cB:
                st.markdown("**中心性**")
                c_measure = st.selectbox(
                    "指標", ["betweenness", "closeness", "eigenvector", "pagerank", "degree"],
                    key="cent_m",
                )
                try:
                    cent = net.centrality(c_measure)
                    cent_df = pd.DataFrame(
                        sorted(cent.items(), key=lambda x: -x[1]),
                        columns=["node", c_measure],
                    ).head(30)
                    st.dataframe(cent_df, width="stretch")
                except Exception as e:
                    st.warning(str(e))


# -----------------------------------------------------------------------------
# Tab 4: 多変量解析 (CA, クラスタリング, SOM, 特徴語)
# -----------------------------------------------------------------------------
with TABS[3]:
    sub = st.radio(
        "手法",
        ["対応分析 (CA)", "クラスタリング + 次元削減", "SOM", "特徴語抽出"],
        horizontal=True, key="mv_sub",
    )

    if sub == "対応分析 (CA)":
        from jp_nlp_toolkit import CorrespondenceAnalysis
        if label_col == "(なし)":
            st.info("カテゴリ列をサイドバーで選択してください。")
        else:
            top_words_ca = st.slider("対象上位語数", 20, 200, 50, key="ca_top")
            if st.button("実行", key="ca_run"):
                try:
                    ca = CorrespondenceAnalysis(n_components=2).fit(
                        tokens_filtered, df_raw[label_col].astype(str).tolist(),
                        top_n_words=top_words_ca,
                    )
                    st.pyplot(ca.plot())
                    wc, lc = ca.get_coordinates()
                    cA, cB = st.columns(2)
                    cA.markdown("**語の座標**")
                    cA.dataframe(wc.head(30), width="stretch")
                    cB.markdown("**カテゴリの座標**")
                    cB.dataframe(lc, width="stretch")
                except ImportError as e:
                    st.error(f"依存ライブラリ不足: {e}")

    elif sub == "クラスタリング + 次元削減":
        from jp_nlp_toolkit import DimensionReducer, DocumentClustering, visualize

        c1, c2, c3 = st.columns(3)
        method = c1.selectbox("クラスタリング", ["kmeans", "hierarchical", "dbscan"])
        k = c2.number_input("k", 2, 30, 5)
        reducer = c3.selectbox("次元削減", ["umap", "pca", "tsne", "mds"])

        if st.button("実行", key="clu_run"):
            with st.spinner("計算中…"):
                clu = DocumentClustering(method=method, n_clusters=int(k))
                labels = clu.fit_predict(tokens_filtered)
                try:
                    st.metric("silhouette", f"{clu.silhouette_score():.3f}")
                except Exception:
                    pass
                try:
                    red = DimensionReducer(method=reducer)
                    coords = red.fit_transform(clu.X_)
                    st.plotly_chart(
                        visualize.scatter_2d(
                            coords, labels=labels.tolist(),
                            texts=[str(i) for i in range(len(coords))],
                        ),
                        width="stretch",
                    )
                except ImportError as e:
                    st.error(str(e))

                if method == "hierarchical":
                    st.pyplot(clu.plot_dendrogram())

                # クラスタ毎の代表文書
                st.subheader("クラスタ毎の代表文書")
                for c in sorted(set(labels)):
                    idxs = [i for i, l in enumerate(labels) if l == c]
                    with st.expander(f"クラスタ {c} ({len(idxs)} 件)"):
                        st.dataframe(df_raw.iloc[idxs].head(5), width="stretch")

    elif sub == "SOM":
        from jp_nlp_toolkit import SOM
        from sklearn.feature_extraction.text import TfidfVectorizer

        x_size = st.number_input("X", 4, 20, 10, key="som_x")
        y_size = st.number_input("Y", 4, 20, 10, key="som_y")
        iters = st.number_input("iterations", 100, 5000, 500, step=100, key="som_it")
        if st.button("SOM 学習", key="som_run"):
            try:
                joined = [" ".join(t) for t in tokens_filtered]
                X = TfidfVectorizer(token_pattern=r"(?u)\S+").fit_transform(joined).toarray()
                som = SOM(x=int(x_size), y=int(y_size)).fit(X, iterations=int(iters))
                st.pyplot(som.plot())
            except ImportError as e:
                st.error(f"依存ライブラリ不足 (pip install minisom): {e}")

    else:  # 特徴語抽出
        from jp_nlp_toolkit import compare_groups
        if label_col == "(なし)":
            st.info("カテゴリ列をサイドバーで選択してください。")
        else:
            measure = st.radio(
                "指標", ["chi2", "log_likelihood", "jaccard"], horizontal=True, key="fw_m",
            )
            top_n_fw = st.slider("上位 N (グループ毎)", 5, 50, 15, key="fw_top")
            if st.button("実行", key="fw_run"):
                out = compare_groups(
                    tokens_filtered,
                    df_raw[label_col].astype(str).tolist(),
                    measure=measure, top_n=top_n_fw,
                )
                st.dataframe(out, width="stretch")


# -----------------------------------------------------------------------------
# Tab 5: トピックモデル (LDA / NMF)
# -----------------------------------------------------------------------------
with TABS[4]:
    from jp_nlp_toolkit import LDATopicModel, NMFTopicModel

    model_type = st.radio("モデル", ["LDA", "NMF"], horizontal=True, key="topic_model")
    c1, c2 = st.columns(2)
    n_topics = c1.number_input("トピック数", 2, 50, 10, key="topic_n")
    n_words_show = c2.number_input("各トピックで表示する語数", 5, 30, 10, key="topic_nw")

    if model_type == "LDA":
        passes = st.number_input("passes", 1, 50, 5, key="topic_passes")
        c3, c4 = st.columns(2)
        run_opt = c3.checkbox("最適トピック数を探索", key="topic_opt")
        if run_opt:
            opt_lo = c4.number_input("探索範囲 開始", 2, 30, 2, key="topic_opt_lo")
            opt_hi = c4.number_input("探索範囲 終了", 4, 50, 20, key="topic_opt_hi")

        if st.button("LDA 実行", key="topic_run_lda"):
            with st.spinner("学習中…"):
                try:
                    lda = LDATopicModel(n_topics=int(n_topics), passes=int(passes)).fit(tokens_filtered)
                    st.subheader("トピック語")
                    st.dataframe(lda.topics_dataframe(n_words=int(n_words_show)), width="stretch")

                    try:
                        st.metric("coherence (c_v)", f"{lda.coherence_score():.3f}")
                    except Exception:
                        pass

                    # pyLDAvis
                    vis_path = str(OUTPUT_DIR / "lda.html")
                    try:
                        lda.visualize(vis_path)
                        st.components.v1.html(
                            Path(vis_path).read_text(encoding="utf-8"),
                            height=800, scrolling=True,
                        )
                    except ImportError:
                        pass

                    if run_opt:
                        st.subheader("最適トピック数探索")
                        df_opt = lda.optimal_n_topics(range_n=(int(opt_lo), int(opt_hi)))
                        st.line_chart(df_opt.set_index("n_topics"))
                        st.dataframe(df_opt, width="stretch")
                except ImportError as e:
                    st.error(str(e))
    else:
        if st.button("NMF 実行", key="topic_run_nmf"):
            with st.spinner("学習中…"):
                nmf = NMFTopicModel(n_topics=int(n_topics)).fit(tokens_filtered)
                st.dataframe(nmf.topics_dataframe(n_words=int(n_words_show)), width="stretch")


# -----------------------------------------------------------------------------
# Tab 6: 埋め込み & 類似検索 (Word2Vec / SBERT / FAISS)
# -----------------------------------------------------------------------------
with TABS[5]:
    sub = st.radio(
        "モード", ["Word2Vec", "SBERT + FAISS 類似検索"],
        horizontal=True, key="emb_sub",
    )

    if sub == "Word2Vec":
        from jp_nlp_toolkit import Word2VecTrainer

        c1, c2, c3 = st.columns(3)
        vs = c1.number_input("vector_size", 50, 500, 200, step=50, key="w2v_vs")
        window = c2.number_input("window", 2, 20, 5, key="w2v_win")
        mc = c3.number_input("min_count", 1, 20, 3, key="w2v_mc")
        if st.button("学習", key="w2v_train"):
            with st.spinner("Word2Vec 学習中…"):
                st.session_state["_w2v"] = Word2VecTrainer(
                    vector_size=int(vs), window=int(window), min_count=int(mc),
                ).fit(tokens_filtered)
                st.success("学習完了")

        w2v = st.session_state.get("_w2v")
        if w2v is not None:
            st.divider()
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**類似語検索**")
                q_word = st.text_input("単語", key="w2v_q")
                if q_word:
                    sim = w2v.most_similar(q_word, topn=15)
                    if sim:
                        st.dataframe(pd.DataFrame(sim, columns=["word", "similarity"]), width="stretch")
                    else:
                        st.warning(f"'{q_word}' は語彙にありません。")
            with cB:
                st.markdown("**アナロジー (king - man + woman = ?)**")
                pos = st.text_input("positive (カンマ区切り)", key="w2v_pos")
                neg = st.text_input("negative (カンマ区切り)", key="w2v_neg")
                if pos:
                    p = [w.strip() for w in pos.split(",") if w.strip()]
                    n = [w.strip() for w in neg.split(",") if w.strip()] if neg else []
                    res = w2v.analogy(positive=p, negative=n, topn=10)
                    if res:
                        st.dataframe(pd.DataFrame(res, columns=["word", "similarity"]), width="stretch")

    else:  # SBERT + FAISS
        from jp_nlp_toolkit import SBERTEncoder, SimilaritySearch

        st.caption("※ 初回は SBERT モデルのダウンロードで数分かかります。")
        lang_model = st.selectbox(
            "SBERT モデル言語",
            ["multi (多言語)", "ja (日本語)", "en (英語)"],
            index=0, key="sbert_lang",
        )
        model_lang = lang_model.split()[0]
        index_type = st.selectbox("FAISS インデックス", ["flat", "ivf", "hnsw"], key="faiss_idx")
        if st.button("インデックス構築", key="faiss_build"):
            with st.spinner("埋め込み生成中 + インデックス構築中…"):
                try:
                    enc = SBERTEncoder(language=model_lang)
                    ss = SimilaritySearch(encoder=enc, index_type=index_type).build_index(raw_texts)
                    st.session_state["_sim_search"] = (ss, df_raw)
                    st.success("構築完了")
                except ImportError as e:
                    st.error(f"依存ライブラリ不足: {e}")

        state = st.session_state.get("_sim_search")
        if state is not None:
            st.divider()
            ss, df_ref = state
            q = st.text_input("検索クエリ", key="faiss_q")
            top_k = st.slider("top_k", 1, 50, 10, key="faiss_topk")
            if q:
                res = ss.search(q, top_k=top_k)
                st.dataframe(res, width="stretch")


# -----------------------------------------------------------------------------
# Tab 7: NER + 係り受け
# -----------------------------------------------------------------------------
with TABS[6]:
    sub = st.radio("機能", ["NER", "係り受け"], horizontal=True, key="lang_sub")

    if sub == "NER":
        from jp_nlp_toolkit import MedicalNER, NERExtractor

        c1, c2 = st.columns(2)
        ner_model = c1.selectbox("モデル", ["ginza", "spacy_en", "rule", "mednerj (医療)"], key="ner_m")
        max_docs = c2.number_input("対象文書数", 10, len(raw_texts), min(100, len(raw_texts)), key="ner_n")

        if st.button("NER 実行", key="ner_run"):
            with st.spinner("NER 実行中…"):
                try:
                    if ner_model.startswith("mednerj"):
                        ner = MedicalNER()
                    else:
                        ner = NERExtractor(model=ner_model)
                    agg = ner.aggregate(raw_texts[:int(max_docs)])
                    st.dataframe(agg.head(100), width="stretch")
                    # ラベル別集計
                    if "label" in agg.columns:
                        label_totals = agg.groupby("label")["count"].sum().sort_values(ascending=False)
                        st.bar_chart(label_totals)
                except (ImportError, RuntimeError) as e:
                    st.error(str(e))

    else:  # 係り受け
        from jp_nlp_toolkit import DependencyParser

        c1, c2 = st.columns(2)
        dep_engine = c1.selectbox("エンジン", ["ginza", "spacy_en"], key="dep_eng")
        relation = c2.selectbox(
            "抽出関係", ["verb_obj", "subject_verb", "adj_noun"], key="dep_rel",
        )

        sample_idx = st.number_input("1 文書を可視化", 0, len(raw_texts) - 1, 0, key="dep_idx")
        if st.button("可視化 + ペア抽出", key="dep_run"):
            try:
                dep = DependencyParser(engine=dep_engine)
                # displacy SVG
                svg = dep.visualize(raw_texts[sample_idx])
                st.components.v1.html(svg, height=250, scrolling=True)
                # 全文書からペア集計
                from collections import Counter
                pairs: Counter = Counter()
                with st.spinner("ペア抽出中…"):
                    for t in raw_texts[: min(len(raw_texts), 200)]:
                        pairs.update(dep.extract_pairs(t, relation=relation))
                pair_df = (
                    pd.DataFrame(pairs.most_common(50), columns=["pair", "count"])
                )
                st.dataframe(pair_df, width="stretch")
            except (ImportError, RuntimeError) as e:
                st.error(str(e))


# -----------------------------------------------------------------------------
# Tab 8: 感情分析
# -----------------------------------------------------------------------------
with TABS[7]:
    from jp_nlp_toolkit import EvaluationExtractor, SentimentAnalyzer

    method = st.radio(
        "手法", ["auto", "oseti (日)", "vader (英)", "custom_dict"],
        horizontal=True, key="sa_m",
    )
    resolved_m = {"oseti (日)": "oseti", "vader (英)": "vader"}.get(method, method)

    custom_dict: dict | None = None
    if method == "custom_dict":
        st.caption("word:score (1 行 1 語、正は positive、負は negative)")
        cdict_text = st.text_area(
            "", "改善:1\n悪化:-1\n効果:1\n副作用:-1", height=150,
            key="sa_cdict", label_visibility="collapsed",
        )
        custom_dict = {}
        for line in cdict_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    custom_dict[k.strip()] = int(v.strip())
                except ValueError:
                    pass

    if st.button("感情分析実行", key="sa_run"):
        with st.spinner("解析中…"):
            try:
                sa = SentimentAnalyzer(method=resolved_m, custom_dict=custom_dict)
                res = sa.analyze_df(df_raw, text_col=text_col)
                show_cols = [text_col, "sentiment_positive", "sentiment_negative", "sentiment_score"]
                st.dataframe(res[show_cols].head(50), width="stretch")
                st.line_chart(res["sentiment_score"].reset_index(drop=True))
                # ヒストグラム
                st.bar_chart(
                    res["sentiment_score"].round(1).value_counts().sort_index()
                )
                st.download_button(
                    "CSV ダウンロード",
                    res[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name="sentiment.csv", mime="text/csv",
                )
            except (ImportError, LookupError) as e:
                st.error(str(e))

    st.divider()
    st.subheader("評価表現抽出 (辞書ベース)")
    if st.button("抽出", key="eval_run"):
        ext = EvaluationExtractor()
        ev = ext.extract_df(df_raw, text_col=text_col)
        st.dataframe(ev.head(100), width="stretch")


# -----------------------------------------------------------------------------
# Tab 9: 時系列
# -----------------------------------------------------------------------------
with TABS[8]:
    from jp_nlp_toolkit import TemporalAnalyzer

    if date_col == "(なし)":
        st.info("日付列をサイドバーで選択してください。")
    else:
        df_ts = df_raw[[date_col]].copy()
        df_ts["_tokens"] = tokens_filtered
        try:
            ta = TemporalAnalyzer(df_ts, text_col="_tokens", date_col=date_col)
        except Exception as e:
            st.error(f"日付パースに失敗: {e}")
            st.stop()

        st.subheader("単語トレンド")
        c1, c2 = st.columns(2)
        freq = c1.selectbox("集計頻度", ["D", "W", "M", "Q", "Y"], index=2, key="ts_freq")
        words_in = c2.text_input("追跡する単語 (カンマ区切り)", key="ts_words")
        metric_ts = st.radio("指標", ["count", "freq"], horizontal=True, key="ts_metric")
        if words_in:
            words = [w.strip() for w in words_in.split(",") if w.strip()]
            st.pyplot(ta.plot_trend(words, freq=freq, metric=metric_ts))

        st.divider()
        st.subheader("急増語検出")
        c1, c2, c3 = st.columns(3)
        base = c1.text_input("基準期間 (例: 2023)", key="ts_base")
        win = c2.text_input("比較期間 (例: 2024)", key="ts_win")
        tn_em = c3.number_input("top_n", 5, 100, 20, key="ts_tn")
        if base and win and st.button("検出", key="ts_em_run"):
            emg = ta.emerging_words(window=win, baseline=base, top_n=int(tn_em))
            st.dataframe(emg, width="stretch")

        st.divider()
        st.subheader("期間別ネットワーク")
        if st.button("構築 (期間毎)", key="ts_net_run"):
            with st.spinner("期間別ネットワーク構築中…"):
                nets = ta.temporal_cooccurrence(freq=freq, min_edge_weight=2, top_n_nodes=30)
                for period, G in list(nets.items())[:8]:
                    st.caption(f"**{period}**: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")


# -----------------------------------------------------------------------------
# Tab 10: コーディング
# -----------------------------------------------------------------------------
with TABS[9]:
    from jp_nlp_toolkit import CodingRule

    st.caption("YAML 形式のコーディングルールを入力。")
    default_rules = """透析関連:
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
"""
    rules_text = st.text_area("ルール", default_rules, height=200, key="code_yaml")
    if st.button("適用", key="code_run"):
        import yaml
        try:
            rules = yaml.safe_load(rules_text)
            cr = CodingRule(rules)
            applied = cr.apply(raw_texts)
            st.subheader("コード別ヒット数")
            st.bar_chart(cr.frequency(raw_texts))
            st.subheader("文書 × コードのヒット")
            st.dataframe(applied.head(30), width="stretch")
            if label_col != "(なし)":
                st.subheader(f"コード × {label_col} クロス集計")
                xt = cr.cross_tab(raw_texts, df_raw[label_col].astype(str).tolist())
                st.dataframe(xt, width="stretch")
            st.subheader("コード間共起")
            G_code = cr.cooccurrence(raw_texts)
            st.write(f"nodes: {G_code.number_of_nodes()}, edges: {G_code.number_of_edges()}")
        except Exception as e:
            st.error(str(e))
