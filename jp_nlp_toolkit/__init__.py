"""jp-nlp-toolkit — 日本語/英語テキスト解析統合ツールキット (LLM非依存)。"""
from __future__ import annotations

__version__ = "0.1.0"

from .preprocess import (
    Tokenizer,
    Normalizer,
    SynonymExpander,
    Token,
    filter_pos,
    remove_stopwords,
    detect_language,
)
from .frequency import (
    word_frequency,
    ngram_frequency,
    tfidf,
    KWIC,
    cooccurrence_stats,
)
from .cooccurrence import CooccurrenceNetwork
from .correspondence import CorrespondenceAnalysis, MultipleCorrespondenceAnalysis
from .clustering import DocumentClustering, DimensionReducer, SOM
from .topic_model import LDATopicModel, NMFTopicModel
from .sentiment import SentimentAnalyzer, EvaluationExtractor
from .ner import NERExtractor, MedicalNER
from .dependency import DependencyParser
from .embedding import Word2VecTrainer, Doc2VecTrainer, SBERTEncoder
from .similarity import SimilaritySearch, cosine_similarity_matrix
from .feature_words import (
    chi2_feature_words,
    log_likelihood_feature_words,
    jaccard_feature_words,
    compare_groups,
)
from .timeseries import TemporalAnalyzer
from .coding import CodingRule, load_rules_yaml
from . import visualize

__all__ = [
    "Tokenizer", "Normalizer", "SynonymExpander", "Token",
    "filter_pos", "remove_stopwords", "detect_language",
    "word_frequency", "ngram_frequency", "tfidf", "KWIC", "cooccurrence_stats",
    "CooccurrenceNetwork",
    "CorrespondenceAnalysis", "MultipleCorrespondenceAnalysis",
    "DocumentClustering", "DimensionReducer", "SOM",
    "LDATopicModel", "NMFTopicModel",
    "SentimentAnalyzer", "EvaluationExtractor",
    "NERExtractor", "MedicalNER",
    "DependencyParser",
    "Word2VecTrainer", "Doc2VecTrainer", "SBERTEncoder",
    "SimilaritySearch", "cosine_similarity_matrix",
    "chi2_feature_words", "log_likelihood_feature_words",
    "jaccard_feature_words", "compare_groups",
    "TemporalAnalyzer",
    "CodingRule", "load_rules_yaml",
    "visualize",
]
