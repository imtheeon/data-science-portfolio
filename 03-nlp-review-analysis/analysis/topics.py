"""Topic extraction via TF-IDF + NMF (Non-negative Matrix Factorization) —
a standard, fully local/free approach to unsupervised topic modeling."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_topic_model(
    texts: list[str],
    n_topics: int = 8,
    n_top_words: int = 10,
    random_state: int = 42,
):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)

    model = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=400)
    W = model.fit_transform(X)

    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for topic in model.components_:
        top_indices = topic.argsort()[::-1][:n_top_words]
        topic_words.append([feature_names[i] for i in top_indices])

    return model, vectorizer, topic_words, W


def dominant_topics(W: np.ndarray) -> np.ndarray:
    return np.argmax(W, axis=1)
