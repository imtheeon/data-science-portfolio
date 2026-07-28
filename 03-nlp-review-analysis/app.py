# app.py
"""Streamlit demo: type a review, see its predicted sentiment and nearest
topic — both computed live by the models trained in run_analysis.py."""

from __future__ import annotations

import joblib
import streamlit as st

from analysis.sentiment import score_sentiment

MODEL_DIR = "model"

st.set_page_config(page_title="NLP Review Analysis", layout="centered")
st.title("NLP Sentiment & Topic Analysis")
st.caption(
    "Real Amazon review data + classic/local NLP (VADER sentiment, NMF "
    "topic modeling) — no external API calls, fully local inference."
)

vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.joblib")
nmf_model = joblib.load(f"{MODEL_DIR}/nmf_model.joblib")
topic_words = joblib.load(f"{MODEL_DIR}/topic_words.joblib")

text = st.text_area(
    "Review text", "This product was amazing, fast shipping and great taste!"
)

if st.button("Analyze"):
    sentiment_df = score_sentiment([text])
    row = sentiment_df.iloc[0]
    st.metric("Predicted sentiment", row["predicted_label"], f"compound={row['compound']:.3f}")

    X = vectorizer.transform([text])
    topic_dist = nmf_model.transform(X)[0]
    top_topic = int(topic_dist.argmax())
    st.write(f"**Nearest topic**: Topic {top_topic} — {', '.join(topic_words[top_topic])}")
