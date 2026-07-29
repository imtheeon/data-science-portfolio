# app.py
"""Streamlit demo: pick a real MovieLens user, see live recommendations
computed by the same item-based CF logic used in run_analysis.py."""

from __future__ import annotations

import streamlit as st

from analysis.collaborative_filtering import (
    build_user_item_matrix,
    compute_item_similarity,
    recommend_for_user,
)
from data.load_movielens import load_movies, load_ratings

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Movie Recommender (Item-Based Collaborative Filtering)")
st.caption(
    "Real MovieLens 100k ratings — recommendations computed live from "
    "real rating patterns, not hardcoded. This demo trains on the full "
    "100k ratings (not the u1.base split used for the published RMSE/"
    "Precision@5 in the README), so it isn't literally the evaluated model."
)


@st.cache_data
def _load():
    ratings = load_ratings()
    movies = load_movies().set_index("item_id")["title"]
    matrix = build_user_item_matrix(ratings)
    similarity = compute_item_similarity(matrix)
    return matrix, similarity, movies


matrix, similarity, movies = _load()

user_id = st.selectbox("Pick a real MovieLens user ID", sorted(matrix.index.tolist()))
n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend"):
    recs = recommend_for_user(matrix, similarity, user_id, n=n)
    st.write(f"**Top {n} recommendations for user {user_id}:**")
    for item_id in recs:
        st.write(f"- {movies.get(item_id, f'item {item_id}')}")

    st.write("**Movies this user already rated highly (>=4):**")
    already = matrix.loc[user_id]
    liked = already[already >= 4].index[:10]
    for item_id in liked:
        st.write(f"- {movies.get(item_id, f'item {item_id}')}")
