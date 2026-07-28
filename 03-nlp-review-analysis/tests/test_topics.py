from analysis.topics import dominant_topics, fit_topic_model


def test_fit_topic_model_separates_distinct_themes():
    dog_docs = ["dog puppy bark leash walk dog park dog treat"] * 10
    pizza_docs = ["pizza cheese pepperoni oven slice pizza box pizza delivery"] * 10
    texts = dog_docs + pizza_docs

    model, vectorizer, topic_words, W = fit_topic_model(texts, n_topics=2, n_top_words=5)

    assert len(topic_words) == 2
    all_words = {w for words in topic_words for w in words}
    assert "dog" in all_words
    assert "pizza" in all_words

    dominant = dominant_topics(W)
    dog_topics = set(dominant[:10])
    pizza_topics = set(dominant[10:])
    assert len(dog_topics) == 1
    assert len(pizza_topics) == 1
    assert dog_topics != pizza_topics
