import re

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="IMDb Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)


# ---------------------------------------------------
# LOAD NLTK RESOURCES
# ---------------------------------------------------
@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


load_nltk_resources()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------------------
def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Stopword removal + lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ---------------------------------------------------
# LOAD AND PREPROCESS DATA
# ---------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("data/imdb_movies_2024_6000.csv")

    # Remove missing storylines
    df = df.dropna(subset=["Storyline"])

    # Remove empty storylines
    df = df[
        df["Storyline"].astype(str).str.strip() != ""
    ]

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Clean storylines
    df["Cleaned_Storyline"] = (
        df["Storyline"]
        .astype(str)
        .apply(clean_text)
    )

    # Remove empty cleaned storylines
    df = df[
        df["Cleaned_Storyline"].str.strip() != ""
    ].reset_index(drop=True)

    return df


df = load_data()


# ---------------------------------------------------
# TF-IDF MODEL
# ---------------------------------------------------
@st.cache_resource
def create_tfidf(cleaned_storylines):

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(cleaned_storylines)

    return tfidf, tfidf_matrix


tfidf, tfidf_matrix = create_tfidf(
    tuple(df["Cleaned_Storyline"])
)


# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------
def recommend_movies(user_storyline, top_n=5):

    cleaned_input = clean_text(user_storyline)

    input_vector = tfidf.transform([cleaned_input])

    similarity_scores = cosine_similarity(
        input_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    recommendations = df.iloc[top_indices][
        ["Movie_Title", "Storyline"]
    ].copy()

    recommendations["Similarity_Score"] = (
        similarity_scores[top_indices]
    )

    return recommendations


# ---------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------
st.title("🎬 IMDb Movie Recommendation System")

st.write(
    "Enter a movie storyline and discover the "
    "5 most similar IMDb movies from 2024."
)

st.divider()

user_storyline = st.text_area(
    "Enter a movie storyline:",
    height=150,
    placeholder=(
        "Example: A young hero begins a dangerous journey "
        "and fights powerful enemies to save his world..."
    )
)

if st.button("Recommend Movies", type="primary"):

    if not user_storyline.strip():

        st.warning("Please enter a movie storyline.")

    else:

        with st.spinner("Finding similar movies..."):

            recommendations = recommend_movies(
                user_storyline,
                top_n=5
            )

        st.subheader("Top 5 Recommended Movies")

        for rank, (_, movie) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            similarity_percentage = (
                movie["Similarity_Score"] * 100
            )

            st.markdown(
                f"### {rank}. {movie['Movie_Title']}"
            )

            st.write(movie["Storyline"])

            st.write(
                f"**Similarity Score:** "
                f"{movie['Similarity_Score']:.4f} "
                f"({similarity_percentage:.2f}%)"
            )

            st.divider()