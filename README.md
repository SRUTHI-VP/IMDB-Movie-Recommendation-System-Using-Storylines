# 🎬 IMDb Movie Recommendation System Using Storylines

## 📌 Project Overview

This project is a **Content-Based Movie Recommendation System** that recommends movies based on the similarity of their storylines.

Movie names and storylines from IMDb 2024 movies are collected using **Selenium Web Scraping**. The storyline text is preprocessed using Natural Language Processing (NLP) techniques and converted into numerical vectors using **TF-IDF Vectorization**.

When a user enters a storyline, **Cosine Similarity** is used to compare the user's storyline with the movie storylines in the dataset. The system then recommends the **Top 5 most similar movies**.

An interactive web application is developed using **Streamlit**.

---

## 🎯 Objective

The main objective of this project is to build a movie recommendation system where:

- The user enters a movie storyline.
- The storyline is preprocessed using NLP techniques.
- The input is converted into a TF-IDF vector.
- Cosine Similarity is calculated between the user input and movie storylines.
- The Top 5 most similar movies are recommended.

---

## 🛠️ Technologies Used

- Python
- Selenium
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Streamlit
- Jupyter Notebook
- VS Code

---

## 📊 Dataset

The dataset contains IMDb movies released in **2024**.

The main columns used are:

| Column | Description |
|---|---|
| Movie_Title | Name of the movie |
| Storyline | Plot/storyline of the movie |

The movie information was scraped from IMDb using Selenium and stored in CSV format.

---

## 🔄 Project Workflow

### 1. Data Collection

Movie titles and storylines were scraped from IMDb using **Selenium**.

The collected data was saved as a CSV file.

### 2. Data Preprocessing

The storyline text was cleaned using NLP preprocessing techniques.

The preprocessing includes:

- Handling missing storylines
- Removing empty storylines
- Removing duplicate records
- Converting text to lowercase
- Removing URLs
- Removing numbers and special characters
- Removing punctuation
- Removing stop words
- Lemmatization

After preprocessing, the dataset contained **5,731 usable movie storylines**.

### 3. TF-IDF Vectorization

The cleaned movie storylines were converted from textual data into numerical vectors using:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
