import os
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

os.makedirs("results", exist_ok=True)

print("=" * 70)
print("STEP 3 - TRAINING CONTENT-BASED MODEL")
print("=" * 70)

movies = pd.read_csv("data/movies.csv")

movies["genres"] = movies["genres"].fillna("(unknown)")

# Genre metadata is the item content feature.
movies["content"] = (
    movies["genres"]
    .str.replace("|", " ", regex=False)
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b"
)

item_vectors = vectorizer.fit_transform(
    movies["content"]
)

item_vectors = normalize(item_vectors)

movie_to_index = {}

for i in range(len(movies)):
    movie_to_index[int(movies.iloc[i]["movieId"])] = i

content_model = {
    "movies": movies,
    "vectorizer": vectorizer,
    "item_vectors": item_vectors,
    "movie_to_index": movie_to_index
}

with open("results/content_model.pkl", "wb") as f:
    pickle.dump(content_model, f)

print("Movies:", len(movies))
print("TF-IDF features:", item_vectors.shape[1])
print("\nContent model saved:")
print("  results/content_model.pkl")
