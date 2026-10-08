import os
import pandas as pd
import numpy as np

DATA_PATH = "data/ml-1m"
os.makedirs("data", exist_ok=True)

print("=" * 70)
print("STEP 1 - PREPARING MOVIELENS DATA")
print("=" * 70)

movies = pd.read_csv(
    DATA_PATH + "/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movieId", "title", "genres"]
)

ratings = pd.read_csv(
    DATA_PATH + "/ratings.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["userId", "movieId", "rating", "timestamp"]
)

ratings["timestamp"] = pd.to_numeric(ratings["timestamp"])
ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

# Keep the last interaction of each user for ranking evaluation.
train_parts = []
test_parts = []

for user_id, group in ratings.groupby("userId"):
    if len(group) >= 2:
        train_parts.append(group.iloc[:-1])
        test_parts.append(group.iloc[-1:])
    else:
        train_parts.append(group)

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

train = train[["userId", "movieId", "rating", "timestamp"]]
test = test[["userId", "movieId", "rating", "timestamp"]]

movies.to_csv("data/movies.csv", index=False)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Movies:", len(movies))
print("Training interactions:", len(train))
print("Test interactions:", len(test))
print("Users:", ratings["userId"].nunique())
print("Items:", ratings["movieId"].nunique())

print("\nTraining interaction count per user:")
print(train.groupby("userId").size().describe())

print("\nTraining interaction count per item:")
print(train.groupby("movieId").size().describe())

print("\nSaved:")
print("  data/movies.csv")
print("  data/train.csv")
print("  data/test.csv")
