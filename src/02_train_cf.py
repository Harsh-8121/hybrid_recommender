import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

os.makedirs("results", exist_ok=True)

print("=" * 70)
print("STEP 2 - TRAINING COLLABORATIVE FILTERING MATRIX FACTORIZATION")
print("=" * 70)

train = pd.read_csv("data/train.csv")

user_ids = sorted(train["userId"].unique())
item_ids = sorted(train["movieId"].unique())

user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}

rows = []
cols = []
values = []

for row in train.itertuples(index=False):
    rows.append(user_to_index[row.userId])
    cols.append(item_to_index[row.movieId])
    values.append(float(row.rating))

R = csr_matrix(
    (values, (rows, cols)),
    shape=(len(user_ids), len(item_ids)),
    dtype=np.float32
)

global_mean = float(train["rating"].mean())

# Center ratings by the global mean.
R_centered = R.copy().astype(np.float32)
R_centered.data = R_centered.data - global_mean

# SVD rank.
k = min(50, min(R_centered.shape) - 1)

print("Users:", len(user_ids))
print("Items:", len(item_ids))
print("Matrix shape:", R.shape)
print("Latent factors:", k)
print("Running sparse SVD...")

U, singular_values, Vt = svds(R_centered, k=k)

# svds returns singular values in ascending order.
order = np.argsort(singular_values)[::-1]
singular_values = singular_values[order]
U = U[:, order]
Vt = Vt[order, :]

sqrt_s = np.sqrt(singular_values)

user_factors = U * sqrt_s
item_factors = Vt.T * sqrt_s

model = {
    "user_ids": user_ids,
    "item_ids": item_ids,
    "user_to_index": user_to_index,
    "item_to_index": item_to_index,
    "user_factors": user_factors.astype(np.float32),
    "item_factors": item_factors.astype(np.float32),
    "global_mean": global_mean
}

with open("results/cf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nCollaborative filtering model saved:")
print("  results/cf_model.pkl")
