import pickle
import numpy as np
import pandas as pd

print("=" * 70)
print("STEP 4 - HYBRID RECOMMENDER")
print("=" * 70)

train = pd.read_csv("data/train.csv")

with open("results/cf_model.pkl", "rb") as f:
    cf = pickle.load(f)

with open("results/content_model.pkl", "rb") as f:
    content = pickle.load(f)

movies = content["movies"]
item_vectors = content["item_vectors"]
movie_to_index = content["movie_to_index"]

user_history = {}

for user_id, group in train.groupby("userId"):
    user_history[int(user_id)] = group["movieId"].astype(int).tolist()

user_counts = {
    user_id: len(history)
    for user_id, history in user_history.items()
}

train_item_counts = (
    train.groupby("movieId").size().to_dict()
)

all_item_ids = [
    int(x) for x in movies["movieId"].tolist()
]

# -------------------------------------------------------------------
# Adaptive blending:
#   0 interactions -> CF 0.00, Content 1.00
#   1 interaction  -> CF 0.20, Content 0.80
#   2 interactions -> CF 0.40, Content 0.60
#   3 interactions -> CF 0.60, Content 0.40
#   4 interactions -> CF 0.80, Content 0.20
#   5+ interactions -> CF 1.00, Content 0.00
# -------------------------------------------------------------------

def get_weights(interaction_count):
    cf_weight = min(interaction_count / 5.0, 1.0)
    content_weight = 1.0 - cf_weight
    return cf_weight, content_weight


def min_max_normalize(score_dict):
    if len(score_dict) == 0:
        return {}

    values = np.array(list(score_dict.values()), dtype=float)
    low = values.min()
    high = values.max()

    if high == low:
        return {key: 0.0 for key in score_dict}

    return {
        key: (value - low) / (high - low)
        for key, value in score_dict.items()
    }


def cf_score(user_id, movie_id):
    if user_id not in cf["user_to_index"]:
        return cf["global_mean"]

    if movie_id not in cf["item_to_index"]:
        return cf["global_mean"]

    u = cf["user_to_index"][user_id]
    i = cf["item_to_index"][movie_id]

    return (
        cf["global_mean"]
        + np.dot(
            cf["user_factors"][u],
            cf["item_factors"][i]
        )
    )


def content_score_for_movie(user_id, movie_id):
    history = user_history.get(user_id, [])

    if len(history) == 0:
        return 0.0

    if movie_id not in movie_to_index:
        return 0.0

    candidate_index = movie_to_index[movie_id]

    similarities = []

    candidate_vector = item_vectors[candidate_index]

    for history_movie in history:
        if history_movie in movie_to_index:
            history_index = movie_to_index[history_movie]
            history_vector = item_vectors[history_index]

            similarity = float(
                candidate_vector.dot(history_vector.T).toarray()[0][0]
            )

            similarities.append(similarity)

    if len(similarities) == 0:
        return 0.0

    return max(similarities)


def recommend(user_id, k=10):
    history = set(user_history.get(user_id, []))

    candidates = [
        movie_id
        for movie_id in all_item_ids
        if movie_id not in history
    ]

    interaction_count = user_counts.get(user_id, 0)

    cf_weight, content_weight = get_weights(
        interaction_count
    )

    # Completely new user: use popularity fallback because
    # there is no user preference information to personalize from.
    if interaction_count == 0:
        popularity = sorted(
            train_item_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = []

        for movie_id, count in popularity:
            if movie_id not in history:
                result.append(movie_id)

            if len(result) == k:
                break

        return result, cf_weight, content_weight

    cf_scores = {}
    content_scores = {}

    for movie_id in candidates:
        cf_scores[movie_id] = cf_score(user_id, movie_id)
        content_scores[movie_id] = content_score_for_movie(
            user_id,
            movie_id
        )

    cf_scores = min_max_normalize(cf_scores)
    content_scores = min_max_normalize(content_scores)

    hybrid_scores = {}

    for movie_id in candidates:
        hybrid_scores[movie_id] = (
            cf_weight * cf_scores.get(movie_id, 0.0)
            + content_weight * content_scores.get(movie_id, 0.0)
        )

    ranked = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True
    )

    return ranked[:k], cf_weight, content_weight


# Demonstrate the adaptive weights.
print("\nAdaptive blending:")
print("Interactions | CF weight | Content weight")

for n in range(0, 7):
    a, b = get_weights(n)
    print(
        f"{n:12d} | {a:9.2f} | {b:14.2f}"
    )

# Demonstrate recommendations for a few users.
print("\nExample recommendations:")

example_users = list(user_history.keys())[:3]

for user_id in example_users:
    recommendations, cf_weight, content_weight = recommend(
        user_id,
        10
    )

    print("\nUser:", user_id)
    print("Training interactions:", user_counts[user_id])
    print("CF weight:", round(cf_weight, 2))
    print("Content weight:", round(content_weight, 2))

    for movie_id in recommendations:
        row = movies[movies["movieId"] == movie_id]

        if len(row) > 0:
            print(
                " ",
                movie_id,
                "-",
                row.iloc[0]["title"]
            )
