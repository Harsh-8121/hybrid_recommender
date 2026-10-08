import os
import pickle
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)

print("=" * 70)
print("STEP 5 - RANKING EVALUATION")
print("=" * 70)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

with open("results/cf_model.pkl", "rb") as f:
    cf = pickle.load(f)

with open("results/content_model.pkl", "rb") as f:
    content = pickle.load(f)

movies = content["movies"]
item_vectors = content["item_vectors"]
movie_to_index = content["movie_to_index"]

# ---------------------------------------------------------------
# Histories and counts
# ---------------------------------------------------------------

user_history = {}

for user_id, group in train.groupby("userId"):
    user_history[int(user_id)] = group["movieId"].astype(int).tolist()

user_counts = {
    user_id: len(history)
    for user_id, history in user_history.items()
}

item_counts = (
    train.groupby("movieId").size().to_dict()
)

all_item_ids = [
    int(x) for x in movies["movieId"].tolist()
]

# ---------------------------------------------------------------
# Popularity baseline
# ---------------------------------------------------------------

popular_movies = [
    int(movie_id)
    for movie_id in (
        train.groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .index
    )
]

# ---------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------

def normalize(score_dict):
    if len(score_dict) == 0:
        return {}

    values = np.array(
        list(score_dict.values()),
        dtype=float
    )

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


def content_score(user_id, movie_id):
    history = user_history.get(user_id, [])

    if len(history) == 0:
        return 0.0

    if movie_id not in movie_to_index:
        return 0.0

    candidate_index = movie_to_index[movie_id]
    candidate_vector = item_vectors[candidate_index]

    best = 0.0

    for history_movie in history:
        if history_movie not in movie_to_index:
            continue

        history_index = movie_to_index[history_movie]
        history_vector = item_vectors[history_index]

        similarity = float(
            candidate_vector.dot(history_vector.T).toarray()[0][0]
        )

        if similarity > best:
            best = similarity

    return best


def get_weights(count):
    cf_weight = min(count / 5.0, 1.0)
    content_weight = 1.0 - cf_weight
    return cf_weight, content_weight


# ---------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------

def hybrid_recommend(user_id, k=10):
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

    # No interaction history at all:
    # no personalized score exists, so use popularity fallback.
    if interaction_count == 0:
        result = []

        for movie_id in popular_movies:
            if movie_id not in history:
                result.append(movie_id)

            if len(result) == k:
                break

        return result

    cf_scores = {}
    content_scores = {}

    for movie_id in candidates:
        cf_scores[movie_id] = cf_score(
            user_id,
            movie_id
        )

        content_scores[movie_id] = content_score(
            user_id,
            movie_id
        )

    cf_scores = normalize(cf_scores)
    content_scores = normalize(content_scores)

    final_scores = {}

    for movie_id in candidates:
        final_scores[movie_id] = (
            cf_weight * cf_scores.get(movie_id, 0.0)
            + content_weight * content_scores.get(movie_id, 0.0)
        )

    ranked = sorted(
        final_scores,
        key=final_scores.get,
        reverse=True
    )

    return ranked[:k]


def popularity_recommend(user_id, k=10):
    history = set(user_history.get(user_id, []))

    result = []

    for movie_id in popular_movies:
        if movie_id not in history:
            result.append(movie_id)

        if len(result) == k:
            break

    return result


# ---------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------

def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    hits = sum(
        1 for movie_id in recommended
        if movie_id in relevant
    )

    return hits / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = sum(
        1 for movie_id in recommended
        if movie_id in relevant
    )

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    dcg = 0.0

    for position, movie_id in enumerate(recommended, start=1):
        if movie_id in relevant:
            dcg += 1.0 / np.log2(position + 1)

    ideal_hits = min(len(relevant), k)

    idcg = 0.0

    for position in range(1, ideal_hits + 1):
        idcg += 1.0 / np.log2(position + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------------
# User slices
# ---------------------------------------------------------------

test_users = [
    int(x) for x in test["userId"].unique()
]

cold_users = []
warm_users = []

for user_id in test_users:
    count = user_counts.get(user_id, 0)

    if count < 5:
        cold_users.append(user_id)
    else:
        warm_users.append(user_id)

# ---------------------------------------------------------------
# Item slices
# ---------------------------------------------------------------

test_items = [
    int(x) for x in test["movieId"].unique()
]

cold_items = []
warm_items = []

for movie_id in test_items:
    count = item_counts.get(movie_id, 0)

    if count < 5:
        cold_items.append(movie_id)
    else:
        warm_items.append(movie_id)

cold_item_set = set(cold_items)

print("\nUSER SLICES")
print("-" * 50)
print("Cold-start users (<5 interactions):", len(cold_users))
print("Warm users (>=5 interactions):", len(warm_users))

print("\nITEM SLICES")
print("-" * 50)
print("Cold-start items (<5 training interactions):", len(cold_items))
print("Warm items (>=5 training interactions):", len(warm_items))

# ---------------------------------------------------------------
# Evaluate users
# ---------------------------------------------------------------

def evaluate_users(users, recommender, k=10):
    precision_values = []
    recall_values = []
    ndcg_values = []

    for user_id in users:
        actual = test[
            test["userId"] == user_id
        ]["movieId"].astype(int).tolist()

        relevant = set(actual)

        recommended = recommender(
            user_id,
            k
        )

        precision_values.append(
            precision_at_k(
                recommended,
                relevant,
                k
            )
        )

        recall_values.append(
            recall_at_k(
                recommended,
                relevant,
                k
            )
        )

        ndcg_values.append(
            ndcg_at_k(
                recommended,
                relevant,
                k
            )
        )

    return {
        "Precision@10": np.mean(precision_values),
        "Recall@10": np.mean(recall_values),
        "NDCG@10": np.mean(ndcg_values)
    }


print("\nEvaluating hybrid model...")
cold_results = evaluate_users(
    cold_users,
    hybrid_recommend,
    10
)

warm_results = evaluate_users(
    warm_users,
    hybrid_recommend,
    10
)

print("\nEvaluating popularity baseline...")
cold_pop_results = evaluate_users(
    cold_users,
    popularity_recommend,
    10
)

warm_pop_results = evaluate_users(
    warm_users,
    popularity_recommend,
    10
)

# ---------------------------------------------------------------
# Item cold-start evaluation
# ---------------------------------------------------------------

def evaluate_cold_item_recall(users, k=10):
    values = []

    for user_id in users:
        actual = test[
            test["userId"] == user_id
        ]["movieId"].astype(int).tolist()

        relevant_cold = set(
            movie_id
            for movie_id in actual
            if movie_id in cold_item_set
        )

        if len(relevant_cold) == 0:
            continue

        recommendations = hybrid_recommend(
            user_id,
            k
        )

        hits = sum(
            1
            for movie_id in recommendations
            if movie_id in relevant_cold
        )

        values.append(
            hits / len(relevant_cold)
        )

    if len(values) == 0:
        return 0.0, 0

    return float(np.mean(values)), len(values)


cold_item_recall, cold_item_users_evaluated = (
    evaluate_cold_item_recall(
        test_users,
        10
    )
)

# ---------------------------------------------------------------
# Print results
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nHYBRID - COLD-START USERS (<5)")
print(
    "Precision@10:",
    round(cold_results["Precision@10"], 6)
)
print(
    "Recall@10:",
    round(cold_results["Recall@10"], 6)
)
print(
    "NDCG@10:",
    round(cold_results["NDCG@10"], 6)
)

print("\nHYBRID - WARM USERS (>=5)")
print(
    "Precision@10:",
    round(warm_results["Precision@10"], 6)
)
print(
    "Recall@10:",
    round(warm_results["Recall@10"], 6)
)
print(
    "NDCG@10:",
    round(warm_results["NDCG@10"], 6)
)

print("\nPOPULARITY - COLD-START USERS")
print(
    "Precision@10:",
    round(cold_pop_results["Precision@10"], 6)
)
print(
    "Recall@10:",
    round(cold_pop_results["Recall@10"], 6)
)
print(
    "NDCG@10:",
    round(cold_pop_results["NDCG@10"], 6)
)

print("\nPOPULARITY - WARM USERS")
print(
    "Precision@10:",
    round(warm_pop_results["Precision@10"], 6)
)
print(
    "Recall@10:",
    round(warm_pop_results["Recall@10"], 6)
)
print(
    "NDCG@10:",
    round(warm_pop_results["NDCG@10"], 6)
)

print("\nITEM COLD-START")
print(
    "Cold items (<5 training interactions):",
    len(cold_items)
)
print(
    "Users with a cold item in test:",
    cold_item_users_evaluated
)
print(
    "Recall@10 for cold test items:",
    round(cold_item_recall, 6)
)

# ---------------------------------------------------------------
# Save results
# ---------------------------------------------------------------

result_rows = [
    [
        "Cold-start users (<5)",
        "Hybrid",
        cold_results["Precision@10"],
        cold_results["Recall@10"],
        cold_results["NDCG@10"]
    ],
    [
        "Warm users (>=5)",
        "Hybrid",
        warm_results["Precision@10"],
        warm_results["Recall@10"],
        warm_results["NDCG@10"]
    ],
    [
        "Cold-start users (<5)",
        "Popularity",
        cold_pop_results["Precision@10"],
        cold_pop_results["Recall@10"],
        cold_pop_results["NDCG@10"]
    ],
    [
        "Warm users (>=5)",
        "Popularity",
        warm_pop_results["Precision@10"],
        warm_pop_results["Recall@10"],
        warm_pop_results["NDCG@10"]
    ]
]

results_df = pd.DataFrame(
    result_rows,
    columns=[
        "User Slice",
        "Model",
        "Precision@10",
        "Recall@10",
        "NDCG@10"
    ]
)

results_df.to_csv(
    "results/ranking_results.csv",
    index=False
)

print("\nSaved:")
print("  results/ranking_results.csv")
