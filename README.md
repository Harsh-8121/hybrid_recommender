# Hybrid Movie Recommender - Cold-Start Recommendation

## Objective

Build a hybrid movie recommender that combines:

1. Collaborative filtering using matrix factorization.
2. Content-based recommendation using movie metadata.
3. Adaptive blending that shifts toward content-based scoring when interaction history is sparse.
4. Ranking evaluation using Precision@10, Recall@10 and NDCG@10.
5. Explicit cold-start evaluation for users with fewer than 5 interactions.
6. Explicit item cold-start evaluation for items with fewer than 5 training interactions.
7. Comparison with a popularity baseline.
8. Failure-case analysis.

Dataset: MovieLens 1M.

---

# 1. Folder Structure

```text
hybrid_recommender/
│
├── data/
│   └── ml-1m/
│       ├── movies.dat
│       ├── ratings.dat
│       └── users.dat
│
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_train_cf.py
│   ├── 03_train_content.py
│   ├── 04_hybrid_recommender.py
│   └── 05_evaluate.py
│
├── results/
│
├── requirements.txt
└── README.md
```

---

# 2. Dataset

Download MovieLens 1M from GroupLens.

Extract the archive and put these files into:

```text
data/ml-1m/
```

Required files:

```text
movies.dat
ratings.dat
users.dat
```

The code expects the original MovieLens 1M `::` file format.

---

# 3. Installation

Open a terminal in the project folder.

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 4. EXACT ORDER TO RUN

Run the following commands one by one.

## STEP 1 - Prepare the dataset

```bash
python src/01_prepare_data.py
```

This creates:

```text
data/movies.csv
data/train.csv
data/test.csv
```

The last interaction of every user is held out as the test interaction.

---

## STEP 2 - Train collaborative filtering

```bash
python src/02_train_cf.py
```

This trains an SVD matrix-factorization model.

Output:

```text
results/cf_model.pkl
```

---

## STEP 3 - Train content model

```bash
python src/03_train_content.py
```

This converts movie genres into TF-IDF vectors.

Output:

```text
results/content_model.pkl
```

---

## STEP 4 - Test the hybrid recommender

```bash
python src/04_hybrid_recommender.py
```

This prints:

- Adaptive CF weight
- Adaptive content weight
- Example recommendations

---

## STEP 5 - Run evaluation

```bash
python src/05_evaluate.py
```

This is the main evaluation script.

It reports:

- Precision@10
- Recall@10
- NDCG@10
- Cold-start user results
- Warm user results
- Popularity baseline
- Item cold-start results

Final results are saved to:

```text
results/ranking_results.csv
```

---

# 5. Adaptive Blending

The hybrid recommender does NOT use a fixed 50/50 weight.

Let:

```text
n = number of training interactions for the user
```

The collaborative filtering weight is:

```text
CF_weight = min(n / 5, 1)
```

The content-based weight is:

```text
Content_weight = 1 - CF_weight
```

Therefore:

| Interactions | CF | Content |
|---:|---:|---:|
| 0 | 0.00 | 1.00 |
| 1 | 0.20 | 0.80 |
| 2 | 0.40 | 0.60 |
| 3 | 0.60 | 0.40 |
| 4 | 0.80 | 0.20 |
| 5+ | 1.00 | 0.00 |

This makes the system rely increasingly on collaborative filtering as user history grows.

For a completely new user with zero interactions, there is no preference information from which to calculate personalized content similarity. The implementation therefore uses a popularity fallback.

---

# 6. Collaborative Filtering

The collaborative filtering component uses sparse matrix factorization.

The user-item rating matrix is centered around the global mean and decomposed using sparse SVD.

The learned user and item latent factors are used to calculate collaborative scores.

---

# 7. Content-Based Component

Movie genres are converted into TF-IDF vectors.

For a candidate movie, its similarity to movies in the user's history is calculated using cosine similarity.

The maximum similarity to the user's history is used as the content score.

This allows item metadata to contribute even when the item itself has little or no interaction history.

---

# 8. Cold-Start Evaluation

## Cold-start users

Users with:

```text
< 5 training interactions
```

## Warm users

Users with:

```text
>= 5 training interactions
```

The evaluation reports these groups separately.

---

# 9. Item Cold Start

The evaluation also calculates:

```text
Cold item = item with <5 training interactions
```

This tests how well the content component can recommend items with little interaction history.

The item cold-start Recall@10 is reported separately.

---

# 10. Ranking Metrics

The project uses:

### Precision@10

Fraction of the top 10 recommendations that are relevant.

### Recall@10

Fraction of relevant test items retrieved in the top 10.

### NDCG@10

Measures ranking quality while giving greater importance to relevant items appearing near the top.

RMSE is not used as the main metric because this project is a ranking/recommendation task rather than a rating-prediction task.

---

# 11. Popularity Baseline

A popularity recommender is included.

It recommends movies with the largest number of training interactions.

The hybrid model is compared against this baseline for:

- Cold-start users
- Warm users

This provides a simple baseline for measuring the value of the hybrid approach.

---

# 12. Failure Case Analysis

The hybrid system can still fail in several cases.

### Extremely sparse users

A user with only one interaction provides very little preference information.

The content component may overfit to that single movie's genre.

### Generic metadata

Genre-only metadata is limited.

Two movies can have the same genre labels but very different plots, tone, actors or quality.

### Popularity bias

Collaborative filtering can favor popular movies because they have more interactions.

### New user with zero history

A completely new user has no known preferences.

The system cannot calculate personalized content similarity without some initial interaction.

The current implementation uses popularity as a fallback.

### Weak cold-start item metadata

If a new item has poor or missing genre metadata, the content model has insufficient information to score it meaningfully.

---

# 13. Reproducibility

Run all five scripts in this exact order:

```bash
python src/01_prepare_data.py
python src/02_train_cf.py
python src/03_train_content.py
python src/04_hybrid_recommender.py
python src/05_evaluate.py
```

The final evaluation file is:

```text
results/ranking_results.csv
```

---

# 14. Expected Submission

The GitHub repository should contain:

```text
README.md
requirements.txt
src/
data/
results/
```

Do NOT upload the MovieLens dataset if the dataset license/distribution terms do not permit redistribution. Instead, document where the evaluator can download it.

Upload the source code and README to GitHub.
