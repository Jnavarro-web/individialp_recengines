# ====================================================================
# INDIVIDUAL_12345_Garcia.py  –  exported from notebook
# ====================================================================

# ── Cell 1 ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
print("✓ Libraries loaded")

# ── Cell 3 ──────────────────────────────────────────────────
# ── Synthetic MovieLens-like dataset ────────────────────────────────────
# Replace these two lines to use real files:
#   ratings = pd.read_csv('ml-latest-small/ratings.csv')
#   movies  = pd.read_csv('ml-latest-small/movies.csv')

np.random.seed(42)
N_USERS, N_MOVIES, N_RATINGS = 671, 9066, 100004

# Power-law item popularity (few movies dominate ratings)
pop_weights = np.random.zipf(1.7, N_MOVIES).astype(float)
pop_weights /= pop_weights.sum()

user_ids  = np.random.randint(1, N_USERS + 1, N_RATINGS)
movie_ids = np.random.choice(np.arange(1, N_MOVIES + 1), N_RATINGS, p=pop_weights)

# Realistic rating distribution (right-skewed toward 3-4)
rating_vals = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
rating_prob = [0.01,0.02,0.03,0.07,0.08,0.18,0.15,0.22,0.13,0.11]
ratings_v   = np.random.choice(rating_vals, N_RATINGS, p=rating_prob)

ratings = (pd.DataFrame({'userId': user_ids, 'movieId': movie_ids, 'rating': ratings_v})
             .drop_duplicates(['userId','movieId'])
             .reset_index(drop=True))

# Genre vocabulary
GENRES = ['Action','Adventure','Animation','Comedy','Crime',
          'Drama','Fantasy','Horror','Romance','Sci-Fi','Thriller']
movie_genres = [
    '|'.join(sorted(np.random.choice(GENRES, np.random.randint(1, 4), replace=False)))
    for _ in range(N_MOVIES)
]
movies = pd.DataFrame({
    'movieId': np.arange(1, N_MOVIES + 1),
    'title'  : [f'Movie {i:05d} ({1970 + i % 54})' for i in range(1, N_MOVIES + 1)],
    'genres' : movie_genres
})

print(f"Ratings : {ratings.shape[0]:>7,} rows")
print(f"Users   : {ratings.userId.nunique():>7,}")
print(f"Movies  : {ratings.movieId.nunique():>7,}")
print(f"\nSample ratings:")
print(ratings.head(5).to_string(index=False))

# ── Cell 5 ──────────────────────────────────────────────────
# ── Basic statistics ────────────────────────────────────────────────────
print("=== Rating Statistics ===")
print(ratings['rating'].describe().round(3))
print(f"\nMissing values: {ratings.isnull().sum().sum()}")
print(f"Sparsity      : {1 - len(ratings)/(N_USERS*N_MOVIES):.4%}")

# Ratings per user / per movie
user_counts  = ratings.groupby('userId')['rating'].count()
movie_counts = ratings.groupby('movieId')['rating'].count()

print(f"\nRatings per user  → mean={user_counts.mean():.1f}  median={user_counts.median():.0f}  max={user_counts.max()}")
print(f"Ratings per movie → mean={movie_counts.mean():.1f}  median={movie_counts.median():.0f}  max={movie_counts.max()}")

# ── Cell 6 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1 – Rating distribution
axes[0].hist(ratings['rating'], bins=10, color='steelblue', edgecolor='white', rwidth=0.85)
axes[0].set(title='Rating Distribution', xlabel='Rating', ylabel='Count')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))

# 2 – Ratings per user (log)
axes[1].hist(user_counts, bins=50, color='teal', edgecolor='white')
axes[1].set(title='Ratings per User', xlabel='# Ratings', ylabel='# Users')
axes[1].set_yscale('log')

# 3 – Ratings per movie (log)
axes[2].hist(movie_counts, bins=50, color='coral', edgecolor='white')
axes[2].set(title='Ratings per Movie', xlabel='# Ratings', ylabel='# Movies')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig('eda_distributions.png', bbox_inches='tight')
plt.show()
print("Fig 1 – Distributions saved")

# ── Cell 7 ──────────────────────────────────────────────────
# ── Top-10 most-rated movies ─────────────────────────────────────────────
top_movies = (ratings.merge(movies, on='movieId')
                      .groupby(['movieId','title'])['rating']
                      .agg(count='count', mean='mean')
                      .nlargest(10, 'count')
                      .reset_index())
print("Top-10 most-rated movies")
print(top_movies[['title','count','mean']].to_string(index=False))

# ── Cell 8 ──────────────────────────────────────────────────
# ── Genre distribution ───────────────────────────────────────────────────
genre_series = (movies['genres'].str.split('|').explode()
                                .value_counts()
                                .reset_index()
                                .rename(columns={'index':'genre','genres':'count'}))
# Adjust for pandas version
if 'genres' in genre_series.columns:
    genre_series.columns = ['genre','count']
else:
    genre_series.columns = ['genre','count']

fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(genre_series['genre'], genre_series['count'], color='steelblue')
ax.set(title='Movie Count by Genre', xlabel='Number of Movies')
plt.tight_layout(); plt.savefig('genres.png', bbox_inches='tight'); plt.show()
print("Fig 2 – Genre distribution saved")

# ── Cell 10 ──────────────────────────────────────────────────
# ── Train / Test split (80/20 per user — temporal proxy) ─────────────────
def train_test_split_per_user(df, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, grp in df.groupby('userId'):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * test_frac))
        test_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)

train, test = train_test_split_per_user(ratings)
print(f"Train : {len(train):,} ratings ({len(train)/len(ratings):.1%})")
print(f"Test  : {len(test):,}  ratings ({len(test)/len(ratings):.1%})")

# ── Cell 11 ──────────────────────────────────────────────────
class BayesianAverageRecommender:
    """
    Non-personalized recommender using a Bayesian (damped) average:
        score(i) = (n_i * μ_i + m * μ_g) / (n_i + m)
    where n_i=ratings for item i, μ_i=mean rating, μ_g=global mean, m=damping factor.
    All users receive the same ranked list.
    """
    def __init__(self, m: int = 10):
        self.m = m  # damping factor (≈ minimum votes for prior)

    def fit(self, df: pd.DataFrame):
        self.global_mean_ = df['rating'].mean()
        stats = df.groupby('movieId')['rating'].agg(['count','mean']).reset_index()
        stats.columns = ['movieId','n','mu']
        stats['score'] = (stats['n'] * stats['mu'] + self.m * self.global_mean_) / (stats['n'] + self.m)
        self.item_scores_ = stats.set_index('movieId')['score']
        return self

    def recommend(self, user_id, n: int = 10, seen_items=None):
        scores = self.item_scores_.copy()
        if seen_items is not None:
            scores = scores.drop(index=seen_items, errors='ignore')
        return scores.nlargest(n).index.tolist()

    def predict(self, user_id, movie_id):
        """Predict rating = Bayesian average for movie (same for all users)."""
        return self.item_scores_.get(movie_id, self.global_mean_)


np_model = BayesianAverageRecommender(m=10)
np_model.fit(train)
print(f"Global mean: {np_model.global_mean_:.3f}")
print(f"Items with scores: {len(np_model.item_scores_):,}")

# ── Cell 12 ──────────────────────────────────────────────────
# ── Top-10 global recommendations ────────────────────────────────────────
top10_ids = np_model.item_scores_.nlargest(10).index
top10 = (movies[movies.movieId.isin(top10_ids)]
           .assign(score=lambda d: d.movieId.map(np_model.item_scores_))
           .sort_values('score', ascending=False)
           [['title','genres','score']])
print("Non-Personalized Top-10 Recommendations")
print(top10.to_string(index=False))

# ── Cell 13 ──────────────────────────────────────────────────
# ── Evaluation helpers ────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / k if k else 0

def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    dcg  = sum(1/np.log2(i+2) for i,r in enumerate(rec_k) if r in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0

def evaluate_ranking(model_fn, test_df, train_df, k=10, sample_users=200):
    """model_fn(user_id, seen_items) → list of recommended movie_ids"""
    users   = test_df['userId'].unique()
    if len(users) > sample_users:
        users = np.random.choice(users, sample_users, replace=False)
    relevant_threshold = 3.5
    P, R, N = [], [], []
    for uid in users:
        relevant = test_df[(test_df.userId==uid) &
                           (test_df.rating >= relevant_threshold)]['movieId'].tolist()
        if not relevant:
            continue
        seen = train_df[train_df.userId==uid]['movieId'].tolist()
        recs = model_fn(uid, seen)
        P.append(precision_at_k(recs, relevant, k))
        R.append(recall_at_k(recs, relevant, k))
        N.append(ndcg_at_k(recs, relevant, k))
    return np.mean(P), np.mean(R), np.mean(N)

print("Evaluation helpers defined ✓")

# ── Cell 14 ──────────────────────────────────────────────────
# ── Non-Personalized: rating prediction metrics ───────────────────────────
test_preds_np = test.apply(lambda r: np_model.predict(r.userId, r.movieId), axis=1)
np_rmse = rmse(test['rating'], test_preds_np)
np_mae  = mae(test['rating'],  test_preds_np)
print(f"Non-Personalized  RMSE={np_rmse:.4f}  MAE={np_mae:.4f}")

# ── Non-Personalized: ranking metrics ────────────────────────────────────
np_rec_fn = lambda uid, seen: np_model.recommend(uid, n=20, seen_items=seen)
np_prec, np_rec, np_ndcg = evaluate_ranking(np_rec_fn, test, train, k=10)
print(f"Non-Personalized  P@10={np_prec:.4f}  R@10={np_rec:.4f}  NDCG@10={np_ndcg:.4f}")

# ── Cell 16 ──────────────────────────────────────────────────
class UserBasedCF:
    """
    User-Based Collaborative Filtering.
    - Similarity  : cosine similarity on mean-centered ratings
    - Prediction  : weighted average of k-nearest neighbours' ratings
    - Cold-start  : falls back to item global mean
    """
    def __init__(self, k: int = 30, min_common: int = 3):
        self.k = k
        self.min_common = min_common

    def fit(self, df: pd.DataFrame):
        # Build user-item matrix
        self.matrix_ = df.pivot_table(index='userId', columns='movieId', values='rating')
        self.global_mean_ = df['rating'].mean()
        self.item_means_  = df.groupby('movieId')['rating'].mean()
        self.user_means_  = self.matrix_.mean(axis=1)

        # Mean-center (subtract user mean)
        centered = self.matrix_.sub(self.user_means_, axis=0).fillna(0)

        # Cosine similarity between all users
        sim_arr = cosine_similarity(centered.values)
        np.fill_diagonal(sim_arr, 0)
        self.similarity_  = pd.DataFrame(sim_arr,
                                          index=self.matrix_.index,
                                          columns=self.matrix_.index)
        self.users_ = set(self.matrix_.index)
        self.items_ = set(self.matrix_.columns)
        return self

    def predict(self, user_id, movie_id):
        if user_id not in self.users_:
            return self.item_means_.get(movie_id, self.global_mean_)
        if movie_id not in self.items_:
            return self.user_means_.get(user_id, self.global_mean_)

        # Users who rated this item
        rated_by = self.matrix_.index[self.matrix_[movie_id].notna()]
        if len(rated_by) == 0:
            return self.user_means_.get(user_id, self.global_mean_)

        sims = self.similarity_.loc[user_id, rated_by]
        top_k = sims.nlargest(self.k)
        top_k = top_k[top_k > 0]
        if top_k.empty:
            return self.item_means_.get(movie_id, self.global_mean_)

        neighbour_ratings = self.matrix_.loc[top_k.index, movie_id]
        neighbour_means   = self.user_means_.loc[top_k.index]
        user_mean         = self.user_means_.get(user_id, self.global_mean_)

        numer = (top_k * (neighbour_ratings - neighbour_means)).sum()
        denom = top_k.abs().sum()
        return user_mean + numer / denom if denom else user_mean

    def recommend(self, user_id, n: int = 10, seen_items=None):
        if user_id not in self.users_:
            # Cold-start: return most popular unseen
            items = self.item_means_
        else:
            # Score all unseen items via prediction
            candidate_items = [i for i in self.items_ if i not in (seen_items or [])]
            scores = {i: self.predict(user_id, i) for i in candidate_items[:500]}  # cap for speed
            items  = pd.Series(scores)
        return items.nlargest(n).index.tolist()


print("UserBasedCF class defined ✓")

# ── Cell 17 ──────────────────────────────────────────────────
# ── Fit CF on training data ───────────────────────────────────────────────
# Note: using a subset for memory efficiency in this demo
SAMPLE_USERS = 671
sample_uids  = train['userId'].unique()[:SAMPLE_USERS]
train_sub    = train[train.userId.isin(sample_uids)]

cf_model = UserBasedCF(k=30)
cf_model.fit(train_sub)
print(f"CF model fitted on {train_sub.shape[0]:,} ratings")
print(f"User-item matrix: {cf_model.matrix_.shape[0]} users × {cf_model.matrix_.shape[1]} movies")
print(f"Density: {cf_model.matrix_.notna().values.mean():.4%}")

# ── Cell 18 ──────────────────────────────────────────────────
# ── Sample CF predictions for user 1 ─────────────────────────────────────
sample_uid = 1
seen = train[train.userId==sample_uid]['movieId'].tolist()
cf_recs = cf_model.recommend(sample_uid, n=10, seen_items=seen)

print(f"CF Top-10 recommendations for user {sample_uid}:")
rec_df = movies[movies.movieId.isin(cf_recs)].copy()
rec_df['pred_rating'] = rec_df.movieId.apply(lambda m: cf_model.predict(sample_uid, m))
print(rec_df[['title','genres','pred_rating']].sort_values('pred_rating', ascending=False)
        .to_string(index=False))

# ── Cell 19 ──────────────────────────────────────────────────
# ── CF: rating prediction metrics ────────────────────────────────────────
test_cf = test[test.userId.isin(sample_uids)].head(3000)  # cap for speed
cf_preds = test_cf.apply(lambda r: cf_model.predict(r.userId, r.movieId), axis=1)
cf_rmse = rmse(test_cf['rating'], cf_preds)
cf_mae  = mae(test_cf['rating'],  cf_preds)
print(f"User-Based CF  RMSE={cf_rmse:.4f}  MAE={cf_mae:.4f}")

# ── CF: ranking metrics ───────────────────────────────────────────────────
cf_rec_fn = lambda uid, seen: cf_model.recommend(uid, n=20, seen_items=seen)
cf_prec, cf_rec, cf_ndcg = evaluate_ranking(cf_rec_fn, test, train, k=10, sample_users=100)
print(f"User-Based CF  P@10={cf_prec:.4f}  R@10={cf_rec:.4f}  NDCG@10={cf_ndcg:.4f}")

# ── Cell 21 ──────────────────────────────────────────────────
# ── Coverage ─────────────────────────────────────────────────────────────
def catalog_coverage(model_recs: set, total_items: int):
    """Fraction of catalog that appears in at least one recommendation list."""
    return len(model_recs) / total_items

K_COV = 10
sample_users_cov = train['userId'].unique()[:100]

np_recs_all, cf_recs_all = set(), set()
for uid in sample_users_cov:
    seen = train[train.userId==uid]['movieId'].tolist()
    np_recs_all.update(np_model.recommend(uid, n=K_COV, seen_items=seen))
    if uid in cf_model.users_:
        cf_recs_all.update(cf_model.recommend(uid, n=K_COV, seen_items=seen))

total_items = ratings['movieId'].nunique()
np_cov = catalog_coverage(np_recs_all, total_items)
cf_cov = catalog_coverage(cf_recs_all, total_items)
print(f"Coverage @{K_COV}  NP={np_cov:.4f}  CF={cf_cov:.4f}")

# ── Cell 22 ──────────────────────────────────────────────────
# ── Diversity (intra-list genre diversity) ────────────────────────────────
# Diversity = mean pairwise Jaccard distance between genre sets in a rec list

genre_sets = movies.set_index('movieId')['genres'].str.split('|').apply(set)

def jaccard_distance(a, b):
    inter = len(a & b)
    union = len(a | b)
    return 1 - inter/union if union else 0

def intra_list_diversity(rec_ids):
    rec_ids = [r for r in rec_ids if r in genre_sets.index]
    if len(rec_ids) < 2:
        return 0
    dists = [jaccard_distance(genre_sets[i], genre_sets[j])
             for ii, i in enumerate(rec_ids)
             for j in rec_ids[ii+1:]]
    return np.mean(dists)

np_div_scores, cf_div_scores = [], []
for uid in sample_users_cov:
    seen = train[train.userId==uid]['movieId'].tolist()
    np_div_scores.append(intra_list_diversity(np_model.recommend(uid, n=10, seen_items=seen)))
    if uid in cf_model.users_:
        cf_div_scores.append(intra_list_diversity(cf_model.recommend(uid, n=10, seen_items=seen)))

np_div = np.mean(np_div_scores)
cf_div = np.mean(cf_div_scores)
print(f"Diversity @10   NP={np_div:.4f}  CF={cf_div:.4f}")

# ── Cell 23 ──────────────────────────────────────────────────
# ── Side-by-side comparison chart ────────────────────────────────────────
metrics_plot = {
    'RMSE (↓)':      [np_rmse,  cf_rmse],
    'MAE (↓)':       [np_mae,   cf_mae],
    'Precision@10':  [np_prec,  cf_prec],
    'Recall@10':     [np_rec,   cf_rec],
    'NDCG@10':       [np_ndcg,  cf_ndcg],
    'Coverage':      [np_cov,   cf_cov],
    'Diversity':     [np_div,   cf_div],
}

labels  = list(metrics_plot.keys())
np_vals = [v[0] for v in metrics_plot.values()]
cf_vals = [v[1] for v in metrics_plot.values()]

x   = np.arange(len(labels))
w   = 0.35
fig, ax = plt.subplots(figsize=(13, 5))
b1 = ax.bar(x - w/2, np_vals, w, label='Non-Personalized', color='steelblue',  alpha=.85)
b2 = ax.bar(x + w/2, cf_vals, w, label='User-Based CF',    color='darkorange', alpha=.85)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_title('Model Comparison Across All Metrics', fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.3f', fontsize=7.5, padding=2)
ax.bar_label(b2, fmt='%.3f', fontsize=7.5, padding=2)
plt.tight_layout()
plt.savefig('comparison.png', bbox_inches='tight')
plt.show()
print("Fig 3 – Comparison chart saved")

# ── Cell 24 ──────────────────────────────────────────────────
# ── Final Metrics Table (mandatory) ──────────────────────────────────────
final_metrics = pd.DataFrame({
    'Approach'   : ['Non-Personalized (Bayesian Avg)', 'Collaborative Filtering (User-Based)'],
    'RMSE'       : [round(np_rmse,  4), round(cf_rmse,  4)],
    'MAE'        : [round(np_mae,   4), round(cf_mae,   4)],
    'Precision@K': [round(np_prec,  4), round(cf_prec,  4)],
    'Recall@K'   : [round(np_rec,   4), round(cf_rec,   4)],
    'NDCG'       : [round(np_ndcg,  4), round(cf_ndcg,  4)],
    'Coverage'   : [round(np_cov,   4), round(cf_cov,   4)],
    'Diversity'  : [round(np_div,   4), round(cf_div,   4)],
})
print(final_metrics.to_string(index=False))
final_metrics
