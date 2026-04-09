# ====================================================================
# INDIVIDUAL_16255_Navarro.py — exported from notebook
# ====================================================================

# ── Cell 1 ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
print("✓ Libraries loaded")

# ── Cell 2 ──────────────────────────────────────────────────
# ── Load raw CSVs ─────────────────────────────────────────────────────────
movies  = pd.read_csv('/content/tmdb_5000_movies.csv')
credits = pd.read_csv('/content/tmdb_5000_credits.csv')

# Align on movie id
credits = credits.rename(columns={'movie_id': 'id'})
df = movies.merge(credits[['id','cast','crew']], on='id', how='left')

print(f"Shape after merge : {df.shape}")
print(f"Columns           : {list(df.columns)}")

# ── Cell 3 ──────────────────────────────────────────────────
# ── Parse embedded JSON columns ───────────────────────────────────────────
def extract_names(json_str, key='name', top_n=None):
    """Extract list of 'name' values from a JSON list-of-dicts string."""
    try:
        items = ast.literal_eval(json_str)
        names = [d[key] for d in items if key in d]
        return names[:top_n] if top_n else names
    except Exception:
        return []

def extract_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        dirs = [d['name'] for d in crew if d.get('job') == 'Director']
        return dirs[0] if dirs else ''
    except Exception:
        return ''

df['genres_list']    = df['genres'].apply(extract_names)
df['keywords_list']  = df['keywords'].apply(extract_names)
df['cast_list']      = df['cast'].apply(lambda x: extract_names(x, top_n=5))
df['director']       = df['crew'].apply(extract_director)
df['release_year']   = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Clean vote data (remove entries with < 10 votes — too noisy)
df_clean = df[(df['vote_count'] >= 10) & (df['vote_average'] > 0)].copy().reset_index(drop=True)
print(f"Movies after quality filter : {len(df_clean):,}  (removed {len(df)-len(df_clean)} low-vote entries)")
print(f"vote_average range          : {df_clean.vote_average.min():.1f} – {df_clean.vote_average.max():.1f}")
print(f"Missing budget (0)          : {(df_clean.budget==0).sum():,}")
print(df_clean[['title','vote_average','vote_count','genres_list']].head(4))

# ── Cell 4 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1 – Vote average distribution
axes[0].hist(df_clean['vote_average'], bins=30, color='steelblue', edgecolor='white')
axes[0].axvline(df_clean['vote_average'].mean(), color='red', linestyle='--', label=f"mean={df_clean.vote_average.mean():.2f}")
axes[0].set(title='Vote Average Distribution', xlabel='vote_average', ylabel='Count')
axes[0].legend()

# 2 – Vote count (log scale)
axes[1].hist(df_clean['vote_count'], bins=50, color='teal', edgecolor='white')
axes[1].set(title='Vote Count Distribution', xlabel='vote_count', ylabel='Count')
axes[1].set_xscale('log')

# 3 – Release year trend
year_counts = df_clean['release_year'].value_counts().sort_index()
axes[2].bar(year_counts.index, year_counts.values, color='coral', width=0.8)
axes[2].set(title='Movies by Release Year', xlabel='Year', ylabel='Count')
axes[2].set_xlim(1960, 2018)

plt.tight_layout()
plt.savefig('eda_distributions.png', bbox_inches='tight')
plt.show()
print("Fig 1 – saved")

# ── Cell 5 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Genre frequency
from collections import Counter
all_genres = Counter(g for gl in df_clean['genres_list'] for g in gl)
genre_df = pd.DataFrame(all_genres.most_common(15), columns=['genre','count'])
axes[0].barh(genre_df['genre'], genre_df['count'], color='steelblue')
axes[0].set(title='Top 15 Genres', xlabel='Number of Movies')
axes[0].invert_yaxis()

# Budget vs Revenue (non-zero only)
fin = df_clean[(df_clean.budget > 0) & (df_clean.revenue > 0)].copy()
fin['budget_m']  = fin['budget']  / 1e6
fin['revenue_m'] = fin['revenue'] / 1e6
axes[1].scatter(fin['budget_m'], fin['revenue_m'], alpha=0.3, s=15, color='darkorange')
axes[1].set(title='Budget vs Revenue (USD millions)', xlabel='Budget ($M)', ylabel='Revenue ($M)')
axes[1].set_xlim(0, 350); axes[1].set_ylim(0, 3000)

plt.tight_layout()
plt.savefig('eda_genre_finance.png', bbox_inches='tight')
plt.show()
print(f"Movies with known budget+revenue: {len(fin):,}")
print(f"Median budget : ${fin.budget_m.median():.0f}M   Median revenue: ${fin.revenue_m.median():.0f}M")

# ── Cell 6 ──────────────────────────────────────────────────
print("=== Dataset Summary ===")
print(f"Total movies (quality-filtered) : {len(df_clean):,}")
print(f"Unique genres                   : {len(all_genres)}")
print(f"Movies with keywords            : {(df_clean.keywords_list.apply(len) > 0).sum():,}")
print(f"Movies with cast data           : {(df_clean.cast_list.apply(len) > 0).sum():,}")
print(f"Movies with director            : {(df_clean.director != '').sum():,}")
print()
print("=== Vote Statistics ===")
print(df_clean['vote_average'].describe().round(3))
print()
# Top directors by average rating (min 3 films)
dir_stats = (df_clean[df_clean.director != '']
             .groupby('director')['vote_average']
             .agg(count='count', mean='mean')
             .query('count >= 3')
             .nlargest(8, 'mean'))
print("Top directors by avg rating (≥3 films):")
print(dir_stats.to_string())

# ── Cell 7 ──────────────────────────────────────────────────
# ── Train / test split (80/20 random, stratified by vote_average bin) ─────
df_clean['rating_bin'] = pd.cut(df_clean['vote_average'], bins=5, labels=False)

train_idx, test_idx = train_test_split(
    df_clean.index, test_size=0.2, random_state=42,
    stratify=df_clean['rating_bin']
)
train = df_clean.loc[train_idx].reset_index(drop=True)
test  = df_clean.loc[test_idx].reset_index(drop=True)

print(f"Train : {len(train):,} movies  avg={train.vote_average.mean():.3f}")
print(f"Test  : {len(test):,}  movies  avg={test.vote_average.mean():.3f}")

# ── Cell 8 ──────────────────────────────────────────────────
class BayesianWeightedRating:
    """
    Non-personalized recommender using the IMDb / TMDb Bayesian formula:

        WR(i) = (v / (v + m)) * R  +  (m / (v + m)) * C

    where v = vote_count(i), R = vote_average(i), m = minimum-vote threshold,
    C = global mean vote.  When v << m the score shrinks toward the global mean,
    penalising films with very few votes. All users receive the same ranked list.

    For rating prediction: predict(movie_id) → WR of that movie.
    """
    def __init__(self, percentile: float = 0.70):
        self.percentile = percentile  # m = this quantile of vote_count

    def fit(self, df: pd.DataFrame):
        self.m_ = df['vote_count'].quantile(self.percentile)
        self.C_ = df['vote_average'].mean()
        df = df.copy()
        df['wr'] = ((df['vote_count'] / (df['vote_count'] + self.m_)) * df['vote_average'] +
                    (self.m_         / (df['vote_count'] + self.m_)) * self.C_)
        self.scores_ = df.set_index('id')[['title','wr','vote_average','vote_count','genres_list']]
        return self

    def recommend(self, n: int = 10, seen_ids=None):
        scores = self.scores_.copy()
        if seen_ids:
            scores = scores[~scores.index.isin(seen_ids)]
        return scores.nlargest(n, 'wr')

    def predict(self, movie_id):
        """Predict vote_average ≈ WR for a movie."""
        if movie_id in self.scores_.index:
            return self.scores_.loc[movie_id, 'wr']
        return self.C_

np_model = BayesianWeightedRating(percentile=0.70)
np_model.fit(train)
print(f"m (min votes threshold) : {np_model.m_:.0f}")
print(f"C (global mean)         : {np_model.C_:.3f}")

# ── Cell 9 ──────────────────────────────────────────────────
top10 = np_model.recommend(n=10)
print("Non-Personalized Top-10 Global Recommendations")
print("=" * 55)
for i, (mid, row) in enumerate(top10.iterrows(), 1):
    genres = ', '.join(row['genres_list'][:3])
    print(f"{i:2}. {row['title']:<40} WR={row['wr']:.3f}  votes={row['vote_count']:,}  genres=[{genres}]")

# ── Cell 10 ──────────────────────────────────────────────────
# ── Evaluation helpers ────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k if k else 0

def recall_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k):
    dcg  = sum(1/np.log2(i+2) for i, r in enumerate(recommended[:k]) if r in set(relevant))
    idcg = sum(1/np.log2(i+2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0

RELEVANT_THRESHOLD = 7.0   # movies rated >= 7.0 are "relevant"
K = 10

# ── NP rating prediction metrics ─────────────────────────────────────────
np_preds = test['id'].apply(np_model.predict)
np_rmse  = rmse(test['vote_average'], np_preds)
np_mae   = mae( test['vote_average'], np_preds)

# ── NP ranking metrics — simulated users within training set ──────────────
# Strategy: split high-rated training movies into (seeds | relevant).
# The user has 'seen' the seed half; the model must surface the 'relevant' half.
train_high_ids = train[train.vote_average >= RELEVANT_THRESHOLD]['id'].tolist()
P_np, R_np, N_np = [], [], []
np.random.seed(42)
for _ in range(200):
    if len(train_high_ids) < 10:
        continue
    perm     = np.random.permutation(train_high_ids)
    n_seed   = max(3, len(train_high_ids) // 2)
    seeds    = perm[:n_seed].tolist()
    relevant = perm[n_seed:].tolist()
    seen     = seeds + list(np.random.choice(
        [i for i in train['id'].tolist() if i not in seeds],
        min(100, len(train) - len(seeds)), replace=False))
    recs = np_model.recommend(n=K * 3, seen_ids=seen).index.tolist()
    if not recs:
        continue
    P_np.append(precision_at_k(recs, relevant, K))
    R_np.append(recall_at_k(recs,    relevant, K))
    N_np.append(ndcg_at_k(recs,      relevant, K))

np_prec, np_rec, np_ndcg = np.mean(P_np), np.mean(R_np), np.mean(N_np)
print(f"Non-Personalized  RMSE={np_rmse:.4f}  MAE={np_mae:.4f}")
print(f"Non-Personalized  P@{K}={np_prec:.4f}  R@{K}={np_rec:.4f}  NDCG@{K}={np_ndcg:.4f}")

# ── Cell 11 ──────────────────────────────────────────────────
# ── Build rich text feature string per movie ─────────────────────────────
# We combine genres, keywords, cast names, and director into a single
# TF-IDF document per movie. This lets us measure item–item similarity
# in a shared feature space — the foundation of item-based CF.

def build_soup(row):
    parts = []
    # Genres (×3 weight via repetition)
    parts += [g.lower().replace(' ','') for g in row['genres_list']] * 3
    # Top keywords (×2)
    parts += [k.lower().replace(' ','') for k in row['keywords_list'][:10]] * 2
    # Top cast (×1)
    parts += [c.lower().replace(' ','') for c in row['cast_list'][:5]]
    # Director (×2)
    if row['director']:
        parts += [row['director'].lower().replace(' ','')] * 2
    return ' '.join(parts)

df_clean['soup'] = df_clean.apply(build_soup, axis=1)

# ── Refresh train and test to include the new 'soup' column ──────────────
# (train/test were created before soup was added to df_clean)
train = df_clean.loc[train_idx].reset_index(drop=True)
test  = df_clean.loc[test_idx].reset_index(drop=True)

print("Sample feature soup for Avatar:")
print(df_clean[df_clean.title=='Avatar']['soup'].values[0][:200])
print(f"\nTrain has 'soup' column: {'soup' in train.columns} ✓")

# ── Cell 12 ──────────────────────────────────────────────────
class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using TF-IDF content features.

    Similarity is computed via cosine distance on a TF-IDF matrix built
    from genre, keyword, cast and director tokens. This is the item-based
    analogue of user-based CF: instead of finding users who liked similar
    movies, we find movies that share similar attributes.

    predict(movie_id) → predicted vote_average as weighted mean of k-NN scores.
    recommend(seed_ids, n) → top-n most similar unseen movies.
    """
    def __init__(self, k: int = 20, max_features: int = 5000):
        self.k = k
        self.max_features = max_features

    def fit(self, df: pd.DataFrame):
        self.df_ = df.set_index('id').copy()
        self.tfidf_ = TfidfVectorizer(max_features=self.max_features, ngram_range=(1,1))
        tfidf_mat = self.tfidf_.fit_transform(df['soup'].fillna(''))
        self.sim_  = cosine_similarity(tfidf_mat)  # shape (n, n)
        self.index_map_ = {mid: i for i, mid in enumerate(df['id'])}
        self.id_list_   = df['id'].tolist()
        return self

    def _get_similar_ids(self, movie_id, exclude_ids=None):
        """Return list of (similar_id, similarity_score) sorted desc."""
        if movie_id not in self.index_map_:
            return []
        idx  = self.index_map_[movie_id]
        sims = self.sim_[idx]
        exclude = set(exclude_ids or []) | {movie_id}
        results = [(self.id_list_[i], s) for i, s in enumerate(sims)
                   if self.id_list_[i] not in exclude]
        return sorted(results, key=lambda x: -x[1])

    def predict(self, movie_id):
        """Predict vote_average as weighted avg of k most similar training items."""
        neighbours = self._get_similar_ids(movie_id)[:self.k]
        if not neighbours:
            return self.df_['vote_average'].mean()
        ids, weights = zip(*neighbours)
        weights = np.array(weights)
        ratings = np.array([self.df_.loc[i, 'vote_average']
                            if i in self.df_.index else self.df_['vote_average'].mean()
                            for i in ids])
        return float(np.dot(weights, ratings) / weights.sum()) if weights.sum() > 0 else self.df_['vote_average'].mean()

    def recommend(self, seed_ids, n=10, seen_ids=None):
        """
        Given a list of seed movie IDs (user's history), aggregate
        similarity scores and return top-n unseen recommendations.
        """
        all_scores = {}
        seen = set(seen_ids or []) | set(seed_ids)
        for sid in seed_ids:
            for mid, score in self._get_similar_ids(sid, exclude_ids=seen):
                all_scores[mid] = all_scores.get(mid, 0) + score
        if not all_scores:
            return []
        return sorted(all_scores, key=lambda x: -all_scores[x])[:n]

print("ItemBasedCF class defined ✓")

# ── Cell 13 ──────────────────────────────────────────────────
# ── Fit on training data ──────────────────────────────────────────────────
cf_model = ItemBasedCF(k=20, max_features=5000)
cf_model.fit(train)

print(f"TF-IDF vocab size : {len(cf_model.tfidf_.vocabulary_):,}")
print(f"Similarity matrix : {cf_model.sim_.shape}")
print(f"Training items    : {len(train):,}")

# ── Cell 14 ──────────────────────────────────────────────────
# ── Sample recommendations seeded from Inception + The Dark Knight ────────
seed_titles = ['Inception', 'The Dark Knight']
seed_ids = df_clean[df_clean.title.isin(seed_titles)]['id'].tolist()

cf_recs = cf_model.recommend(seed_ids, n=10)
rec_df = df_clean[df_clean.id.isin(cf_recs)][['title','vote_average','genres_list']].copy()
rec_df['genres'] = rec_df['genres_list'].apply(lambda x: ', '.join(x[:3]))

print(f"CF recommendations for users who liked: {seed_titles}")
print("=" * 60)
for _, row in rec_df.iterrows():
    print(f"  • {row['title']:<40}  ★{row['vote_average']:.1f}  [{row['genres']}]")

# ── Cell 15 ──────────────────────────────────────────────────
# ── CF: rating prediction on held-out movies ─────────────────────────────
cf_preds_val = test['id'].apply(cf_model.predict)
cf_rmse = rmse(test['vote_average'], cf_preds_val)
cf_mae  = mae( test['vote_average'], cf_preds_val)
print(f"Item-Based CF  RMSE={cf_rmse:.4f}  MAE={cf_mae:.4f}")

# ── CF: ranking evaluation via simulated users (within training set) ──────
# Same protocol as NP: split high-rated train movies into seeds | relevant.
# CF uses the seeds to find similar unseen movies from the training catalogue.
np.random.seed(42)
P_cf, R_cf, N_cf = [], [], []
train_high_ids_cf = train[train.vote_average >= RELEVANT_THRESHOLD]['id'].tolist()

for _ in range(200):
    if len(train_high_ids_cf) < 10:
        continue
    perm     = np.random.permutation(train_high_ids_cf)
    n_seed   = max(3, len(train_high_ids_cf) // 2)
    seeds    = perm[:n_seed].tolist()
    relevant = perm[n_seed:].tolist()
    seen     = seeds + list(np.random.choice(
        [i for i in train['id'].tolist() if i not in seeds],
        min(100, len(train) - len(seeds)), replace=False))
    recs = cf_model.recommend(seeds[:10], n=K * 3, seen_ids=seen)
    if not recs:
        continue
    P_cf.append(precision_at_k(recs, relevant, K))
    R_cf.append(recall_at_k(recs,    relevant, K))
    N_cf.append(ndcg_at_k(recs,      relevant, K))

cf_prec = np.mean(P_cf)
cf_rec  = np.mean(R_cf)
cf_ndcg = np.mean(N_cf)
print(f"Item-Based CF  P@{K}={cf_prec:.4f}  R@{K}={cf_rec:.4f}  NDCG@{K}={cf_ndcg:.4f}")

# ── Cell 16 ──────────────────────────────────────────────────
# ── Catalogue Coverage ────────────────────────────────────────────────────
# Fraction of the training catalogue that appears in at least one rec list.
total          = len(train)
train_high_cov = train[train.vote_average >= RELEVANT_THRESHOLD]['id'].tolist()

# NP: same ranked list for every user (regardless of seen_ids)
np_recs_global = set(np_model.recommend(n=200).index.tolist())
np_cov = len(np_recs_global) / total

# CF: different list per simulated user — take the union across 100 users
cf_recs_all = set()
for _ in range(100):
    seeds = list(np.random.choice(train_high_cov,
                                  min(5, len(train_high_cov)), replace=False))
    seen  = list(np.random.choice(train['id'].tolist(), 100, replace=False))
    cf_recs_all.update(cf_model.recommend(seeds, n=K, seen_ids=seen))

cf_cov = len(cf_recs_all) / total
print(f"Catalogue Coverage  NP={np_cov:.4f}  CF={cf_cov:.4f}  (training catalog={total:,})")

# ── Cell 17 ──────────────────────────────────────────────────
# ── Intra-list Genre Diversity ────────────────────────────────────────────
genre_lookup = df_clean.set_index('id')['genres_list'].apply(set)

def jaccard_dist(a, b):
    u = len(a | b)
    return 1 - len(a & b) / u if u else 0

def intra_diversity(rec_ids):
    valid = [i for i in rec_ids if i in genre_lookup.index]
    if len(valid) < 2:
        return 0
    return np.mean([jaccard_dist(genre_lookup[i], genre_lookup[j])
                    for ii, i in enumerate(valid) for j in valid[ii+1:]])

train_high_div = train[train.vote_average >= RELEVANT_THRESHOLD]['id'].tolist()
np_div_scores, cf_div_scores = [], []

for _ in range(100):
    seeds = list(np.random.choice(train_high_div, min(5, len(train_high_div)), replace=False))
    seen  = list(np.random.choice(train['id'].tolist(), 100, replace=False))
    np_list = np_model.recommend(n=K, seen_ids=seen).index.tolist()
    cf_list = cf_model.recommend(seeds, n=K, seen_ids=seen)
    np_div_scores.append(intra_diversity(np_list))
    cf_div_scores.append(intra_diversity(cf_list))

np_div = np.mean(np_div_scores)
cf_div = np.mean(cf_div_scores)
print(f"Intra-list Diversity  NP={np_div:.4f}  CF={cf_div:.4f}")

# ── Cell 18 ──────────────────────────────────────────────────
metrics_plot = {
    'RMSE (↓)'   : [np_rmse,  cf_rmse],
    'MAE (↓)'    : [np_mae,   cf_mae],
    'Precision@10': [np_prec,  cf_prec],
    'Recall@10'  : [np_rec,   cf_rec],
    'NDCG@10'    : [np_ndcg,  cf_ndcg],
    'Coverage'   : [np_cov,   cf_cov],
    'Diversity'  : [np_div,   cf_div],
}
labels  = list(metrics_plot.keys())
np_vals = [v[0] for v in metrics_plot.values()]
cf_vals = [v[1] for v in metrics_plot.values()]

x = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(13, 5))
b1 = ax.bar(x-w/2, np_vals, w, label='Non-Personalized', color='steelblue',  alpha=.85)
b2 = ax.bar(x+w/2, cf_vals, w, label='Item-Based CF',    color='darkorange', alpha=.85)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_title('Model Comparison — TMDB 5000', fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.3f', fontsize=7.5, padding=2)
ax.bar_label(b2, fmt='%.3f', fontsize=7.5, padding=2)
plt.tight_layout()
plt.savefig('comparison.png', bbox_inches='tight')
plt.show()
print("Fig 3 – Comparison chart saved")

# ── Cell 19 ──────────────────────────────────────────────────
# ── Final Metrics Table (mandatory) ──────────────────────────────────────
final_metrics = pd.DataFrame({
    'Approach'   : ['Non-Personalized (Bayesian WR)', 'Item-Based CF (TF-IDF Cosine)'],
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
