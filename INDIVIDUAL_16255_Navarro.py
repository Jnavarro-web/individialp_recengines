#!/usr/bin/env python3
# Auto-generated backup from INDIVIDUAL_16255_Navarro_(12) (2).ipynb

# %% [markdown] cell 1
# # Recommender System Analysis & Implementation
# **ID / Name:** 16255 · Navarro &nbsp;|&nbsp; **Dataset:** TMDB 5000 + MovieLens &nbsp;|&nbsp; **Date:** April 2026
#
# ## 1. Introduction
#
# Recommender systems are one of the most common real-world uses of machine learning, even if most users do not notice them directly. The core problem is simple: there is a large catalogue, the user has limited time, and the system has to suggest something useful before the user loses interest.
#
# This notebook first evaluates the two required recommender approaches and then extends the comparison a bit further. TMDB 5000 provides the content side of the problem: genres, keywords, cast, crew, budgets and aggregate vote scores for roughly 4,800 films. MovieLens provides the individual user ratings (user behavior data). Together, the two datasets make it possible to move from general film quality to something closer to personal preference.
#
# The assignment requires a NP Bayesian Weighted Rating model and a Collaborative Filtering model. These are implemented first and evaluated in the required way. After that, I extend the comparison with two hybrid versions, because with the TMDB only we cannot really tell whether a recommendation meets a specific user's preferences or is just globally popular. The extension keeps the same model order as the rest of the notebook, but adds real user data from MovieLens to test personal relevance more directly.
#
# Metrics: RMSE · MAE · Precision@K · Recall@K · NDCG@K · Coverage · Diversity · Genre Mismatch Rate

# %% cell 2
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

# %% [markdown] cell 3
# ## 2. Data Exploration
#
# ### TMDB 5000

# %% cell 4
# ── Load raw CSVs ─────────────────────────────────────────────────────────
movies  = pd.read_csv('/content/tmdb_5000_movies.csv')
credits = pd.read_csv('/content/tmdb_5000_credits.csv')

# Align on movie id
credits = credits.rename(columns={'movie_id': 'id'})
df = movies.merge(credits[['id','cast','crew']], on='id', how='left')

print(f"Shape after merge : {df.shape}")
print(f"Columns           : {list(df.columns)}")

# %% cell 5
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

# %% [markdown] cell 6
# The two TMDB files were merged using the shared `id` column, and the metadata fields were parsed into Python lists so they could later be used as recommender features. A minimum of 10 votes quality filter removed 411 films with little information in their scores.
#
# That leaves **4,392 films** with vote averages ranging from 1.9 to 8.5. The 744 films with a recorded budget of zero are treated as missing financial data rather than genuinely zero-budget productions, because in this dataset zero usually means the information was not collected. The main point at this stage is that not every rating is equally reliable, so vote count is going to matter almost as much as vote average.

# %% [markdown] cell 7
# ### Exploratory Charts

# %% cell 8
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

# %% [markdown] cell 9
# The vote average chart shows that the typical film sits around **6.23** (std = 0.89), with most films grouped between about 5.7 and 6.8. This matters later because both core models are pulled toward the global mean, so RMSE can look fairly good even if the models are not learning much beyond that central value.
#
# The vote count chart needs a log scale because the distribution is very uneven. *Avatar* has 11,800 votes, while most films have fewer than 300. A film rated 5.0 by two people is not as reliable as a film rated 5.0 by several thousand people, even though the averages are the same. This is why using a raw average alone would be weak, and why the Bayesian model adjusts for vote count.
#
# The release year chart shows that the dataset is concentrated mostly between 2000 and 2016, with much fewer older films. Therefore, popularity, will be more weighted naturally toward modern films.

# %% cell 10
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Genre frequency
from collections import Counter
all_genres = Counter(g for gl in df_clean['genres_list'] for g in gl)
genre_df = pd.DataFrame(all_genres.most_common(15), columns=['genre','count'])
axes[0].barh(genre_df['genre'], genre_df['count'], color='steelblue')
axes[0].set(title='Top 15 most represented Genres', xlabel='Number of Movies')
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

# %% [markdown] cell 11
# The left chart shows how many films in the dataset belong to each genre. Drama has the highest count, followed by Comedy, Thriller and Action.
#
# In the budget vs revenue scatter there is a positive relationship overall, but the variance is very wide. There are many films below the diagonal, meaning they earned less than their budget, and some of those are expensive productions. A small number of franchise blockbusters on the far right make very large returns and pull the averages upward, which hides how risky the typical high budget film actually is. It is also worth noting that a large share of films have no financial data at all, so they don't appear here.

# %% cell 12
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

# %% [markdown] cell 13
# The mean vote average is 6.227 with a standard deviation of 0.893, so most films sit fairly close to the same centre. Genre counts are led by Drama, followed by Comedy, Thriller and Action, and most films have two or three genre labels at the same time.
#
# The director summary is also a useful check. The highest averages among directors with enough films include Hayao Miyazaki, Sergio Leone and Christopher Nolan, which suggests that the vote average is capturing some real quality signal rather than just random noise.

# %% [markdown] cell 14
# ## 3. Non-Personalized Recommender

# %% cell 15
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

# %% [markdown] cell 16
# The split gives **3,513 training films** and **879 test films**. It is stratified by `vote_average` quintile so that the test set still covers the full quality range of movies instead of being dominated by average films or very good ones only. The means of 6.227 and 6.228 are almost identical, which suggests the split worked as intended.

# %% cell 17
class BayesianWeightedRating:
    """
    Non-personalized recommender using the IMDb / TMDb Bayesian formula:

        WR(i) = (v / (v + m)) * R  +  (m / (v + m)) * C

    v = vote_count(i)
    R = vote_average(i)
    m = minimum-vote threshold
    C = global mean vote

    When v << m the score shrinks toward the global mean,
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

# %% [markdown] cell 18
# Sorting by raw average alone would be a weak choice here. For example, a film with only two ratings of 10.0 could rank above *The Shawshank Redemption*, which would not be very convincing. The Bayesian Weighted Rating formula handles this by pulling low-vote scores back toward the global mean of **6.227**.
#
# Here, **m = 662**, which is the 70th percentile of vote counts in the training data. So a film needs a fairly strong amount of voting evidence before its score is allowed to stand on its own. Below that point, the formula keeps pulling it back toward the average.

# %% cell 19
top10 = np_model.recommend(n=10)
print("Non-Personalized Top-10 Global Recommendations")
print("=" * 55)
for i, (mid, row) in enumerate(top10.iterrows(), 1):
    genres = ', '.join(row['genres_list'][:3])
    print(f"{i:2}. {row['title']:<40} WR={row['wr']:.3f}  votes={row['vote_count']:,}  genres=[{genres}]")

# %% [markdown] cell 20
# The top 10 look like what would normally be expected from a quality-based ranking: *The Shawshank Redemption* (WR=8.330, 8,205 votes), *The Godfather* (8.181), *Fight Club* (8.164), *Pulp Fiction* (8.149) and *Forrest Gump* (8.048), among others. Which are all highly recognized films, so the model is producing a sensible global list.
#
# On the other hand, every user gets the same ranking. That means the model is not personalised at all, but that is also not really its job. Its main value is in the cold-start case, where a new user has no rating history and the system still needs a reasonable first list.

# %% cell 21
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

# %% [markdown] cell 22
# **RMSE = 0.9169, MAE = 0.7219** on a 0-10 scale means predictions are usually less than one point away from the real value. Both core models end up with the same result because neither one has access to real individual preference at this stage.
#
# **P@10 = 1.000, NDCG@10 = 1.000** look ideal, but they need to be read carefully. In this setup, relevance is defined as vote_average > 7.0, which is exactly what the Non-Personalized model is designed to rank highly. So these scores show that the model is consistent with the evaluation rule, not necessarily that it is personally useful.
#
# **R@10 = 0.027** is still low because there are many highly rated films in the dataset, and the model only returns ten recommendations at a time.

# %% [markdown] cell 23
# ## 4. Collaborative Filtering Recommender

# %% cell 24
# ── Build rich text feature string per movie ─────────────────────────────
# We combine genres, keywords, cast names, and director into a single TF-IDF document per movie. 

def build_soup(row):
    parts = []
    # Genres (×3 weight)
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

# Refresh train and test to include the new 'soup' column 

train = df_clean.loc[train_idx].reset_index(drop=True)
test  = df_clean.loc[test_idx].reset_index(drop=True)

print("Sample feature soup for Avatar:")
print(df_clean[df_clean.title=='Avatar']['soup'].values[0][:200])
print(f"\nTrain has 'soup' column: {'soup' in train.columns} ✓")

# %% [markdown] cell 25
# Avatar's feature soup shows the weighting working as intended: genre tokens appear three times each, keywords appear twice, and then cast and director follow. When this is passed into TF-IDF, the repeated genre tokens end up carrying more weight in the vector, which makes sense because genre is usually one of the strongest signals of similarity for films.
#
# TF-IDF also improves on a simple genre-match method because it down-weights very common words like "drama" and gives more weight to more distinctive features, such as a specific keyword or director. So two films sharing a rarer keyword can end up closer together than two films that only share a broad genre label.

# %% cell 26
class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using TF-IDF content features.

    Similarity is computed via cosine distance on a TF-IDF matrix built
    from genre, keyword, cast and director tokens (finding movies that share similar attributes.

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

# %% [markdown] cell 27
# TF-IDF represents each film as a vector where common tokens get less weight and more distinctive ones get more weight. Cosine similarity then compares the angle between those vectors: 1.0 means the content is very similar, while 0.0 means they have very little in common. One advantage here is that cosine similarity is less affected by the total number of features, so films with lots of keywords are not automatically pushed far away from shorter ones.
#
# For predictions, the model takes the 20 most similar training films and returns a similarity-weighted average of their vote averages. For recommendations, it starts from the user's seed films, combines the similarity scores across them, and then returns the strongest unseen matches. In simple terms, a film ranks higher if it is similar to several things the user liked, not just one.

# %% cell 28
# ── Fit on training data ──────────────────────────────────────────────────
cf_model = ItemBasedCF(k=20, max_features=5000)
cf_model.fit(train)

print(f"TF-IDF vocab size : {len(cf_model.tfidf_.vocabulary_):,}")
print(f"Similarity matrix : {cf_model.sim_.shape}")
print(f"Training items    : {len(train):,}")

# %% [markdown] cell 29
# The model is fitted on **3,513 training films**. The TF-IDF vocabulary is capped at 5,000 features, which is large enough to capture more specific keywords, cast names and directors while still staying manageable computationally. Since the matrix is built once and then reused, recommendation itself stays relatively fast.

# %% cell 30
# ── Sample recommendations seeded from top-rated training movies ─────────
# We seed from films confirmed to be in the training set (from the Top-10 above)
seed_titles = ['The Dark Knight Rises', 'Interstellar']
seed_ids = [mid for mid in train['id'].tolist()
            if train.loc[train['id']==mid, 'title'].values[0] in seed_titles
            if mid in cf_model.index_map_] if False else []

# Robust seed: pick top-2 highest-rated movies in training set
top_train = train.nlargest(5, 'vote_average')[['id','title']].values
seed_ids = [int(top_train[0][0]), int(top_train[1][0])]
seed_names = [top_train[0][1], top_train[1][1]]

cf_recs = cf_model.recommend(seed_ids, n=10)
rec_df = train[train.id.isin(cf_recs)][['title','vote_average','genres_list']].copy()
rec_df['genres'] = rec_df['genres_list'].apply(lambda x: ', '.join(x[:3]))
rec_df = rec_df.sort_values('vote_average', ascending=False)

print(f"CF recommendations seeded from: {seed_names}")
print("=" * 62)
for _, row in rec_df.iterrows():
    print(f"  • {row['title']:<42}  ★{row['vote_average']:.1f}  [{row['genres']}]")

# %% [markdown] cell 31
# Seeding from *The Shawshank Redemption* and *The Godfather*, both strong Crime/Drama films, returns *The Godfather Part II* at the top, followed by other Crime and Drama titles. So the genre and keyword matching is behaving in the expected way.
#
# A clear issue shows up lower down the list. *The Rise of the Krays* and *Adulterers* appear because their tokens match the seeds, even though their vote averages are much weaker. This shows one of the main limits of content-based similarity: it optimises for similarity, not for quality. Two films can share a genre or director and still be very different overall. The hybrid extension later in **Section 5** is meant to deal with this more directly.

# %% cell 32
# CF: rating prediction on held-out movies 
cf_preds_val = test['id'].apply(cf_model.predict)
cf_rmse = rmse(test['vote_average'], cf_preds_val)
cf_mae  = mae( test['vote_average'], cf_preds_val)
print(f"Item-Based CF  RMSE={cf_rmse:.4f}  MAE={cf_mae:.4f}")

# CF: ranking evaluation via simulated users (within training set)
# Same protocol as NP: split high-rated train movies into seeds (relevant ones)
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

# %% [markdown] cell 33
# **RMSE = 0.9169, MAE = 0.7219** are identical to the Non-Personalized model. In other words, using content similarity alone does not give new information, because the model still has no real individual preference signal to learn from.
#
# **P@10 = 0.126, NDCG@10 = 0.136** are much lower than NP. CF is not ranking by quality but by similarity to the seed films. So a film can be quite similar to the seeds while still only being average overall. The trade-off is that CF explores taste and catalogue more than the NP model.
#
# **R@10 = 0.003** is also lower because CF only searches inside the pool of films that look similar to the seeds, and that may not overlap much with the wider set of globally high-rated films.

# %% [markdown] cell 34
# ## 5. Evaluation & Comparison

# %% cell 35
# Catalogue Coverage 
# Fraction of the training catalogue that appears in at least one rec list.
total          = len(train)
train_high_cov = train[train.vote_average >= RELEVANT_THRESHOLD]['id'].tolist()

# NP: same ranked list for every user (regardless of seen_ids)
np_recs_global = set(np_model.recommend(n=200).index.tolist())
np_cov = len(np_recs_global) / total

# CF: different list per simulated user take the union across 100 users
cf_recs_all = set()
for _ in range(100):
    seeds = list(np.random.choice(train_high_cov,
                                  min(5, len(train_high_cov)), replace=False))
    seen  = list(np.random.choice(train['id'].tolist(), 100, replace=False))
    cf_recs_all.update(cf_model.recommend(seeds, n=K, seen_ids=seen))

cf_cov = len(cf_recs_all) / total
print(f"Catalogue Coverage  NP={np_cov:.4f}  CF={cf_cov:.4f}  (training catalog={total:,})")

# %% [markdown] cell 36
# **NP = 0.057, CF = 0.091.** CF achieves ~1.6× higher catalogue coverage than NP (9.1% vs 5.7%). The NP model mainly serves the same set of globally popular films to everyone, so once those are exhausted it does not have much variety left to offer.
#
# CF pushes different users toward different content clusters depending on their seeds, so more diverse films start to appear. In a much larger streaming catalogue, the difference between 5.7% and 9.1% coverage could mean thousands of extra titles actually reaching users.

# %% cell 37
# Intra-list Genre Diversity 
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

# %% [markdown] cell 38
# **NP = 0.742, CF = 0.585.** The NP model's diversity score is high, but not because it is doing anything particularly thoughtful — it gives everyone the same 10 films, and those films just happen to span several genres naturally. CF's lists are more genre-homogeneous because they cluster around whatever genre the seed films belong to. Seed from two Crime/Drama films and you mostly get Crime/Drama back. 
#
# A Jaccard distance of 0.585 still suggests some variety inside CF lists, but the filter-bubble tendency is visible here. This is a known trade-off: the more a model focuses on what you already like, the more it risks showing you the same type of thing repeatedly. Section 6 partially addresses this by anchoring the CF search across the user's top three genres simultaneously rather than a single seed.

# %% cell 39
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

# %% [markdown] cell 40
# Both models are doing something useful, but in different ways.
#
# NP dominates Precision@10, Recall@10, NDCG@10 and Diversity. That is expected because NP ranks by quality, and the evaluation rule also treats quality as relevance. So it makes sense that NP looks stronger on those metrics.
#
# CF wins on Coverage. Even without real user data, sending different simulated users through different seed films makes CF explore a wider part of the catalogue. This is the first real sign of the personalisation advantage that becomes clearer later in **Section 5**.
#
# RMSE and MAE are tied at 0.917 and 0.722. That suggests neither model is learning much beyond the global mean when only aggregate TMDB data is available. This is basically the ceiling of the TMDB-only setup, and it is the reason MovieLens is added later.

# %% cell 41
# ── Core Two-Model Metrics Table ─────────────────────────────────────────
core_metrics = pd.DataFrame({
    'Approach'   : ['Non-Personalized (Bayesian WR)', 'Item-Based CF (TF-IDF Cosine)'],
    'RMSE'       : [round(np_rmse,  4), round(cf_rmse,  4)],
    'MAE'        : [round(np_mae,   4), round(cf_mae,   4)],
    'Precision@K': [round(np_prec,  4), round(cf_prec,  4)],
    'Recall@K'   : [round(np_rec,   4), round(cf_rec,   4)],
    'NDCG'       : [round(np_ndcg,  4), round(cf_ndcg,  4)],
    'Coverage'   : [round(np_cov,   4), round(cf_cov,   4)],
    'Diversity'  : [round(np_div,   4), round(cf_div,   4)],
})
print(core_metrics.to_string(index=False))
core_metrics

# %% [markdown] cell 42
# ### Required Model Comparison
#
# NP wins on 5 out of 7 metrics, but that does not automatically mean it is the better model overall. The evaluation uses vote_average > 7.0 as the relevance threshold, and that is already close to what NP is designed to prioritise. So this result shows that the model matches the evaluation rule well, more than it proves it is better in every practical sense.
#
# The bigger limitation is what this table cannot show, whether either model is actually recommending films that suit the user, or just films that score well in general. 
#
# ### Motivation for the Extension
#
# The brief only asks for the two models above, but the TMDB only setup has something missing. A film can be globally highly rated and still be a poor recommendation for a specific person if it belongs to a genre they usually dislike. Linking MovieLens through `links.csv` makes it possible to test this properly through a **genre mismatch rate**, rather than only looking at global quality scores.

# %% [markdown] cell 43
# ### Real User Behaviour and Baseline Mismatch
#
# The two required models above were evaluated against simulated users. Through MovieLens we add this layer, real users, real rating behaviour and enough overlap with TMDB to test whether a recommendation respects taste rather than only popularity.

# %% cell 44
# Load MovieLens files 
ml_ratings = pd.read_csv('/content/ratings.csv')
ml_links   = pd.read_csv('/content/links.csv')

# Clean the tmdbId column 
ml_links['tmdbId'] = pd.to_numeric(ml_links['tmdbId'], errors='coerce')
ml_links = ml_links.dropna(subset=['tmdbId'])
ml_links['tmdbId'] = ml_links['tmdbId'].astype(int)

print(f"MovieLens ratings : {len(ml_ratings):,}")
print(f"Unique users      : {ml_ratings.userId.nunique():,}")
print(f"Unique ML movies  : {ml_ratings.movieId.nunique():,}")
print(f"Rating scale      : {ml_ratings.rating.min()} – {ml_ratings.rating.max()}")
print()
print(ml_ratings.head(5).to_string(index=False))

# %% [markdown] cell 45
# MovieLens contributes **100,836 individual ratings** from **610 users** on a 0.5-5.0 half-star scale. The main difference from TMDB, therefore this individual variation is what makes it possible to build user taste profiles instead of assuming that everyone wants the same globally popular content.

# %% cell 46
# Merge MovieLens ratings with TMDB metadata 
# through links.csv: MovieLens movieId > TMDB id
# We then pull in genre, title and vote info from our existing df_clean

ratings_rich = (ml_ratings
    .merge(ml_links[['movieId','tmdbId']], on='movieId', how='inner')
    .merge(df_clean[['id','title','genres_list','vote_average','vote_count']],
           left_on='tmdbId', right_on='id', how='inner'))

print(f"Ratings with full metadata : {len(ratings_rich):,}")
print(f"Users retained             : {ratings_rich.userId.nunique():,}")
print(f"Movies retained            : {ratings_rich.tmdbId.nunique():,}")
print(f"(Movies lost in merge: films in MovieLens not present in TMDB 5000)")
print()
print(ratings_rich[['userId','title','rating','genres_list']].head(6).to_string(index=False))

# %% [markdown] cell 47
# After linking through `links.csv`, the overlap used here contains **70,149 ratings across 610 users and 3,505 movies** that also exist in the TMDB catalogue.
#
# Even a quick look at one user's history shows why this matters. Films inside the same broad genre can receive very different scores from the same person. That kind of within-genre variation is exactly what aggregate vote averages flatten out, and it is why individual profiles are worth building.

# %% cell 48
# Build a personal genre preference profile for every user 
# For each user we compute their average rating per genre AND how many
# times they have rated that genre. Both are needed for the smoothed filter.

from collections import defaultdict

def build_user_genre_profile(user_df):
    """Returns ({genre: mean_rating}, {genre: count}) for one user."""
    genre_scores = defaultdict(list)
    for _, row in user_df.iterrows():
        for g in row['genres_list']:
            genre_scores[g].append(row['rating'])
    profile = {g: round(sum(v) / len(v), 3) for g, v in genre_scores.items()}
    counts  = {g: len(v) for g, v in genre_scores.items()}
    return profile, counts

user_profiles     = {}
user_genre_counts = {}
for uid, grp in ratings_rich.groupby('userId'):
    profile, counts = build_user_genre_profile(grp)
    user_profiles[uid]     = profile
    user_genre_counts[uid] = counts

# Show two contrasting user profiles
for uid in [1, 5]:
    profile   = user_profiles[uid]
    counts    = user_genre_counts[uid]
    n_ratings = len(ratings_rich[ratings_rich.userId == uid])
    print(f"User {uid}  ({n_ratings} ratings total)")
    print("-" * 48)
    for genre, score in sorted(profile.items(), key=lambda x: -x[1]):
        bar   = '█' * int(score * 2)
        liked = 'yes' if score >= 3.5 else 'no'
        print(f"  {liked} {genre:<20} {score:.2f}  {bar}  (n={counts[genre]})")
    print()

# %% [markdown] cell 49
# For each user, a genre profile is built from rating history. Two thresholds are used for different purposes. For the **mismatch metric**, a recommendation counts as a clash only when the user's average for that primary genre is **below 3.0**. For the **hybrid filters**, the rule is stricter but also smoothed: a genre is blocked only when the user averages **at or below 3.5** across **at least 3 films**.
#
# **User 1** has rated 147 films and seems to have a broad taste. Their strongest genres include Music, Western and Animation, while even their weakest category is not strongly negative. This kind of profile gives the personalised models more room to explore.
#
# **User 5** is much more selective with only 32 ratings. They show positive preferece for Western, History and Animation, but have much weaker scores in high volume genres such as Drama, Comedy, Action and Romance. This is a better example of the kind of user for whom a list based on popularity becomes less useful.

# %% cell 50
# Popularity bias detector 
# We measure the "genre mismatch rate": out of every 10 recommendations,
# how many go to a user who has historically DISLIKED that genre?
# (defined as: average rating for that genre below 3.0 in their history)
#
# Improvement of TMDB

DISLIKE_THRESHOLD = 3.0
movie_genres_lookup = df_clean.set_index('id')['genres_list']

def genre_mismatch_rate(recommendations, user_profile, movie_genres):
    mismatches = 0
    for mid in recommendations:
        if mid not in movie_genres.index:
            continue
        primary_genre = movie_genres[mid][0] if movie_genres[mid] else None
        if not primary_genre:
            continue
        user_score = user_profile.get(primary_genre)
        # Only flag as mismatch if user has rated this genre AND dislikes it
        if user_score is not None and user_score < DISLIKE_THRESHOLD:
            mismatches += 1
    return mismatches / len(recommendations) if recommendations else 0

# Test on 50 users
np.random.seed(42)
sample_users = ratings_rich['userId'].unique()[:50]
np_mismatch_rates = []
cf_mismatch_rates = []

for uid in sample_users:
    profile = user_profiles.get(uid, {})
    if not profile:
        continue
    seen = ratings_rich[ratings_rich.userId == uid]['tmdbId'].tolist()

    # NP: same global top-10 for everyone
    np_recs = np_model.recommend(n=10, seen_ids=seen).index.tolist()

    # CF: seeded from this user's actually liked films
    liked_ids = ratings_rich[
        (ratings_rich.userId == uid) & (ratings_rich.rating >= 3.5)
    ]['tmdbId'].tolist()
    liked_in_model = [m for m in liked_ids if m in cf_model.index_map_]
    cf_recs = cf_model.recommend(liked_in_model[:5], n=10, seen_ids=seen) if liked_in_model else []

    np_mismatch_rates.append(genre_mismatch_rate(np_recs,  profile, movie_genres_lookup))
    cf_mismatch_rates.append(genre_mismatch_rate(cf_recs,  profile, movie_genres_lookup))

print(f"Genre mismatch rate (fraction of recs clashing with user taste):")
print(f"  Non-Personalized : {np.mean(np_mismatch_rates):.1%}")
print(f"  Item-Based CF    : {np.mean(cf_mismatch_rates):.1%}")
print()
print("Lower = the model better respects what the user actually likes.")
print("NP recommends the same list to everyone so it inevitably")
print("clashes with individual tastes more often than CF does.")

# %% [markdown] cell 51
# **NP mismatch = 15.4%, CF mismatch = 9.4%.** In practice, that means the Non-Personalized model offers around 1 to 2 recommendations in every 10 into a genre the user has usually rated poorly. CF improves on that by seeding from films the user actually liked, but the mismatch rate is still close to one in ten.
#
#
# ### Hybrid v1 - CF + Smoothed Genre Filter

# %% cell 52
# Hybrid Recommender: CF with smoothed genre filter 
# Three rules:
#   1. CF: films similar to what this user has actually liked
#   2. Smoothed genre filter: only block a genre if the user has rated it
#      below 3.5 AND has done so across at least 3 films. One or two bad
#      experiences in a genre do not permanently disregard it
#   3. NP fallback: fill remaining slots from the global quality list
#      (same filter applied)
#
# Unknown genres are always allowed through.

DISLIKE_THRESHOLD = 3.5
MIN_GENRE_RATINGS = 3   # minimum ratings in a genre before the block triggers

def passes_genre_filter(tmdb_id, profile, genre_counts):
    """True = film is allowed through the genre filter."""
    genres = movie_genres_lookup.get(tmdb_id, [])
    if not genres:
        return True
    primary = genres[0]
    score   = profile.get(primary)
    count   = genre_counts.get(primary, 0)
    if score is None:
        return True                              # never rated this genre — allow
    if score <= DISLIKE_THRESHOLD and count >= MIN_GENRE_RATINGS:
        return False                             # consistent dislike — block
    return True                                  # too few ratings to be confident — allow

# Genre coverage lookup: maps each genre to its training film IDs 
# 
# Used to inject proxy seeds for genres the user loves but hasn't rated yet
genre_to_train_films = defaultdict(list)
for _, _row in train.iterrows():
    for _g in _row['genres_list']:
        genre_to_train_films[_g].append(_row['id'])

def build_seeds_with_coverage(user_id, liked_in_model, profile, n_seeds=5):
    """
    Guarantees representation of the user's top 3 loved genres in the seed list.
    If the user has no rated films in a loved genre (sparse history), injects
    the highest-WR unseen training film in that genre as a proxy seed.
    This ensures CF is anchored to all three top preferences, not just the ones
    the user happens to have already watched.
    """
    seen = set(ratings_rich[ratings_rich.userId == user_id]['tmdbId'].tolist())
    top_genres = [g for g, _ in sorted(profile.items(), key=lambda x: -x[1])[:3]]

    seeds = []
    used  = set()

    for genre in top_genres:
        # Try a film the user has already liked in this genre
        genre_liked = [m for m in liked_in_model
                       if m not in used and genre in (movie_genres_lookup.get(m, []))]
        if genre_liked:
            seeds.append(genre_liked[0])
            used.add(genre_liked[0])
        else:
            # No rated films in this genre > use best unseen training film as proxy
            candidates = [fid for fid in genre_to_train_films[genre]
                          if fid not in seen and fid in cf_model.index_map_ and fid not in used]
            if candidates:
                best = max(candidates,
                           key=lambda x: np_model.scores_.loc[x, 'wr']
                           if x in np_model.scores_.index else 0)
                seeds.append(best)
                used.add(best)

    # Fill remaining slots with preference-sorted liked films
    def genre_score(tmdb_id):
        genres = movie_genres_lookup.get(tmdb_id, [])
        return profile.get(genres[0], 0) if genres else 0

    remaining = sorted([m for m in liked_in_model if m not in used],
                       key=genre_score, reverse=True)
    seeds.extend(remaining[:n_seeds - len(seeds)])
    return seeds[:n_seeds]

def hybrid_recommend(user_id, n=10):
    profile = user_profiles.get(user_id, {})
    counts  = user_genre_counts.get(user_id, {})
    seen    = set(ratings_rich[ratings_rich.userId == user_id]['tmdbId'].tolist())

    liked = ratings_rich[
        (ratings_rich.userId == user_id) & (ratings_rich.rating >= 3.5)
    ]['tmdbId'].tolist()
    liked_in_model = [m for m in liked if m in cf_model.index_map_]
    seeds = build_seeds_with_coverage(user_id, liked_in_model, profile, n_seeds=5)
    cf_candidates  = cf_model.recommend(seeds, n=50, seen_ids=list(seen)) \
                     if seeds else []

    filtered = [m for m in cf_candidates if passes_genre_filter(m, profile, counts)]

    if len(filtered) < n:
        for m in np_model.recommend(n=200, seen_ids=list(seen)).index.tolist():
            if m not in filtered and passes_genre_filter(m, profile, counts):
                filtered.append(m)
            if len(filtered) >= n:
                break

    return filtered[:n]


# Demo: two users with different taste profiles
for uid in [1, 5]:
    recs     = hybrid_recommend(uid, n=8)
    rec_rows = df_clean[df_clean.id.isin(recs)][['title','genres_list']].values
    profile  = user_profiles[uid]
    top3     = [g for g, _ in sorted(profile.items(), key=lambda x: -x[1])[:3]]
    bottom3  = [g for g, _ in sorted(profile.items(), key=lambda x:  x[1])[:3]]

    print(f"User {uid}  |  Loves: {top3}  |  Dislikes: {bottom3}")
    print("-" * 64)
    for title, genres in rec_rows:
        print(f"  • {title:<42} [{', '.join(genres[:2])}]")
    print()

# %% [markdown] cell 53
# Hybrid v1 combines three main ideas. First, CF generates candidates from films similar to what the user liked. Second, a smoothed genre filter removes recommendations whose primary genre is consistently rated badly. Third, a Non-Personalized fallback fills any remaining slots so the system can still return a full list.
#
# The coverage-aware seed approach is what makes this stronger than just filtering a normal CF list. For genres where the user has clear positive history, the model seeds from films they already liked. Where history is more limited but still positive, it injects the highest-WR unseen training film from that genre as a proxy seed.
#
# For **User 1**, the recommendations stay centred on Music, Western and Animation. For **User 5**, the list leans more toward Animation and Western titles. This is the kind of correction the earlier global models could not really make.

# %% cell 54
# Hybrid v2: CF + smoothed genre filteer + quality gate 
# Third condition on top of the original Hybrid:
# every recommended film must also pass a minimum Bayesian WR threshold.
#
# This solves the quality problem: a film with distinctive
# but rare tokens can score high on content similarity without being good.
# Now it also needs to have earned a decent quality score.
#
# WR threshold = 6.3 — high enough to block the lower quality films
# (e.g. The Rise of the Krays at WR=6.18, Adulterers at WR=6.20)
# that slipped through on genre token matches alone.
#
# A film now passes only if it clears all three filters:
#   Filter 1 — content similarity to the user's liked films (CF)
#   Filter 2 — primary genre not consistently disliked (smoothed genre filter)
#   Filter 3 — Bayesian WR >= 6.3 (quality threshold)

WR_THRESHOLD = 6.3

def hybrid_v2_recommend(user_id, n=10, wr_threshold=WR_THRESHOLD):
    profile = user_profiles.get(user_id, {})
    counts  = user_genre_counts.get(user_id, {})
    seen    = set(ratings_rich[ratings_rich.userId == user_id]['tmdbId'].tolist())

    def passes_all_filters(tmdb_id):
        # Gate 1 + 2: smoothed genre filter
        if not passes_genre_filter(tmdb_id, profile, counts):
            return False
        # Gate 3: minimum quality
        if tmdb_id in np_model.scores_.index:
            if np_model.scores_.loc[tmdb_id, 'wr'] < wr_threshold:
                return False
        return True

    liked = ratings_rich[
        (ratings_rich.userId == user_id) & (ratings_rich.rating >= 3.5)
    ]['tmdbId'].tolist()
    liked_in_model = [m for m in liked if m in cf_model.index_map_]
    seeds = build_seeds_with_coverage(user_id, liked_in_model, profile, n_seeds=5)
    cf_candidates  = cf_model.recommend(seeds, n=80, seen_ids=list(seen)) \
                     if seeds else []

    filtered = [m for m in cf_candidates if passes_all_filters(m)]

    if len(filtered) < n:
        for m in np_model.recommend(n=200, seen_ids=list(seen)).index.tolist():
            if m not in filtered and passes_all_filters(m):
                filtered.append(m)
            if len(filtered) >= n:
                break

    return filtered[:n]


# Demo: same two users — now with WR shown for each recommendation 
for uid in [1, 5]:
    recs     = hybrid_v2_recommend(uid, n=8)
    rec_rows = df_clean[df_clean.id.isin(recs)][['id','title','genres_list']].copy()
    rec_rows['wr'] = rec_rows['id'].apply(
        lambda x: round(np_model.scores_.loc[x, 'wr'], 2)
        if x in np_model.scores_.index else None)
    profile = user_profiles[uid]
    top3    = [g for g, _ in sorted(profile.items(), key=lambda x: -x[1])[:3]]

    print(f"User {uid}  |  Loves: {top3}")
    print("-" * 66)
    for _, row in rec_rows.iterrows():
        print(f"  • {row['title']:<42} WR={row['wr']}  [{', '.join(row['genres_list'][:2])}]")
    print()

# %% [markdown] cell 55
# ### Hybrid v2 - CF + Smoothed Genre Filter + WR Quality Gate
#
# Hybrid v2 keeps the Hybrid v1 logic and adds one more requirement: every recommendation must also pass **WR >= 6.3**. This is meant to deal with the weakness seen in raw CF, where a film can look similar because of shared tokens but not meet the quality standards.
#
# With this quality filter added, the final lists look cleaner. **User 1's list** still covers the user's preferred genres, but now every title also passes the same quality threshold. **User 5's list** is still personalised, while the weaker content matches are mostly removed.

# %% cell 56
# Full four-model comparison
hybrid_mismatch_rates   = []
hybridv2_mismatch_rates = []
hybrid_recs_all         = set()
hybridv2_recs_all       = set()

# Mean WR of recommended films — measures quality of what gets surfaced
def mean_wr_of_recs(rec_ids):
    wrs = [np_model.scores_.loc[m, 'wr']
           for m in rec_ids if m in np_model.scores_.index]
    return round(np.mean(wrs), 4) if wrs else 0

np_wr_scores, cf_wr_scores, h_wr_scores, h2_wr_scores = [], [], [], []

for uid in sample_users:
    profile = user_profiles.get(uid, {})
    if not profile:
        continue
    seen     = ratings_rich[ratings_rich.userId == uid]['tmdbId'].tolist()
    liked    = ratings_rich[(ratings_rich.userId==uid)&(ratings_rich.rating>=3.5)]['tmdbId'].tolist()
    liked_in = [m for m in liked if m in cf_model.index_map_]

    np_recs  = np_model.recommend(n=10, seen_ids=seen).index.tolist()
    cf_recs  = cf_model.recommend(liked_in[:5], n=10, seen_ids=seen) if liked_in else []
    h_recs   = hybrid_recommend(uid, n=10)
    h2_recs  = hybrid_v2_recommend(uid, n=10)

    hybrid_mismatch_rates.append(genre_mismatch_rate(h_recs,  profile, movie_genres_lookup))
    hybridv2_mismatch_rates.append(genre_mismatch_rate(h2_recs, profile, movie_genres_lookup))
    hybrid_recs_all.update(h_recs)
    hybridv2_recs_all.update(h2_recs)

    np_wr_scores.append(mean_wr_of_recs(np_recs))
    cf_wr_scores.append(mean_wr_of_recs(cf_recs))
    h_wr_scores.append(mean_wr_of_recs(h_recs))
    h2_wr_scores.append(mean_wr_of_recs(h2_recs))

# Coverage
total        = len(df_clean)
np_cov_ext   = len(set(np_model.recommend(n=200).index.tolist())) / total
cf_recs_union = set()
for uid in sample_users:
    liked    = ratings_rich[(ratings_rich.userId==uid)&(ratings_rich.rating>=3.5)]['tmdbId'].tolist()
    liked_in = [m for m in liked if m in cf_model.index_map_]
    if liked_in:
        seen = ratings_rich[ratings_rich.userId==uid]['tmdbId'].tolist()
        cf_recs_union.update(cf_model.recommend(liked_in[:5], n=10, seen_ids=seen))
cf_cov_ext   = len(cf_recs_union) / total
hybrid_cov   = len(hybrid_recs_all) / total
hybridv2_cov = len(hybridv2_recs_all) / total

print("=" * 72)
print(f"{'Model':<28} {'Mismatch':>10} {'Coverage':>10} {'Mean WR':>10}")
print("-" * 72)
print(f"{'Non-Personalized':<28} {np.mean(np_mismatch_rates):>9.1%} {np_cov_ext:>10.4f} {np.mean(np_wr_scores):>10.4f}")
print(f"{'Item-Based CF':<28} {np.mean(cf_mismatch_rates):>9.1%} {cf_cov_ext:>10.4f} {np.mean(cf_wr_scores):>10.4f}")
print(f"{'Hybrid (CF+filter)':<28} {np.mean(hybrid_mismatch_rates):>9.1%} {hybrid_cov:>10.4f} {np.mean(h_wr_scores):>10.4f}")
print(f"{'Hybrid v2 (CF+filter+WR)':<28} {np.mean(hybridv2_mismatch_rates):>9.1%} {hybridv2_cov:>10.4f} {np.mean(h2_wr_scores):>10.4f}")
print("=" * 72)
print()
print("Mismatch = fraction clashing with user genre history  (lower = better)")
print("Coverage = fraction of catalogue reached              (higher = better)")
print("Mean WR  = avg Bayesian quality of recommended films  (higher = better)")

# %% [markdown] cell 57
# ### Final Four-Model Comparison
#
# This is the main result of the notebook.
#
# **NP** has the highest mismatch at 15.4%, which means around 1 to 2 recommendations in every 10 clash with the user's genre history. Its mean WR of 7.96 is the highest of all models because it is directly sorted by WR, but that comes with the downside that it ignores who the user is.
#
# **CF** lowers mismatch to 9.4% by seeding from liked films, but its mean WR falls to 6.27, which is the lowest of all models. This supports the earlier point that content similarity alone can return films that look relevant but are not especially strong overall.
#
# **Hybrid v1** drops mismatch to 0.2% through the smoothed genre filter and coverage-aware seeds, while recovering some coverage through the Non-Personalized fallback. It is the strongest model for pure personal fit.
#
# **Hybrid v2** raises the quality threshold to WR >= 6.3, which increases mismatch slightly (also higher chances for discovery) to 0.6% but also pushes mean WR up to 6.70. So the trade-off is  a small loss on mismatch in exchange for noticeably better recommendation quality.
#
# ### Overall Judgment
#
# The main takeaway is not that Hybrid v2 beats the Non-Personalized model on every metric, because the two systems are not trying to do exactly the same job. The more important result is that the hybrid approach brings personal relevance and recommendation quality closer together. Hybrid v1 shows that mismatch can almost disappear, while Hybrid v2 shows that this can be done without letting quality drop too far.

# %% cell 58
#Visualisation: four-model comparison 
models   = ['Non-Personalized', 'Item-Based CF', 'Hybrid', 'Hybrid v2']
mis_vals = [np.mean(np_mismatch_rates), np.mean(cf_mismatch_rates),
            np.mean(hybrid_mismatch_rates), np.mean(hybridv2_mismatch_rates)]
cov_vals = [np_cov_ext, cf_cov_ext, hybrid_cov, hybridv2_cov]
wr_vals  = [np.mean(np_wr_scores), np.mean(cf_wr_scores),
            np.mean(h_wr_scores),  np.mean(h2_wr_scores)]
colors   = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

specs = [
    (mis_vals, 'Genre Mismatch Rate\n(lower = better)',       'Fraction mismatched',    ':.1%'),
    (cov_vals, 'Catalogue Coverage\n(higher = better)',       'Fraction of catalogue',  ':.3f'),
    (wr_vals,  'Mean WR of Recommendations\n(higher = better)', 'Avg Bayesian WR',      ':.3f'),
]

for ax, (vals, title, ylabel, fmt) in zip(axes, specs):
    bars = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, v in zip(bars, vals):
        label = (f'{v:.1%}' if fmt == ':.1%' else f'{v:.3f}')
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                label, ha='center', fontweight='bold', fontsize=8.5)

plt.tight_layout()
plt.savefig('four_model_comparison.png', bbox_inches='tight')
plt.show()
print("Chart saved ✓")

# %% [markdown] cell 59
# Before the rubric table, it is worth keeping the two hybrid models visible as well because they answer a slightly different part of the problem. **Hybrid v1** is the stronger pure-personalisation version, almost removing genre mismatch by filtering out genres the user has consistently rated badly. **Hybrid v2** keeps that same idea but adds a minimum WR quality gate, so mismatch rises slightly while the average quality of recommendations improves quite a lot.
#
# So in simple terms: **Hybrid v1** is better if the priority is matching user taste as closely as possible, while **Hybrid v2** is the better balanced option if the goal is to keep the recommendations personal but also make sure the films are strong enough overall.

# %% cell 60
# ── Hybrid extension summary + mandatory final metrics table ─────────────
hybrid_metrics = pd.DataFrame({
    'Approach'      : ['Hybrid v1 (CF + filter)', 'Hybrid v2 (CF + filter + WR)'],
    'Mismatch Rate' : [round(np.mean(hybrid_mismatch_rates), 4), round(np.mean(hybridv2_mismatch_rates), 4)],
    'Coverage'      : [round(hybrid_cov, 4), round(hybridv2_cov, 4)],
    'Mean WR'       : [round(np.mean(h_wr_scores), 4), round(np.mean(h2_wr_scores), 4)],
})

display(hybrid_metrics)

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

final_metrics

