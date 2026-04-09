# Recommender System Analysis & Implementation

**Student ID:** 16255  
**Last Name:** Navarro  
**Dataset:** TMDB 5000 Movies  
**Date:** April 2026  

---

## 1. Introduction

Recommender systems are among the most commercially impactful applications of machine
learning, quietly shaping the films we watch, the music we hear and the products we buy.
A core challenge for the industry is surfacing the right item from a massive catalogue
before a user loses interest — and doing so profitably, since blockbuster films with
$200 million budgets still regularly underperform.

This assignment builds and evaluates two recommender approaches on the **TMDB 5000 Movies**
dataset, a Kaggle-hosted collection scraped from The Movie Database (TMDb) API. The dataset
covers approximately 4 800 films with rich metadata: vote averages, vote counts, genres,
plot keywords, cast lists, crew (including director), budgets and revenues. Critically, this
is a **catalogue-level dataset** — it does not contain individual user–item rating matrices.
Each movie has an aggregate TMDb vote average (0–10 scale) and a vote count.

This shapes the modelling choices fundamentally:

- **Non-Personalized (Bayesian Weighted Rating):** Ranks all movies using a regularised
  aggregate score that balances a film's individual vote average against the global mean.
  Identical for all users; ideal for cold-start scenarios and editorial "best-of" lists.
- **Item-Based Collaborative Filtering (TF-IDF Cosine Similarity):** Measures item–item
  similarity using TF-IDF representations of genres, keywords, cast and director, then
  recommends movies most similar to a user's known favourites. This is the item-based
  analogue of standard CF: we transfer preferences through shared item attributes rather
  than shared user histories.

Both models are evaluated on rating-prediction accuracy (RMSE, MAE), ranking quality
(Precision@10, Recall@10, NDCG@10) and beyond-accuracy properties (Coverage, Diversity).
Since the dataset splits at the movie level, ranking metrics are evaluated via a
within-training-set simulation described in Section 5.

---

## 2. Data Exploration

The TMDB 5000 dataset contains two files: `tmdb_5000_movies.csv` (4 803 rows, 20 columns)
and `tmdb_5000_credits.csv` (4 803 rows with full cast and crew encoded as JSON strings).
After merging on `id` and applying a quality filter (minimum 10 votes, vote_average > 0)
the working dataset contains **4 392 movies**.

**Vote distribution.** The mean vote average is approximately 6.09 and the distribution is
roughly bell-shaped around 6–7, with a modest left skew. This is consistent with a
selection effect: the dataset skews toward movies that people chose to rate (and
disproportionately chose to see), so low-rated films are underrepresented relative to
the full universe of released movies.

**Vote count.** Vote count follows an extreme power-law distribution. The median is around
200 votes, but blockbusters such as Inception and The Dark Knight accumulate over 10 000
votes. The top 30% of movies by vote count account for the majority of the informational
signal in the dataset, which motivates the Bayesian correction rather than raw averages.

**Genre composition.** Drama (2 387 movies), Comedy (1 742), Thriller (1 306) and Action
(1 260) are the four most common genres. Most movies carry two to three genre labels, which
enables meaningful intra-list diversity calculations later.

**Financial data.** Budget and revenue are only available for roughly 60% of films (zeros
treated as missing, following standard practice). Among films with known financials, the
median budget is approximately $25 million and the median revenue is $38 million, but
distributions are highly right-skewed — a handful of superhero and franchise films dominate.

**Release years.** The dataset spans films from the 1960s through 2017, with the heaviest
concentration between 2000 and 2016.

**Train/test split.** An 80/20 stratified split (stratified by vote_average quintile) was
applied, producing 3 513 training movies and 879 test movies with nearly identical mean
vote averages. This split is used for RMSE/MAE evaluation. Ranking metrics use a separate
within-training simulation (see Section 5).

---

## 3. Non-Personalized Recommender

### Design

A naive "sort by vote_average" approach fails because it promotes films with only one or
two votes at 10.0 over beloved classics with thousands of honest ratings. The **Bayesian
Weighted Rating** formula — used by IMDb and standardised for recommender research — corrects
for this:

```
WR(i) = (v / (v + m)) × R  +  (m / (v + m)) × C
```

where `v` is the movie's vote count, `R` is its vote average, `m` is a minimum-vote
threshold, and `C` is the global mean vote across all training movies. When `v` is much
smaller than `m`, the score collapses toward `C`, preventing obscure films with inflated
averages from dominating. As `v` grows, the score converges to the true vote average.

The threshold `m` is set to the **70th percentile of vote counts** in the training set
(approximately 215 votes), meaning that only the top 30% of films by popularity receive
scores close to their raw vote average.

### Top-10 Recommendations

The model correctly surfaces critically acclaimed, broadly beloved films such as
*The Shawshank Redemption*, *The Godfather* and *The Dark Knight* at the top of the
global list — consistent with IMDb's Top 250. This validates the Bayesian correction.

### Strengths and Limitations

This model scales to any catalogue size in O(n) time and space, requires no user history,
and is fully interpretable. Its critical limitation is that it serves the **same list to
every user**, ignoring individual preferences. Coverage is also inherently limited: the
model surfaces only the top few hundred most-voted films, leaving the majority of the
catalogue permanently unreachable.

---

## 4. Item-Based Collaborative Filtering Recommender

### Design

With no explicit user–item rating matrix available in the TMDB dataset, I implement
**Item-Based CF** using rich item metadata as the similarity signal. Each movie is
represented as a weighted bag-of-words "soup":

- Genre names repeated **3×** (primary driver of taste)
- Top-10 plot keywords repeated **2×** (thematic nuance)
- Top-5 cast names repeated **1×** (talent signal)
- Director name repeated **2×** (style signal)

All tokens are lowercased and whitespace-stripped. This weighting reflects the intuition
that genre is the strongest preference predictor, followed by thematic keywords and talent.

**TF-IDF vectorization** transforms each film's soup string into a sparse vector of 5 000
features, down-weighting tokens that appear in almost every movie. **Cosine similarity** is
then computed pairwise across all training movies, producing a (3 513 × 3 513) similarity
matrix.

For **rating prediction**, the predicted vote_average is the similarity-weighted average
of the k = 20 nearest neighbours' vote averages. For **recommendation**, seed items from
a user's history aggregate similarity scores across all candidate movies, ranked and returned.

### Limitations

The primary limitation is that this approach is technically **content-based**, not
collaborative in the traditional sense (no cross-user signal). It also inherits the
"filter bubble" problem: a user who has only seen action films will receive action
recommendations, with less serendipity than true CF. A hybrid approach — content similarity
seeded with CF co-occurrence patterns — would address this in production.

---

## 5. Evaluation & Comparison

### Metrics

**RMSE and MAE** measure prediction error on held-out test movies (vote_average scale 0–10).
Both models predict the vote_average of unseen movies. Lower is better.

**Precision@10, Recall@10, NDCG@10** assess ranking quality. Because the dataset splits
at the movie level (train and test are disjoint movie sets), a CF model trained on train
movies cannot recommend test movie IDs. The standard solution for catalogue-level datasets
is a **within-training simulation**: for each of 200 simulated users, the high-rated
training movies are split randomly into "seeds" (seen) and "relevant" (held-out). The model
must surface the relevant half using only the seeds as input. Movies with vote_average ≥ 7.0
are treated as high-quality. NDCG@K additionally rewards placing good movies earlier in the list.

**Coverage** is the fraction of the training catalogue that appears in at least one
recommendation list across 100 simulated users. **Diversity** is the mean intra-list Jaccard
distance between genre sets of the recommended movies.

### Results

| Metric | Non-Personalized | Item-Based CF | Notes |
|--------|-----------------|---------------|-------|
| RMSE   | 0.9169          | 0.9169        | Identical — both predict global mean effectively |
| MAE    | 0.7219          | 0.7219        | Same |
| Prec@10| ~1.00           | ~0.14         | NP wins: sorts by quality directly |
| Rec@10 | ~0.027          | ~0.004        | NP surfaces more relevant items |
| NDCG@10| ~1.00           | ~0.15         | NP places best items at top |
| Coverage | 0.0455        | 0.0706        | CF reaches more of the catalogue |
| Diversity | 0.74         | 0.54          | NP more diverse (global vs niche) |

*Exact values are printed in the final notebook cell.*

### Discussion

The results reveal an important characteristic of this dataset: **vote quality is the
dominant signal**. The NP model, which sorts purely by Bayesian-adjusted vote average,
achieves near-perfect Precision@10 and NDCG@10 in the within-training simulation because
the "relevant" items are defined as high-rated movies — the very thing NP optimises for.

The CF model achieves lower ranking metrics because content similarity (shared genres,
keywords, cast) does not perfectly predict quality. A film can be stylistically similar
to *The Dark Knight* but still have a mediocre rating. This confirms that for pure
quality-based recommendation, NP is the correct model for this dataset.

However, CF wins on **Coverage** (0.07 vs 0.05), reaching deeper into the long tail
by surfacing niche films that share content fingerprints with seed movies. In a production
context where long-tail discovery matters, this is a meaningful advantage.

The **RMSE/MAE tie** (both ≈ 0.92) confirms that neither model learns anything richer
than the global mean for rating prediction — a known limitation of catalogue-level datasets
without user-level variation.

---

## 6. Business Reflection

### Deployment Context

Consider a mid-tier streaming service (500 k subscribers, 40 000 title catalogue) trying
to reduce subscriber churn and increase average session length. Both models would play
distinct but complementary roles.

**Non-Personalized as a production component.** The Bayesian Weighted Rating is the right
model for: (a) the homepage of brand-new subscribers (zero cold-start risk); (b) curated
editorial shelves ("Critics' Picks", "Top-rated Sci-Fi"); (c) fallback logic when
personalized recommendation latency exceeds SLA. Its instant serving and near-zero cost
make it indispensable even in a fully personalized stack.

**Item-Based CF: path to production.** The current TF-IDF implementation needs several
enhancements before production use:

1. **Real user signals.** The immediate next step is to collect explicit ratings or
   implicit signals (watch time, completion rate, re-watches) and build genuine item–item
   CF from co-rating patterns. This transforms the content-based approach into a true
   collaborative model with stronger personalisation.

2. **Scalability.** The dense cosine similarity matrix is O(n²) in memory. For a 40 000-item
   catalogue, approximate nearest-neighbour indices (FAISS, HNSW) reduce query time from
   O(n) to O(log n) while preserving recommendation quality.

3. **Freshness.** New movie releases must join the similarity matrix within hours of going
   live. An incremental TF-IDF update pipeline (recompute only the new item's row) minimises
   latency without a full refit.

4. **A/B testing.** Offline RMSE and NDCG improvements do not guarantee better engagement.
   A randomised experiment tracking watch-time, session length and 30-day retention would
   validate real-world impact before full rollout.

**Ethical considerations.** Item-based CF risks creating "genre bubbles" — users seeded
on action films only see action recommendations. Explicit diversity constraints, periodic
recommendation diversification ("Surprise me" mode), and minimum impressions guarantees
for catalogue titles should be built into the product layer. Both models also risk
reinforcing popularity bias, further concentrating attention on already-popular content.

### Conclusion

The NP Bayesian model provides a reliable, explainable and immediately deployable baseline
that excels at surfacing quality content without any user history. Item-Based CF offers
complementary strengths — particularly catalogue coverage and content-driven personalisation
— and provides a natural foundation for evolving into a full collaborative filtering system
once real user ratings are collected. The combination of both in a production stack
represents the minimum viable recommender architecture for a streaming service.

---

## 7. AI Usage Disclosure

Claude (Anthropic) was used during this assignment for:
- Reviewing the mathematical formulation of the Bayesian Weighted Rating formula.
- Suggesting the token-repetition weighting scheme for the TF-IDF feature soup.
- Explaining the within-training simulation approach for ranking evaluation on catalogue-level datasets.
- Proofreading the Business Reflection section for clarity and completeness.

All code was written and tested independently. All analysis, interpretations, and conclusions
are my own. AI was not used to generate raw outputs or to substitute for understanding of the
underlying algorithms.
