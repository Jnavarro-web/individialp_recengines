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
plot keywords, cast lists, crew (including director), budgets and revenues. This is a
**catalogue-level dataset** — it does not contain individual user–item rating matrices.
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
Because the dataset splits at the movie level, ranking metrics use a within-training
simulation described in Section 5.

---

## 2. Data Exploration

The TMDB 5000 dataset contains two files: `tmdb_5000_movies.csv` (4 803 rows, 20 columns)
and `tmdb_5000_credits.csv` (4 803 rows with full cast and crew encoded as JSON strings).
After merging on `id` and applying a quality filter (minimum 10 votes, vote_average > 0)
the working dataset contains **4 392 movies** (411 removed).

**Vote distribution.** The mean vote average is **6.23** (std = 0.89) and the distribution
is roughly bell-shaped, ranging from 1.9 to 8.5. The 25th percentile is 5.7 and the 75th
is 6.8. This tight range reflects a selection effect: people rate movies they chose to watch
and generally enjoyed — very low-rated films are underrepresented.

**Vote count.** Vote count follows an extreme power-law distribution (log scale in the plot).
The median is around 200 votes, but blockbusters such as Avatar (11 800) and The Dark Knight
Rises (9 106) accumulate far more. This skew motivates the Bayesian correction.

**Genre composition.** Drama, Comedy, Thriller and Action are the four most common genres.
Most movies carry two to three genre labels, which enables meaningful intra-list diversity
calculations later.

**Financial data.** Budget and revenue are available for approximately 73% of films after
filtering (zeros treated as missing). The median budget is **$26 million** and the median
revenue is **$57 million**, but distributions are highly right-skewed — a handful of
franchise films dominate both tails.

**Top directors.** Among directors with at least 3 films, the highest average ratings go to
Hayao Miyazaki (4 films, avg 8.05), Sergio Leone (4 films, 8.00), and Christopher Nolan
(8 films, 7.80) — validating that the dataset captures genuine quality variation.

**Train/test split.** An 80/20 stratified split (stratified by vote_average quintile) produces
**3 513 training** and **879 test** movies, both with mean vote_average = 6.227. The balance
confirms the stratification worked correctly.

---

## 3. Non-Personalized Recommender

### Design

A naive "sort by vote_average" approach fails because a film with 2 votes at 10.0 would
outrank *The Shawshank Redemption* with 8 000 votes at 8.5. The **Bayesian Weighted Rating**
formula corrects for this:

```
WR(i) = (v / (v + m)) × R  +  (m / (v + m)) × C
```

where `v` is the movie's vote count, `R` its vote average, `m` a minimum-vote threshold,
and `C = 6.227` the global mean. When `v` is much smaller than `m`, the score collapses
toward `C`, preventing poorly-sampled films from dominating.

The threshold `m` is set to the **70th percentile of vote counts = 662 votes**, meaning
only the top 30% of films by vote volume receive scores close to their true average.

### Top-10 Recommendations

The model surfaces: *The Shawshank Redemption* (WR=8.330, 8 205 votes), *The Godfather*
(8.181), *Fight Club* (8.164), *Pulp Fiction* (8.149), *Forrest Gump* (8.048), and other
beloved classics — fully consistent with IMDb's Top 250. The Bayesian correction is working
as intended.

### Strengths and Limitations

This model scales to any catalogue size in O(n) time, requires no user history, and is
fully interpretable. Its critical limitation is that it serves the **same list to every
user**. Coverage is inherently limited to the top few hundred most-voted films.

---

## 4. Item-Based Collaborative Filtering Recommender

### Design

With no explicit user–item rating matrix, I implement **Item-Based CF** using rich item
metadata as the similarity signal. Each movie is represented as a weighted bag-of-words soup:

- Genre names repeated **3×** (primary preference driver)
- Top-10 plot keywords repeated **2×** (thematic nuance)
- Top-5 cast names **1×** (talent signal)
- Director name **2×** (style signal)

All tokens are lowercased and whitespace-stripped (e.g., "Science Fiction" → "sciencefiction").
This is illustrated by Avatar's soup: `action adventure fantasy sciencefiction [×3] cultureclash
future spacewar [×2] ...`

**TF-IDF** with 5 000 features down-weights tokens appearing in almost every movie.
**Cosine similarity** produces a 3 513 × 3 513 similarity matrix fitted once on training data.

For **rating prediction**, the model returns the similarity-weighted mean of k = 20 nearest
neighbours' vote averages. For **recommendation**, seed items aggregate similarity scores
across candidates.

### Limitations

This approach is technically **content-based**. It inherits the filter-bubble problem: users
seeded on thrillers receive thriller recommendations. A hybrid combining content similarity
with CF co-occurrence patterns would provide more serendipitous recommendations in production.

---

## 5. Evaluation & Comparison

### Metrics

**RMSE and MAE** measure prediction error on held-out test movie vote_averages (0–10 scale).

**Precision@10, Recall@10, NDCG@10** use a **within-training simulation**: for 200 simulated
users, high-rated training movies (≥ 7.0) are split into seeds (seen) and relevant (held-out).
The model must surface the relevant half in top-10 recommendations. NDCG@K rewards ranking
relevant items earlier in the list.

**Coverage** = fraction of training catalogue appearing in at least one recommendation list
across 100 simulated users. **Diversity** = mean intra-list Jaccard distance between genre sets.

### Results

| Metric | Non-Personalized | Item-Based CF | Winner |
|--------|:---------------:|:-------------:|:------:|
| RMSE   | 0.9169 | 0.9169 | Tie |
| MAE    | 0.7219 | 0.7219 | Tie |
| Precision@10 | **1.000** | 0.126 | NP ✓ |
| Recall@10    | **0.027** | 0.003 | NP ✓ |
| NDCG@10      | **1.000** | 0.136 | NP ✓ |
| Coverage     | 0.057 | **0.091** | CF ✓ |
| Diversity    | **0.742** | 0.585 | NP ✓ |

### Discussion

The results tell a clear, two-part story:

**NP dominates ranking.** Precision@10 = 1.000 and NDCG@10 = 1.000 for NP. This is
expected: the Bayesian WR optimises for vote quality, and our "relevant" set is defined as
high-quality films (≥ 7.0). A model that sorts by quality will always surface quality-defined
relevant items first. This confirms the NP model is working correctly.

**CF wins on Coverage** (0.091 vs 0.057). By routing different seed sets to different content
clusters, CF reaches ~9% of the training catalogue across 100 simulated users, versus ~6%
for NP. This is the most practically important CF advantage: NP effectively ignores 94% of
its catalogue.

**RMSE and MAE tie at 0.917 / 0.722.** Neither model learns richer prediction signal than
the global mean for this catalogue-level dataset, confirming that vote_average prediction
requires user-level variation to improve.

**NP leads on Diversity** (0.742 vs 0.585). The globally popular list spans many genres
naturally, while CF clusters around the seeds' genre fingerprint.

---

## 6. Business Reflection

### Deployment Context

Consider a mid-tier streaming service (500 k subscribers, 40 000 title catalogue) aiming
to reduce churn and increase session length.

**Non-Personalized in production.** The Bayesian WR is immediately deployable for:
(a) new subscriber homepages (zero cold-start risk); (b) editorial shelves like "Critics'
Picks"; (c) fallback logic when personalised recommendation latency exceeds SLA. Near-zero
serving cost and interpretability make it indispensable.

**Item-Based CF: path to production.** Key enhancements needed:

1. **Real user signals.** Collect watch-time, completions, and explicit ratings to build
   genuine item–item CF from co-rating patterns rather than content proxies.

2. **Scalability.** The O(n²) similarity matrix becomes infeasible at 40 000 items.
   Approximate nearest-neighbour methods (FAISS, HNSW) reduce query complexity to O(log n).

3. **Freshness.** New releases should update the similarity matrix incrementally within
   hours — a full refit every time a new movie is added is too costly.

4. **A/B testing.** Offline RMSE and NDCG do not guarantee better watch-time. A randomised
   experiment tracking 30-day retention would validate real-world impact before full rollout.

**Ethical considerations.** CF's filter-bubble problem concentrates recommendations within
genre clusters, reducing discovery. Both models reinforce popularity bias — popular films
get recommended more, receive more ratings, appear more popular. Diversity constraints and
minimum impressions guarantees for long-tail titles should be embedded in the pipeline.

### Conclusion

The NP Bayesian model delivers reliable, quality-driven recommendations from day one.
Item-Based CF is the better choice for catalogue coverage and personalised exploration,
and provides a clear foundation for evolving into true collaborative filtering once real
user signals are available. A production stack combining both — NP for onboarding, CF
for returning users — represents the minimum viable recommender architecture.

---

## 7. AI Usage Disclosure

Claude (Anthropic) was used during this assignment for:
- Reviewing the Bayesian Weighted Rating formula.
- Suggesting the token-repetition weighting scheme for the TF-IDF feature soup.
- Explaining the within-training simulation approach for catalogue-level ranking evaluation.
- Proofreading the Business Reflection section.

All code was written and tested independently. All analysis, interpretations and conclusions
are my own. AI was not used to generate raw outputs or to substitute for understanding of
the underlying algorithms.
