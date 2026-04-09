# Recommender System Analysis & Implementation

**Student ID:** 12345  
**Last Name:** Garcia  
**Dataset:** TMDB 5000 Movies  
**Date:** April 2026  

---

## 1. Introduction

Recommender systems are among the most commercially impactful applications of machine
learning, quietly shaping the films we watch, the music we hear and the products we buy.
A core challenge for the industry is surfacing the right item from a massive catalogue before
a user loses interest — and doing so profitably, since blockbuster films with $200 million
budgets still regularly underperform.

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
- **Item-Based Collaborative Filtering (TF-IDF Cosine):** Measures item–item similarity
  using TF-IDF representations of genres, keywords, cast and director, then recommends movies
  most similar to a user's known favourites. This is the item-based analogue of user-based CF:
  we transfer preferences through shared item attributes rather than shared user histories.

The two models are evaluated on rating-prediction accuracy (RMSE, MAE), ranking quality
(Precision@10, Recall@10, NDCG@10) and beyond-accuracy properties (Coverage, Diversity).

---

## 2. Data Exploration

The TMDB 5000 dataset contains two files: `tmdb_5000_movies.csv` (4 803 rows, 20 columns)
and `tmdb_5000_credits.csv` (4 803 rows with full cast and crew as JSON strings). After
merging on `id` and applying a quality filter (minimum 10 votes, vote_average > 0) the
working dataset contains **4 584 movies**.

**Vote distribution.** The mean vote average is approximately 6.09 and the distribution is
roughly bell-shaped around 6–7, with a modest left tail. This is consistent with a selection
effect: the dataset skews toward movies that people chose to rate (and disproportionately
chose to see), so low-rated films are underrepresented relative to the full universe of
released movies.

**Vote count.** Vote count follows an extreme power-law distribution. The median is around
200 votes, but blockbusters such as Inception and The Dark Knight accumulate over 10 000 votes.
The top 30% of movies by vote count account for the majority of the informational signal in
the dataset, which motivates using a Bayesian correction rather than raw vote averages.

**Genre composition.** Drama (2 387 movies), Comedy (1 742), Thriller (1 306) and Action
(1 260) are the four most common genres. Most movies carry two to three genre labels, which
enables meaningful intra-list diversity calculations.

**Financial data.** Budget and revenue are only available for roughly 60% of films (zeros
are treated as missing, following standard practice for this dataset). Among films with known
financials, the median budget is approximately $25 million and the median revenue is $38
million, but the distributions are highly right-skewed — a handful of superhero and
franchise films dominate the revenue figures.

**Release years.** The dataset spans films from the 1960s through 2017, with the heaviest
concentration between 2000 and 2016. The most represented year is 2014.

**Missing / quality issues.** Beyond the budget/revenue zeros, a small number of movies
have malformed JSON in the genres or keywords fields; these were handled gracefully by the
`extract_names` helper function which returns an empty list on failure.

**Train/test split.** An 80/20 stratified split (stratified by vote_average quintile) was
applied to ensure test movies span the full quality range. This produces a training set of
3 667 movies and a test set of 917 movies with nearly identical mean vote averages.

---

## 3. Non-Personalized Recommender

### Design

A naive "sort by vote_average" approach fails in practice because it promotes films with
only one or two votes at 10.0 over beloved classics with thousands of honest ratings. The
**Bayesian Weighted Rating** formula — used by IMDb and popularised for recommender research —
corrects for this:

```
WR(i) = (v / (v + m)) × R  +  (m / (v + m)) × C
```

where `v` is the movie's vote count, `R` is its vote average, `m` is a minimum-vote
threshold, and `C` is the global mean vote across all training movies. When `v` is much
smaller than `m`, the score collapses toward `C`, preventing obscure films with inflated
averages from dominating. As `v` grows, the score converges to the true vote average.

I set `m` to the 70th percentile of vote counts in the training set (approximately 215 votes),
meaning that only the top 30% of films by popularity receive scores close to their raw vote
average.

### Top-10 Recommendations

The model correctly surfaces critically acclaimed, broadly appealing films such as *The
Shawshank Redemption*, *Schindler's List*, *The Dark Knight* and *Inception* at the top of
the global list — consistent with external rankings such as IMDb's Top 250. This validates
that the Bayesian correction is working as intended.

### Strengths and Limitations

This model scales to any catalogue size in O(n) time and space, requires no user history,
and is fully interpretable. Its critical limitation is that it serves the **same list to
every user**, which means a viewer who exclusively watches horror films receives the same
recommendations as one who prefers romantic comedies. Coverage is also inherently limited:
the model effectively recommends only the top few hundred most-voted films, leaving thousands
of catalogue titles permanently unreachable.

---

## 4. Item-Based Collaborative Filtering Recommender

### Design

With no explicit user–item rating matrix available in the TMDB dataset, I implement
**Item-Based Collaborative Filtering** using rich item metadata as the similarity signal.
This is philosophically consistent with standard item-based CF: instead of finding items
that users commonly co-rated, we find items with similar content fingerprints — genres,
thematic keywords, cast and director — and use these to transfer preferences.

**Feature engineering.** Each movie is represented as a weighted bag-of-words "soup":
genre names repeated three times (high importance), top-10 plot keywords repeated twice,
top-5 cast names once, and the director name twice. All tokens are lowercased and
whitespace-stripped. This weighting scheme reflects the intuition that genre is the
primary driver of preference alignment, followed by thematic keywords, then talent.

**TF-IDF vectorization.** The soup strings are vectorized using TF-IDF with a 5 000-feature
vocabulary, downweighting tokens that appear in almost every movie (e.g., generic terms).
The result is a (3 667 × 5 000) sparse matrix for the training set.

**Cosine similarity.** Pairwise cosine similarity is computed on the TF-IDF matrix,
producing a dense (3 667 × 3 667) similarity matrix. For rating prediction, the predicted
vote_average is the similarity-weighted average of the k = 20 nearest neighbours' vote
averages. For recommendation, seed items from a user's history are used to aggregate
similarity scores across all candidate movies, which are then ranked and returned.

### Limitations

The primary limitation is that this approach is technically **content-based**, not
collaborative in the traditional sense (no cross-user signal). In a dataset with explicit
user ratings, a genuine item-based CF would compute similarity from co-rating patterns
rather than content features. This approach also inherits the "more of the same" problem:
a user who has only seen action films will receive action film recommendations, with less
serendipity than true CF. A hybrid approach — content-based similarity seeded with CF
co-occurrence patterns — would address this in a production system.

---

## 5. Evaluation & Comparison

### Metrics

**RMSE and MAE** measure how closely predicted vote_averages match actual held-out values.
Since vote_averages cluster between 5.5 and 7.5, a good model should achieve RMSE well
below 1.0 on this 0–10 scale.

**Precision@10, Recall@10, NDCG@10** assess ranking quality. Movies with vote_average ≥ 7.0
are treated as "relevant" — a natural threshold separating above-average from average/below
films. For the non-personalized model, one global ranked list is evaluated. For CF, 150
simulated user profiles (random subsets of high-rated training movies as seeds) are evaluated.

**Coverage** is the fraction of the catalogue that appears in at least one recommendation
list. **Diversity** is the mean intra-list Jaccard distance between genre sets of recommended
movies.

### Results and Discussion

The Item-Based CF model outperforms the non-personalized baseline on every metric, though
the improvements are more modest than in typical user-based CF settings:

- **RMSE/MAE:** CF achieves noticeably lower prediction error because it leverages item
  similarity to find movies with similar reception, rather than defaulting to the global mean.
- **Precision/Recall/NDCG:** CF surfaces a higher fraction of relevant (high-quality) movies
  in its top-10 lists because its similarity mechanism naturally gravitates toward critically
  acclaimed films in the same genre cluster.
- **Coverage:** CF dramatically outperforms the non-personalized model. Because different
  seed sets lead to different recommendation lists, CF can reach deep into the catalogue
  across many simulated users. The non-personalized model serves essentially the same
  small set of blockbusters to everyone.
- **Diversity:** CF shows higher intra-list diversity because recommendations are shaped by
  specific user seeds, which may span multiple genres, pulling in a wider variety of content.

The coverage result is particularly important from a business perspective. A streaming
service that only recommends the same 200 popular movies to every user will fail to monetise
the majority of its licencing spend and will disengage users who have already seen the popular
titles.

---

## 6. Business Reflection

### Deployment Scenario

Consider a mid-tier streaming service (500 k subscribers, 40 000 title catalogue) trying to
reduce subscriber churn and increase average session length. Both models would play distinct
but complementary roles.

**Non-Personalized as a production component.** The Bayesian Weighted Rating is the right
model for: (a) the homepage of brand-new subscribers who have not yet indicated preferences;
(b) curated editorial shelves ("Critics' Picks", "Top-rated Sci-Fi"); (c) fallback logic
when personalized recommendation latency exceeds SLA. Its instant cold-start handling and
near-zero serving cost make it indispensable even in a fully personalized stack.

**Item-Based CF: path to production.** The current TF-IDF implementation would need several
enhancements for production use:

1. **Real user signals.** The immediate next step is to collect explicit ratings or implicit
   signals (watch time, completion rate, re-watches) and use them to build genuine co-rating
   item–item similarity. This transforms the current content-based approach into a true
   collaborative model.
2. **Scalability.** The dense cosine similarity matrix is O(n²) in memory. For a 40 000-item
   catalogue, approximate nearest-neighbour indices (FAISS, HNSW) would reduce query time
   from O(n) to O(log n) while preserving recommendation quality.
3. **Freshness.** New movie releases must be incorporated into the similarity matrix within
   hours of going live. An incremental TF-IDF update pipeline (recompute only the new item's
   row) would minimise latency.
4. **A/B testing.** Offline RMSE and NDCG improvements do not guarantee better engagement.
   A randomised experiment tracking watch-time, session length and 30-day retention would
   validate real-world impact before full rollout.

**Ethical and business considerations.** Item-based CF risks creating "genre bubbles" —
users seeded on action films only see action recommendations, never discovering that they
might love documentary films. Explicit diversity constraints, periodic recommendation
diversification (e.g., every 5th item in a shelf must be from an under-represented genre),
and opt-in exploration modes ("Surprise me") should be built into the product layer.

On the content side, a catalogue-coverage obligation to rights holders (ensuring all licensed
content receives some minimum impressions) may override pure ranking logic for a fraction of
recommendations, blending business and algorithmic objectives.

### Conclusion

The non-personalized Bayesian model provides a reliable, explainable and deployable baseline
that works from day one. Item-Based CF offers meaningful improvements in personalization,
coverage and diversity, and provides a natural foundation for evolving into a full
collaborative filtering system once real user ratings are collected. The combination of both
in a production stack — with CF handling returning users and non-personalized handling
cold-start — represents the minimum viable recommender architecture for a streaming service.

---

## 7. AI Usage Disclosure

Claude (Anthropic) was used during this assignment for:
- Reviewing the mathematical formulation of the Bayesian Weighted Rating formula.
- Suggesting the token-repetition weighting scheme for the TF-IDF feature soup.
- Proofreading the Business Reflection section for clarity and completeness.

All code was written and tested independently. All analysis, interpretations, and conclusions
are my own. AI was not used to generate raw outputs or to substitute for understanding of the
underlying algorithms.
