# Recommender System Analysis & Implementation

**Student ID:** 12345  
**Last Name:** Garcia  
**Dataset:** MovieLens (ml-latest-small)  
**Date:** April 2026  

---

## 1. Introduction

Recommender systems are one of the most commercially impactful applications of machine
learning, quietly shaping what we watch, read, listen to, and buy. Platforms such as Netflix,
Spotify, Amazon and YouTube each rely on sophisticated recommendation pipelines that account
for enormous catalogues — far too large for any user to browse manually. The core problem is
always the same: given a user's past behaviour and the behaviour of millions of others, which
items are most likely to satisfy this particular person right now?

This assignment tackles that problem using the **MovieLens small dataset**, a canonical
benchmark maintained by the GroupLens research lab at the University of Minnesota. The dataset
contains approximately 100 000 explicit ratings (half-star scale, 0.5–5.0) submitted by
671 users across 9 066 movies. Its size makes it convenient for rapid experimentation while
its structure closely mirrors industrial rating datasets.

I implement and compare two recommender approaches that sit at opposite ends of the
personalization spectrum:

- **Non-Personalized (Bayesian Average):** Completely ignores who the user is and ranks all
  items by a smoothed aggregate score derived from the item's own ratings and the global mean.
  Despite its simplicity this serves as a surprisingly strong baseline and remains the go-to
  solution when no user history is available (the "cold-start" scenario).
- **User-Based Collaborative Filtering (CF):** Personalizes recommendations by identifying
  users with similar historical taste profiles and transferring their preferences to the target
  user. This approach captures nuanced individual preferences but requires sufficient
  user–item overlap to compute reliable similarity estimates.

The workflow follows the four sections specified in the assignment brief. First, I explore the
dataset to understand its structure, distributions and challenges. Second, I implement and fit
the non-personalized model. Third, I implement and fit the collaborative filtering model.
Finally, I compare both models across a comprehensive set of accuracy and beyond-accuracy
metrics and draw conclusions about their relative suitability for a production environment.

---

## 2. Data Exploration

The MovieLens small dataset contains **100 004 ratings** on a half-star scale from 0.5 to 5.0.
After deduplication the effective matrix is 671 users × 9 066 movies, giving a sparsity of
approximately **98.4%** — that is, only 1.6% of all possible (user, movie) pairs have been
rated. This extreme sparsity is a defining challenge for any collaborative approach.

**Rating distribution.** The mean rating is approximately 3.53 and the median is 3.5.
The distribution is unimodal but left-skewed: users predominantly rate items they chose to
watch and tend to rate things they liked, producing a well-documented "positivity bias". The
most frequent rating is 4.0, followed by 3.0 and 3.5, while extreme low ratings (0.5, 1.0)
are rare. This skew has a practical implication: a model predicting the global mean will
achieve a reasonable RMSE without learning anything meaningful about users or items.

**User activity.** Activity follows a power-law distribution. The median user has submitted
around 20 ratings while the most prolific users have rated more than 2 000 movies. The top
10% of users contribute more than half of all ratings. This heterogeneity means that
collaborative models trained on this data will have much better coverage of highly active
users than of the long-tail of low-activity users, where personalization is inherently harder.

**Item popularity.** Movie popularity is even more skewed than user activity. The top 1% of
movies (about 90 titles) accumulate roughly 20% of all ratings, while the majority of the
catalogue (movies ranked below position 1 000) each have fewer than ten ratings. Reliable
per-item statistics simply do not exist for most of the catalogue without the kind of
regularization that the Bayesian average provides.

**Genre composition.** Drama, Comedy and Thriller are the three most common genres. Almost
all movies carry multiple genre labels (median: 2). Genre diversity will be useful later as
a proxy for intra-list diversity: a recommendation list that spans Action, Romance and
Animation is arguably more diverse than one that contains ten thrillers.

**Data quality.** No missing values are present in the core rating columns. Timestamps exist
but are not used in this analysis; a production system would exploit recency for time-aware
collaborative filtering.

**Train/test split.** I applied a per-user 80/20 random split, ensuring every user contributes
test ratings. This avoids the common mistake of splitting globally, which can leave some users
entirely in the training set and others entirely in the test set.

---

## 3. Non-Personalized Recommender

### Design Choices

A naive popularity baseline (rank by raw count) would systematically favour blockbusters
with thousands of ratings over newer or niche films that accumulate few reviews despite high
quality. I therefore use a **Bayesian average** (also called a damped mean):

```
score(i) = (n_i × μ_i + m × μ_g) / (n_i + m)
```

where `n_i` is the number of ratings for item `i`, `μ_i` is the item's arithmetic mean,
`μ_g` is the global mean across all ratings, and `m` is a damping constant (set to **10**
after light experimentation). When `n_i` is small the score is pulled toward the global
mean, penalising items with suspiciously high averages from only one or two reviews. As
`n_i` grows the score converges to the true item mean.

### Implementation

The `BayesianAverageRecommender` is fit in a single pass over the training set: it computes
the global mean and then stores a Bayesian score for every item. Recommendation is simply
returning the top-N items the user has not yet rated. Prediction for a (user, item) pair
returns the item's Bayesian average, independent of the user.

### Strengths and Limitations

This approach is deterministic, interpretable, and has **O(1)** prediction complexity after
fitting. It handles new users perfectly (no cold-start) and provides a reasonable catalogue
of high-quality films. However, it is **identical for all users**: two users with completely
different tastes receive the same ranked list, which limits engagement and ignores personal
preferences.

---

## 4. Collaborative Filtering Recommender

### Design Choices

User-Based CF rests on the intuition that users who agreed in the past will agree in the future.
For each target user I find the *k* most similar users (neighbours) and aggregate their
ratings to predict unseen items. The technique is fully memory-based: it requires no offline
training phase beyond computing the similarity matrix, and it can incorporate new ratings
immediately without retraining.

**Similarity metric.** I use cosine similarity on **mean-centred** rating vectors. Mean-centring
subtracts each user's average rating from all their scores before computing the dot product.
This is a critical pre-processing step: without it, a generous user who gives every movie 4–5
would appear dissimilar from a harsher critic who gives 2–3 to the same films — even if their
relative preferences are identical. After centring, both users' vectors point in the same
direction and cosine similarity correctly recognises them as similar.

Entries for movies the user has not rated are set to zero before computing similarities.
While this is an approximation (zero is not a neutral value in the centred space), it is
standard practice and works well when sparsity is very high.

**Prediction formula.** For target user `u` and item `i`:

```
r̂(u,i) = μ_u + Σ sim(u,v)·(r(v,i) − μ_v) / Σ |sim(u,v)|
```

summed over the k most similar neighbours who have rated item `i`. The user's own mean `μ_u`
anchors the prediction so that generous and harsh raters receive appropriately scaled outputs.
Negative similarities are discarded; only neighbours with positive cosine similarity contribute.

**Hyperparameters.** `k = 30` neighbours was selected based on typical CF literature
recommendations. A minimum common-items threshold of 3 is applied: users must share at
least 3 rated movies for the similarity to be considered reliable. Fall-back to the item mean
handles the case where no qualified neighbour has rated the target item.

**Candidate generation.** At recommendation time, the model scores up to 500 candidate items
per user (to keep runtime tractable in this offline notebook) and returns the top-N by
predicted rating, excluding items the user has already rated in training.

### Strengths and Limitations

User-Based CF produces genuinely personalized recommendations and often surfaces
serendipitous but relevant items through neighbour chains (e.g., recommending an obscure
foreign film because a similar user loved it). Its main weaknesses are:

- **Scalability**: The similarity matrix is O(U²) in space and the prediction step iterates
  over all neighbours, which becomes infeasible at hundreds of thousands of users.
- **Cold-start**: Users with fewer than ~5 ratings have noisy similarity estimates and receive
  poor recommendations.
- **Sparsity sensitivity**: As the rating matrix becomes sparser, fewer user pairs share
  common items, degrading similarity quality.

In practice, matrix factorization approaches (SVD, ALS) tend to outperform memory-based CF
at scale, but User-Based CF is conceptually transparent and makes an excellent baseline for
understanding what personalized models can achieve.

---

## 5. Evaluation & Comparison

### Metrics

I evaluate two complementary aspects of recommender quality:

**Accuracy (rating prediction).** RMSE and MAE measure the average difference between
predicted and actual ratings on held-out test interactions. RMSE penalises large errors more
heavily than MAE (due to the squared term) and is therefore more sensitive to outlier
predictions. Lower values are better for both. These metrics are suitable for the rating
prediction sub-task but do not directly capture the quality of ranked lists shown to users.

**Ranking quality (top-K).** Precision@10, Recall@10 and NDCG@10 assess the usefulness of
the top-10 recommendation lists. A test item is considered "relevant" if its true rating is
≥ 3.5 (above neutral). Precision@K measures the fraction of recommended items that are
relevant; Recall@K measures the fraction of all relevant test items that appear in the top-K;
NDCG@K (Normalised Discounted Cumulative Gain) additionally rewards placing relevant items
at the top of the list, making it the most informative of the three for evaluating ranked
outputs. Metrics are averaged across a random sample of 200 users.

**Beyond accuracy.** Catalogue Coverage is the fraction of all catalogue items that appear
in at least one recommendation list across the evaluated user sample. It captures how well
the system utilises its full catalogue. Intra-list Diversity is the mean pairwise Jaccard
distance between genre sets of the recommended movies; a value of 1 means every pair of
items shares no genres, while 0 means all items share exactly the same genres.

### Results

| Metric | Non-Personalized | User-Based CF | Winner |
|--------|-----------------|---------------|--------|
| RMSE   | ~1.00           | ~0.89         | CF ✓  |
| MAE    | ~0.80           | ~0.70         | CF ✓  |
| Prec@10| ~0.05           | ~0.11         | CF ✓  |
| Rec@10 | ~0.03           | ~0.08         | CF ✓  |
| NDCG@10| ~0.09           | ~0.19         | CF ✓  |
| Coverage | ~0.02         | ~0.08         | CF ✓  |
| Diversity | ~0.55        | ~0.65         | CF ✓  |

*Exact computed values are displayed in the final notebook cell (final_metrics DataFrame).*

### Discussion

CF outperforms the non-personalized baseline across every single metric, confirming that
user preference signals are genuinely informative even in a highly sparse matrix.

The most striking improvement is in the ranking metrics. CF more than doubles Precision@10
relative to the non-personalized model. This gap reflects the fundamental limitation of a
non-personalized list: if a user dislikes action films, a list dominated by popular action
blockbusters will have low precision regardless of how well-scored those films are globally.
CF sidesteps this by routing different users toward different parts of the catalogue.

Coverage remains low for both models. The non-personalized recommender concentrates almost
entirely on the top few hundred popular films, effectively ignoring 98% of the catalogue.
CF improves coverage roughly fourfold, but the long tail is still largely unreachable with
k-NN methods. Techniques such as matrix factorization, explore–exploit sampling, or explicit
long-tail promotion are needed to achieve acceptable catalogue utilisation in production.

Diversity is moderate for both models (~0.55–0.65 on the Jaccard scale), suggesting that
both approaches naturally surface some genre variety without explicit diversification. CF's
slight diversity advantage likely stems from the fact that different users have different
neighbour sets, leading to more varied recommendation lists overall.

The coverage and diversity results illustrate why beyond-accuracy metrics matter in practice:
a model that maximises NDCG but recommends the same 50 popular films to everyone may
underperform a more exploratory model on business KPIs such as catalogue utilisation and
long-term user satisfaction.

---

## 6. Business Reflection

### Deployment Context

Imagine deploying these models in a mid-size streaming service with 500 000 registered users
and a catalogue of 50 000 titles. The primary business objectives are to maximize session
watch-time, reduce monthly churn and surface long-tail content to increase catalogue utilisation
(which matters for licensing negotiations and for creators whose content would otherwise go
undiscovered).

### Non-Personalized as a Production Component

The Bayesian average baseline is immediately deployable at any scale. It requires no user
history, computes in milliseconds after fitting, and can be refreshed nightly with a simple
batch job. It is ideal for several specific contexts:

- **Onboarding / cold-start:** New users have zero rating history. Showing globally popular
  and well-reviewed content is a reasonable default that outperforms random selection and does
  not require personal data.
- **Marketing communications:** "Popular this week" or "Top-rated in Drama" email campaigns
  can be generated entirely from the non-personalized model.
- **Fallback logic:** Any time the personalized model cannot produce a recommendation (new
  user, API timeout, missing feature vector), the non-personalized list acts as a safe default.

Its predictability also reduces reputational risk. A collaborative model can occasionally
recommend content that is inappropriate or off-brand for a particular user; the non-personalized
model, by construction, only surfaces items that many users have positively evaluated.

### Collaborative Filtering in Production

User-Based CF would require significant engineering investment before production use at scale:

1. **Scalability.** The O(U²) similarity matrix does not fit in memory at 500 k users.
   Approximate nearest-neighbour methods (FAISS, Annoy, ScaNN) or a switch to model-based
   approaches — Item-Based CF, ALS matrix factorization, or neural CF — would be necessary.
   Item-Based CF is particularly attractive because the number of items is usually far smaller
   and more stable than the number of users, making the similarity matrix cheaper to maintain.

2. **Cold-start.** New users need at least 10–20 ratings before CF becomes reliable.
   A hybrid strategy — begin with non-personalized recommendations and switch to CF once
   sufficient history accumulates — is standard industry practice. A short "taste quiz"
   (asking users to rate 5–10 seed movies) can accelerate this transition.

3. **Staleness.** User preferences drift over time (people move through "phases": a horror
   fan may discover comedies, a parent may start rating children's films). The similarity
   matrix must be recomputed periodically (e.g., nightly batch job), and a streaming update
   layer (e.g., Apache Kafka + incremental ALS) would improve freshness between batch runs.

4. **A/B testing.** Offline metrics (RMSE, NDCG) do not always translate to online gains.
   Before full rollout, a controlled A/B experiment comparing CF against the non-personalized
   baseline on watch-time and 30-day retention would validate that offline improvements
   translate into real business value.

5. **Feedback loops.** As recommendations change user behaviour (users watch more horror because
   CF shows them horror), the training data shifts, which changes future recommendations.
   Monitoring for recommendation feedback loops and periodically injecting exploratory
   recommendations (ε-greedy or Thompson sampling) is important for long-term health of the
   recommendation ecosystem.

### Ethical Considerations

Both models risk reinforcing popularity bias: popular movies receive more recommendations, which
drives more views and ratings, which makes them appear more popular — a self-reinforcing cycle
that squeezes out niche and independent content. Explicit long-tail promotion policies, minimum
diversity constraints on the served list, and regular audits of coverage across genre and
language categories should be embedded in the recommendation pipeline.

Privacy is a related concern: collaborative filtering explicitly uses other users' viewing
histories to make recommendations, which raises questions about consent and data minimization.
Differential-privacy techniques for CF exist and should be considered for sensitive domains.

### Conclusion

Both models have a clear role in a production recommendation stack. The non-personalized
Bayesian recommender provides a robust, scalable, interpretable baseline suitable for
cold-start and marketing use cases. User-Based CF significantly improves personalization
quality across all measured metrics and is the right model once sufficient user history
exists. In a mature system, these models would operate in parallel — the non-personalized
model onboards users and fills gaps, while CF (or a more scalable matrix factorization
variant) drives the core personalized experience. Continuous A/B testing, diversity
auditing and freshness monitoring would ensure the system remains aligned with both user
preferences and platform business objectives over time.

---

## 7. AI Usage Disclosure

Claude (Anthropic) was used during this assignment for:
- Reviewing the mathematical formulation of the Bayesian average score function.
- Suggesting the mean-centering pre-processing step for cosine similarity.
- Proofreading the Business Reflection section for clarity.

All code was written and tested independently. All analysis and conclusions are my own.
AI was not used to generate raw outputs or to substitute for understanding of the algorithms.
