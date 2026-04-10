# Recommender System Analysis & Implementation

**Student ID:** 16255  
**Last Name:** Navarro  
**Dataset:** TMDB 5000 Movies + MovieLens ml-latest-small  
**Date:** April 2026  

---

## 1. Introduction

Recommender systems are among the most commercially impactful applications of machine
learning, shaping the films we watch, the music we hear and the products we buy. The core
challenge is surfacing the right item from a massive catalogue before a user loses interest.

This assignment builds and evaluates three recommender approaches across two datasets.
The **TMDB 5000 Movies** dataset (~4 800 films with genres, keywords, cast, crew, budget and
vote scores) provides the content fingerprint for every movie. The **MovieLens ml-latest-small**
dataset (100 836 ratings from 610 real users) provides the individual behaviour data that TMDB
alone cannot supply. The two are joined through a shared TMDB ID bridge in `links.csv`.

The three models are:

- **Non-Personalized (Bayesian Weighted Rating):** Ranks all movies by a regularised aggregate
  score. Identical for all users. Ideal for cold-start and editorial lists.
- **Item-Based Collaborative Filtering (TF-IDF Cosine Similarity):** Measures item–item
  similarity from genres, keywords, cast and director. Personalised by seeding from a user's
  liked films.
- **Hybrid (CF + Genre Hard-Filter):** CF recommendations filtered by each user's real genre
  history from MovieLens. Never recommends genres the user has actively shown they dislike.

---

## 2. Data Exploration

### TMDB 5000

After merging the movies and credits files on `id` and applying a quality filter (≥10 votes,
vote_average > 0), the working dataset contains **4 392 movies** (411 removed).

The mean vote_average is **6.227** (std = 0.893), ranging from 1.9 to 8.5. The distribution
is roughly bell-shaped with a slight left skew — most films cluster between the 25th percentile
(5.7) and 75th (6.8). No film reaches 10.0, which already hints at the convergence effect the
Bayesian formula exploits.

Vote count follows an extreme power-law distribution: Avatar has 11 800 votes, The Dark Knight
Rises has 9 106, while most films have under 300. This is the core problem motivating the
Bayesian correction — a 2-vote film at 10.0 should not outrank an 8 000-vote film at 8.5.

Genre composition: Drama leads, followed by Comedy, Thriller and Action. Most films carry 2–3
genre labels simultaneously. The top directors by average rating (minimum 3 films) are
Hayao Miyazaki (4 films, avg 8.05), Sergio Leone (4 films, 8.00) and Christopher Nolan
(8 films, 7.80) — a useful sanity check confirming the vote signal captures genuine quality.

Budget/revenue data is available for ~73% of films. Median budget: $26M, median revenue: $57M,
with a highly right-skewed distribution — a handful of franchise films dominate both tails.

**Train/test split:** Stratified 80/20 by vote_average quintile → 3 513 training, 879 test,
both with mean 6.227. The identical means confirm the stratification worked correctly.

### MovieLens

100 836 ratings from 610 real users on a 0.5–5.0 half-star scale. The most common rating is
4.0, followed by 3.0 — a realistic distribution reflecting that people mostly rate films they
chose to watch. After merging through `links.csv`, we retain **70 149 ratings** across
**610 users and 3 505 movies** — the subset of MovieLens films present in the TMDB catalogue.

---

## 3. Non-Personalized Recommender

### Design

The Bayesian Weighted Rating formula:

```
WR(i) = (v / (v + m)) × R  +  (m / (v + m)) × C
```

where `v` = vote count, `R` = vote average, `m` = minimum-vote threshold (662, the 70th
percentile), `C` = global mean (6.227). Films with fewer than 662 votes are progressively
pulled toward 6.227 — they must earn their ranking through volume, not a handful of enthusiastic
early ratings.

### Top-10 Results

1. The Shawshank Redemption — WR=8.330, 8 205 votes (Drama, Crime)
2. The Godfather — WR=8.181, 5 893 votes (Drama, Crime)
3. Fight Club — WR=8.164, 9 413 votes (Drama)
4. Pulp Fiction — WR=8.149, 8 428 votes (Thriller, Crime)
5. Forrest Gump — WR=8.048, 7 927 votes (Comedy, Drama, Romance)
6. The Lord of the Rings: The Return of the King — WR=7.958
7. The Godfather: Part II — WR=7.957, 3 338 votes
8. Star Wars — WR=7.930, 6 624 votes
9. The Green Mile — WR=7.923, 4 048 votes
10. Se7en — WR=7.907, 5 765 votes

Fully consistent with IMDb's Top 250. The Bayesian correction correctly places Part II
(only 3 338 votes) below films with stronger evidence, even if its raw average is comparable.

### Evaluation Results

| Metric | Score |
|--------|-------|
| RMSE | 0.9169 |
| MAE | 0.7219 |
| Precision@10 | 1.000 |
| Recall@10 | 0.027 |
| NDCG@10 | 1.000 |
| Coverage | 0.057 |
| Diversity | 0.742 |

RMSE of 0.9169 means predictions are less than one point off on a 0–10 scale. The perfect
Precision and NDCG scores reflect that the simulation defines "relevant" as vote_average ≥ 7.0 —
exactly what this model optimises for. This is expected and informative, but not the full
picture: it says nothing about whether these recommendations match individual tastes.

---

## 4. Item-Based Collaborative Filtering Recommender

### Design

Each movie is represented as a weighted bag-of-words feature soup:
- Genre names repeated **3×** — primary preference driver
- Top-10 plot keywords **2×** — thematic nuance
- Top-5 cast names **1×** — talent signal
- Director name **2×** — style signal

All tokens lowercased and whitespace-stripped. TF-IDF with 5 000 features down-weights common
tokens (e.g. `drama`) and up-weights distinctive ones (specific keywords, uncommon genres).
Cosine similarity produces a **3 513 × 3 513** similarity matrix over training films.

For rating prediction: similarity-weighted mean of k=20 nearest neighbours' vote_averages.
For recommendation: seed items from the user's liked history aggregate similarity scores across
all candidate films.

### Sample Recommendations (seeded from The Shawshank Redemption + The Godfather)

- The Godfather: Part II ★8.3 (Drama, Crime)
- Escape from Alcatraz ★7.2 (Crime, Drama)
- Ajami ★6.8 (Crime, Drama)
- The Rainmaker ★6.7 (Drama, Crime, Thriller)
- The Cotton Club ★6.6 (Music, Drama, Crime)

Content similarity is working: every recommendation shares at least one seed genre. However,
quality drops toward the bottom of the list — *The Rise of the Krays* at ★4.5 matches genre
tokens without matching quality. This is a known limitation of content-based similarity,
addressed by the Hybrid model.

### Evaluation Results

| Metric | Score |
|--------|-------|
| RMSE | 0.9169 |
| MAE | 0.7219 |
| Precision@10 | 0.126 |
| Recall@10 | 0.003 |
| NDCG@10 | 0.136 |
| Coverage | 0.091 |
| Diversity | 0.585 |

RMSE and MAE are identical to NP — neither model learns beyond the global mean from aggregate
data alone. CF trades ranking quality (lower P@10, NDCG) for catalogue reach: it covers 9.1%
of the training catalogue across 100 simulated users vs NP's 5.7%, a 60% wider reach.

---

## 5. Evaluation & Comparison

### Metric Summary

| Metric | Non-Personalized | Item-Based CF | Winner |
|--------|:---:|:---:|:---:|
| RMSE | 0.9169 | 0.9169 | Tie |
| MAE | 0.7219 | 0.7219 | Tie |
| Precision@10 | **1.000** | 0.126 | NP |
| Recall@10 | **0.027** | 0.003 | NP |
| NDCG@10 | **1.000** | 0.136 | NP |
| Coverage | 0.057 | **0.091** | CF |
| Diversity | **0.742** | 0.585 | NP |

### Discussion

NP dominates ranking metrics because the evaluation defines relevance as high vote_average —
exactly what NP optimises for. CF wins only on coverage, reaching 60% more of the catalogue.
RMSE/MAE tie because predicting aggregate vote scores reduces to the global mean regardless
of the model. The fundamental limitation of both models is that they have no real user
histories to draw from — Section 6 addresses this directly.

---

## 6. Extended Analysis — Real User Behaviour with MovieLens

### Why This Section Exists

The TMDB-only evaluation cannot answer whether recommendations match individual taste.
By merging MovieLens individual ratings with TMDB metadata, we can measure something new:
the **genre mismatch rate** — how often a model recommends a film in a genre the user has
historically disliked.

### User Genre Profiles

Built from each user's real rating history. Two contrasting examples:

**User 1** (147 ratings): loves Music (5.00), Western (4.67), Animation (4.57), Crime (4.43).
Only dislikes Horror (3.38). A broad-taste user — any model does reasonably well for them.

**User 5** (32 ratings): loves Western (5.00) and History (4.50) but rates Drama (3.47),
Comedy (3.44), Action (3.14), Romance (3.00), Horror (3.00) and Science Fiction (3.00) all
near or below the dislike threshold of 3.0. The global NP top-10 (dominated by Drama and Crime)
is technically high quality but personally irrelevant for User 5.

### Popularity Bias Measurement

| Model | Mismatch Rate |
|-------|:---:|
| Non-Personalized | 15.4% |
| Item-Based CF | 9.4% |
| Hybrid (CF + filter) | **0.0%** |

Out of every 10 NP recommendations, roughly 1.5 clash with the user's genre preferences.
CF cuts that to 0.94 per 10 by seeding from the user's liked films. The Hybrid eliminates
mismatches entirely by applying a hard filter: if a user has historically rated a genre below
3.0, no film from that genre appears in their recommendations — regardless of popularity.

### Hybrid Recommender Design

1. **CF step:** find films similar to the user's liked history
2. **Hard filter:** remove any film whose primary genre the user has rated below 3.0
3. **NP fallback:** fill remaining slots from the global quality list (same filter applied)
4. Unknown genres (never rated) are allowed through — the filter only blocks confirmed dislikes

### Sample Results

**User 1** (loves Crime, Drama, Music — dislikes Horror, Romance, Mystery):
*Zulu, Shinjuku Incident, Exiled, The Betrayed, Ajami* — all Crime/Drama, no Horror or Romance.

**User 5** (loves Western, History, Animation — dislikes Romance, Horror, Sci-Fi):
*Toy Story 3, Quest for Camelot, Babe: Pig in the City* (Animation) alongside Crime/Drama
fallbacks. No Sci-Fi, Romance or Action-heavy films despite many ranking highly globally.

### Three-Model Comparison

| Model | Mismatch Rate | Coverage |
|-------|:---:|:---:|
| Non-Personalized | 15.4% | 0.0455 |
| Item-Based CF | 9.4% | 0.0316 |
| Hybrid (CF + filter) | **0.0%** | 0.0357 |

The Hybrid achieves 0% mismatch while maintaining 3.57% coverage — better than raw CF's
3.16% because the NP fallback fills gaps when the CF pool runs short after filtering.
NP's apparent 4.55% coverage is misleading: it's the same 200 films for everyone, not
genuine catalogue exploration.

### Conclusion

A recommender should be evaluated on both axes simultaneously. Perfect mismatch avoidance
with minimal coverage is useless. Great coverage with high mismatch is frustrating. The
Hybrid achieves the right balance: it never recommends genres users actively dislike, uses
content similarity to personalise, and falls back to global quality when needed.

---

## 7. Business Reflection

### Deployment Context

A mid-tier streaming service (500 k subscribers, 40 000 title catalogue) aiming to reduce
churn and increase session length.

**Non-Personalized** is immediately deployable for new subscriber homepages (zero cold-start
risk), editorial "Critics' Picks" shelves, and as a fallback when personalised recommendation
latency exceeds SLA. Near-zero serving cost makes it indispensable even in a fully personalised
stack.

**Item-Based CF + Hybrid** requires real user signal to unlock its full potential. Minimum
enhancements for production:

1. **User signal collection.** Watch time, completion rate and explicit ratings feed the
   genre profile. Even 10–20 interactions are enough to build a meaningful preference filter.
2. **Scalability.** The O(n²) similarity matrix needs approximate nearest-neighbour indexing
   (FAISS, HNSW) for catalogues above ~10 000 items.
3. **Freshness.** New releases should update the similarity matrix incrementally within hours,
   not require a full nightly refit.
4. **A/B testing.** The 0.0% mismatch result is an offline metric. A randomised experiment
   tracking watch-time and 30-day retention validates that it translates to real engagement.

**Ethical considerations.** The hard genre filter raises a design question: should users be
locked into their historical tastes? A user who only watched Horror last year might want to
explore Comedy. Production systems typically soften the filter (allow genres rated 2.0–3.0 in
a small fraction of slots) or add an explicit "Surprise me" mode that bypasses the filter
entirely. The goal is to serve taste without trapping users inside it.

---

## 7. AI Usage Disclosure

Claude (Anthropic) was used during this assignment for:
- Reviewing the Bayesian Weighted Rating formula.
- Suggesting the token-repetition weighting scheme for the TF-IDF feature soup.
- Explaining the within-training simulation approach for ranking evaluation on catalogue-level datasets.
- Explaining the genre mismatch rate concept and hard-filter hybrid design.
- Proofreading and structuring the report.

All code was written and tested independently. All analysis, interpretations and conclusions
are my own. AI was not used to generate raw outputs or to substitute for understanding of the
underlying algorithms.
