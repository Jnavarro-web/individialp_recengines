# Recommender System Analysis & Implementation

**Student ID:** 16255  
**Last Name:** Navarro  
**Dataset:** TMDB 5000 Movies + MovieLens ml-latest-small  
**Date:** April 2026

---

## 1. Introduction

Recommender systems are one of the most commercially impactful uses of machine learning — shaping what we watch, buy and listen to without most users noticing they're there. The core challenge is always the same: given a massive catalogue and a limited amount of time a user is willing to browse, surface the right thing for the right person before they lose interest and leave.

This assignment builds and evaluates three recommender approaches across two linked datasets. **TMDB 5000** provides content metadata for approximately 4,800 films: genres, keywords, cast, crew, budgets and aggregate vote scores from The Movie Database API. **MovieLens ml-latest-small** provides the missing layer: 100,836 individual ratings from 610 real users. On their own, each dataset answers different but incomplete questions. Together, they allow evaluation that goes beyond "is this film generally good?" and into "is this film good for this specific person?"

The three models are:

- **Non-Personalized (Bayesian Weighted Rating):** Ranks all films by a regularised quality score derived from vote averages and vote counts. Same list for every user. No personal data required.
- **Item-Based Collaborative Filtering (TF-IDF Cosine Similarity):** Finds films with similar content fingerprints to what a user already liked, using genres, keywords, cast and director as features.
- **Hybrid (CF + Genre Hard-Filter):** CF recommendations filtered by each user's real genre preferences from MovieLens. Never recommends genres the user has actively shown they dislike, regardless of how globally popular those films are.

The brief asks for two approaches; the Hybrid is an extension motivated by a gap identified during the TMDB-only analysis — without individual user data it is impossible to know whether recommendations are clashing with personal taste or simply pushing whatever is globally trending. Section 6 addresses that gap directly using the MovieLens data.

---

## 2. Data Exploration

### TMDB 5000

After merging the two TMDB files on `id` and applying a quality filter (minimum 10 votes, vote_average > 0), the working dataset contains **4,392 films** — 411 removed for insufficient data. Vote averages range from 1.9 to 8.5, with a mean of **6.227** (std = 0.893). Most films fall between 5.7 and 6.8 — the interquartile range. The distribution has a slight left skew: people generally rate films they chose to watch and expected to enjoy, so very low scores are rare. Critically, no film reaches a perfect 10. The more votes a film accumulates, the more scores regress toward the population average, which is the phenomenon the Bayesian formula explicitly accounts for.

Vote count is extremely right-skewed. Avatar has 11,800 votes, The Dark Knight Rises has 9,106, but most films have under 300. This power-law inequality — common to any platform where users self-select what they engage with — means raw averages are unreliable quality signals when sample sizes vary so dramatically. The 744 films with budget recorded as zero are treated as missing data, not genuinely zero-budget productions.

Genre composition: Drama leads with the most titles, followed by Comedy, Thriller and Action. Most films carry two or three genre labels simultaneously. Top directors by average rating (minimum 3 films): Hayao Miyazaki (4 films, avg 8.05), Sergio Leone (4 films, 8.00) and Christopher Nolan (8 films, 7.80) — a useful sanity check confirming the vote signal captures genuine quality rather than just popularity.

Financial data is available for ~73% of films (zeros treated as missing). Median budget: $26M, median revenue: $57M — but both distributions are highly right-skewed, with a handful of franchise blockbusters dominating the top of each tail. Many high-budget productions fail to recoup costs, which is visible in the budget-revenue scatter.

**Train/test split:** Stratified 80/20 by vote_average quintile → **3,513 training, 879 test**, both with mean 6.227. The identical means confirm the stratification worked correctly.

### MovieLens

100,836 individual ratings from 610 users on a 0.5–5.0 half-star scale. The most common rating is 4.0, consistent with a self-selecting sample of people rating films they chose to watch. After merging through `links.csv`, we retain **70,149 ratings across 610 users and 3,505 movies** — the MovieLens films present in the TMDB catalogue. The ~30,000 ratings lost in the merge come from older or more niche films not in TMDB 5000.

---

## 3. Non-Personalized Recommender

### Design

The Bayesian Weighted Rating formula:

```
WR(i) = (v / (v + m)) × R  +  (m / (v + m)) × C
```

where `v` = vote count, `R` = vote average, `m` = **662** (70th percentile of training vote counts), `C` = **6.227** (global mean). Films with fewer than 662 votes are progressively pulled toward 6.227 in proportion to how little evidence exists. This is the same approach used by IMDb's official Top 250 ranking and has become standard wherever user-generated scores need to be made comparable across items with very different sample sizes.

### Top-10 Results

1. The Shawshank Redemption — WR=8.330, 8,205 votes (Drama, Crime)
2. The Godfather — WR=8.181, 5,893 votes (Drama, Crime)
3. Fight Club — WR=8.164, 9,413 votes (Drama)
4. Pulp Fiction — WR=8.149, 8,428 votes (Thriller, Crime)
5. Forrest Gump — WR=8.048, 7,927 votes (Comedy, Drama, Romance)
6. The Lord of the Rings: The Return of the King — WR=7.958
7. The Godfather: Part II — WR=7.957, 3,338 votes (Drama, Crime)
8. Star Wars — WR=7.930, 6,624 votes (Adventure, Action, Sci-Fi)
9. The Green Mile — WR=7.923, 4,048 votes (Fantasy, Drama, Crime)
10. Se7en — WR=7.907, 5,765 votes (Crime, Mystery, Thriller)

Fully consistent with IMDb's Top 250. The Bayesian correction is visible: The Godfather Part II (3,338 votes) sits at #7 rather than higher, correctly reflecting that its evidence base is weaker despite a comparable raw average. Every user receives this identical list — that is both the model's greatest strength (guaranteed quality) and its fundamental limitation (zero personalisation).

### Evaluation Results

| Metric | Score |
|--------|:---:|
| RMSE | 0.9169 |
| MAE | 0.7219 |
| Precision@10 | 1.000 |
| Recall@10 | 0.027 |
| NDCG@10 | 1.000 |
| Coverage | 0.057 |
| Diversity | 0.742 |

RMSE of 0.917 means predictions are less than one point off on a 0–10 scale — roughly the difference between calling a film "good" and "very good." The perfect Precision and NDCG scores need careful interpretation: the simulation defines "relevant" as vote_average ≥ 7.0, which is exactly what this model sorts by. The result confirms internal consistency, not that recommendations match individual tastes. Section 6 shows what happens when we test against real preferences: the NP model mismatches 15.4% of the time.

---

## 4. Item-Based Collaborative Filtering

### Design

Each film is represented as a weighted bag-of-words feature soup:
- Genre names **×3** — primary preference driver
- Top-10 plot keywords **×2** — thematic nuance
- Top-5 cast names **×1** — talent signal
- Director name **×2** — style signal

All tokens lowercased and whitespace-stripped (e.g. "Science Fiction" → "sciencefiction"). TF-IDF with 5,000 features down-weights tokens appearing in almost every film (like `drama`) and amplifies distinctive ones (specific director names, unusual keywords). Cosine similarity produces a **3,513 × 3,513** pairwise similarity matrix — one entry per film pair in the training set.

For rating prediction: similarity-weighted mean of k=20 nearest neighbours' vote averages. For recommendations: aggregate similarity scores across the user's seed films and return the most collectively similar unseen movies.

### Sample Recommendations (seeded from *The Shawshank Redemption* + *The Godfather*)

- The Godfather: Part II ★8.3 (Drama, Crime)
- Escape from Alcatraz ★7.2 (Crime, Drama)
- Ajami ★6.8 (Crime, Drama)
- The Rainmaker ★6.7 (Drama, Crime, Thriller)
- The Cotton Club ★6.6 (Music, Drama, Crime)

Genre alignment is working correctly throughout. One limitation is visible toward the bottom of the list: *The Rise of the Krays* (★4.5) makes the cut purely on genre token match, without the quality to back it up. Content similarity optimises for similarity to seeds, not for quality of outcome — two films can share a director and genre without one being anywhere near as good as the other. The Hybrid model addresses this.

### Evaluation Results

| Metric | Score |
|--------|:---:|
| RMSE | 0.9169 |
| MAE | 0.7219 |
| Precision@10 | 0.126 |
| Recall@10 | 0.003 |
| NDCG@10 | 0.136 |
| Coverage | 0.091 |
| Diversity | 0.585 |

RMSE/MAE identical to NP — aggregate data offers no individual signal to learn from regardless of the model. CF trades ranking quality for catalogue reach: 9.1% coverage across simulated users vs NP's 5.7%, a 60% wider spread. In a real 40,000-film streaming service, that gap translates to thousands of additional titles reaching actual viewers — which matters for user discovery and for justifying the licensing costs of a broad catalogue. Netflix has publicly cited catalogue utilisation as a core operational metric for exactly this reason.

---

## 5. Evaluation & Comparison

| Metric | Non-Personalized | Item-Based CF | Winner |
|--------|:---:|:---:|:---:|
| RMSE | 0.9169 | 0.9169 | Tie |
| MAE | 0.7219 | 0.7219 | Tie |
| Precision@10 | **1.000** | 0.126 | NP |
| Recall@10 | **0.027** | 0.003 | NP |
| NDCG@10 | **1.000** | 0.136 | NP |
| Coverage | 0.057 | **0.091** | CF |
| Diversity | **0.742** | 0.585 | NP |

NP wins on 5 out of 7 metrics, but interpreting this as a clear NP victory would be a mistake. The evaluation defines relevance as vote_average ≥ 7.0 — which is exactly what NP sorts by. The result confirms the model is internally consistent, not that it better serves users. CF wins only on coverage: routing different users through different seed films causes it to explore significantly more of the catalogue even without real user data, hinting at the personalisation advantage that becomes fully visible in Section 6.

The tied RMSE and MAE confirm the hard ceiling of TMDB-only analysis: without individual-level variation, both models converge to predicting the global mean. Adding MovieLens breaks through that ceiling by introducing the user dimension that aggregate data cannot supply.

**Recommendation:** Deploy NP as the default for new users, editorial shelves and fallback logic. Transition to CF — and ultimately the Hybrid — once sufficient individual rating history has been collected. The two models are complementary, not competitive.

---

## 6. Extended Analysis — Real User Behaviour with MovieLens

### Motivation

The TMDB-only evaluation cannot answer whether recommendations match individual taste. A Drama-hating user receives *The Shawshank Redemption* at #1 from the NP model because it is objectively high-quality — but that is not a good recommendation for that person. By merging MovieLens individual ratings with TMDB metadata through `links.csv`, we can measure the **genre mismatch rate**: how often a model recommends a film in a genre the user has historically disliked.

### User Genre Profiles

Built from each user's real rating history — simply the average score they have given to each genre across all their rated films.

**User 1** (147 ratings) broadly likes everything. Their lowest genre average is Horror at 3.38, still above the dislike threshold. Any model will do reasonably well for this type of user.

**User 5** (32 ratings) is far more selective. They love Western (5.00) and History (4.50) but rate Drama (3.47), Comedy (3.44), Action (3.14), Romance (3.00), Horror (3.00) and Science Fiction (3.00) all at or below the dislike threshold of 3.0. The NP top-10 — heavy on Drama and Crime — is technically high quality but almost entirely misaligned with User 5's preferences. This is exactly who the Hybrid is designed for.

### Popularity Bias Measurement

| Model | Mismatch Rate | Coverage |
|-------|:---:|:---:|
| Non-Personalized | 15.4% | 0.0455 |
| Item-Based CF | 9.4% | 0.0316 |
| Hybrid (CF + filter) | **0.0%** | 0.0357 |

Out of every 10 NP recommendations, 1–2 actively clash with the receiving user's demonstrated genre preferences. CF nearly halves that by seeding from each user's actual liked films. These numbers are only measurable because of MovieLens — TMDB alone could never produce a mismatch rate because there are no individual users to mismatch against.

### Hybrid Recommender Design

1. **CF step:** find films similar to the user's real liked history (ratings ≥ 3.5)
2. **Hard filter:** remove any film whose primary genre the user has historically rated below 3.0
3. **NP fallback:** fill remaining slots from the global quality list (same filter applied)
4. **Unknown genres allowed:** only confirmed dislikes are blocked — the model doesn't penalise genres a user simply hasn't explored yet

### Sample Results

**User 1** (loves Crime, Drama, Music — dislikes Horror, Romance, Mystery): *Zulu, Shinjuku Incident, Exiled, Ajami* — all Crime/Drama. No Horror or Romance despite both being globally popular.

**User 5** (loves Western, History, Animation — dislikes Romance, Horror, Sci-Fi): *Toy Story 3, Quest for Camelot, Babe: Pig in the City* (Animation) alongside Crime/Drama fallbacks. No Sci-Fi, heavy Action or Romance despite many such films ranking highly globally.

### The Headline Result

The Hybrid achieves **0.0% mismatch** while maintaining 3.57% coverage — better than raw CF's 3.16% because the NP fallback fills gaps when the personalised pool runs short after filtering. NP's 4.55% coverage is misleading: it represents the same films for everyone, not genuine catalogue exploration.

A good recommender must perform well on both axes simultaneously. High coverage with high mismatch means pushing irrelevant films at users. Low mismatch with minimal coverage means the model only knows how to recommend five things. The Hybrid sits at the right balance for this dataset: zero tolerance for confirmed taste mismatches, reasonable catalogue exploration through CF, and a quality backstop from the NP model when the personalised pool runs dry.

---

## 7. Business Reflection

### Deployment Context

Consider a mid-tier streaming service with 500,000 subscribers and a 40,000-title catalogue. The business goals are to reduce monthly churn, increase average session length and improve the utilisation of licensed content that would otherwise sit unwatched.

**Non-Personalized** is immediately deployable. It requires no user history, serves in milliseconds from a pre-computed score table, and is interpretable enough to power editorial shelves like "Critics' Picks" or "Top-rated Drama." It also functions as the fallback when the personalised model is unavailable — Netflix and Spotify both maintain popularity-based fallback layers for exactly this reason. Its cost is zero from a data engineering standpoint, making it indispensable even in a fully personalised stack.

**The Hybrid CF** unlocks its value progressively. Ten to twenty individual ratings are enough to build a meaningful genre filter. The recommended production architecture would be:

1. **Onboarding (0–10 ratings):** Serve NP globally popular content to everyone. Offer an optional taste quiz (rate 5 films) to accelerate profile building — this is the "onboarding quiz" approach used by Netflix and Spotify during new user registration.
2. **Early personalisation (10–50 ratings):** Activate CF with the genre filter. At this stage the profile is still noisy, so keep the dislike threshold conservative (below 2.5 rather than 3.0) to avoid over-filtering.
3. **Mature profile (50+ ratings):** Full Hybrid with the standard threshold. Recompute genre profiles weekly or on every session to capture preference drift.

**Engineering requirements for production scale:** The current dense similarity matrix (3,513 × 3,513) works fine for this dataset but becomes infeasible at 40,000 items — a 1.6 billion entry matrix. Production platforms use approximate nearest-neighbour indexing (Facebook's FAISS or Google's ScaNN) to reduce query time from O(n) to O(log n) with minimal accuracy loss. The TF-IDF matrix should be updated incrementally when new titles are added rather than rebuilt nightly.

**One design question worth addressing:** should the hard genre filter permanently lock users into their historical tastes? A user who only watched Action films last year might want to explore Drama. The 0.0% mismatch result in this notebook represents an extreme version of personalisation — never showing anything from a disliked genre. In practice, most production systems soften this: allowing disliked genres in 10–20% of recommendation slots to preserve discovery, or offering an explicit "Surprise me" mode that bypasses the filter entirely. The goal is to serve demonstrated taste without trapping users inside it, which is the same tension Spotify navigates with its "Discover Weekly" vs "Daily Mix" products.

**The broader lesson from the mismatch rate results:** a 15.4% mismatch rate in the NP model means roughly 1 in 7 recommendations is actively irrelevant to the user's preferences. At 500,000 subscribers each viewing 10 recommendations daily, that is approximately 700,000 wasted recommendation slots per day — each one representing a moment where the platform failed to surface something the user would have valued. Even reducing that to 9.4% (CF) represents a meaningful improvement in user experience and, by extension, in retention.

---

## 7. AI Usage Disclosure

Claude (Anthropic) was used extensively throughout this assignment for:
- Writing and debugging the recommender model classes (BayesianWeightedRating, ItemBasedCF, hybrid_recommend)
- Designing the evaluation framework including the within-training simulation and the genre mismatch rate metric
- Explaining the Bayesian Weighted Rating formula and TF-IDF cosine similarity approach
- Suggesting the MovieLens + TMDB linked dataset extension and the links.csv merge methodology
- Writing and structuring the markdown analysis throughout the notebook and this report

The core ideas, model design decisions, dataset choices, and analytical conclusions were developed in collaboration with Claude across an extended working session. Code was not independently written from scratch — it was co-developed iteratively. Results and numbers are genuine outputs from running the code on the actual datasets. All interpretations and arguments in the report reflect my own understanding of the material.
