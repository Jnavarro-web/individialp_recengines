# Recommender System Analysis & Implementation

**Student ID:** 16255  
**Last Name:** Navarro  
**Dataset:** TMDB 5000 Movies + MovieLens ml-latest-small  
**Date:** April 2026

---

## 1. Introduction

Recommender systems are one of the most commercially important uses of machine learning, transforming and directing in the background what we watch, buy and listen to. The problem main challenge is given a huge catalogue and a user with limited patience, recommend something worth clicking before they lose interest.

This project evaluates two required recommender approaches and then extends them to test a more realistic question. TMDB 5000 supplies the content side of the problem: genres, keywords, cast, crew, budgets and aggregate vote scores for roughly 4,800 films. MovieLens supplies the 100,836 ratings from 610 real users. On their own, each dataset is incomplete. Together, they allow the analysis to move from general film quality to real user preference.

The brief requires a Non-Personalized Bayesian Weighted Rating model and a Collaborative Filtering model. Those two models are implemented first and evaluated exactly as required. However, the notebook also showed that only a TMDB evaluation cannot tell whether a recommendation is personally suitable or simply globally popular. For that reason, two hybrid extensions were developed later in the analysis. They keep the same logic and ordering as the core models, but add real user taste information from MovieLens so the project can test personal relevance as well as quality.

---

## 2. Data Exploration

### TMDB 5000

After merging the two TMDB files and applying a minimum 10 votes quality filter, the working dataset contains 4,392 films, with 411 removed for insufficient data. Vote averages range from 1.9 to 8.5 with a mean of 6.227 and a standard deviation of 0.893. Most films fall between 5.7 and 6.8. That clustering around the middle makes sense, highly anticipated films are often rated harshly when they disappoint, while lower expectation discoveries are rated generously when they turn out better than expected.

Vote count is extremely right-skewed. *Avatar* has 11,800 votes, while most films have fewer than 300. A film rated 5.0 by two people is not as trustworthy as a film rated 5.0 by seven thousand people, even if the averages match. This is the main reason a raw average is not enough and why the Bayesian formula later becomes necessary. It is also worth noting early that both required models have the global mean structurally embedded in their predictions, so their RMSE will naturally look decent. That does not automatically mean they are learning anything useful, which is why the ranking metrics and later mismatch analysis matter more.

Genre composition is led by Drama, followed by Comedy, Thriller and Action, with most films carrying two or three genre labels at the same time. Looking at directors with at least three films, Hayao Miyazaki averages 8.05, Sergio Leone 8.00 and Christopher Nolan 7.80, which suggests `vote_average` is capturing genuine film quality rather than just noise. Financial data is available for around 73% of films, with a median budget of $26M and median revenue of $57M. Both are strongly right-skewed, and many expensive films still fail to meet their budgets.

**Train/test split:** stratified 80/20 by `vote_average` quintile, producing 3,513 training films and 879 test films. Both subsets keep the same mean of 6.227, confirming the split is balanced.

### MovieLens

MovieLens adds 100,836 individual ratings from 610 users on a 0.5-5.0 half-star scale. After joining through `links.csv`, 70,149 ratings remain across 610 users and 3,505 movies that also exist in TMDB 5000. The ratings lost in the merge mainly belong to older or more niche titles not present in the TMDB subset. Even with that loss, the linked data adds valuable information, enough evidence of what different users actually like or dislike.

---

## 3. Non-Personalized Recommender

### Design

The Non-Personalized model ranks films using the Bayesian Weighted Rating formula:

```
WR(i) = (v / (v + m)) × R  +  (m / (v + m)) × C
```

where `v` is vote count, `R` is vote average, `m = 662` is the 70th percentile of training vote counts, and `C = 6.227` is the global mean. Films with fewer than 662 votes are pulled progressively towards 6.227, so a high score supported by only a few ratings is treated as weaker evidence than a slightly lower score backed by thousands.

### Top-10 Results

The top ten films are *The Shawshank Redemption* (WR = 8.330, 8,205 votes), *The Godfather* (8.181), *Fight Club* (8.164), *Pulp Fiction* (8.149), *Forrest Gump* (8.048), *The Lord of the Rings: The Return of the King* (7.958), *The Godfather Part II* (7.957), *Star Wars* (7.930), *The Green Mile* (7.923) and *Se7en* (7.907).

This is the kind of list the model should produce: strong, and trusted films with a large volume of support. Every user receives the same recommendations, so the value of the model is not personalisation. Its strength is in the cold start, when a user has no history yet, a popularity-quality list like this is a sensible first layer.

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

The apparently perfect Precision and NDCG scores need to be interpreted. In the notebook, a recommendation is treated as relevant when `vote_average > 7.0`, and that is exactly what this model is designed to rank highest. So the scores show internal consistency, not genuine user fit. The model is doing what it was built to do, but at this stage there is still no way to know whether the films suit the individual seeing them.

---

## 4. Collaborative Filtering Recommender

### Design

The collaborative filtering stage is item-based and content-driven. Each film is converted into a weighted feature string made from genres, keywords, cast and director. Genre names are repeated three times, top keywords and director twice, and top cast once. TF-IDF with 5,000 features then down-weights common terms like "drama" and gives more weight to distinctive tokens. Cosine similarity on those vectors produces a 3,513 x 3,513 item-item similarity matrix.

For prediction, the model takes the similarity-weighted mean of the 20 most similar training films' `vote_average` values. For recommendation, it aggregates similarity scores across the user's seed films and returns the highest-scoring unseen titles.

### Sample Recommendations

Using *The Shawshank Redemption* and *The Godfather* as seeds, the model returns films such as *The Godfather Part II* (8.3), *Escape from Alcatraz* (7.2), *Ajami* (6.8), *The Rainmaker* (6.7) and *The Cotton Club* (6.6). The Crime/Drama pattern is exactly what should happen.

The limitation appears lower down the list. *The Rise of the Krays* (4.5) can still appear because it matches the same tokens, even though its overall quality is not very good. In other words, similarity can retrieve films that seem right on the math without being especially good. That weakness becomes important later and why it motivates the second hybrid extension.

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

RMSE and MAE are identical to the Non-Personalized model because both are still trying to predict aggregate quality without access to true individual variation. Where CF differs is in recommendation behaviour, it sacrifices ranking quality for catalogue reach. Coverage rises to 9.1% compared with 5.7% for the Non-Personalized model, which means it explores a meaningfully wider slice of the catalogue.

---

## 5. Evaluation & Comparison

### Required Model Comparison

| Metric | Non-Personalized | Item-Based CF | Winner |
|--------|:---:|:---:|:---:|
| RMSE | 0.9169 | 0.9169 | Tie |
| MAE | 0.7219 | 0.7219 | Tie |
| Precision@10 | **1.000** | 0.126 | NP |
| Recall@10 | **0.027** | 0.003 | NP |
| NDCG@10 | **1.000** | 0.136 | NP |
| Coverage | 0.057 | **0.091** | CF |
| Diversity | **0.742** | 0.585 | NP |

On paper, the Non-Personalized model wins five of the seven metrics. But that result needs context. Precision, Recall and NDCG all reward the same quality-based ranking rule that the model is already using, so it was always likely to dominate there. CF wins on coverage, which is the first hint that it is doing something the NP model cannot: moving different users through different parts of the catalogue.

The tie on RMSE and MAE is the key limitation. Both models hit essentially the same floor because neither one has access to real personal preference in only the TMDB evaluation. That is the point where the extension becomes justified rather than optional, if we want a real recommendator.

### Motivation for the Extension

The assignment only requires the two models above, but the notebook made a clear weakness not easy to ignore. A globally good film can still be a poor recommendation if it belongs to a genre a user consistently dislikes. TMDB only evaluation cannot measure that problem because it has no user histories. Linking MovieLens through `links.csv` allows us to test recommendations against real rating behaviour and calculate a **genre mismatch rate**: the proportion of recommended films whose primary genre the user has historically rated poorly.

This extension does not replace the required models. It builds directly on them in the same order as the earlier sections. First, the two required systems are tested against real users. Then two hybrid versions are added to see whether personal relevance can be improved without abandoning quality.

### Real User Behaviour and Baseline Mismatch

For each user, a genre profile is built from their rating history. Two thresholds are used for different purposes. For the **mismatch metric**, a recommendation is counted as a clash only when the user's average score for that primary genre is **below 3.0**. For the **hybrid filters**, the rule is more conservative and smoothed: a genre is blocked only when the user's average is **at or below 3.5** and the user has rated **at least 3 films** in that genre. This prevents one or two bad experiences from disregarding a genre too early.

The contrast between users is useful. **User 1** (147 ratings) has broad taste, with strongest genres in Music (5.00), Western (4.67) and Animation (4.57), and no category collapsing badly. **User 5** (32 ratings) is much narrower, with positive metrics in Western, History and Animation, but weaker scores in high-volume genres such as Drama, Comedy, Action and Romance. That second user is exactly the kind of case where a universal popularity list starts to fail.

Testing the required models against these real profiles shows why the extension matters. The **Non-Personalized** model produces a **15.4% mismatch rate**, while **Item-Based CF** reduces that to **9.4%** by seeding recommendations from films the user already liked. That is an improvement, but it still means almost one in ten CF recommendations can fall into a genre the user has repeatedly rated badly.

### Hybrid v1 - CF + Smoothed Genre Filter

#### Design

Hybrid v1 combines three elements. First, CF generates a candidate pool from films similar to what the user liked. Second, a smoothed genre filter blocks any film whose primary genre the user has rated at or below 3.5 across at least 3 watched titles. Third, a Non-Personalized fallback fills any remaining gaps from the global WR ranking so the system can still return a full recommendation list.

The model also uses a coverage-aware seed mechanism. Thus, rather than building recommendations only from the films the user most liked, it makes sure the user's top three genres are represented in the seed set. If one of those genres has little rating history, the highest WR unseen training film from that genre is used as a proxy seed. It helps keep the recommendations more balanced and stops the model from leaning too heavily toward whichever genre dominates the user's history. And adds room for discovery. 

#### Sample Results

For **User 1**, the recommendations correctly represent the users prefered genres: Music, Western and Animation, including *Spirit: Stallion of the Cimarron*, *Fantasia 2000*, *Legends of Oz: Dorothy's Return*, *Paint Your Wagon* and *Beat the World*. For **User 5**, the list deviates towards the user's most liked genres: Animation and Western titles such as *Toy Story 3*, *Spirit: Stallion of the Cimarron*, *The Pirates! In an Adventure with Scientists!* and *Quest for Camelot*. In both cases, the recommendations feel visibly tied to the user's taste profile rather than just to global popularity.

#### Evaluation Results

| Metric | Hybrid v1 |
|--------|:---:|
| Mismatch Rate | **0.2%** |
| Coverage | 0.0462 |
| Mean WR | 6.3220 |

Hybrid v1 almost eliminates genre clashes while recovering coverage to slightly above the NP baseline. Its weakness is quality control. It respects taste extremely well, but some of the remaining titles are only decent rather than genuinely strong films.

### Hybrid v2 - CF + Smoothed Genre Filter + WR Quality Gate

#### Design

Hybrid v2 keeps the full Hybrid v1 pipeline and adds one final condition: every recommendation must also have a **Bayesian Weighted Rating of at least 6.3**. This targets the specific failure case seen in the raw CF model, where a film can be highly similar because of token overlap while not meeting quality overall.

The final logic then becomes: recommend films that are **similar to what the user liked**, **not in a consistently disliked genre**, and **good enough on WR**. If the filtered candidate pool becomes too small, the fallback uses only films that also satisfy the same genre and WR conditions.

#### Sample Results

For **User 1**, the output is higuer in quality: *Toy Story 3* (WR = 7.43), *Spirit: Stallion of the Cimarron* (6.88), *Fantasia 2000* (6.46), *Rango* (6.51), *Cars* (6.55) and *Step Up Revolution* (6.46). For **User 5**, the final list includes *Toy Story 3* (7.43), *Up* (7.57), *Frozen* (7.18), *Space Jam* (6.41), *Quest for Camelot* (6.38), *A Monster in Paris* (6.31) and *Space Pirate Captain Harlock* (6.32). Their prefered tastes remain there, but the weaker "only content" matches are gone.

#### Evaluation Results

| Metric | Hybrid v2 |
|--------|:---:|
| Mismatch Rate | 0.6% |
| Coverage | **0.0471** |
| Mean WR | **6.6958** |

Hybrid v2 gives up a very small amount of mismatch performance relative to Hybrid v1, but it improves the average quality of recommendations substantially. The trade-off becomes an advantage: a tiny increase in mismatch (also room for discovery) in exchange for a higuer quality floor.

### Final Four-Model Comparison

| Model | Mismatch Rate | Coverage | Mean WR |
|-------|:---:|:---:|:---:|
| Non-Personalized | 15.4% | 0.0455 | **7.9612** |
| Item-Based CF | 9.4% | 0.0316 | 6.2708 |
| Hybrid v1 (CF + filter) | **0.2%** | 0.0462 | 6.3220 |
| Hybrid v2 (CF + filter + WR) | 0.6% | **0.0471** | 6.6958 |

Overall NP has the highest mean WR because it is explicitly ranking by quality and ignoring taste. Item-Based CF improves relevance, but loses both coverage and quality. Eventhough, Hybrid v1 is the strongest model for pure personal fit, Hybrid v2 is the best overall balance: mismatch remains close to zero and allows users to discover new films, coverage is the highest of the personalised models, and average recommendation quality rises meaningfully above both CF and Hybrid v1.

### Overall Judgment

The main result is not that Hybrid v2 beats the NP model on every number, because the two systems are dealing with different problems. The conclusion to make here is that the hybrid approach closes the gap between **relevance** and **quality**. Hybrid v1 proves that respecting user taste can almost eliminate mismatch. Hybrid v2 shows that this can be done without settling for mediocre titles. Therefore, their use cases would be applied differently to all, but all achieving to meet the issue they are dealing with, if we are in a cold strat position a NP model would be a good fit but, if the goal is a personalised recommender, **Hybrid v2** is the strongest model in the project. 

---

## 6. Business Reflection

To place the results in a realistic setting, lets take a mid-sized streaming platform with 300,000 subscribers and a 40,000-title catalogue. In that environment, the Non-Personalized model is immediately useful. It's fast, easy to pre-compute, and the best for cold start situations such as new users, editorial shelves or fallback recommendations when no profile is available. A popularity quality layer like this would help get a clear picture of what users actually would prefer from the diverse set.

The personalised models would be introduced gradually. During the beggining, the system could rely on the Non-Personalized model while collecting the first 10 to 20 ratings, perhaps supported by a short taste quiz. Once enough data exists, CF with the smoothed genre filter becomes viable. As the profile matures, Hybrid v2 becomes the better production choice because it keeps recommendations aligned with taste without allowing similarity alone to drag quality down.

There is also a clear operational reason to care about the mismatch numbers. A 15.4% mismatch rate means that out of every 10 recommendations made by the Non-Personalized model, 1 to 2 are likely to land in a genre the user actively dislikes. At the scale of thousands of users, that also becomes hundreds of thousands of poor homepage recommendations per day. And repeated oftenly, could damage trust. 

The main technical limitation is scalability. A dense similarity matrix becomes expensive once the catalogue grows into the tens of thousands of items, so a real deployment would need approximate nearest neighbour retrieval or a more compressed candidate generation layer. There is also a product observation, if the genre filter becomes too strong, it may over-protect the user from discovery. The current smoothing rule already reduces that risk, but balancing familiarity and exploration would need monitoring in production.

---

## 7. AI Usage Disclosure

AI tools were used throughout this assignment as development support rather than as a substitute for running the analysis. Claude (Anthropic) was used to help structure and debug model classes, refine code, fix the mismatch metric, support the coverage-aware seed mechanism, and help polish parts of the written analysis in the report.

The idea for extending the project with MovieLens-based mismatch testing came from the observation that the TMDB only setup could not distinguish between recommending something because a user would genuinely enjoy it and recommending it because it was simply popular overall. AI assistance then helped translate that idea into working implementation more efficiently.

All reported figures come from the actual notebook outputs on the linked datasets. The interpretation of the results, baseline of the code, the choice to extend the analysis, and the conclusions drawn from the models are my own.
