# Personalised Movie Recommendation System — Collaborative Filtering
HCLTech Career Shaper · Project 28 · Digital-Data-AI / AIML

User-based CF, item-based CF and biased matrix-factorisation (SVD) recommenders for Indian regional cinema,
evaluated with RMSE and Precision/Recall/NDCG@K, with diversity metrics, a re-ranking step against filter bubbles,
a group-fairness audit, and a Streamlit dashboard (`app_ui.py`) and a Flask REST API (`app.py`).

## ⚠️ Read this first — what is and isn't real
* **The supplied `dataset.csv` could not support a recommender**: 500 rows, only **8 films**, 309 users with ~1.6 ratings each,
  and corrupted metadata (every title appears with 6 genres, 4 languages and 15 release years). 27 user–film pairs were duplicated.
* **The extra 4,500 rows are synthetic.** MovieLens-100K could not be downloaded in the build environment, so
  `01_generate_data.py` simulates users and 62 films (real titles; metadata compiled from general knowledge — **spot-check it**).
  Ratings come from a documented latent-factor model (see `data/generation_config.json`).
  **Therefore every accuracy number below describes how well the models recover *my generator*, not real viewer behaviour.**
  Disclose this in your report. To use real data, place MovieLens in the same schema
  (`user_id, movie_id, movie_title, genre, language, release_year, rating, timestamp`) as `data/dataset_clean.csv` and re-run steps 3–4.
* The 500 original rows are kept (10 % of the data) but their ratings are not tied to the repaired metadata.

## Run it
```bash
pip install -r requirements.txt
bash run_all.sh        # data → clean → EDA → evaluation → tests   (~2 min, fully seeded)
streamlit run app_ui.py     # dashboard UI  → http://localhost:8501   (run from this folder)
python3 app.py              # REST API + minimal page → http://127.0.0.1:5000
```
| Step | File | Output |
|---|---|---|
| 1 | `01_generate_data.py` | `data/dataset_extended.csv` (5,000 rows, same 7 columns, sorted by timestamp), `anomaly_labels.csv` (130 injected faults = 2.6 %) |
| 2 | `02_preprocess.py` | `data/dataset_clean.csv` (4,843 rows), `results/cleaning_report.json` |
| 3 | `03_eda.py` | `figures/01–04`, `results/eda_summary.json` |
| 4 | `04_evaluate.py` | `results/*.csv|json`, `figures/05–07` |
| – | `recsys/` (`models.py`, `metrics.py`, `engine.py`, `catalog.py`), `app_ui.py` (Streamlit), `app.py` (Flask API), `tests/` (21 tests) | |

## Data pipeline
Cleaning rules (in order): bad timestamps → rating outside 1–5 → timestamp after the 2025-12-01 cut-off → unknown titles →
**metadata repaired from the catalogue** (498 rows affected) → rated-before-release → exact duplicates → rating-bomb users
(≥10 ratings, std ≤ 0.25) → repeat ratings (keep latest).
The rules caught all 130 injected faults — **expected, since I created those faults and they are simple; it is not evidence of robustness on real data.**
Final: 500 users × 62 films, 4,843 ratings, 84 % sparse, median 7 ratings/user, mean rating 3.69.

## Models and protocol
Pure-NumPy implementations (`recsys/models.py`): GlobalMean, BiasBaseline, UserKNN, ItemKNN (shrunk, bias-removed cosine),
SVD (biased MF trained by SGD — Surprise's algorithm), Popularity, and **ItemKNN+Pop** (hybrid: item-CF rating prediction + popularity prior).
My BiasBaseline and SVD were cross-checked against scikit-surprise on the same split (test RMSE 0.810 vs 0.812; 0.815 vs 0.811).

Random 80/20 split (seed 42). Hyper-parameters **and the choice of final recommender** were selected by 5-fold CV on the 80 % train part only;
the 20 % test set was used once. “Relevant” = test rating ≥ 4. Candidates = all films the user didn't rate in train.
Ranking metrics use the 276 test users with ≥ 1 relevant item (avg 1.8 relevant items → Precision@10 can't exceed ≈ 0.18).

## Results (test set; 95 % bootstrap CI)
| Model | RMSE | NDCG@10 | Hit-rate@10 | Recall@10 |
|---|---|---|---|---|
| GlobalMean | 0.882 | – | – | – |
| BiasBaseline | 0.810 | 0.150 | 0.355 | 0.238 |
| UserKNN | 0.801 | 0.166 | 0.384 | 0.267 |
| **ItemKNN** | **0.799** [0.762, 0.829] | 0.175 | 0.435 | 0.301 |
| SVD | 0.815 | 0.155 | 0.370 | 0.248 |
| Popularity | – | 0.257 | 0.576 | 0.435 |
| Random | – | 0.109 | 0.294 | 0.199 |
| **ItemKNN+Pop (final)** | 0.799 | **0.287** [0.256, 0.320] | **0.645** | **0.504** |

What the numbers do and don't say:
* **Rating prediction:** ItemKNN beats the bias baseline by 0.011 RMSE (paired CI [−0.018, −0.004]) and UserKNN by 0.009 ([−0.017, −0.001]) — real but small.
  **SVD does *not* beat the bias baseline** (+0.005, CI [−0.001, +0.012]). With ~8 ratings per user there is too little data to learn
  latent factors; this is a legitimate finding, not a bug (Surprise's SVD agrees). The brief names SVD, so report it as-is.
* **Ranking:** plain rating-prediction models lose to **Popularity** because test items are mostly films users were exposed to (popularity-driven),
  and rating predictors favour niche, high-bias films. Blending a popularity prior (β=2, chosen by CV) beats Popularity by +0.031 NDCG@10 (paired CI [0.013, 0.049]).
  How much of this is an artefact of my exposure model in the generator is unknowable without real data.
* CIs on individual models overlap heavily (only ~970 test ratings); rely on the paired tests in `results/metrics.json`.

## Diversity, filter bubbles & fairness (brief §8)
`results/diversity_tradeoff.csv`, `figures/06`, `figures/07`, `results/group_audit.csv`, `results/language_exposure.csv`.
* **Filter-bubble finding:** the final hybrid recommends Hindi films in **59 %** of slots vs 40 % of training ratings; Kannada 6.6 % and Malayalam 7.7 % vs ~12 % of ratings.
  Plain ItemKNN is worse for those two (1.4 % / 2.8 %). Popularity-driven systems amplify the majority-language bubble.
* **Mitigation:** the `diversity` parameter applies MMR re-ranking (genre+language similarity). It has a real cost:

| λ (diversity) | NDCG@10 | Catalogue coverage | Genre ILD | Non-Hindi share |
|---|---|---|---|---|
| 1.0 (off) | 0.287 | 63 % | 0.75 | 41 % |
| 0.6 (`diversity=0.4`) | 0.270 | 81 % | 0.85 | 48 % |
| 0.4 (`diversity=0.6`) | 0.259 | 92 % | 0.86 | 56 % |

  At `diversity=0.6` Hindi falls to 44 % and Kannada/Malayalam rise to 10 % each. Pick the point on the curve in your report and justify it.
* **Demographic bias:** the data has **no demographic attributes** (just `user_id`), so demographic parity cannot be tested — say so rather than claim "no bias".
  I audited what exists: error/NDCG by user activity, by user's dominant language, and by item popularity. Tail films have higher RMSE (0.835 vs 0.795 head);
  Tamil-dominant users get lower NDCG@10 (0.18) than Hindi-dominant (0.31). Sub-groups are small (35–69 users) and untested for significance — treat as flags, not conclusions.
* Language is **not** a model input (pure CF); it is used only for evaluation and for re-ranking. Explanations ("Because you rated X") are returned with every recommendation.

## Dashboard (`streamlit run app_ui.py`)
Sidebar: existing user (or 🎲 random) / new user (pick films + rate them), method, *Diversify* slider, list length.
Tabs: Recommendations (cards + genre/language mix vs catalogue) · Why these? · Taste profile · Similar films · How good is it? · Ethics & diversity.

## API
```
GET  /api/recommend/<user_id>?k=10&method=hybrid|itemknn|userknn|svd|popularity&diversity=0..1
POST /api/recommend/new     {"ratings": {"1": 5, "3": 4.5}, "k": 10, "diversity": 0.4}     # user not in the data
GET  /api/movies/<id>/similar?k=5      GET /api/movies?language=&genre=
GET  /api/users/<id>/history           GET /api/metrics            GET /api/health        GET /  (demo page)
```
Unknown users get popularity-based recommendations flagged `cold_start: true`. Invalid input returns 400/404 JSON errors.

## Limitations
Synthetic data (above) · 62 films is a tiny catalogue · random (not temporal) split · one genre per film · ratings-only, no side information
(a content-based hybrid would help cold-start) · SGD-SVD tuned on a small grid · no online/A-B evaluation.

## Brief deliverables checklist
| Deliverable | Status |
|---|---|
| Working prototype | ✅ `app_ui.py` (dashboard) + `app.py` (API) + `recsys/` |
| EDA with 3+ visualisations | ✅ `figures/01–04` (+ 05–07 for results) |
| Cleaned `dataset.csv` | ✅ `data/dataset_clean.csv` |
| `data_dictionary.csv` | ✅ filled |
| Project report (≥ 8 pages) | ❌ not written |
| Project synopsis | ❌ template not filled |
| Demo video / expo demo | ❌ yours to record |
| Signed ethical declaration | ❌ yours to sign — AI-tool use must be disclosed in the report appendix |
