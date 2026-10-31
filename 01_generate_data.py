"""Step 1 - expand dataset.csv from 500 to 5,000 rows (synthetic) + inject labelled faults.

Output
  data/dataset_extended.csv   5,000 rows, EXACT same 7 columns as the supplied dataset.csv,
                              sorted by timestamp (YYYY-MM-DD HH:MM:SS)
  data/anomaly_labels.csv     which rows are injected faults (~2.6 % of rows) and of what type
  data/movie_catalog.csv      canonical title -> language/genre/year table
  data/generation_config.json parameters + seed (reproducibility)

IMPORTANT: the 4,500 new rows are SYNTHETIC. MovieLens-100K could not be downloaded in the
build environment (grouplens.org is not reachable), so ratings come from a documented
latent-factor model (users have taste vectors, genre/language preferences and a rating bias;
films have quality, popularity and latent attributes). Disclose this in the report.
"""
import json
import numpy as np
import pandas as pd
from recsys.catalog import catalog_frame, GENRES, LANGUAGES

SEED = 42
N_USERS = 500
N_TOTAL, N_ORIGINAL = 5000, 500
N_NEW = N_TOTAL - N_ORIGINAL            # 4500 rows to add (normal + injected faults)
WINDOW_START, WINDOW_END = pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2025-12-01 23:59:59")
FMT = "%Y-%m-%d %H:%M:%S"

rng = np.random.default_rng(SEED)
cat = catalog_frame()
n_items = len(cat)
cat.to_csv("data/movie_catalog.csv", index=False)

# ------------------------------------------------------------------ original rows
orig = pd.read_csv("data/dataset_original.csv")
orig["timestamp"] = pd.to_datetime(orig["timestamp"]).dt.strftime(FMT)   # date-only -> 00:00:00
orig["_anom"] = ""

# ------------------------------------------------------------------ latent model
K = 4
lang_pop = np.array([0.35, 0.20, 0.20, 0.12, 0.13])            # population prior over languages
lang_idx = {l: i for i, l in enumerate(LANGUAGES)}
gen_idx = {g: i for i, g in enumerate(GENRES)}
item_lang = cat.language.map(lang_idx).to_numpy()
item_gen = cat.genre.map(gen_idx).to_numpy()

user_bias = rng.normal(0, 0.35, N_USERS + 1)
P = rng.normal(0, 0.55, (N_USERS + 1, K))                      # user taste vectors
Q = rng.normal(0, 0.55, (n_items, K))                          # film attribute vectors
item_bias = rng.normal(0, 0.30, n_items) + np.where(cat.fame_rank <= 15, 0.15, 0.0)
genre_logit = rng.normal(0, 1.0, (N_USERS + 1, len(GENRES)))
primary_lang = rng.choice(len(LANGUAGES), N_USERS + 1, p=lang_pop)
lang_logit = rng.normal(0, 0.5, (N_USERS + 1, len(LANGUAGES)))
lang_logit[np.arange(N_USERS + 1), primary_lang] += 1.5
g_z = genre_logit - genre_logit.mean(1, keepdims=True)
l_z = lang_logit - lang_logit.mean(1, keepdims=True)

popularity = cat.fame_rank.to_numpy(float) ** -0.55            # Zipf-like: famous films rated more


def draw_rating(u, i):
    raw = (3.33 + user_bias[u] + item_bias[i] + 0.35 * g_z[u, item_gen[i]]
           + 0.30 * l_z[u, item_lang[i]] + 0.8 * P[u] @ Q[i] + rng.normal(0, 0.5))
    return float(np.clip(np.round(raw * 2) / 2, 1.0, 5.0))


def rand_ts(lo, hi):
    return lo + pd.Timedelta(seconds=int(rng.integers(0, int((hi - lo).total_seconds()))))


# ------------------------------------------------------------------ injected faults (labelled)
N_BOT_USERS, BOT_ROWS = 4, 15
N_RANGE, N_FUTURE, N_PRE, N_DUP = 20, 20, 10, 20
N_INJECT = N_BOT_USERS * BOT_ROWS + N_RANGE + N_FUTURE + N_PRE + N_DUP     # 130 = 2.6 % of 5000
N_NORMAL = N_NEW - N_INJECT

# ------------------------------------------------------------------ normal synthetic rows
# heavy-tailed user activity
w = rng.lognormal(0, 0.8, N_USERS)
counts = np.maximum(2, np.round(w / w.sum() * N_NORMAL)).astype(int)
while counts.sum() != N_NORMAL:                                 # fix rounding drift
    j = rng.integers(0, N_USERS)
    counts[j] += 1 if counts.sum() < N_NORMAL else (-1 if counts[j] > 2 else 0)

seen_orig = orig.groupby("user_id")["movie_title"].apply(set).to_dict()
title_to_i = {t: i for i, t in enumerate(cat.movie_title)}
rows = []
for u in range(1, N_USERS + 1):
    already = {title_to_i[t] for t in seen_orig.get(u, set()) if t in title_to_i}
    expo = popularity * np.exp(1.2 * l_z[u, item_lang] + 0.8 * g_z[u, item_gen])   # exposure
    expo[list(already)] = 0
    n_u = min(counts[u - 1], int((expo > 0).sum()))
    items = rng.choice(n_items, n_u, replace=False, p=expo / expo.sum())
    for i in items:
        lo = max(WINDOW_START, pd.Timestamp(cat.release_date[i]))
        rows.append(dict(user_id=u, movie_title=cat.movie_title[i], genre=cat.genre[i],
                         language=cat.language[i], release_year=int(cat.release_year[i]),
                         rating=draw_rating(u, i), timestamp=rand_ts(lo, WINDOW_END).strftime(FMT),
                         _anom=""))
gen = pd.DataFrame(rows)
# top up if min(counts, available) left us short
while len(gen) < N_NORMAL:
    u = int(rng.integers(1, N_USERS + 1)); i = int(rng.integers(0, n_items))
    if cat.movie_title[i] in set(gen[gen.user_id == u].movie_title) | seen_orig.get(u, set()):
        continue
    lo = max(WINDOW_START, pd.Timestamp(cat.release_date[i]))
    gen.loc[len(gen)] = [u, cat.movie_title[i], cat.genre[i], cat.language[i],
                         int(cat.release_year[i]), draw_rating(u, i), rand_ts(lo, WINDOW_END).strftime(FMT), ""]
gen = gen.iloc[:N_NORMAL].copy()


def meta(i):
    return dict(movie_title=cat.movie_title[i], genre=cat.genre[i], language=cat.language[i],
                release_year=int(cat.release_year[i]))


inj = []
# 1) rating-bomb bots: new user ids, constant extreme rating, all within one hour
for b in range(N_BOT_USERS):
    uid = N_USERS + 1 + b
    val = 5.0 if b % 2 == 0 else 1.0
    t0 = rand_ts(WINDOW_START + pd.Timedelta(days=60), WINDOW_END - pd.Timedelta(days=5))
    for i in rng.choice(n_items, BOT_ROWS, replace=False):
        inj.append(dict(user_id=uid, **meta(i), rating=val,
                        timestamp=(t0 + pd.Timedelta(seconds=int(rng.integers(0, 3600)))).strftime(FMT),
                        _anom="rating_bomb_bot"))
# 2) rating outside the 1-5 scale
for _ in range(N_RANGE):
    u, i = int(rng.integers(1, N_USERS + 1)), int(rng.integers(0, n_items))
    inj.append(dict(user_id=u, **meta(i), rating=float(rng.choice([0.0, 0.0, -1.0, 5.5, 6.0, 10.0])),
                    timestamp=rand_ts(max(WINDOW_START, pd.Timestamp(cat.release_date[i])), WINDOW_END).strftime(FMT),
                    _anom="out_of_range_rating"))
# 3) timestamp after the data-collection cut-off
for _ in range(N_FUTURE):
    u, i = int(rng.integers(1, N_USERS + 1)), int(rng.integers(0, n_items))
    inj.append(dict(user_id=u, **meta(i), rating=draw_rating(u, i),
                    timestamp=rand_ts(pd.Timestamp("2026-11-01"), pd.Timestamp("2027-12-31")).strftime(FMT),
                    _anom="future_timestamp"))
# 4) rated before the film was released (2024 releases only)
late = [title_to_i["Kalki 2898 AD"], title_to_i["Manjummel Boys"]]
for k in range(N_PRE):
    u, i = int(rng.integers(1, N_USERS + 1)), late[k % 2]
    inj.append(dict(user_id=u, **meta(i), rating=draw_rating(u, i),
                    timestamp=rand_ts(WINDOW_START, pd.Timestamp(cat.release_date[i]) - pd.Timedelta(days=1)).strftime(FMT),
                    _anom="rated_before_release"))
# 5) exact duplicate rows
for j in rng.choice(len(gen), N_DUP, replace=False):
    r = gen.iloc[j].to_dict(); r["_anom"] = "exact_duplicate"; inj.append(r)

full = pd.concat([orig, gen, pd.DataFrame(inj)], ignore_index=True)
assert len(full) == N_TOTAL, len(full)

# sequential timestamps (sorted, stable) ------------------------------------------------
full["_ts"] = pd.to_datetime(full["timestamp"])
full = full.sort_values("_ts", kind="stable").reset_index(drop=True)
full.insert(0, "row_number", np.arange(1, len(full) + 1))

cols = ["user_id", "movie_title", "genre", "language", "release_year", "rating", "timestamp"]
full[cols].to_csv("data/dataset_extended.csv", index=False)
labels = full[full["_anom"] != ""][["row_number", "user_id", "movie_title", "_anom"]].rename(columns={"_anom": "anomaly_type"})
labels.to_csv("data/anomaly_labels.csv", index=False)

json.dump(dict(seed=SEED, n_users=N_USERS, n_total=N_TOTAL, n_original=N_ORIGINAL, n_synthetic=N_NEW,
               n_injected_faults=N_INJECT, pct_faults=round(100 * N_INJECT / N_TOTAL, 2),
               latent_dims=K, rating_formula="3.33+b_u+b_i+0.35*genre_pref+0.30*lang_pref+0.8*p_u.q_i+N(0,0.5), "
                                             "rounded to 0.5, clipped [1,5]",
               popularity="fame_rank^-0.55 x exp(1.2*lang_pref + 0.8*genre_pref), sampled w/o replacement",
               window=[str(WINDOW_START), str(WINDOW_END)]),
          open("data/generation_config.json", "w"), indent=2)

print(f"wrote {len(full)} rows | faults injected: {len(labels)} ({100*len(labels)/len(full):.1f}%)")
print(labels.anomaly_type.value_counts().to_string())
print("ts range:", full.timestamp.iloc[0], "->", full.timestamp.iloc[-1],
      "| sorted:", full["_ts"].is_monotonic_increasing)
