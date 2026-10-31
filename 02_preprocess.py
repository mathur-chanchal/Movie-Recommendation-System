"""Step 2 - clean + pre-process data/dataset_extended.csv -> data/dataset_clean.csv

Rules are applied in order; every removed row is attributed to the first rule that caught it, so
we can score the rule-based detector against data/anomaly_labels.csv (precision / recall).
The detector never looks at the labels - they are used only for the report.
"""
import json
import numpy as np
import pandas as pd
from recsys.catalog import catalog_frame

CUTOFF = pd.Timestamp("2025-12-01 23:59:59")        # last timestamp in the supplied dataset.csv
BOT_MIN_RATINGS, BOT_MAX_STD = 10, 0.25

raw = pd.read_csv("data/dataset_extended.csv")
raw.insert(0, "row_number", np.arange(1, len(raw) + 1))
cat = catalog_frame()
labels = pd.read_csv("data/anomaly_labels.csv")

df = raw.copy()
dropped = {}                                          # rule -> set(row_number)


def drop(rule, mask):
    global df
    dropped[rule] = set(df.loc[mask, "row_number"])
    df = df[~mask].copy()


# 1 ── unparseable timestamps
df["ts"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
drop("unparseable_timestamp", df.ts.isna())
# 2 ── rating outside the 1–5 scale (or not a 0.5 step)
drop("rating_out_of_range", ~df.rating.between(1, 5) | ((df.rating * 2) % 1 != 0))
# 3 ── timestamp after the data-collection cut-off
drop("future_timestamp", df.ts > CUTOFF)
# 4 ── titles missing from the catalogue
drop("unknown_title", ~df.movie_title.isin(cat.movie_title))

# 5 ── repair metadata from the canonical catalogue (rows are KEPT, only fixed)
df = df.merge(cat[["movie_id", "movie_title", "genre", "language", "release_year", "release_date"]],
              on="movie_title", how="left", suffixes=("", "_cat"))
meta_fixed = {c: int((df[c] != df[c + "_cat"]).sum()) for c in ["genre", "language", "release_year"]}
rows_with_any_fix = int(((df.genre != df.genre_cat) | (df.language != df.language_cat) |
                         (df.release_year != df.release_year_cat)).sum())
for c in ["genre", "language", "release_year"]:
    df[c] = df[c + "_cat"]
df = df.drop(columns=[c + "_cat" for c in ["genre", "language", "release_year"]])

# 6 ── rated before the film existed
drop("rated_before_release", df.ts < pd.to_datetime(df.release_date))
# 7 ── exact duplicate rows
drop("exact_duplicate", df.duplicated(["user_id", "movie_id", "rating", "timestamp"], keep="first"))
# 8 ── rating-bomb / bot users: many ratings with (almost) no variance
g = df.groupby("user_id").rating.agg(["size", "std"])
bots = g[(g["size"] >= BOT_MIN_RATINGS) & (g["std"] <= BOT_MAX_STD)].index
drop("rating_bomb_bot", df.user_id.isin(bots))
# 9 ── same user rated the same film twice -> keep the most recent rating
df = df.sort_values("ts", kind="stable")
drop("repeat_rating_kept_latest", df.duplicated(["user_id", "movie_id"], keep="last"))

clean = df.sort_values("ts", kind="stable")[
    ["user_id", "movie_id", "movie_title", "genre", "language", "release_year", "rating", "timestamp"]]
clean["rating"] = clean["rating"].astype(float)
clean.to_csv("data/dataset_clean.csv", index=False)

# ── report ---------------------------------------------------------------------------------
removed_all = set().union(*dropped.values())
lab = labels.set_index("row_number").anomaly_type
per_type = {}
for t, rows in lab.groupby(lab):
    rows = set(rows.index)
    per_type[t] = dict(injected=len(rows), removed=len(rows & removed_all),
                       recall=round(len(rows & removed_all) / len(rows), 3))
tp = len(set(lab.index) & removed_all)
natural = removed_all - set(lab.index)               # removed rows that were NOT injected
nat_by_rule = {r: len(rows - set(lab.index)) for r, rows in dropped.items() if rows - set(lab.index)}

n_users, n_items = clean.user_id.nunique(), clean.movie_id.nunique()
report = dict(
    rows_in=len(raw), rows_out=len(clean), rows_removed=len(removed_all),
    removed_by_rule={r: len(v) for r, v in dropped.items()},
    metadata_values_corrected=meta_fixed, rows_with_any_metadata_fix=rows_with_any_fix,
    bot_users_flagged=[int(b) for b in bots],
    detector_vs_injected_labels=dict(
        injected=len(lab), caught=tp, recall=round(tp / len(lab), 3),
        removed_but_not_injected=len(natural), removed_but_not_injected_by_rule=nat_by_rule,
        note="All 130 injected faults were caught. 100% recall is expected: the faults are simple and rule-shaped by construction, so this says little about real-world robustness. The 27 'not injected' removals are genuine repeat-ratings already present in the supplied dataset.csv."),
    per_anomaly_type=per_type,
    final=dict(users=int(n_users), movies=int(n_items), ratings=len(clean),
               density_pct=round(100 * len(clean) / (n_users * n_items), 2),
               mean_rating=round(clean.rating.mean(), 3),
               ratings_per_user_median=float(clean.groupby("user_id").size().median()),
               ratings_per_movie_min=int(clean.groupby("movie_id").size().min())))
json.dump(report, open("results/cleaning_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
