"""Step 3 - exploratory data analysis (figures -> figures/01..04, numbers -> results/eda_summary.json)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
orig = pd.read_csv("data/dataset_original.csv")
raw = pd.read_csv("data/dataset_extended.csv")
clean = pd.read_csv("data/dataset_clean.csv", parse_dates=["timestamp"])
labels = pd.read_csv("data/anomaly_labels.csv")
rep = json.load(open("results/cleaning_report.json"))
LANG_COL = {"Hindi": "#e15759", "Tamil": "#4c78a8", "Telugu": "#59a14f", "Kannada": "#f28e2b", "Malayalam": "#b07aa1"}

# ---- 01 data quality ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
meta = orig.groupby("movie_title").agg(genres=("genre", "nunique"), languages=("language", "nunique"), years=("release_year", "nunique"))
x = np.arange(len(meta)); w = 0.27
for k, (c, col) in enumerate(zip(["genres", "languages", "years"], ["#4c78a8", "#f28e2b", "#59a14f"])):
    ax[0].bar(x + (k - 1) * w, meta[c], w, label=c, color=col)
ax[0].set_xticks(x); ax[0].set_xticklabels(meta.index, rotation=40, ha="right"); ax[0].axhline(1, color="k", ls="--", lw=.8)
ax[0].set_title("Supplied dataset.csv: distinct values per title\n(a film should have exactly 1 - dashed line)"); ax[0].set_ylim(0, 19); ax[0].legend(fontsize=8, ncol=3, loc="upper center")
t = labels.anomaly_type.value_counts()
ax[1].barh(t.index, t.values, color="#e15759"); ax[1].set_title(f"Injected faults ({len(labels)} rows = {100*len(labels)/len(raw):.1f}% of 5,000)")
rm = pd.Series(rep["removed_by_rule"]); rm = rm[rm > 0]
ax[2].barh(rm.index, rm.values, color="#4c78a8"); ax[2].set_title(f"Rows removed by cleaning ({rep['rows_removed']})")
plt.tight_layout(); plt.savefig("figures/01_data_quality.png"); plt.close()

# ---- 02 rating distributions ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
vc = clean.rating.value_counts().sort_index()
ax[0].bar(vc.index.astype(str), vc.values, color="#4c78a8")
ax[0].set_title(f"Rating distribution (mean {clean.rating.mean():.2f}, sd {clean.rating.std():.2f})"); ax[0].set_xlabel("rating")
g = clean.groupby("language").rating.agg(["mean", "size"]).sort_values("mean")
ax[1].barh(g.index, g["mean"], color=[LANG_COL[l] for l in g.index]); ax[1].set_xlim(3, 4.2)
for y, (m, n) in enumerate(zip(g["mean"], g["size"])): ax[1].text(m + .01, y, f"{m:.2f} (n={n})", va="center", fontsize=8)
ax[1].set_title("Mean rating by film language")
g2 = clean.groupby("genre").rating.agg(["mean", "size"]).sort_values("mean")
ax[2].barh(g2.index, g2["mean"], color="#59a14f"); ax[2].set_xlim(3, 4.2)
for y, (m, n) in enumerate(zip(g2["mean"], g2["size"])): ax[2].text(m + .01, y, f"{m:.2f} (n={n})", va="center", fontsize=8)
ax[2].set_title("Mean rating by genre")
plt.tight_layout(); plt.savefig("figures/02_rating_distributions.png"); plt.close()

# ---- 03 long tail / activity -----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
per_user = clean.groupby("user_id").size()
ax[0].hist(per_user, bins=range(1, per_user.max() + 2, 1), color="#4c78a8"); ax[0].set_title(f"Ratings per user (median {per_user.median():.0f})")
ax[0].set_xlabel("# ratings"); ax[0].set_ylabel("users")
pm = clean.groupby(["movie_id", "language"]).size().reset_index(name="n").sort_values("n", ascending=False).reset_index(drop=True)
ax[1].bar(range(len(pm)), pm.n, color=[LANG_COL[l] for l in pm.language])
ax[1].set_title("Ratings per film (long tail)"); ax[1].set_xlabel("film rank by popularity")
ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in LANG_COL.values()], labels=list(LANG_COL), fontsize=7)
lang_share = clean.language.value_counts(normalize=True); cat = pd.read_csv("data/movie_catalog.csv").language.value_counts(normalize=True)
d = pd.DataFrame({"share of ratings": lang_share, "share of catalogue": cat}).loc[list(LANG_COL)]
d.plot.bar(ax=ax[2], color=["#4c78a8", "#bbbbbb"]); ax[2].set_title("Language: ratings vs catalogue"); ax[2].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig("figures/03_activity_long_tail.png"); plt.close()

# ---- 04 sparsity + time -------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
M = clean.pivot_table(index="user_id", columns="movie_id", values="rating").notna().astype(int)
M = M.loc[M.sum(axis=1).sort_values(ascending=False).index, M.sum(axis=0).sort_values(ascending=False).index]
ax[0].imshow(M.values, aspect="auto", cmap="Blues", interpolation="nearest")
ax[0].set_title(f"User x film matrix ({M.shape[0]}x{M.shape[1]}, {100*M.values.mean():.1f}% observed)")
ax[0].set_xlabel("films (most -> least rated)"); ax[0].set_ylabel("users (most -> least active)")
mo = clean.set_index("timestamp").resample("MS").size().iloc[:-1]   # drop Dec-2025 (only 1 day of data)
ax[1].plot(mo.index, mo.values, "o-", color="#4c78a8"); ax[1].set_title("Ratings per month (Dec 2025 omitted: 1 day only)"); ax[1].set_ylim(0)
plt.tight_layout(); plt.savefig("figures/04_sparsity_and_time.png"); plt.close()

# ---- numbers -----------------------------------------------------------------------------------
summary = dict(
    supplied_rows=len(orig), supplied_users=int(orig.user_id.nunique()), supplied_films=int(orig.movie_title.nunique()),
    supplied_ratings_per_user_mean=round(len(orig) / orig.user_id.nunique(), 2),
    supplied_duplicate_user_film_pairs=int(orig.duplicated(["user_id", "movie_title"]).sum()),
    supplied_distinct_genres_per_title=int(meta.genres.min()), supplied_distinct_years_per_title=int(meta.years.min()),
    clean_rows=len(clean), clean_users=int(clean.user_id.nunique()), clean_films=int(clean.movie_id.nunique()),
    sparsity_pct=round(100 * (1 - M.values.mean()), 2), rating_mean=round(clean.rating.mean(), 3), rating_sd=round(clean.rating.std(), 3),
    share_ratings_ge_4=round(float((clean.rating >= 4).mean()), 3), top10_films_share_of_ratings=round(float(pm.n.head(10).sum() / pm.n.sum()), 3),
    users_with_le3_ratings=int((per_user <= 3).sum()),
    language_share_of_ratings=lang_share.round(3).to_dict(), language_share_of_catalogue=cat.round(3).to_dict())
json.dump(summary, open("results/eda_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
