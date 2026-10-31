"""Step 4 - tune, evaluate and audit the recommenders.

Protocol
  1. Random 80/20 hold-out split of ratings (seed 42). The 20 % TEST set is not touched until step 3.
  2. Hyper-parameters tuned by 5-fold CV (RMSE) on the 80 % TRAIN part only.
  3. Models refit on full TRAIN, scored once on TEST.
       rating accuracy : RMSE, MAE (95 % bootstrap CI over test ratings)
       ranking         : Precision/Recall/HitRate/NDCG @5,@10. "Relevant" = test rating >= 4.0.
                         Candidates = every film the user did not rate in TRAIN. 95 % CI over users.
  4. Diversity / filter-bubble metrics of the top-10 lists + MMR re-ranking trade-off curve.
  5. Group audit: error and hit-rate by user activity, user's dominant language, item popularity.
  6. Cross-check against scikit-surprise (if installed).
"""
import itertools, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recsys.models import (Interactions, GlobalMean, BiasBaseline, UserKNN, ItemKNN, ItemKNNPop, SVD, Popularity,
                           metadata_similarity, mmr_rerank)
from recsys.metrics import (rmse, mae, top_k_lists, ranking_per_user, summarise, diversity_report, bootstrap_ci)

SEED, KS, REL_THR = 42, (5, 10), 4.0
t_start = time.time()

df = pd.read_csv("data/dataset_clean.csv")
cat = pd.read_csv("data/movie_catalog.csv").set_index("movie_id")
ix = Interactions(df)
genre = cat.loc[ix.item_ids, "genre"].to_numpy()
language = cat.loc[ix.item_ids, "language"].to_numpy()

# ------------------------------------------------------------------ 1. split
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(df))
cut = int(0.8 * len(df))
train, test = df.iloc[perm[:cut]].reset_index(drop=True), df.iloc[perm[cut:]].reset_index(drop=True)
R_tr, M_tr = ix.to_matrices(train)
R_te, M_te = ix.to_matrices(test)
tu, ti, ty = ix.encode(test)
print(f"train {len(train)} | test {len(test)} | users {ix.n_users} | movies {ix.n_items}")

# ------------------------------------------------------------------ 2. tuning (5-fold CV on TRAIN)
GRIDS = {
    "BiasBaseline": (BiasBaseline, dict(reg_i=[5, 10, 20], reg_u=[5, 15, 30])),
    "UserKNN": (UserKNN, dict(k=[10, 20, 40], shrink=[2, 5, 15], damping=[0.5, 2.0])),
    "ItemKNN": (ItemKNN, dict(k=[5, 10, 20, 40], shrink=[2, 5, 15], damping=[0.5, 2.0])),
    "SVD": (SVD, dict(n_factors=[10, 20], reg=[0.1, 0.2, 0.5, 1.0], reg_bias=[0.02, 0.05], n_epochs=[30])),
}
folds = np.array_split(rng.permutation(len(train)), 5)


def cv_rmse(cls, params):
    errs = []
    for f in range(5):
        va = train.iloc[folds[f]]
        tr = train.iloc[np.concatenate([folds[g] for g in range(5) if g != f])]
        R, M = ix.to_matrices(tr)
        m = cls(**params).fit(R, M)
        u, i, y = ix.encode(va)
        errs.append(rmse(y, m.predict_all()[u, i]))
    return float(np.mean(errs)), float(np.std(errs))


best, tuning_rows = {}, []
for name, (cls, grid) in GRIDS.items():
    keys = list(grid)
    res = []
    for vals in itertools.product(*grid.values()):
        params = dict(zip(keys, vals))
        mean, sd = cv_rmse(cls, params)
        res.append((mean, params))
        tuning_rows.append(dict(model=name, **params, cv_rmse=round(mean, 4), cv_rmse_sd=round(sd, 4)))
    best[name] = min(res, key=lambda x: x[0])[1]
    print(f"tuned {name:13s} -> {best[name]}  CV-RMSE {min(res, key=lambda x: x[0])[0]:.4f}")
pd.DataFrame(tuning_rows).to_csv("results/tuning_grid.csv", index=False)
gm_cv = np.mean([rmse(ix.encode(train.iloc[folds[f]])[2],
                      GlobalMean().fit(*ix.to_matrices(train.iloc[np.concatenate([folds[g] for g in range(5) if g != f])]))
                      .predict_all()[ix.encode(train.iloc[folds[f]])[0], ix.encode(train.iloc[folds[f]])[1]])
                 for f in range(5)])

# ---- model selection for TOP-N ranking, also by CV on TRAIN (never on test)
BETAS = [0, 0.25, 0.5, 1.0, 2.0, 4.0]
cv_ndcg = {n: [] for n in ["UserKNN", "ItemKNN", "SVD"]} | {f"ItemKNN+Pop(b={b})": [] for b in BETAS if b > 0}
for f in range(5):
    va = train.iloc[folds[f]]
    tr = train.iloc[np.concatenate([folds[g] for g in range(5) if g != f])]
    Rf, Mf = ix.to_matrices(tr); Rv, Mv = ix.to_matrices(va)
    for n in ["UserKNN", "ItemKNN", "SVD"]:
        m = GRIDS[n][0](**best[n]).fit(Rf, Mf)
        cv_ndcg[n].append(np.nanmean(ranking_per_user(top_k_lists(m.score_all(), Mf, 10), Rv, Mv, 10, REL_THR)["ndcg"]))
    for b in BETAS:
        if b > 0:
            m = ItemKNNPop(**best["ItemKNN"], beta=b).fit(Rf, Mf)
            cv_ndcg[f"ItemKNN+Pop(b={b})"].append(np.nanmean(ranking_per_user(top_k_lists(m.score_all(), Mf, 10), Rv, Mv, 10, REL_THR)["ndcg"]))
cv_ndcg = {k: float(np.mean(v)) for k, v in cv_ndcg.items()}
best_beta = max([b for b in BETAS if b > 0], key=lambda b: cv_ndcg[f"ItemKNN+Pop(b={b})"])
best["ItemKNN+Pop"] = dict(**best["ItemKNN"], beta=best_beta)
cv_ndcg_sel = {"UserKNN": cv_ndcg["UserKNN"], "ItemKNN": cv_ndcg["ItemKNN"], "SVD": cv_ndcg["SVD"],
               "ItemKNN+Pop": cv_ndcg[f"ItemKNN+Pop(b={best_beta})"]}
best_rank_model = max(cv_ndcg_sel, key=cv_ndcg_sel.get)
print("CV NDCG@10 (train only):", {k: round(v, 4) for k, v in cv_ndcg.items()}, "->", best_rank_model)

# ------------------------------------------------------------------ 3. final fit + test
models = {"GlobalMean": GlobalMean().fit(R_tr, M_tr), "Popularity": Popularity().fit(R_tr, M_tr)}
for name, (cls, _) in GRIDS.items():
    models[name] = cls(**best[name]).fit(R_tr, M_tr)
models["ItemKNN+Pop"] = ItemKNNPop(**best["ItemKNN+Pop"]).fit(R_tr, M_tr)

pop_share = M_tr.sum(0) / ix.n_users                         # share of users who rated each film (train)
results, per_user_store, lists_store = {}, {}, {}
rnd_scores = np.random.default_rng(SEED).random((ix.n_users, ix.n_items))

for name, m in list(models.items()) + [("Random", None)]:
    row = {}
    if m is not None and name != "Popularity":
        pred = m.predict_all()[tu, ti]
        lo, hi = bootstrap_ci((ty - pred) ** 2, seed=1)
        row.update(rmse=rmse(ty, pred), rmse_ci=(float(np.sqrt(lo)), float(np.sqrt(hi))), mae=mae(ty, pred))
    elif name == "Popularity":
        pred = m.predict_all()[tu, ti]
        row.update(rmse=rmse(ty, pred), mae=mae(ty, pred))
    if name == "GlobalMean":       # constant scores = no ranking; index tie-break would just return the catalogue's first (most famous) films
        results[name] = row
        continue
    scores = rnd_scores if m is None else m.score_all()
    for k in KS:
        L = top_k_lists(scores, M_tr, k)
        pu = ranking_per_user(L, R_te, M_te, k, REL_THR)
        for metric, vals in pu.items():
            row[f"{metric}@{k}"] = float(np.nanmean(vals))
            row[f"{metric}@{k}_ci"] = bootstrap_ci(vals, seed=2)
        if k == 10:
            per_user_store[name] = pu
            lists_store[name] = L
            row["diversity@10"] = diversity_report(L, genre, language, pop_share, ix.n_items)
    results[name] = row

n_eval_users = int((~np.isnan(per_user_store["Random"]["precision"])).sum())
print(f"ranking evaluated on {n_eval_users} users with >=1 relevant test item")

# paired bootstrap: is the best personalised model really better than the bias baseline?
def paired_rmse_diff(a, b, n_boot=1000):
    ea = (ty - models[a].predict_all()[tu, ti]) ** 2
    eb = (ty - models[b].predict_all()[tu, ti]) ** 2
    r = np.random.default_rng(3)
    d = []
    for _ in range(n_boot):
        s = r.integers(0, len(ty), len(ty))
        d.append(np.sqrt(ea[s].mean()) - np.sqrt(eb[s].mean()))
    return float(np.mean(d)), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


personalised = ["UserKNN", "ItemKNN", "SVD"]      # rating-prediction comparison
paired = {m: paired_rmse_diff(m, "BiasBaseline") for m in personalised}
best_rating_model = min(personalised, key=lambda n: results[n]["rmse"])

# paired bootstrap over users for NDCG@10 differences
def paired_ndcg_diff(a, b, n_boot=2000):
    x, y = per_user_store[a]["ndcg"], per_user_store[b]["ndcg"]
    ok = ~np.isnan(x)
    d = (x - y)[ok]
    r = np.random.default_rng(4)
    bs = [r.choice(d, len(d)).mean() for _ in range(n_boot)]
    return float(d.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


paired_ndcg = {f"{a} - {b}": paired_ndcg_diff(a, b) for a, b in
               [("ItemKNN+Pop", "Popularity"), ("ItemKNN", "BiasBaseline"), ("ItemKNN", "Random"), ("SVD", "BiasBaseline")]}

# ------------------------------------------------------------------ 4. diversity: MMR re-ranking
sim = metadata_similarity(genre, language)
mmr_rows = []
base_scores = models[best_rank_model].score_all()
for lam in [1.0, 0.8, 0.6, 0.4, 0.2]:
    L = np.array([mmr_rerank(base_scores[u], ~M_tr[u], sim, 10, lam) for u in range(ix.n_users)])
    pu = ranking_per_user(L, R_te, M_te, 10, REL_THR)
    d = diversity_report(L, genre, language, pop_share, ix.n_items)
    mmr_rows.append(dict(lam=lam, **{f"{k}@10": float(np.nanmean(v)) for k, v in pu.items()}, **d))
mmr = pd.DataFrame(mmr_rows)
mmr_lang = {}
for lam in [1.0, 0.6, 0.4]:
    Lx = np.array([mmr_rerank(base_scores[u], ~M_tr[u], sim, 10, lam) for u in range(ix.n_users)])
    mmr_lang[f"recs_{best_rank_model}_MMR_lam{lam}"] = pd.Series(language[Lx.ravel()]).value_counts(normalize=True)
mmr.to_csv("results/diversity_tradeoff.csv", index=False)

# ------------------------------------------------------------------ 5. group audit (best rating model)
bm = models[best_rating_model]
err = (ty - bm.predict_all()[tu, ti]) ** 2
n_tr_user = M_tr.sum(1)
Lmat = np.stack([(language == l).astype(float) for l in sorted(set(language))], 1)
dom_lang = np.where(n_tr_user > 0, np.array(sorted(set(language)))[(M_tr @ Lmat).argmax(1)], "no_history")
act_q = np.quantile(n_tr_user[n_tr_user > 0], [1 / 3, 2 / 3])
act_grp = np.where(n_tr_user <= act_q[0], "light", np.where(n_tr_user <= act_q[1], "medium", "heavy"))
ipop = M_tr.sum(0)
pq = np.quantile(ipop, [1 / 3, 2 / 3])
pop_grp = np.where(ipop <= pq[0], "tail", np.where(ipop <= pq[1], "mid", "head"))
audit_rows = []
for gname, garr, idx_on in [("user_activity", act_grp, tu), ("dominant_language", dom_lang, tu), ("item_popularity", pop_grp, ti)]:
    for g in sorted(set(garr)):
        sel = garr[idx_on] == g
        row = dict(group_type=gname, group=g, n_test_ratings=int(sel.sum()),
                   rmse=float(np.sqrt(err[sel].mean())) if sel.sum() else np.nan)
        if gname != "item_popularity":
            pu = per_user_store[best_rank_model]
            members = garr == g
            row.update(n_users=int(members.sum()), ndcg10=float(np.nanmean(pu["ndcg"][members])) if members.any() else np.nan,
                       hit_rate10=float(np.nanmean(pu["hit_rate"][members])) if members.any() else np.nan)
        audit_rows.append(row)
audit = pd.DataFrame(audit_rows)
audit.to_csv("results/group_audit.csv", index=False)

# share of top-10 recommendation slots by language vs the language mix of the training ratings
lang_mix = pd.DataFrame({
    "train_ratings_share": pd.Series(language[np.nonzero(M_tr)[1]]).value_counts(normalize=True),
    **{f"recs_{n}": pd.Series(language[lists_store[n].ravel()]).value_counts(normalize=True)
       for n in ["Popularity", "ItemKNN", best_rank_model] if n != best_rank_model or True}}).fillna(0).round(3)
lang_mix = lang_mix.join(pd.DataFrame(mmr_lang)).fillna(0).round(3)
lang_mix.to_csv("results/language_exposure.csv")

# ------------------------------------------------------------------ 6. Surprise cross-check
cross = {}
try:
    from surprise import Dataset, Reader, SVD as SSVD, BaselineOnly
    tr_s = Dataset.load_from_df(train[["user_id", "movie_id", "rating"]], Reader(rating_scale=(1, 5))).build_full_trainset()
    pairs = list(zip(test.user_id, test.movie_id, test.rating))
    p = best["SVD"]
    for nm, algo in [("Surprise BaselineOnly", BaselineOnly(verbose=False)),
                     ("Surprise SVD (same params)", SSVD(n_factors=p["n_factors"], n_epochs=p["n_epochs"], lr_all=0.01,
                                                         reg_all=p["reg"], random_state=SEED))]:
        algo.fit(tr_s)
        cross[nm] = rmse(ty, [algo.predict(u, i).est for u, i, _ in pairs])
except Exception as e:                                    # surprise is optional
    cross["skipped"] = str(e)

# ------------------------------------------------------------------ save
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return round(float(o), 4) if isinstance(o, (float, np.floating)) else o


summary = dict(split=dict(train=len(train), test=len(test), eval_users_for_ranking=n_eval_users, relevance_threshold=REL_THR),
               best_params=best, cv_ndcg10=cv_ndcg, global_mean_cv_rmse=round(float(gm_cv), 4), best_rating_model=best_rating_model,
               best_ranking_model=best_rank_model, test_results=results, paired_rmse_vs_BiasBaseline=paired, paired_ndcg10_diff=paired_ndcg,
               surprise_crosscheck_rmse=cross, runtime_sec=round(time.time() - t_start))
json.dump(clean(summary), open("results/metrics.json", "w"), indent=2)

tbl = pd.DataFrame({n: {"RMSE": r.get("rmse"), "MAE": r.get("mae"), **{k: r[k] for k in r if "@" in k and not k.endswith("_ci") and "diversity" not in k}}
                    for n, r in results.items()}).T.round(4)
tbl.to_csv("results/model_comparison.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n=== TEST RESULTS ===\n", tbl[["RMSE", "MAE", "precision@10", "recall@10", "hit_rate@10", "ndcg@10"]])
print("\npaired RMSE diff vs BiasBaseline (negative = better):", {k: (round(v[0], 4), tuple(round(x, 4) for x in v[1])) for k, v in paired.items()})
print("paired NDCG@10 diffs (mean, 95% CI):", {k: (round(v[0], 4), tuple(round(x, 4) for x in v[1])) for k, v in paired_ndcg.items()})
print("surprise cross-check:", cross)
print("\n=== DIVERSITY @10 ===")
print(pd.DataFrame({n: results[n]["diversity@10"] for n in results if "diversity@10" in results[n]}).T.round(3))
print("\n=== MMR trade-off ===\n", mmr[["lam", "precision@10", "ndcg@10", "hit_rate@10", "coverage", "ild_genre", "ild_language", "regional_share", "novelty"]].round(3))
print("\n=== GROUP AUDIT ===\n", audit.round(3))
print("\n=== LANGUAGE EXPOSURE ===\n", lang_mix)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
order = ["GlobalMean", "BiasBaseline", "UserKNN", "ItemKNN", "SVD"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
vals = [results[n]["rmse"] for n in order]
errs = [[results[n]["rmse"] - results[n]["rmse_ci"][0] for n in order], [results[n]["rmse_ci"][1] - results[n]["rmse"] for n in order]]
ax[0].bar(order, vals, yerr=errs, color=["#bbb", "#9ab", "#4c78a8", "#59a14f", "#e15759"], capsize=4)
ax[0].set_ylim(min(vals) - 0.08, max(vals) + 0.05); ax[0].set_ylabel("Test RMSE (lower is better)")
ax[0].set_title("Rating prediction (95% CI)"); ax[0].tick_params(axis="x", rotation=25)
rk = ["Random", "Popularity", "UserKNN", "ItemKNN", "SVD", "ItemKNN+Pop"]
ax[1].bar(rk, [results[n]["ndcg@10"] for n in rk],
          yerr=[[results[n]["ndcg@10"] - results[n]["ndcg@10_ci"][0] for n in rk], [results[n]["ndcg@10_ci"][1] - results[n]["ndcg@10"] for n in rk]],
          color=["#ddd", "#f28e2b", "#4c78a8", "#59a14f", "#e15759", "#b07aa1"], capsize=4)
ax[1].set_ylabel("NDCG@10 (higher is better)"); ax[1].set_title("Top-10 ranking (95% CI over users)"); ax[1].tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.savefig("figures/05_model_comparison.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, (col, lab) in zip(ax, [("ild_genre", "Genre diversity within a list (ILD)"), ("coverage", "Catalogue coverage"), ("regional_share", "Share of non-Hindi recommendations")]):
    a.plot(mmr[col], mmr["ndcg@10"], "o-", color="#e15759")
    for _, r in mmr.iterrows():
        a.annotate(f"λ={r.lam}", (r[col], r["ndcg@10"]), textcoords="offset points", xytext=(4, 5), fontsize=8)
    a.set_xlabel(lab); a.set_ylabel("NDCG@10")
plt.suptitle(f"Accuracy vs diversity trade-off (MMR re-ranking of {best_rank_model})", y=1.02)
plt.tight_layout(); plt.savefig("figures/06_diversity_tradeoff.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, gt in zip(ax, ["user_activity", "dominant_language", "item_popularity"]):
    s = audit[audit.group_type == gt]
    a.bar(s.group, s.rmse, color="#4c78a8"); a.set_title(gt.replace("_", " ")); a.set_ylim(0.6, max(audit.rmse) + 0.1)
    for x, (v, n) in enumerate(zip(s.rmse, s.n_test_ratings)):
        a.text(x, v + 0.01, f"n={n}", ha="center", fontsize=8)
ax[0].set_ylabel(f"Test RMSE ({best_rating_model})")
plt.suptitle("Error by group (fairness audit)", y=1.02); plt.tight_layout(); plt.savefig("figures/07_group_audit.png", bbox_inches="tight"); plt.close()
print(f"\nfinished in {time.time() - t_start:.0f}s")
