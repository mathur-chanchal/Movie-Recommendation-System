"""Accuracy, ranking and diversity metrics."""
import numpy as np


# ----------------------------------------------------------------------------- rating prediction
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


# ----------------------------------------------------------------------------- top-K lists
def top_k_lists(scores, train_mask, k):
    """Top-k unseen items per user (items already rated in training are excluded)."""
    s = np.where(train_mask, -np.inf, scores)
    # tiny deterministic tie-break by item index so equal scores don't depend on sort internals
    s = s - np.arange(s.shape[1])[None, :] * 1e-9
    idx = np.argsort(-s, axis=1)[:, :k]
    return idx


def ranking_per_user(lists, test_R, test_M, k, rel_threshold=4.0):
    """Per-user Precision@k, Recall@k, HitRate@k, NDCG@k. A test item is 'relevant' if rating >= threshold.
    Users with no relevant test item are excluded (returned as NaN)."""
    rel = test_M & (test_R >= rel_threshold)
    n_rel = rel.sum(1)
    hits = np.take_along_axis(rel, lists[:, :k], 1)                      # (users, k) booleans
    prec = hits.sum(1) / k
    rec = np.where(n_rel > 0, hits.sum(1) / np.maximum(n_rel, 1), np.nan)
    hit = (hits.sum(1) > 0).astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    ideal = np.array([disc[:min(n, k)].sum() for n in n_rel])
    ndcg = np.where(n_rel > 0, dcg / np.maximum(ideal, 1e-12), np.nan)
    valid = n_rel > 0
    out = {n: np.where(valid, v, np.nan) for n, v in
           dict(precision=prec, recall=rec, hit_rate=hit, ndcg=ndcg).items()}
    return out


def summarise(per_user):
    return {m: float(np.nanmean(v)) for m, v in per_user.items()}


# ----------------------------------------------------------------------------- diversity / filter-bubble
def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _entropy(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def diversity_report(lists, genre, language, item_pop, n_items, hindi="Hindi"):
    """Metrics describing the *set* of recommendations (not accuracy).

    coverage        share of the catalogue that appears in at least one user's list
    gini            inequality of item exposure (0 = every film shown equally, 1 = one film gets everything)
    ild_genre/lang  intra-list diversity: avg. share of item pairs in a list that DIFFER in genre / language
    entropy_*       Shannon entropy of the genre / language mix across all lists (bits)
    novelty         mean self-information  -log2(popularity)  (higher = less mainstream picks)
    avg_popularity  mean share of users who rated a recommended film
    regional_share  share of recommended films that are not Hindi
    """
    genre, language = np.asarray(genre), np.asarray(language)
    n_users, k = lists.shape
    flat = lists.ravel()
    exposure = np.bincount(flat, minlength=n_items)

    def ild(attr):
        vals = []
        for row in lists:
            a = attr[row]
            same = (a[:, None] == a[None, :]).sum() - len(a)
            vals.append(1 - same / (len(a) * (len(a) - 1)))
        return float(np.mean(vals))

    pop = np.clip(item_pop, 1e-6, None)
    return dict(coverage=float((exposure > 0).mean()), gini=gini(exposure),
                ild_genre=ild(genre), ild_language=ild(language),
                entropy_genre=_entropy(genre[flat]), entropy_language=_entropy(language[flat]),
                novelty=float(np.mean(-np.log2(pop[flat]))), avg_popularity=float(np.mean(item_pop[flat])),
                regional_share=float(np.mean(language[flat] != hindi)))


def bootstrap_ci(values, n_boot=500, seed=0, alpha=0.05):
    """Percentile CI of the mean over users (NaNs ignored)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
