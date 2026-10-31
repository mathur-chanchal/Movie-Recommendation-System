"""Collaborative-filtering models (pure NumPy - no compiled dependencies).

All models share one interface, working on dense (users x items) matrices:

    model.fit(R, M)            R = ratings (0 where unknown), M = boolean "is observed" mask
    model.predict_all() -> (n_users, n_items) matrix of predicted ratings (clipped to 1..5)

Models
    GlobalMean      - predicts the training mean (floor baseline)
    BiasBaseline    - mu + b_u + b_i, regularised (strong baseline)
    UserKNN         - user-based CF on bias-removed residuals, shrunk cosine similarity
    ItemKNN         - item-based CF (adjusted cosine on residuals), shrunk
    SVD             - biased matrix factorisation trained with SGD (Funk-SVD / Koren), as in Surprise
    Popularity      - ranking-only baseline (score = number of training ratings)
"""
import numpy as np

R_MIN, R_MAX = 1.0, 5.0


class Interactions:
    """Maps raw user_id / movie_id values to dense matrix indices."""

    def __init__(self, df, user_ids=None, item_ids=None):
        self.user_ids = np.sort(df.user_id.unique()) if user_ids is None else np.asarray(user_ids)
        self.item_ids = np.sort(df.movie_id.unique()) if item_ids is None else np.asarray(item_ids)
        self.u_map = {u: k for k, u in enumerate(self.user_ids)}
        self.i_map = {i: k for k, i in enumerate(self.item_ids)}
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)

    def encode(self, df):
        return (df.user_id.map(self.u_map).to_numpy(), df.movie_id.map(self.i_map).to_numpy(),
                df.rating.to_numpy(float))

    def to_matrices(self, df):
        u, i, r = self.encode(df)
        R = np.zeros((self.n_users, self.n_items))
        M = np.zeros((self.n_users, self.n_items), bool)
        R[u, i] = r
        M[u, i] = True
        return R, M


# ----------------------------------------------------------------------------- baselines
class GlobalMean:
    name = "GlobalMean"

    def fit(self, R, M):
        self.mu, self.shape = R[M].mean(), R.shape
        return self

    def predict_all(self):
        return np.full(self.shape, self.mu)

    def score_all(self):
        return self.predict_all()


class BiasBaseline:
    """r_ui ~ mu + b_u + b_i with L2-regularised alternating estimation (Koren, 2010)."""
    name = "BiasBaseline"

    def __init__(self, reg_i=10.0, reg_u=15.0, n_iter=10):
        self.reg_i, self.reg_u, self.n_iter = reg_i, reg_u, n_iter

    def fit(self, R, M):
        self.mu = R[M].mean()
        bu, bi = np.zeros(R.shape[0]), np.zeros(R.shape[1])
        for _ in range(self.n_iter):
            bi = ((R - self.mu - bu[:, None]) * M).sum(0) / (self.reg_i + M.sum(0))
            bu = ((R - self.mu - bi[None, :]) * M).sum(1) / (self.reg_u + M.sum(1))
        self.bu, self.bi = bu, bi
        return self

    def predict_all(self):
        return np.clip(self.mu + self.bu[:, None] + self.bi[None, :], R_MIN, R_MAX)

    def score_all(self):
        return self.predict_all()


class Popularity:
    """Ranking baseline: recommend the most-rated unseen films. Rating prediction = item mean."""
    name = "Popularity"

    def fit(self, R, M):
        self.count = M.sum(0).astype(float)
        mu = R[M].mean()
        self.item_mean = np.where(M.sum(0) > 0, (R * M).sum(0) / np.maximum(M.sum(0), 1), mu)
        self.n_users = R.shape[0]
        return self

    def predict_all(self):
        return np.tile(self.item_mean, (self.n_users, 1))

    def score_all(self):
        return np.tile(self.count, (self.n_users, 1))


# ----------------------------------------------------------------------------- neighbourhood CF
def _shrunk_cosine(E, M, shrink):
    """Cosine similarity between rows of E over co-rated columns, shrunk by n_common/(n_common+shrink)."""
    Ec = E * M
    num = Ec @ Ec.T
    sq = Ec ** 2
    Mf = M.astype(float)
    den = np.sqrt((sq @ Mf.T) * (Mf @ sq.T))
    sim = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    n_common = Mf @ Mf.T
    sim *= n_common / (n_common + shrink)
    np.fill_diagonal(sim, 0.0)
    return sim


def _knn_residuals(E, M, S, k, damping):
    """For every (row a, column b): weighted mean of residuals E[v, b] over the k most similar rows v
    that observed column b. Positive similarities only; `damping` shrinks low-evidence predictions."""
    A, B = E.shape
    P = np.zeros((A, B))
    for b in range(B):
        raters = np.flatnonzero(M[:, b])
        if raters.size == 0:
            continue
        S_sub = np.clip(S[:, raters], 0, None)
        kk = min(k, raters.size)
        if kk < raters.size:
            idx = np.argpartition(-S_sub, kk - 1, axis=1)[:, :kk]
            sims = np.take_along_axis(S_sub, idx, 1)
            vals = E[raters, b][idx]
        else:
            sims, vals = S_sub, np.tile(E[raters, b], (A, 1))
        P[:, b] = (sims * vals).sum(1) / (sims.sum(1) + damping)
    return P


class UserKNN:
    name = "UserKNN"

    def __init__(self, k=30, shrink=5.0, damping=1.0):
        self.k, self.shrink, self.damping = k, shrink, damping

    def fit(self, R, M):
        self.base = BiasBaseline().fit(R, M)
        E = (R - self.base.predict_all()) * M
        S = _shrunk_cosine(E, M, self.shrink)
        self.resid = _knn_residuals(E, M, S, self.k, self.damping)
        self.sim = S
        return self

    def predict_all(self):
        return np.clip(self.base.predict_all() + self.resid, R_MIN, R_MAX)

    score_all = predict_all


class ItemKNN:
    name = "ItemKNN"

    def __init__(self, k=20, shrink=5.0, damping=1.0):
        self.k, self.shrink, self.damping = k, shrink, damping

    def fit(self, R, M):
        self.base = BiasBaseline().fit(R, M)
        E = (R - self.base.predict_all()) * M
        S = _shrunk_cosine(E.T, M.T, self.shrink)                  # item x item
        self.resid = _knn_residuals(E.T, M.T, S, self.k, self.damping).T
        self.sim = S
        return self

    def predict_all(self):
        return np.clip(self.base.predict_all() + self.resid, R_MIN, R_MAX)

    score_all = predict_all


class ItemKNNPop(ItemKNN):
    """Hybrid for TOP-N ranking: standardised ItemKNN predicted rating + beta * standardised log-popularity.
    Rating prediction is identical to ItemKNN; only score_all() (used for ranking) changes.
    Rationale: users mostly rate films they were exposed to, so 'will they watch it' depends on popularity as
    well as 'will they like it'. beta=0 recovers plain ItemKNN."""
    name = "ItemKNN+Pop"

    def __init__(self, k=20, shrink=5.0, damping=1.0, beta=1.0):
        super().__init__(k, shrink, damping)
        self.beta = beta

    def fit(self, R, M):
        super().fit(R, M)
        self.count = M.sum(0).astype(float)
        return self

    def score_all(self):
        P = self.predict_all()
        z_pred = (P - P.mean()) / (P.std() + 1e-12)
        lp = np.log1p(self.count)
        z_pop = (lp - lp.mean()) / (lp.std() + 1e-12)
        return z_pred + self.beta * z_pop[None, :]


# ----------------------------------------------------------------------------- matrix factorisation
class SVD:
    """Biased matrix factorisation:  r_ui ~ mu + b_u + b_i + p_u . q_i, trained by SGD.
    (This is the algorithm scikit-surprise calls `SVD`; it is NOT a plain linear-algebra SVD.)"""
    name = "SVD"

    def __init__(self, n_factors=20, n_epochs=30, lr=0.01, reg=0.05, reg_bias=None, seed=0):
        self.f, self.epochs, self.lr, self.reg, self.seed = n_factors, n_epochs, lr, reg, seed
        self.reg_bias = reg if reg_bias is None else reg_bias      # Koren: biases need less shrinkage than factors

    def fit(self, R, M):
        rng = np.random.default_rng(self.seed)
        nu, ni = R.shape
        u_idx, i_idx = np.nonzero(M)
        r = R[u_idx, i_idx]
        self.mu = r.mean()
        bu, bi = np.zeros(nu), np.zeros(ni)
        P = rng.normal(0, 0.1, (nu, self.f))
        Q = rng.normal(0, 0.1, (ni, self.f))
        lr, reg, rb = self.lr, self.reg, self.reg_bias
        self.history = []
        for _ in range(self.epochs):
            for n in rng.permutation(len(r)):
                u, i = u_idx[n], i_idx[n]
                err = r[n] - (self.mu + bu[u] + bi[i] + P[u] @ Q[i])
                bu[u] += lr * (err - rb * bu[u])
                bi[i] += lr * (err - rb * bi[i])
                pu = P[u].copy()
                P[u] += lr * (err * Q[i] - reg * P[u])
                Q[i] += lr * (err * pu - reg * Q[i])
            pred = np.clip(self.mu + bu[u_idx] + bi[i_idx] + (P[u_idx] * Q[i_idx]).sum(1), R_MIN, R_MAX)
            self.history.append(float(np.sqrt(np.mean((r - pred) ** 2))))
        self.bu, self.bi, self.P, self.Q = bu, bi, P, Q
        return self

    def predict_all(self):
        return np.clip(self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T, R_MIN, R_MAX)

    score_all = predict_all


# ----------------------------------------------------------------------------- diversity re-ranking
def metadata_similarity(genre, language):
    """Item-item similarity used for diversification: 0.5*same_genre + 0.5*same_language."""
    g, l = np.asarray(genre), np.asarray(language)
    return 0.5 * (g[:, None] == g[None, :]) + 0.5 * (l[:, None] == l[None, :])


def mmr_rerank(scores, candidate_mask, item_sim, k, lam):
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998).
    Greedily picks the item maximising  lam*relevance - (1-lam)*max_similarity_to_already_picked.
    lam=1 -> plain top-k;  lower lam -> more diverse lists. Relevance is min-max scaled per user."""
    cand = np.flatnonzero(candidate_mask)
    if cand.size == 0:
        return np.array([], int)
    s = scores[cand]
    rel = (s - s.min()) / (s.max() - s.min() + 1e-12)
    chosen, remaining = [], list(range(cand.size))
    while remaining and len(chosen) < k:
        if not chosen:
            j = remaining[int(np.argmax(rel[remaining]))]
        else:
            red = item_sim[np.ix_(cand[remaining], cand[chosen])].max(1)
            j = remaining[int(np.argmax(lam * rel[remaining] - (1 - lam) * red))]
        chosen.append(j)
        remaining.remove(j)
    return cand[chosen]
