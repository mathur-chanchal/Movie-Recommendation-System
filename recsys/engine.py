"""Serving-time recommender: loads the cleaned data, fits the models once, answers queries.

Methods
  hybrid      ItemKNN rating prediction + popularity prior   (default - best top-N ranking in evaluation)
  itemknn     item-based CF
  userknn     user-based CF
  svd         biased matrix factorisation
  popularity  most-rated films (also the cold-start fallback)

`diversity` in [0, 1] controls MMR re-ranking: 0 = plain ranking, 1 = strongly diversified (lambda = 1 - diversity).
"""
import json
import os
import numpy as np
import pandas as pd

from .models import (Interactions, ItemKNN, ItemKNNPop, UserKNN, SVD, Popularity, BiasBaseline,
                     metadata_similarity, mmr_rerank)

METHODS = ("hybrid", "itemknn", "userknn", "svd", "popularity")


class RecommenderEngine:
    def __init__(self, data_dir="data", results_dir="results"):
        self.df = pd.read_csv(os.path.join(data_dir, "dataset_clean.csv"))
        cat = pd.read_csv(os.path.join(data_dir, "movie_catalog.csv")).set_index("movie_id")
        self.ix = Interactions(self.df)
        self.cat = cat.loc[self.ix.item_ids]
        self.titles = self.cat.movie_title.to_numpy()
        self.R, self.M = self.ix.to_matrices(self.df)
        try:                                           # hyper-parameters tuned in 04_evaluate.py
            self.params = json.load(open(os.path.join(results_dir, "metrics.json")))["best_params"]
        except FileNotFoundError:
            self.params = {}
        p = self.params
        self.models = {
            "hybrid": ItemKNNPop(**p.get("ItemKNN+Pop", {})).fit(self.R, self.M),
            "itemknn": ItemKNN(**p.get("ItemKNN", {})).fit(self.R, self.M),
            "userknn": UserKNN(**p.get("UserKNN", {})).fit(self.R, self.M),
            "svd": SVD(**p.get("SVD", {})).fit(self.R, self.M),
            "popularity": Popularity().fit(self.R, self.M),
        }
        self.scores = {k: m.score_all() for k, m in self.models.items()}
        self.pred = {k: m.predict_all() for k, m in self.models.items()}
        self.item_sim = metadata_similarity(self.cat.genre, self.cat.language)
        self.knn = self.models["itemknn"]              # similarity matrix + biases for explanations / new users
        self.base = self.knn.base
        self.count = self.M.sum(0)

    # ------------------------------------------------------------------ helpers
    def _movie(self, j, **extra):
        r = self.cat.iloc[j]
        return dict(movie_id=int(self.ix.item_ids[j]), title=r.movie_title, genre=r.genre, language=r.language,
                    release_year=int(r.release_year), **extra)

    def has_user(self, user_id):
        return user_id in self.ix.u_map and self.M[self.ix.u_map[user_id]].any()

    def _rank(self, scores_u, exclude, k, diversity):
        cand = ~exclude
        if diversity > 0:
            return mmr_rerank(scores_u, cand, self.item_sim, k, 1.0 - diversity)
        s = np.where(cand, scores_u, -np.inf)
        order = np.argsort(-s)
        return order[np.isfinite(s[order])][:k]

    def _reason(self, u, j):
        """'Because you rated X' - the already-rated film that contributes most to the prediction for j."""
        rated = np.flatnonzero(self.M[u])
        if rated.size == 0:
            return "Popular with other viewers"
        resid = self.R[u, rated] - self.base.predict_all()[u, rated]
        contrib = np.clip(self.knn.sim[j, rated], 0, None) * resid     # same positive-similarity rule as the model
        t = int(np.argmax(contrib))
        if contrib[t] <= 0:
            return "Popular with viewers whose taste is similar to yours"
        return f"Because you rated '{self.titles[rated[t]]}' {self.R[u, rated[t]]:g}"

    # ------------------------------------------------------------------ public API
    def recommend(self, user_id, k=10, method="hybrid", diversity=0.0):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not self.has_user(user_id):                 # cold start: popularity, optionally diversified
            u_scores, excl, method_used, cold = self.scores["popularity"][0], np.zeros(self.ix.n_items, bool), "popularity", True
            u = None
        else:
            u = self.ix.u_map[user_id]
            u_scores, excl, method_used, cold = self.scores[method][u], self.M[u], method, False
        picks = self._rank(u_scores, excl, k, diversity)
        out = []
        for rank, j in enumerate(picks, 1):
            out.append(self._movie(j, rank=rank, score=round(float(u_scores[j]), 4),
                                   predicted_rating=None if cold else round(float(self.pred[method_used][u, j]), 2),
                                   reason="Popular with other viewers" if cold else self._reason(u, j)))
        return dict(user_id=user_id, method=method_used, diversity=diversity, cold_start=cold,
                    n_ratings_known=0 if cold else int(self.M[u].sum()), recommendations=out)

    def recommend_for_new_user(self, ratings, k=10, diversity=0.0):
        """ratings: {movie_id: rating} for a user who is not in the data. Uses item-item similarity."""
        idx = {self.ix.i_map[int(m)]: float(r) for m, r in ratings.items() if int(m) in self.ix.i_map}
        if not idx:
            return self.recommend(-1, k, "popularity", diversity)
        rated = np.array(list(idx))
        r = np.array(list(idx.values()))
        mu, bi = self.base.mu, self.base.bi
        bu = (r - mu - bi[rated]).sum() / (len(r) + 15.0)                       # shrunk user bias
        resid = r - (mu + bu + bi[rated])
        S = np.clip(self.knn.sim[:, rated], 0, None)
        top = np.argsort(-S, axis=1)[:, :self.knn.k]
        w = np.take_along_axis(S, top, 1)
        pred = np.clip(mu + bu + bi + (w * resid[top]).sum(1) / (w.sum(1) + self.knn.damping), 1, 5)
        z = (pred - self.pred["hybrid"].mean()) / (self.pred["hybrid"].std() + 1e-12)
        lp = np.log1p(self.count)
        scores = z + self.models["hybrid"].beta * (lp - lp.mean()) / (lp.std() + 1e-12)
        excl = np.zeros(self.ix.n_items, bool)
        excl[rated] = True
        picks = self._rank(scores, excl, k, diversity)
        out = []
        for rank, j in enumerate(picks, 1):
            c = S[j] * resid
            reason = (f"Because you rated '{self.titles[rated[int(np.argmax(c))]]}' {r[int(np.argmax(c))]:g}"
                      if c.max() > 0 else "Popular with other viewers")
            out.append(self._movie(j, rank=rank, score=round(float(scores[j]), 4), predicted_rating=round(float(pred[j]), 2), reason=reason))
        return dict(user_id=None, method="hybrid (new user)", diversity=diversity, cold_start=False,
                    n_ratings_known=len(idx), recommendations=out)

    def similar(self, movie_id, k=5):
        if movie_id not in self.ix.i_map:
            raise KeyError(movie_id)
        j = self.ix.i_map[movie_id]
        s = self.knn.sim[j].copy()
        s[j] = -np.inf
        return [self._movie(i, similarity=round(float(s[i]), 3)) for i in np.argsort(-s)[:k]]

    def movies(self, language=None, genre=None):
        rows = []
        for j in range(self.ix.n_items):
            m = self._movie(j, n_ratings=int(self.count[j]), mean_rating=round(float((self.R[:, j] * self.M[:, j]).sum() / max(self.count[j], 1)), 2))
            if (language is None or m["language"].lower() == language.lower()) and (genre is None or m["genre"].lower() == genre.lower()):
                rows.append(m)
        return rows

    def history(self, user_id):
        if user_id not in self.ix.u_map:
            raise KeyError(user_id)
        u = self.ix.u_map[user_id]
        return [self._movie(j, rating=float(self.R[u, j])) for j in np.flatnonzero(self.M[u])]
