"""Run:  pytest -q      (from the project root)"""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recsys.metrics import gini, ranking_per_user, top_k_lists, rmse, diversity_report
from recsys.models import BiasBaseline, GlobalMean, ItemKNN, UserKNN, mmr_rerank, metadata_similarity
from recsys.engine import RecommenderEngine
import app as appmod


# ----------------------------------------------------------------------------- data
def test_clean_data_invariants():
    d = pd.read_csv("data/dataset_clean.csv")
    assert d.rating.between(1, 5).all() and ((d.rating * 2) % 1 == 0).all()
    assert not d.duplicated(["user_id", "movie_id"]).any()
    assert pd.to_datetime(d.timestamp).max() <= pd.Timestamp("2025-12-01 23:59:59")
    assert d.groupby("movie_title")[["genre", "language", "release_year"]].nunique().max().max() == 1
    assert d.isna().sum().sum() == 0


def test_extended_dataset_matches_spec():
    e = pd.read_csv("data/dataset_extended.csv")
    o = pd.read_csv("data/dataset_original.csv")
    assert len(e) == 5000 and list(e.columns) == list(o.columns)
    assert pd.to_datetime(e.timestamp).is_monotonic_increasing
    lab = pd.read_csv("data/anomaly_labels.csv")
    assert 0.02 <= len(lab) / len(e) <= 0.05


# ----------------------------------------------------------------------------- metrics (hand-computed)
def test_ranking_metrics_hand_example():
    # 1 user, 4 items. test: item0=5 (relevant), item1=2 (not), item2 unseen-in-test. train excludes item3.
    test_R = np.array([[5.0, 2.0, 0, 0]]); test_M = np.array([[True, True, False, False]])
    lists = np.array([[2, 0, 1]])                          # relevant item sits at rank 2
    m = ranking_per_user(lists, test_R, test_M, 3)
    assert m["precision"][0] == pytest.approx(1 / 3)
    assert m["recall"][0] == 1.0 and m["hit_rate"][0] == 1.0
    assert m["ndcg"][0] == pytest.approx((1 / np.log2(3)) / 1.0)


def test_top_k_excludes_train_items():
    scores = np.array([[0.9, 0.8, 0.7, 0.1]]); train = np.array([[True, False, False, False]])
    assert 0 not in top_k_lists(scores, train, 3)[0]


def test_gini_and_rmse():
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75)
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))


def test_diversity_report_uniform_vs_collapsed():
    genre = np.array(["A", "B", "C", "D"]); lang = np.array(["x", "y", "x", "y"])
    uniform = diversity_report(np.array([[0, 1], [2, 3]]), genre, lang, np.full(4, .25), 4)
    collapsed = diversity_report(np.array([[0, 1], [0, 1]]), genre, lang, np.full(4, .25), 4)
    assert uniform["coverage"] == 1.0 and collapsed["coverage"] == 0.5
    assert uniform["gini"] < collapsed["gini"]


# ----------------------------------------------------------------------------- models
@pytest.fixture(scope="module")
def eng():
    return RecommenderEngine()


def test_models_beat_global_mean_and_stay_in_range(eng):
    R, M = eng.R, eng.M
    rng = np.random.default_rng(0)
    hold = M & (rng.random(M.shape) < 0.2)
    tr = M & ~hold
    gm = GlobalMean().fit(R, tr).predict_all()[hold]
    for model in (BiasBaseline(), ItemKNN(), UserKNN()):
        p = model.fit(R, tr).predict_all()
        assert p.min() >= 1 and p.max() <= 5
        assert rmse(R[hold], p[hold]) < rmse(R[hold], gm)


def test_mmr_lambda_one_equals_topk_and_diversifies(eng):
    s = eng.scores["hybrid"][0]
    mask = np.ones_like(s, bool)
    assert list(mmr_rerank(s, mask, eng.item_sim, 5, 1.0)) == list(np.argsort(-s)[:5])
    div = mmr_rerank(s, mask, eng.item_sim, 5, 0.2)
    assert eng.item_sim[np.ix_(div, div)].mean() <= eng.item_sim[np.ix_(np.argsort(-s)[:5], np.argsort(-s)[:5])].mean()


# ----------------------------------------------------------------------------- engine
def test_recommendations_exclude_seen_and_respect_k(eng):
    for uid in (7, 42, 250):
        seen = {h["movie_id"] for h in eng.history(uid)}
        for method in ("hybrid", "itemknn", "userknn", "svd", "popularity"):
            for div in (0.0, 0.6):
                r = eng.recommend(uid, 8, method, div)["recommendations"]
                assert len(r) == 8 and not seen & {x["movie_id"] for x in r}
                assert len({x["movie_id"] for x in r}) == 8


def test_cold_start_unknown_user(eng):
    r = eng.recommend(999999, 5)
    assert r["cold_start"] and len(r["recommendations"]) == 5 and r["method"] == "popularity"


def test_diversity_slider_increases_list_diversity(eng):
    def avg_ild(div):
        v = []
        for uid in eng.ix.user_ids[:120]:
            g = [x["genre"] for x in eng.recommend(int(uid), 10, "hybrid", div)["recommendations"]]
            v.append(1 - sum(a == b for i, a in enumerate(g) for b in g[i + 1:]) / 45)
        return np.mean(v)
    assert avg_ild(0.8) > avg_ild(0.0)


def test_new_user_and_similar(eng):
    r = eng.recommend_for_new_user({1: 5, 3: 4.5, 4: 5}, 5, 0.3)
    ids = {x["movie_id"] for x in r["recommendations"]}
    assert len(ids) == 5 and not ids & {1, 3, 4}
    sim = eng.similar(1, 5)
    assert len(sim) == 5 and 1 not in {x["movie_id"] for x in sim}


# ----------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    return appmod.app.test_client()


def test_api_happy_paths(client):
    assert client.get("/api/health").json["status"] == "ok"
    assert len(client.get("/api/movies").json) == 62
    assert len(client.get("/api/movies?language=Malayalam").json) == 10
    r = client.get("/api/recommend/7?k=3&diversity=0.5")
    assert r.status_code == 200 and len(r.json["recommendations"]) == 3
    assert client.get("/").status_code == 200
    assert client.get("/api/metrics").status_code == 200
    p = client.post("/api/recommend/new", json={"ratings": {"1": 5, "3": 4}, "k": 4})
    assert p.status_code == 200 and len(p.json["recommendations"]) == 4


@pytest.mark.parametrize("url", ["/api/recommend/7?k=0", "/api/recommend/7?k=abc", "/api/recommend/7?k=500",
                                 "/api/recommend/7?diversity=2", "/api/recommend/7?method=magic"])
def test_api_rejects_bad_params(client, url):
    assert client.get(url).status_code == 400


def test_api_errors(client):
    assert client.get("/api/movies/9999/similar").status_code == 404
    assert client.get("/api/users/999999/history").status_code == 404
    assert client.post("/api/recommend/new", json={}).status_code == 400
    assert client.post("/api/recommend/new", json={"ratings": {"1": 9}}).status_code == 400
    assert client.post("/api/recommend/new", json={"ratings": {"9999": 4}}).status_code == 404


def test_explanations_only_cite_films_the_user_liked(eng):
    """Regression: an explanation must never cite a film rated BELOW the model's own expectation for the user."""
    import re
    checked = 0
    for uid in eng.ix.user_ids[:150]:
        u = eng.ix.u_map[int(uid)]
        for rec in eng.recommend(int(uid), 10, "hybrid", 0.3)["recommendations"]:
            m = re.match(r"Because you rated '(.+)' ([\d.]+)$", rec["reason"])
            if m:
                j = int(np.flatnonzero(eng.titles == m.group(1))[0])
                assert eng.M[u, j] and eng.R[u, j] > eng.base.predict_all()[u, j]
                checked += 1
    assert checked > 100


# ----------------------------------------------------------------------------- Streamlit UI
def test_streamlit_ui_runs_in_all_modes():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(os.path.abspath("app_ui.py"), default_timeout=120).run()
    assert not at.exception
    cards = lambda: sum('class="card"' in m.value for m in at.markdown)
    assert cards() == 10
    for method in ("itemknn", "userknn", "svd", "popularity"):
        at.session_state["method"] = method
        at.run()
        assert not at.exception and cards() == 10
    at.radio[0].set_value("New user").run()
    assert not at.exception and cards() == 0                      # asks the user to pick films
    next(b for b in at.button if "sample" in b.label).click().run()
    assert not at.exception and cards() == 10
