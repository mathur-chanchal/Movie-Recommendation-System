"""Flask REST API + minimal web UI for the movie recommender.

    python app.py                      ->  http://127.0.0.1:5000

GET  /                                   web demo
GET  /api/health
GET  /api/movies?language=&genre=        catalogue with rating stats
GET  /api/users/<id>/history             a user's ratings
GET  /api/recommend/<user_id>?k=10&method=hybrid&diversity=0.4
POST /api/recommend/new                  {"ratings": {"1": 5, "7": 4}, "k": 10, "diversity": 0.4}   (new user, no id)
GET  /api/movies/<id>/similar?k=5
GET  /api/metrics                        evaluation summary produced by 04_evaluate.py
"""
import json
import os
from flask import Flask, jsonify, request, Response
from recsys.engine import RecommenderEngine, METHODS

app = Flask(__name__)
engine = RecommenderEngine()


def _err(msg, code=400):
    return jsonify(error=msg), code


def _int(name, default, lo, hi):
    try:
        v = int(request.args.get(name, default))
    except ValueError:
        raise ValueError(f"'{name}' must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"'{name}' must be between {lo} and {hi}")
    return v


def _div(raw):
    try:
        d = float(raw)
    except (TypeError, ValueError):
        raise ValueError("'diversity' must be a number between 0 and 1")
    if not 0 <= d <= 1:
        raise ValueError("'diversity' must be between 0 and 1")
    return d


@app.get("/api/health")
def health():
    return jsonify(status="ok", users=engine.ix.n_users, movies=engine.ix.n_items, ratings=int(engine.M.sum()),
                   methods=list(METHODS))


@app.get("/api/movies")
def movies():
    return jsonify(engine.movies(request.args.get("language"), request.args.get("genre")))


@app.get("/api/users/<int:user_id>/history")
def history(user_id):
    try:
        return jsonify(user_id=user_id, ratings=engine.history(user_id))
    except KeyError:
        return _err(f"unknown user {user_id}", 404)


@app.get("/api/recommend/<int:user_id>")
def recommend(user_id):
    try:
        k = _int("k", 10, 1, 50)
        d = _div(request.args.get("diversity", 0))
        method = request.args.get("method", "hybrid")
        if method not in METHODS:
            raise ValueError(f"'method' must be one of {list(METHODS)}")
    except ValueError as e:
        return _err(str(e))
    return jsonify(engine.recommend(user_id, k, method, d))


@app.post("/api/recommend/new")
def recommend_new():
    body = request.get_json(silent=True) or {}
    ratings = body.get("ratings")
    if not isinstance(ratings, dict) or not ratings:
        return _err("body must be JSON like {\"ratings\": {\"<movie_id>\": <1-5>, ...}}")
    try:
        clean = {int(m): float(r) for m, r in ratings.items()}
        if any(not 1 <= r <= 5 for r in clean.values()):
            raise ValueError("ratings must be between 1 and 5")
        k = int(body.get("k", 10))
        if not 1 <= k <= 50:
            raise ValueError("'k' must be between 1 and 50")
        d = _div(body.get("diversity", 0))
    except (ValueError, TypeError) as e:
        return _err(str(e))
    unknown = [m for m in clean if m not in engine.ix.i_map]
    if unknown:
        return _err(f"unknown movie_id(s): {unknown}", 404)
    return jsonify(engine.recommend_for_new_user(clean, k, d))


@app.get("/api/movies/<int:movie_id>/similar")
def similar(movie_id):
    try:
        k = _int("k", 5, 1, 30)
        return jsonify(movie_id=movie_id, similar=engine.similar(movie_id, k))
    except KeyError:
        return _err(f"unknown movie {movie_id}", 404)
    except ValueError as e:
        return _err(str(e))


@app.get("/api/metrics")
def metrics():
    path = os.path.join("results", "metrics.json")
    if not os.path.exists(path):
        return _err("run 04_evaluate.py first", 404)
    return jsonify(json.load(open(path)))


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Regional Movie Recommender</title>
<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#222}
label{margin-right:1rem}input,select{padding:.3rem}button{padding:.4rem 1rem}
table{border-collapse:collapse;width:100%;margin-top:1rem}td,th{border-bottom:1px solid #ddd;padding:.4rem;text-align:left;font-size:.92rem}
.tag{background:#eef;border-radius:4px;padding:1px 6px;font-size:.8rem}.muted{color:#777}</style>
<h2>Personalised Movie Recommender <span class=muted style="font-size:.6em">collaborative filtering - Indian cinema</span></h2>
<p>
<label>User ID <input id=u type=number value=7 min=1 style="width:5em"></label>
<label>Method <select id=m><option value=hybrid>hybrid (default)</option><option>itemknn</option><option>userknn</option><option>svd</option><option>popularity</option></select></label>
<label>Diversity <input id=d type=range min=0 max=1 step=0.1 value=0.4> <span id=dv>0.4</span></label>
<button onclick=go()>Recommend</button></p>
<div id=info class=muted></div><table id=t></table>
<script>
d.oninput=()=>dv.textContent=d.value;
async function go(){
 const r=await fetch(`/api/recommend/${u.value}?k=10&method=${m.value}&diversity=${d.value}`);const j=await r.json();
 if(!r.ok){info.textContent=j.error;t.innerHTML='';return}
 info.textContent=j.cold_start?'Unknown user - showing popular films (cold start).':`Based on ${j.n_ratings_known} ratings - method: ${j.method}`;
 t.innerHTML='<tr><th>#<th>Film<th>Language<th>Genre<th>Year<th>Why</tr>'+j.recommendations.map(x=>
 `<tr><td>${x.rank}<td>${x.title}<td><span class=tag>${x.language}</span><td>${x.genre}<td>${x.release_year}<td class=muted>${x.reason}</tr>`).join('')}
go();
</script>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
