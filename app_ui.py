"""CineMatch - Streamlit front-end for the regional-cinema recommender.

    streamlit run app_ui.py          (run from the project root)

Uses the same engine as the Flask API (recsys/engine.py); nothing is re-implemented here.
"""
import html
import json
import os
import random

import altair as alt
import pandas as pd
import streamlit as st

from recsys.engine import RecommenderEngine

st.set_page_config(page_title="CineMatch · Regional Movie Recommender", page_icon="🎬", layout="wide")

LANG_COL = {"Hindi": "#FF6B6B", "Tamil": "#4DABF7", "Telugu": "#51CF66", "Kannada": "#FFA94D", "Malayalam": "#B197FC"}
METHODS = {"hybrid": "Hybrid ★ recommended", "itemknn": "Item-based CF", "userknn": "User-based CF",
           "svd": "SVD matrix factorisation", "popularity": "Most popular (baseline)"}
ACCENT, MUTED = "#8B6CFF", "#5B6080"

st.markdown("""
<style>
.block-container{padding-top:3.4rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#2A1B5E 0%,#5B2A86 45%,#B23A7A 100%);border-radius:18px;padding:22px 28px;margin-bottom:18px}
.hero h1{margin:0;font-size:2rem;letter-spacing:.3px;color:#fff}.hero p{margin:.25rem 0 0;color:#E9DDFB;font-size:.95rem}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:6px}
.kpi{background:#171A30;border:1px solid #262B4A;border-radius:14px;padding:12px 16px}
.kpi .l{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;color:#9AA0C3}.kpi .v{font-size:1.6rem;font-weight:700;margin-top:2px}
.card{background:#171A30;border:1px solid #262B4A;border-radius:14px;padding:14px 16px;margin-bottom:12px;min-height:128px;transition:border-color .15s}
.card:hover{border-color:#8B6CFF}
.top{display:flex;align-items:center;gap:10px}.rank{background:#8B6CFF;color:#fff;width:28px;height:28px;border-radius:50%;display:flex;
align-items:center;justify-content:center;font-weight:700;font-size:.85rem;flex:none}
.title{font-weight:650;font-size:1.05rem;line-height:1.2}.chips{margin:8px 0 6px}
.chip{display:inline-block;border-radius:999px;padding:1px 10px;font-size:.74rem;margin-right:6px;font-weight:600}
.chip.g{background:#262B4A;color:#C9CCE6}
.bar{height:5px;background:#262B4A;border-radius:4px;margin:8px 0 6px}.bar>div{height:5px;border-radius:4px;background:linear-gradient(90deg,#8B6CFF,#FF7AB6)}
.why{font-size:.8rem;color:#9AA0C3;font-style:italic}.star{float:right;font-weight:700;color:#FFD166}
section[data-testid=stSidebar]{border-right:1px solid #262B4A}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the models (a few seconds, first load only)…")
def load_engine():
    return RecommenderEngine()


eng = load_engine()
cat = eng.cat.reset_index()
titles = sorted(eng.cat.movie_title)
title_to_id = dict(zip(eng.cat.movie_title, eng.cat.index))


def ild(vals):
    vals = list(vals)
    pairs = len(vals) * (len(vals) - 1) / 2
    same = sum(a == b for i, a in enumerate(vals) for b in vals[i + 1:])
    return 1 - same / pairs if pairs else 0.0


def chip(lang):
    c = LANG_COL.get(lang, "#888")
    return f'<span class="chip" style="background:{c}26;color:{c};border:1px solid {c}66">{html.escape(lang)}</span>'


def card(r):
    pr = r.get("predicted_rating")
    star = f'<span class="star">★ {pr:.2f}</span>' if pr else ""
    bar = f'<div class="bar"><div style="width:{pr / 5 * 100:.0f}%"></div></div>' if pr else '<div class="bar"></div>'
    return (f'<div class="card"><div class="top"><div class="rank">{r["rank"]}</div>'
            f'<div class="title">{html.escape(r["title"])}</div>{star}</div>'
            f'<div class="chips">{chip(r["language"])}<span class="chip g">{html.escape(r["genre"])}</span>'
            f'<span class="chip g">{r["release_year"]}</span></div>{bar}'
            f'<div class="why">{html.escape(r["reason"])}</div></div>')


def mix_chart(a, b, a_label, b_label, title):
    d = pd.DataFrame({a_label: pd.Series(a).value_counts(normalize=True) * 100,
                      b_label: pd.Series(b).value_counts(normalize=True) * 100}).fillna(0)
    d.index.name = "category"
    d = d.reset_index().melt("category", var_name="source", value_name="pct")
    return (alt.Chart(d, title=title).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("category:N", title=None, sort="-x"), x=alt.X("pct:Q", title="% of list"), yOffset="source:N",
        color=alt.Color("source:N", title=None, scale=alt.Scale(domain=[a_label, b_label], range=[ACCENT, MUTED])),
        tooltip=["category", "source", alt.Tooltip("pct:Q", format=".1f")]).properties(height=210))


# ------------------------------------------------------------------ sidebar
SAMPLE = {"KGF": 5.0, "Baahubali": 4.5, "RRR": 5.0, "Pushpa: The Rise": 4.0}
if "uid" not in st.session_state:
    st.session_state.uid = 7
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    mode = st.radio("Viewer", ["Existing user", "New user"], horizontal=True)
    new_ratings = {}
    if mode == "Existing user":
        c1, c2 = st.columns([3, 1])
        c1.selectbox("User ID", [int(u) for u in eng.ix.user_ids], key="uid")
        c2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if c2.button("🎲", help="Pick a random user"):
            st.session_state.uid = int(random.choice(list(eng.ix.user_ids)))
            st.rerun()
        method = st.selectbox("Method", list(METHODS), format_func=METHODS.get, key="method")
    else:
        st.caption("Tell us a few films you've watched and how you rated them.")
        if st.button("Use a sample taste"):
            st.session_state.new_titles = list(SAMPLE)
            for t, v in SAMPLE.items():
                st.session_state[f"r_{t}"] = v
        picked = st.multiselect("Films you've watched", titles, key="new_titles")
        for t in picked:
            new_ratings[title_to_id[t]] = st.slider(t, 1.0, 5.0, 4.0, 0.5, key=f"r_{t}")
        method = "hybrid"
        st.caption("New users are served by the recommended hybrid model.")
    diversity = st.slider("Diversify results", 0.0, 1.0, 0.3, 0.1,
                          help="0 = most relevant only. Higher values re-rank (MMR) so the list mixes genres and languages.")
    k = st.slider("Number of recommendations", 4, 20, 10, 2)
    st.divider()
    st.caption("Collaborative filtering on a synthetic, regional-cinema dataset. See the Ethics tab.")

# ------------------------------------------------------------------ compute
if mode == "Existing user":
    res = eng.recommend(int(st.session_state.uid), k, method, diversity)
elif new_ratings:
    res = eng.recommend_for_new_user(new_ratings, k, diversity)
else:
    res = None

st.markdown('<div class="hero"><h1>🎬 CineMatch</h1><p>Personalised recommendations across Hindi, Tamil, Telugu, Kannada and Malayalam cinema '
            '— with a built-in guard against filter bubbles.</p></div>', unsafe_allow_html=True)

if res is None:
    st.info("👈 Pick a few films you've watched in the sidebar (or press **Use a sample taste**) to get recommendations.")
    st.stop()

recs = pd.DataFrame(res["recommendations"])
source = "Cold start · popular" if res["cold_start"] else ("Personalised" if res["method"] != "popularity" else "Popular films")
variety = (ild(recs.genre) + ild(recs.language)) / 2
nonhindi = (recs.language != "Hindi").mean() * 100
st.markdown(f"""<div class="kpis">
<div class="kpi"><div class="l">Ratings by this viewer</div><div class="v">{res['n_ratings_known']}</div></div>
<div class="kpi"><div class="l">List variety (0–1)</div><div class="v">{variety:.2f}</div></div>
<div class="kpi"><div class="l">Languages in list</div><div class="v">{recs.language.nunique()} <span style="font-size:.9rem;color:#9AA0C3">/ 5 · {nonhindi:.0f}% non-Hindi</span></div></div>
<div class="kpi"><div class="l">Source</div><div class="v" style="font-size:1.25rem;padding-top:6px">{source}</div></div></div>""", unsafe_allow_html=True)
if res["cold_start"]:
    st.warning("This user has no rating history, so we fall back to popular films.")

tabs = st.tabs(["⭐ Recommendations", "💡 Why these?", "👤 Taste profile", "🔎 Similar films", "📈 How good is it?", "⚖️ Ethics & diversity"])

# ---- 1. recommendations
with tabs[0]:
    for i in range(0, len(recs), 2):
        cols = st.columns(2)
        for col, (_, r) in zip(cols, recs.iloc[i:i + 2].iterrows()):
            col.markdown(card(r), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.altair_chart(mix_chart(recs.genre, cat.genre, "This list", "Catalogue", "Genre mix: list vs catalogue"), width="stretch")
    c2.altair_chart(mix_chart(recs.language, cat.language, "This list", "Catalogue", "Language mix: list vs catalogue"), width="stretch")
    st.caption("Tip: raise **Diversify results** and watch the two charts move toward the catalogue mix — usually at a small cost in relevance.")

# ---- 2. explanations
with tabs[1]:
    st.write("Every recommendation comes with a reason, so the system is not a black box.")
    for _, r in recs.iterrows():
        with st.expander(f"{r['rank']}. {r['title']}  ·  {r['language']} · {r['genre']}"):
            st.markdown(f"**Why:** {r['reason']}")
            m1, m2 = st.columns(2)
            if r["predicted_rating"]:
                m1.metric("Predicted rating", f"{r['predicted_rating']:.2f} / 5")
            m2.metric("Ranking score", f"{r['score']:.2f}")
            sim = pd.DataFrame(eng.similar(int(r["movie_id"]), 3))[["title", "language", "genre", "similarity"]]
            st.caption("Films with similar audiences:")
            st.dataframe(sim, hide_index=True, width="stretch")

# ---- 3. taste profile
with tabs[2]:
    if mode == "Existing user":
        hist = pd.DataFrame(eng.history(int(st.session_state.uid)))
    else:
        hist = pd.DataFrame([dict(eng._movie(eng.ix.i_map[m]), rating=v) for m, v in new_ratings.items()])
    if hist.empty:
        st.info("No history for this viewer.")
    else:
        c1, c2 = st.columns([3, 2])
        c1.dataframe(hist.sort_values("rating", ascending=False)[["title", "language", "genre", "release_year", "rating"]],
                     hide_index=True, width="stretch",
                     column_config={"rating": st.column_config.ProgressColumn("rating", min_value=0, max_value=5, format="%.1f ★"),
                                    "release_year": st.column_config.NumberColumn("year", format="%d")})
        c2.altair_chart(mix_chart(hist.language, recs.language, "Watched", "Recommended", "Language: watched vs recommended"), width="stretch")
        st.caption("If the 'Recommended' bars simply copy the 'Watched' bars, the viewer is being kept in a bubble.")

# ---- 4. similar films
with tabs[3]:
    pick = st.selectbox("Pick a film", titles, index=titles.index("KGF"))
    sim = pd.DataFrame(eng.similar(int(title_to_id[pick]), 8))[["title", "language", "genre", "release_year", "similarity"]]
    st.dataframe(sim, hide_index=True, width="stretch",
                 column_config={"similarity": st.column_config.ProgressColumn("similarity", min_value=0, max_value=1, format="%.2f"),
                                "release_year": st.column_config.NumberColumn("year", format="%d")})
    st.caption("Similarity = shrunk cosine over how viewers rated each film relative to their own average (item-based CF).")

# ---- 5. performance
with tabs[4]:
    st.warning("The ratings are largely **synthetic** (generated from a documented latent-factor model), so these numbers show how well "
               "the models recover that generator — not how real audiences behave.")
    try:
        M = json.load(open("results/metrics.json"))
        rows = []
        for n, r in M["test_results"].items():
            rows.append({"Model": n, "RMSE ↓": r.get("rmse"), "NDCG@10 ↑": r.get("ndcg@10"), "Hit-rate@10 ↑": r.get("hit_rate@10"),
                         "Recall@10 ↑": r.get("recall@10")})
        tbl = pd.DataFrame(rows).round(3).astype(object).where(lambda d: d.notna(), "—")
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.markdown(f"**Test set:** {M['split']['test']} ratings · ranking scored on {M['split']['eval_users_for_ranking']} users · "
                    f"final model chosen by cross-validation on the training data only.")
        st.markdown("- **Item-based CF** predicts ratings best; **SVD does not beat the simple bias baseline** at ~8 ratings per user.\n"
                    "- Plain **Popularity** beats the CF models at top-10 ranking; adding a popularity prior to item-CF beats both (paired test, 95% CI above 0).")
    except FileNotFoundError:
        st.info("Run `python 04_evaluate.py` to generate results/metrics.json.")
    if os.path.exists("figures/05_model_comparison.png"):
        st.image("figures/05_model_comparison.png")

# ---- 6. ethics
with tabs[5]:
    st.subheader("Filter bubbles and who gets recommended")
    try:
        le = pd.read_csv("results/language_exposure.csv", index_col=0)
        show = le[["train_ratings_share", "recs_ItemKNN+Pop", "recs_ItemKNN+Pop_MMR_lam0.6", "recs_ItemKNN+Pop_MMR_lam0.4"]] * 100
        show.columns = ["Share of ratings", "Recs: no diversify", "Recs: diversify 0.4", "Recs: diversify 0.6"]
        long = show.rename_axis("language").reset_index().melt("language", var_name="series", value_name="pct")
        order = list(show.columns)
        st.altair_chart(alt.Chart(long, title="Share of recommendation slots by language (all users)").mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("language:N", title=None, sort=list(LANG_COL), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("series:N", sort=order), y=alt.Y("pct:Q", title="% of slots"),
            color=alt.Color("series:N", title=None, sort=order, scale=alt.Scale(domain=order, range=[MUTED, "#B8A4FF", ACCENT, "#FF7AB6"])),
            tooltip=["language", "series", alt.Tooltip("pct:Q", format=".1f")]).properties(height=300), width="stretch")
        st.dataframe(pd.read_csv("results/diversity_tradeoff.csv")[["lam", "ndcg@10", "coverage", "ild_genre", "regional_share"]].round(3)
                     .rename(columns={"lam": "λ (1 = off)", "coverage": "catalogue coverage", "ild_genre": "genre variety",
                                      "regional_share": "non-Hindi share"}), hide_index=True, width="stretch")
    except FileNotFoundError:
        st.info("Run `python 04_evaluate.py` first.")
    st.markdown("""
- **The bubble is real:** without diversification Hindi films get far more recommendation slots than their share of ratings, while Kannada and Malayalam get fewer than theirs. Popularity-driven systems amplify the majority.
- **Mitigation:** the *Diversify* slider applies MMR re-ranking. It costs some ranking accuracy (see table) — the point is to let the viewer and the product owner choose the trade-off knowingly.
- **No demographic data:** the dataset has only anonymous `user_id`s, so demographic fairness cannot be tested; we audit by activity level, dominant language and film popularity instead.
- **Privacy:** no names, contacts or other PII are stored. Every recommendation is explained.
- **Data honesty:** most ratings are synthetic; film metadata was compiled by hand and should be spot-checked.""")
