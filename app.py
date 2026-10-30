"""Streamlit app — Movie Recommendation System (item-based collaborative filtering).

Run locally:  pip install -r requirements.txt && streamlit run app.py
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading 100K ratings and building the item–item similarity matrix…")
def build(min_count=5):
    r = pd.read_csv("data/ratings.dat", sep="::", engine="python", names=["user_id", "movie_id", "rating", "ts"])
    m = pd.read_csv("data/movies.dat", sep="::", engine="python", names=["movie_id", "title", "genres"], encoding="utf-8")
    m["genres"] = m.genres.fillna("")
    while True:  # 5-core filter, as in the notebook
        uc, ic = r.user_id.value_counts(), r.movie_id.value_counts()
        nr = r[r.user_id.isin(uc[uc >= min_count].index) & r.movie_id.isin(ic[ic >= min_count].index)]
        if len(nr) == len(r): break
        r = nr
    users, items = np.sort(r.user_id.unique()), np.sort(r.movie_id.unique())
    ui = {u: i for i, u in enumerate(users)}; ii = {x: j for j, x in enumerate(items)}
    B = sp.csr_matrix((np.ones(len(r)), (r.user_id.map(ui), r.movie_id.map(ii))), shape=(len(users), len(items)))
    Bn = normalize(B, axis=0)
    S = (Bn.T @ Bn).toarray(); np.fill_diagonal(S, 0)
    meta = m.set_index("movie_id").loc[items].reset_index()
    stats = r.groupby("movie_id").rating.agg(["count", "mean"]).loc[items].reset_index()
    meta = meta.merge(stats, on="movie_id")
    meta["year"] = meta.title.str.extract(r"\((\d{4})\)\s*$").astype(float)
    return meta, S, len(r), len(users)


meta, S, n_ratings, n_users = build()
titles = meta.title.tolist()
pop_order = meta["count"].values.argsort()[::-1]

st.title("🎬 Movie Recommender")
st.caption(f"Item-based collaborative filtering on {n_ratings:,} real IMDb ratings from {n_users:,} MovieTweetings users "
           f"({len(meta):,} movies). Item-kNN was the best model in the notebook evaluation (NDCG@10 +12% over popularity).")

tab1, tab2, tab3 = st.tabs(["⭐ Get recommendations", "🔎 Similar movies", "🔥 Most popular"])

with tab1:
    default = [t for t in ["Iron Man 3 (2013)", "Gravity (2013)", "The Conjuring (2013)"] if t in titles][:2] or titles[pop_order[0]:pop_order[0] + 1]
    liked = st.multiselect("Pick movies you liked", options=[titles[i] for i in pop_order], default=default)
    c1, c2, c3 = st.columns(3)
    n = c1.slider("How many recommendations", 5, 30, 10)
    genres = sorted({g for gs in meta.genres for g in gs.split("|") if g})
    gfilter = c2.multiselect("Only these genres (optional)", genres)
    min_count = c3.slider("Hide movies with fewer ratings than", 5, 50, 5)
    if liked:
        idx = [titles.index(t) for t in liked]
        score = S[idx].sum(0)
        because = S[idx].argmax(0)
        res = meta.assign(score=score, because=[liked[b] for b in because])
        res = res.drop(index=idx)
        res = res[res["count"] >= min_count]
        if gfilter:
            res = res[res.genres.apply(lambda g: any(x in g.split("|") for x in gfilter))]
        res = res[res.score > 0].sort_values("score", ascending=False).head(n)
        if res.empty:
            st.warning("No recommendations with these filters — try fewer filters.")
        else:
            for _, row in res.iterrows():
                st.markdown(f"**{row.title}** · _{row.genres.replace('|', ', ')}_  \n"
                            f"⭐ {row['mean']:.1f}/10 from {int(row['count'])} ratings · because you liked *{row.because}* "
                            f"(similarity {row.score:.2f})")
    else:
        st.info("Select at least one movie to get personalised recommendations.")

with tab2:
    q = st.selectbox("Movie", [titles[i] for i in pop_order])
    j = titles.index(q)
    sim = pd.DataFrame({"movie": titles, "cosine similarity": S[j], "ratings": meta["count"], "avg rating": meta["mean"].round(1),
                        "genres": meta.genres}).sort_values("cosine similarity", ascending=False).head(12)
    st.write(f"People who rated **{q}** also rated:")
    st.dataframe(sim.round(3), use_container_width=True, hide_index=True)

with tab3:
    top = meta.sort_values("count", ascending=False).head(25)[["title", "genres", "count", "mean"]]
    st.write("The popularity baseline: the same list for everyone. Item-kNN recommended from **66%** of the catalogue in the "
             "evaluation, while popularity covered only **1.6%**.")
    st.dataframe(top.rename(columns={"count": "ratings", "mean": "avg rating"}).round(2), use_container_width=True, hide_index=True)
