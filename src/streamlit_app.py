from urllib.parse import quote_plus
import streamlit as st
import pandas as pd
import requests
import re


from recommender import MovieRecommender
import streamlit as st

API_KEY = st.secrets.get("TMDB_API_KEY")

if not API_KEY:
    st.error("API key not loaded!")
# =========================
# 🎬 GET POSTER (FIXED)
# =========================
@st.cache_data
def get_poster(movie_title):
    try:
        import re

        # 🔥 REMOVE RANDOM NUMBERS (your main issue)
        movie_title = re.sub(r"\d+$", "", movie_title)
        movie_title = re.sub(r"\(\d{4}\)", "", movie_title)
        movie_title = movie_title.strip()

        query = quote_plus(movie_title)

        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"

        res = requests.get(url)
        if res.status_code != 200:
            return None

        data = res.json()
        results = data.get("results", [])

        # ✅ just take first valid poster (more reliable)
        for r in results:
            if r.get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{r['poster_path']}"

        return None

    except Exception as e:
        print("Poster error:", e)
        return None


# =========================
# 🧱 CARDS
# =========================
def render_recommendation_cards(results):
    if not results:
        st.warning("No recommendations found.")
        return

    cols = st.columns(3)

    for idx, rec in enumerate(results):
        with cols[idx % 3]:
            poster = get_poster(rec.title)

            if poster:
                st.image(poster, width="stretch")
            else:
                st.write("No image")

            year = f" ({int(rec.year)})" if pd.notna(rec.year) else ""
            st.markdown(f"**{rec.title}{year}**")
            st.caption(f"Score: {rec.score:.4f}")


# =========================
# 🎥 MOVIE FOCUS
# =========================
def render_movie_focus(title, row):
    col1, col2 = st.columns([1, 2])

    with col1:
        poster = get_poster(title)
        if poster:
            st.image(poster, width="stretch")

    with col2:
        st.markdown(f"## {title}")
        st.write(f"**Year:** {int(row['year']) if pd.notna(row.get('year')) else 'N/A'}")
        st.write(f"**Genres:** {row.get('genres', 'N/A')}")
        st.write(f"**Director:** {row.get('director', 'N/A')}")


# =========================
# 📦 LOAD MODEL
# =========================
def load_from_csv(path):
    df = pd.read_csv(path)
    model = MovieRecommender()
    model.fit(df)
    return model


# =========================
# 🎨 UI CONFIG
# =========================
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")


# =========================
# ⚙️ SIDEBAR (TRAINING)
# =========================
with st.sidebar:
    st.header("⚙️ Model Settings")

    data_path = st.text_input("Dataset", "data/movies_tmdb_clean.csv")
    model_path = st.text_input("Model Path", "models/movie_recommender.joblib")

    if st.button("🚀 Train + Save"):
        st.session_state.recommender = load_from_csv(data_path)
        st.session_state.recommender.save(model_path)
        st.success("Model trained & saved!")

    if st.button("📂 Load Model"):
        st.session_state.recommender = MovieRecommender.load(model_path)
        st.success("Model loaded!")


# =========================
# 🧠 LOAD DEFAULT
# =========================
if "recommender" not in st.session_state:
    st.session_state.recommender = load_from_csv("data/movies_tmdb_clean.csv")

recommender = st.session_state.recommender
titles = recommender.available_titles()


# =========================
# 🔍 SEARCH
# =========================
search = st.text_input("Search movie")

filtered = [t for t in titles if search.lower() in t.lower()] if search else titles

if not filtered:
    st.warning("No movies found")
    st.stop()


# =========================
# 🎬 MAIN UI
# =========================
col1, col2 = st.columns(2)

# -------- LEFT --------
with col1:
    st.subheader("🎯 Similar Movies")

    selected = st.selectbox("Select Movie", filtered)

    row = recommender.movies_df[recommender.movies_df["title"] == selected].iloc[0]

    render_movie_focus(selected, row)

    top_n = st.slider("Top N", 1, 20, 10)

    if st.button("Recommend Similar"):
        results = recommender.recommend_similar(selected, top_n)

        st.subheader("Results")
        render_recommendation_cards(results)


# -------- RIGHT --------
with col2:
    st.subheader("🔥 Personalized")

    liked = st.multiselect("Select liked movies", filtered)

    top_n_p = st.slider("Top N Personalized", 1, 20, 10)

    if st.button("Generate Personalized"):
        if not liked:
            st.warning("Select at least one movie")
        else:
            results = recommender.recommend_for_user(liked, top_n_p)

            st.subheader("Results")
            render_recommendation_cards(results)