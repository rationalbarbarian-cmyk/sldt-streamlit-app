# app.py
import streamlit as st
import json
import numpy as np

from sldt import run_sldt
from axes import AXES

# -----------------------------
# 1. Define seed phrases (4 axes)
# -----------------------------
SEED_PHRASES = {
    "Valence": {
        "positive": [
            "I'm so grateful for this.",
            "This is wonderful.",
            "Everything feels hopeful."
        ],
        "negative": [
            "Everything is falling apart.",
            "This is unbearable.",
            "Nothing will work."
        ],
    },
    "Power": {
        "positive": [
            "Do exactly as I say.",
            "You must obey.",
            "I'm in charge here."
        ],
        "negative": [
            "Please don't be upset.",
            "I'm sorry, I didn't mean to.",
            "I'll do whatever you want."
        ],
    },
    "Agency": {
        "positive": [
            "I decided to take action.",
            "I chose this path.",
            "I will make it happen."
        ],
        "negative": [
            "I was forced to do this.",
            "It just happened to me.",
            "I had no choice."
        ],
    },
    "EvaluativeForce": {
        "positive": [
            "This was a careless and damaging decision.",
            "That behavior is unacceptable.",
            "This is deeply wrong."
        ],
        "negative": [
            "This is a description of events.",
            "Here is what happened.",
            "It was neither good nor bad."
        ],
    },
}

# -----------------------------
# 2. Optional: load 100 openings
# -----------------------------
@st.cache_data
def load_openings(path: str = "openings.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)  # {id: text}
    except FileNotFoundError:
        return {}

OPENINGS = load_openings()

# -----------------------------
# 3. Streamlit layout
# -----------------------------
st.set_page_config(page_title="Semantic Legal Dynamics Tool", layout="wide")
st.title("Semantic Legal Dynamics Tool (SLDT)")

tab_single, tab_corpus = st.tabs(["Single Text Analysis", "Corpus (100 Openings)"])

# -----------------------------
# 4. Single text analysis tab
# -----------------------------
with tab_single:
    st.subheader("Analyze a single text")

    default_text = "I really think things will work out."
    text = st.text_area("Enter text to analyze:", value=default_text, height=150)

    if st.button("Run SLDT on this text"):
        result = run_sldt(text, SEED_PHRASES)

        alpha = result["alpha"]
        beta = result["beta"]
        coherence = result["coherence"]
        curvature = result["curvature"]
        rotation = result["rotation"]
        interpretation = result["interpretation"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Static Profile (α)")
            st.write(alpha)

            st.markdown("#### Dynamic Motion (β)")
            st.write(beta)

            st.markdown("#### Coherence")
            st.write({"coherence": coherence})

        with col2:
            st.markdown("#### Curvature (per token)")
            st.line_chart(curvature)

            st.markdown("#### Phase Rotation")
            if len(rotation) > 0:
                st.line_chart(rotation)
            else:
                st.write("Not enough tokens to compute rotation.")

        st.markdown("#### Interpretive Diagnostics")
        for line in interpretation:
            st.write(f"- {line}")

# -----------------------------
# 5. Corpus analysis tab (100 openings)
# -----------------------------
with tab_corpus:
    st.subheader("Analyze your corpus of openings")

    if not OPENINGS:
        st.info("No openings.json found. Place a JSON file named 'openings.json' in the app directory.")
    else:
        keys = list(OPENINGS.keys())
        choice = st.selectbox("Choose an opening:", keys)

        if st.button("Run SLDT on selected opening"):
            text = OPENINGS[choice]
            st.markdown("**Selected text:**")
            st.write(text)

            result = run_sldt(text, SEED_PHRASES)

            st.markdown("#### Static Profile (α)")
            st.write(result["alpha"])

            st.markdown("#### Dynamic Motion (β)")
            st.write(result["beta"])

            st.markdown("#### Coherence")
            st.write({"coherence": result["coherence"]})

            st.markdown("#### Interpretive Diagnostics")
            for line in result["interpretation"]:
                st.write(f"- {line}")

        if st.button("Run batch analysis on all openings"):
            batch_results = {}
            for key, text in OPENINGS.items():
                batch_results[key] = run_sldt(text, SEED_PHRASES)["alpha"]

            st.markdown("#### Static α for all openings")
            st.dataframe(batch_results)
