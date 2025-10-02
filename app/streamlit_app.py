# streamlit_app.py
import streamlit as st
import pandas as pd
from src.hybrid_inference import recommend_for_user
from pathlib import Path

st.set_page_config(layout="wide", page_title="Fashion Recommender Demo")
st.title("Personalized Fashion Recommender — Demo")

DATA = Path("data")
items_meta = pd.read_csv(DATA/"items_meta.csv")

st.markdown("### Select a user to view recommendations")
user_id = st.number_input("user_id (0..)", min_value=0, max_value=200000, value=0, step=1)

topk = st.slider("Number of recommendations", 1, 20, 8)
alpha = st.slider("Hybrid weight (collaborative vs content)", 0.0, 1.0, 0.6)

if st.button("Get Recommendations"):
    try:
        recs = recommend_for_user(int(user_id), topk=topk, alpha=alpha)
        cols = st.columns(4)
        for idx, row in recs.reset_index(drop=True).iterrows():
            col = cols[idx % 4]
            with col:
                st.image(row["image_url"], caption=f"{row['title']} (score: {row['score']:.3f})", use_column_width=True)
                st.write(f"**Category:** {row['category']}  ")
                st.write(f"**Price:** ₹{int(row['price'])}")
    except Exception as e:
        st.error(f"Error: {e}\nMake sure you trained the model and files exist in /models and /data")
