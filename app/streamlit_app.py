# streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path

# ✅ Import recommender (if available)
try:
    from src.hybrid_inference import recommend_for_user
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(layout="wide", page_title="Fashion Recommender Demo")
st.title("👗 Personalized Fashion Recommender — Demo")

# Load metadata
DATA = Path("data")
items_meta = pd.DataFrame()
try:
    items_meta = pd.read_csv(DATA / "items_meta.csv")
    # Clean up image URLs
    items_meta["image_url"] = items_meta["image_url"].astype(str).str.strip()
except Exception as e:
    st.warning(f"⚠️ Could not load items_meta.csv, fallback will be empty. Error: {e}")

# -------------------------------
# User input section
# -------------------------------
st.markdown("### Select a user to view recommendations")
user_id = st.number_input("user_id (0..)", min_value=0, max_value=200000, value=0, step=1)

topk = st.slider("Number of recommendations", 1, 20, 8)
alpha = st.slider("Hybrid weight (collaborative vs content)", 0.0, 1.0, 0.6)

# -------------------------------
# Button click: Recommendations
# -------------------------------
recs = None

if st.button("Get Recommendations"):
    try:
        if MODEL_AVAILABLE:
            recs = recommend_for_user(int(user_id), topk=topk, alpha=alpha)
            if recs is None or recs.empty:
                raise ValueError("Model returned no results.")
        else:
            raise ImportError("No trained model available.")
    except Exception as e:
        st.info("ℹ️ Showing demo recommendations (sample data only).")

        if not items_meta.empty:
            recs = items_meta.sample(n=topk, replace=True).copy()
            recs["score"] = 0.5
        else:
            recs = pd.DataFrame()

    # -------------------------------
    # Display recommendations as cards
    # -------------------------------
    if recs is not None and not recs.empty:
        st.markdown("## 🎯 Recommended Products")
        cols = st.columns(4)
        for idx, row in recs.reset_index(drop=True).iterrows():
            col = cols[idx % 4]
            with col:
                img_path = row.get("image_url", "")
                if pd.notna(img_path):
                    # ✅ Show both local and online images
                    if img_path.startswith("http"):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.image(str(DATA / img_path), use_container_width=True)

                st.markdown(f"**{row.get('title','Item')}**")
                st.markdown(f"🛒 *Category:* {row.get('category','N/A')}")
                st.markdown(f"💰 **₹{int(row.get('price',0))}**")
                st.markdown("---")
    else:
        st.error("❌ No recommendations available.")
