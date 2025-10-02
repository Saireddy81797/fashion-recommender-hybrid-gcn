# Button for recommendations
if st.button("Get Recommendations"):
    try:
        # 🔹 Try real model first
        recs = recommend_for_user(int(user_id), topk=topk, alpha=alpha)

        # If recommend_for_user returns nothing or fails, fallback
        if recs is None or recs.empty:
            raise ValueError("No recommendations from model, using fallback.")

    except Exception as e:
        st.warning(f"⚠️ Using fallback demo mode because: {e}")

        # 🔹 Fallback: pick random items from metadata
        recs = items_meta.sample(n=topk, replace=True)

        # Add dummy scores
        recs["score"] = 0.5  

    # Display recommendations
    cols = st.columns(4)
    for idx, row in recs.reset_index(drop=True).iterrows():
        col = cols[idx % 4]
        with col:
            st.image(
                row["image_url"],
                caption=f"{row['title']} (score: {row['score']:.3f})",
                use_column_width=True,
            )
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Price:** ₹{int(row['price'])}")
