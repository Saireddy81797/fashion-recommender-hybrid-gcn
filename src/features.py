# features.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA = Path("data")
EMB_OUT = DATA / "item_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight & fast

def build_item_embeddings(items_csv=DATA/"items.csv"):
    df = pd.read_csv(items_csv)
    texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_OUT, embeddings)
    # save mapping
    df[["item_id","title","image_url","category","price"]].to_csv(DATA/"items_meta.csv", index=False)
    print("Saved embeddings:", EMB_OUT)
    return embeddings

if __name__ == "__main__":
    build_item_embeddings()
