# hybrid_inference.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import heapq

DATA = Path("data")
MODEL = Path("models")

def load_all():
    items_meta = pd.read_csv(DATA/"items_meta.csv")
    item_emb = np.load(MODEL/"item_emb.npy")  # learned from LightGCN (item embeddings)
    content_emb = np.load(DATA/"item_embeddings.npy")  # content embedding from SBERT
    user_emb = np.load(MODEL/"user_emb.npy")
    return items_meta, user_emb, item_emb, content_emb

def recommend_for_user(user_id, topk=10, alpha=0.6):
    items_meta, user_emb, item_emb, content_emb = load_all()
    n_items = item_emb.shape[0]
    # collaborative score = dot(user_emb, item_emb)
    uvec = user_emb[user_id]
    coll_scores = item_emb.dot(uvec)
    # content similarity: find top items similar to user's previously interacted items
    # For simplicity compute item-to-item similarity arithmetically
    # final_score = alpha * coll + (1-alpha) * content_sim
    # Compute content_sim as max cosine similarity with items user interacted (approximation)
    content_norm = content_emb / (np.linalg.norm(content_emb, axis=1, keepdims=True)+1e-9)
    coll_norm = coll_scores / (np.linalg.norm(coll_scores)+1e-9)
    # For demo, a simple content boost is just the content embedding's dot with a "user content preference"
    # We'll approximate user content preference by averaging content embeddings of top-collaborative items
    top_c_idx = np.argsort(-coll_scores)[:30]
    user_content_pref = content_emb[top_c_idx].mean(axis=0)
    content_scores = content_emb.dot(user_content_pref)
    # normalize
    content_scores = content_scores / (np.linalg.norm(content_scores)+1e-9)
    final = alpha * coll_norm + (1-alpha) * content_scores
    topk_idx = np.argsort(-final)[:topk]
    results = items_meta.iloc[topk_idx].copy()
    results["score"] = final[topk_idx]
    return results

if __name__ == "__main__":
    res = recommend_for_user(0, topk=8)
    print(res[["item_id","title","score"]])
