# train.py
import numpy as np
import pandas as pd
from pathlib import Path
import random
import torch
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, diags
from lightgcn import LightGCN
import joblib
from tqdm import tqdm

DATA = Path("data")
MODEL_OUT = Path("models")
MODEL_OUT.mkdir(exist_ok=True)

def load_data():
    df_inter = pd.read_csv(DATA/"interactions.csv")
    df_items = pd.read_csv(DATA/"items.csv")
    users = df_inter["user_id"].unique()
    items = df_items["item_id"].unique()
    n_users = int(df_inter["user_id"].max()+1)
    n_items = int(df_inter["item_id"].max()+1)
    return df_inter, n_users, n_items

def build_adj(df_inter, n_users, n_items):
    # Build bipartite adjacency matrix
    rows = df_inter["user_id"].values
    cols = df_inter["item_id"].values
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(n_users, n_items))
    # Build symmetric adjacency for propagation
    upper = coo_matrix(A)
    lower = coo_matrix(A.T)
    # adjacency for graph: [[0, A],[A.T, 0]]
    top_row = coo_matrix((np.hstack([np.zeros(1)]), ([0],[0])))  # not used
    # combine
    adj = coo_matrix(
        (np.hstack([upper.data, lower.data]),
         (np.hstack([upper.row, lower.row + n_users]),
          np.hstack([upper.col + n_users, lower.col])))
    , shape=(n_users+n_items, n_users+n_items))
    # normalize adjacency (symmetric normalization)
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = diags(d_inv_sqrt)
    adj_norm = D_inv_sqrt.dot(adj).dot(D_inv_sqrt)
    return adj_norm

def train_lightgcn(epochs=20, emb_size=64, lr=0.001, device="cpu"):
    df_inter, n_users, n_items = load_data()
    adj_norm = build_adj(df_inter, n_users, n_items)
    model = LightGCN(n_users, n_items, emb_size=emb_size, device=device)
    opt = optim.Adam(model.parameters(), lr=lr)
    # Build user->set of positive items for sampling
    user_pos = df_inter.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items = set(range(n_items))

    def sample_pair():
        u = random.choice(list(user_pos.keys()))
        pos = random.choice(list(user_pos[u]))
        neg = random.choice(list(all_items - user_pos[u]))
        return u, pos, neg

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for _ in range(2000):  # steps per epoch (small for demo)
            u, i_pos, i_neg = sample_pair()
            u_t = torch.tensor([u], dtype=torch.long, device=device)
            pos_t = torch.tensor([i_pos], dtype=torch.long, device=device)
            neg_t = torch.tensor([i_neg], dtype=torch.long, device=device)
            # get embeddings via propagation
            u_out, i_out = model(adj_norm)
            user_vec = u_out[u_t]
            pos_vec = i_out[pos_t]
            neg_vec = i_out[neg_t]
            pos_score = (user_vec * pos_vec).sum(dim=1)
            neg_score = (user_vec * neg_vec).sum(dim=1)
            loss = -F.logsigmoid(pos_score - neg_score).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep}/{epochs} - loss: {np.mean(losses):.4f}")
    # Save model embeddings (we'll not dump full torch module for portability)
    u_emb, i_emb = model(adj_norm)
    np.save(MODEL_OUT/"user_emb.npy", u_emb.detach().cpu().numpy())
    np.save(MODEL_OUT/"item_emb.npy", i_emb.detach().cpu().numpy())
    print("Saved embeddings:", MODEL_OUT)
    return

if __name__ == "__main__":
    train_lightgcn(epochs=12, emb_size=64)
