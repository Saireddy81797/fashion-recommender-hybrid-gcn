# lightgcn.py
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_size=64, n_layers=3, device="cpu"):
        super().__init__()
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_items, emb_size)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.to(device)

    def forward(self, adj_norm):
        """
        adj_norm: scipy.sparse normalized adjacency (shape (n_users+n_items, n_users+n_items))
        We'll use sparse matrix multiplication at numpy then convert to torch for speed on CPU.
        """
        u_e = self.user_emb.weight
        i_e = self.item_emb.weight
        all_emb = torch.cat([u_e, i_e], dim=0)  # (n_users+n_items, dim)
        embs = [all_emb]
        # Propagate using sparse multiplication
        # Convert to torch sparse once:
        if isinstance(adj_norm, np.ndarray):
            adj_t = torch.tensor(adj_norm).to(self.device)
            for _ in range(self.n_layers):
                all_emb = torch.matmul(adj_t, all_emb)
                embs.append(all_emb)
        else:
            # assume scipy sparse
            adj = adj_norm.tocsr()
            emb_np = all_emb.detach().cpu().numpy()
            for _ in range(self.n_layers):
                emb_np = adj.dot(emb_np)
                embs.append(torch.tensor(emb_np, device=self.device, dtype=all_emb.dtype))
        embs = torch.stack(embs, dim=1)  # (N, n_layers+1, dim)
        out = embs.mean(dim=1)  # average of layers
        u_out, i_out = out[:self.n_users, :], out[self.n_users:, :]
        return u_out, i_out

    def get_user_item_embeddings(self):
        return self.user_emb.weight.detach(), self.item_emb.weight.detach()

    def predict(self, user_indices, item_indices):
        u = self.user_emb(user_indices)
        i = self.item_emb(item_indices)
        return (u * i).sum(dim=1)

