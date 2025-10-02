# data_simulation.py
import random
import json
from pathlib import Path
import pandas as pd
import numpy as np

RNG = random.Random(42)
np.random.seed(42)

OUT = Path("data")
OUT.mkdir(exist_ok=True)

def gen_items(n_items=2000):
    categories = ["Tops","Bottoms","Dresses","Shoes","Accessories","Outerwear"]
    styles = ["casual","formal","sport","ethnic","boho","minimal"]
    colors = ["red","blue","black","white","green","yellow","pink","beige"]
    items = []
    for i in range(n_items):
        title = f"{RNG.choice(colors).capitalize()} {RNG.choice(styles).capitalize()} {RNG.choice(categories)} {i}"
        desc = f"A {RNG.choice(styles)} {RNG.choice(categories).lower()} in {RNG.choice(colors)} color with comfortable fit and modern design."
        price = float(RNG.randint(299, 4999))
        items.append({"item_id": i, "title": title, "description": desc, "category": RNG.choice(categories), "price": price, "image_url": f"https://picsum.photos/seed/{i}/300/400"})
    df_items = pd.DataFrame(items)
    df_items.to_csv(OUT / "items.csv", index=False)
    print("items saved:", df_items.shape)
    return df_items

def gen_users(n_users=500):
    users = []
    for u in range(n_users):
        age = RNG.randint(16, 55)
        gender = RNG.choice(["M","F","O"])
        users.append({"user_id": u, "age": age, "gender": gender})
    df_users = pd.DataFrame(users)
    df_users.to_csv(OUT / "users.csv", index=False)
    print("users saved:", df_users.shape)
    return df_users

def gen_interactions(df_users, df_items, interactions_per_user=(10, 80)):
    rows = []
    for _, u in df_users.iterrows():
        n = RNG.randint(*interactions_per_user)
        pos_items = RNG.sample(list(df_items["item_id"]), n)
        for i in pos_items:
            timestamp = int(np.random.randint(1650000000, 1700000000))
            rating = RNG.choice([1])  # implicit feedback
            rows.append({"user_id": u["user_id"], "item_id": int(i), "timestamp": timestamp, "rating": rating})
    df_inter = pd.DataFrame(rows)
    df_inter = df_inter.sort_values("timestamp").reset_index(drop=True)
    df_inter.to_csv(OUT / "interactions.csv", index=False)
    print("interactions saved:", df_inter.shape)
    return df_inter

if __name__ == "__main__":
    items = gen_items(1500)
    users = gen_users(800)
    inter = gen_interactions(users, items, (5,60))
    print("Done.")
