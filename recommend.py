import numpy as np
import pandas as pd

def recommend_for_user(user_id, top_n=5):
    # Load trained embeddings
    user_embeddings = np.load("user_embeddings.npy")
    anime_components = np.load("anime_components.npy")
    anime_index_to_id = np.load("anime_index_to_id.npy", allow_pickle=True).item()

    # Build reverse mapping
    anime_id_to_index = {v: k for k, v in anime_index_to_id.items()}

    # Load dataset
    df = pd.read_parquet("user_watches_filtered.parquet")

    # Rebuild user mapping
    user_cat = df["user_id"].astype("category")
    user_index_to_id = dict(enumerate(user_cat.cat.categories))
    user_id_to_index = {v: k for k, v in user_index_to_id.items()}

    if user_id not in user_id_to_index:
        print("User not found.")
        return

    user_index = user_id_to_index[user_id]

    # Get user embedding
    user_vector = user_embeddings[user_index]

    # Compute scores (dot product with all anime)
    scores = np.dot(user_vector, anime_components)

    # Get watched anime
    watched_ids = df[df["user_id"] == user_id]["anime_id"].values

    watched_indices = [
        anime_id_to_index[a]
        for a in watched_ids
        if a in anime_id_to_index
    ]

    # Remove watched
    scores[watched_indices] = -np.inf

    # Get top N
    top_indices = np.argsort(scores)[-top_n:][::-1]

    recommended_ids = [anime_index_to_id[i] for i in top_indices]

    print(f"\nTop {top_n} recommendations for User {user_id}:\n")
    for aid in recommended_ids:
        print(aid)


if __name__ == "__main__":
    recommend_for_user(user_id=1, top_n=5)