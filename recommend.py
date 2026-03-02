import numpy as np
import pandas as pd

def recommend_for_user(user_id, top_n=5):
    # Load trained embeddings
    user_embeddings = np.load("user_embeddings.npy")
    anime_components = np.load("anime_components.npy")
    anime_index_to_id = np.load("anime_index_to_id.npy", allow_pickle=True).item()

    # Reverse mapping
    anime_id_to_index = {v: k for k, v in anime_index_to_id.items()}

    # Load datasets
    df = pd.read_parquet("user_watches_filtered.parquet")
    anime_df = pd.read_csv("animes.csv")

    # Create proper title column
    anime_df["display_title"] = (
        anime_df["title_english"]
        .fillna(anime_df["title"])
        .fillna(anime_df["title_japanese"])
    )

    anime_id_to_name = dict(zip(anime_df["anime_id"], anime_df["display_title"]))

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

    # Compute scores
    scores = np.dot(user_vector, anime_components)

    # Remove watched anime
    watched_ids = df[df["user_id"] == user_id]["anime_id"].values
    watched_indices = [
        anime_id_to_index[a]
        for a in watched_ids
        if a in anime_id_to_index
    ]
    scores[watched_indices] = -np.inf

    # Get top N
    top_indices = np.argsort(scores)[-top_n:][::-1]
    recommended_ids = [anime_index_to_id[i] for i in top_indices]

    print(f"\nTop {top_n} recommendations for User {user_id}:\n")

    for aid in recommended_ids:
        name = anime_id_to_name.get(aid, "Unknown Title")
        print(f"{name} (ID: {aid})")


if __name__ == "__main__":
    recommend_for_user(user_id=1, top_n=5)