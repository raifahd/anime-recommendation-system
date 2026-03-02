import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np

print("Loading dataset...")

df = pd.read_parquet("user_watches_filtered.parquet")

# Encode users and anime into indices
user_cat = df["user_id"].astype("category")
anime_cat = df["anime_id"].astype("category")

user_ids = user_cat.cat.codes
anime_ids = anime_cat.cat.codes

# Save mappings
user_mapping = dict(enumerate(user_cat.cat.categories))
anime_mapping = dict(enumerate(anime_cat.cat.categories))

np.save("anime_index_to_id.npy", anime_mapping)

rows = user_ids
cols = anime_ids
values = df["score"].values

# Build sparse matrix directly
sparse_matrix = csr_matrix((values, (rows, cols)))

print("Sparse matrix shape:", sparse_matrix.shape)

# Train SVD on sparse matrix
svd = TruncatedSVD(n_components=50)
latent_matrix = svd.fit_transform(sparse_matrix)

print("Training complete.")
print("Latent shape:", latent_matrix.shape)

# Save model
np.save("user_embeddings.npy", latent_matrix)
np.save("anime_components.npy", svd.components_)

print("Model saved successfully.")