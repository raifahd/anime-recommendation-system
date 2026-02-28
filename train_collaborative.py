import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np

print("Loading dataset...")

df = pd.read_parquet("user_watches_filtered.parquet")

# Encode users and anime into indices
user_ids = df["user_id"].astype("category").cat.codes
anime_ids = df["anime_id"].astype("category").cat.codes

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