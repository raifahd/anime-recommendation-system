import pandas as pd

input_file = "user_watches.csv"
output_file = "user_watches_filtered.parquet"

chunksize = 2_000_000  # process 2M rows at a time
filtered_chunks = []

for chunk in pd.read_csv(input_file, chunksize=chunksize):

    # Keep only useful rows
    chunk = chunk[chunk["score"] > 0]

    # Keep only important columns
    chunk = chunk[["user_id", "anime_id", "score"]]

    # Reduce memory
    chunk["user_id"] = chunk["user_id"].astype("int32")
    chunk["anime_id"] = chunk["anime_id"].astype("int32")
    chunk["score"] = chunk["score"].astype("int8")

    filtered_chunks.append(chunk)

# Combine everything
final_df = pd.concat(filtered_chunks)

# Save compressed version
final_df.to_parquet(output_file, index=False)

print("Optimization complete.")
print("Saved as:", output_file)
print("Final rows:", len(final_df))