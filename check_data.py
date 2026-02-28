import pandas as pd

file_path = "user_watches.csv"

# Count rows WITHOUT loading everything
row_count = sum(1 for _ in open(file_path)) - 1  # minus header

print("Total Rows:", row_count)

# Read only header
df = pd.read_csv(file_path, nrows=5)

print("\nColumns:")
print(df.columns)

print("\nSample:")
print(df.head())