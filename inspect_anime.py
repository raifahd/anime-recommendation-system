import pandas as pd

df = pd.read_csv("animes.csv")

print("Columns:")
print(df.columns)

print("\nSample:")
print(df.head())