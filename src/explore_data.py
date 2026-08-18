import pandas as pd

df = pd.read_csv("data/Symptom2Disease.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nUnique diseases:", df['label'].nunique())
print("\nDisease list:\n", df['label'].unique())
print("\nSample rows:\n", df.head())