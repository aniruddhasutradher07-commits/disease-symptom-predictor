import pandas as pd
import re

df = pd.read_csv("data/Symptom2Disease.csv")

df = df.drop(columns=["Unnamed: 0"])

df["text"] = df["text"].str.lower()

df["text"] = df["text"].apply(
    lambda x: re.sub(r"[^a-zA-Z\s]", "", x)
)

df["text"] = df["text"].apply(
    lambda x: re.sub(r"\s+", " ", x).strip()
)

print("\n--- Cleaned Dataset ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Missing Values ---")
print(df.isnull().sum())