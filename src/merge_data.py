import pandas as pd
import os
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load original dataset
original_path = os.path.join(BASE_DIR, "data", "Symptom2Disease.csv")
original_df = pd.read_csv(original_path)
original_df = original_df[['text', 'label']]
original_df['label'] = original_df['label'].str.lower().str.strip()

print("Original dataset:", original_df.shape)
print("Original diseases:", sorted(original_df['label'].unique()))

# Load new dataset from Hugging Face
print("\nDownloading new dataset from Hugging Face...")
hf_dataset = load_dataset("gretelai/symptom_to_diagnosis")

# Combine train + test splits from new dataset
new_train = hf_dataset['train'].to_pandas()
new_test = hf_dataset['test'].to_pandas()
new_df = pd.concat([new_train, new_test], ignore_index=True)

# Rename columns to match original schema
new_df = new_df.rename(columns={'input_text': 'text', 'output_text': 'label'})
new_df = new_df[['text', 'label']]
new_df['label'] = new_df['label'].str.lower().str.strip()

print("New dataset:", new_df.shape)
print("New diseases:", sorted(new_df['label'].unique()))

# Merge both datasets
merged_df = pd.concat([original_df, new_df], ignore_index=True)

# Remove exact duplicate rows if any
before = len(merged_df)
merged_df = merged_df.drop_duplicates(subset=['text', 'label'])
after = len(merged_df)
print(f"\nRemoved {before - after} exact duplicates")

print("\nMerged dataset:", merged_df.shape)
print("\nExamples per disease:\n", merged_df['label'].value_counts())

# Save merged dataset
output_path = os.path.join(BASE_DIR, "data", "merged_symptom_disease.csv")
merged_df.to_csv(output_path, index=False)
print(f"\nSaved merged dataset to {output_path}")