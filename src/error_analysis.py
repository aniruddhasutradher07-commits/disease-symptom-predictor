import pandas as pd
import os
from preprocess import clean_text
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "Symptom2Disease.csv")

df = pd.read_csv(csv_path)
df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "text": X_test.values
})

errors = results[(results['actual'] == 'drug reaction') & (results['predicted'] != 'drug reaction')]
print("Misclassified 'drug reaction' cases:\n")
for _, row in errors.iterrows():
    print(f"Predicted as: {row['predicted']}")
    print(f"Text: {row['text']}\n")