import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "Symptom2Disease.csv")

df = pd.read_csv(csv_path)

# Split data (raw text, no manual cleaning needed)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Load pretrained sentence embedding model
print("Loading embedding model (first time may download ~80MB)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to embeddings
print("Generating embeddings...")
X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = embedder.encode(X_test.tolist(), show_progress_bar=True)

# Train classifier on embeddings
model = LogisticRegression(max_iter=1000)
model.fit(X_train_emb, y_train)

# Evaluate
y_pred = model.predict(X_test_emb)
acc = accuracy_score(y_test, y_pred)
print("\nEmbeddings-based Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model (embedder itself doesn't need saving, it's pretrained/downloaded automatically)
joblib.dump(model, os.path.join(BASE_DIR, "model_embeddings.pkl"))
print("\nEmbeddings model saved as model_embeddings.pkl")