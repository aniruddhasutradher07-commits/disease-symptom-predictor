import pandas as pd
import os
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "Symptom2Disease.csv")

df = pd.read_csv(csv_path)
df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy (with bigrams):", nb_acc)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy (with bigrams):", lr_acc)

if lr_acc >= nb_acc:
    best_model = lr_model
    best_pred = lr_pred
    print("\n>>> Logistic Regression performed better. Using it as final model.")
else:
    best_model = nb_model
    best_pred = nb_pred
    print("\n>>> Naive Bayes performed better. Using it as final model.")

print("\nClassification Report (best model):\n", classification_report(y_test, best_pred))

labels = sorted(df['label'].unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
print("\nConfusion matrix saved as confusion_matrix.png")

joblib.dump(best_model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))
print("Model and vectorizer saved!")