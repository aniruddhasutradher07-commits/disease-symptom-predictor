import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/Symptom2Disease.csv")

df = df.drop(columns=["Unnamed: 0"])

df["text"] = df["text"].str.lower()

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

symptoms = input("\nEnter your symptoms: ")

symptoms_tfidf = vectorizer.transform([symptoms.lower()])

prediction = model.predict(symptoms_tfidf)[0]

probabilities = model.predict_proba(symptoms_tfidf)[0]
confidence = probabilities.max()


print("\n==============================")
print("PREDICTION")
print("==============================")

print("Predicted disease:", prediction)
print(f"Model confidence: {confidence:.2%}")

print("\n⚠️ This is an educational ML prediction")
print("and is NOT a medical diagnosis.")