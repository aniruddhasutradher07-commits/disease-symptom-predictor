import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report


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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_tfidf, y_train)

logistic_predictions = logistic_model.predict(X_test_tfidf)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print("Accuracy:", logistic_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_predictions = nb_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

print("\n==============================")
print("MULTINOMIAL NAIVE BAYES")
print("==============================")

print("Accuracy:", nb_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        nb_predictions
    )
)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    f"Logistic Regression: {logistic_accuracy:.4f}"
)

print(
    f"Naive Bayes:         {nb_accuracy:.4f}"
)