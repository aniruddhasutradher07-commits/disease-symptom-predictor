import joblib
import os
from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

def predict_disease(symptom_text):
    cleaned = clean_text(symptom_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction