# Disease Symptom Predictor

An NLP-based machine learning project that predicts a possible disease from natural-language symptom descriptions, served through a Flask web application.

## Project Overview

The Disease Symptom Predictor uses Natural Language Processing (NLP) and machine learning to classify symptom descriptions into one of 24 disease categories.

Instead of requiring users to select symptoms from predefined checkboxes, the system accepts symptoms written in natural language through a web interface, making diagnosis-style prediction more intuitive.

> **Disclaimer:** This project is developed for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

## Objectives

* Process natural-language symptom descriptions.
* Apply NLP techniques to convert text into numerical features.
* Train and compare multiple machine learning models for disease classification.
* Analyze model errors to understand real-world limitations.
* Evaluate whether semantic (embeddings-based) understanding improves on keyword-based methods.
* Serve predictions through a Flask web application.

## Dataset

The project uses the **Symptom2Disease** dataset containing:

* **1,200 symptom descriptions**
* **24 disease classes**
* Natural-language symptom descriptions
* `text` — symptom description
* `label` — disease category

The dataset is kept locally in the `data/` directory and is excluded from the Git repository using `.gitignore`.

## Diseases Covered

Acne, Arthritis, Bronchial Asthma, Cervical spondylosis, Chicken pox, Common Cold, Dengue, Dimorphic Hemorrhoids, Fungal infection, Hypertension, Impetigo, Jaundice, Malaria, Migraine, Pneumonia, Psoriasis, Typhoid, Varicose Veins, Allergy, Diabetes, Drug reaction, Gastroesophageal reflux disease, Peptic ulcer disease, Urinary tract infection.

## Methodology

```text
Natural-Language Symptoms
          ↓
     Text Cleaning
          ↓
   TF-IDF Vectorization (unigrams + bigrams)
          ↓
    Train/Test Split (80/20, stratified)
          ↓
 Model Comparison (Naive Bayes vs Logistic Regression)
          ↓
 Best Model Selected Automatically
          ↓
 Flask Web Application
```

### 1. Text Preprocessing
Lowercasing, punctuation removal, and English stopword removal (`src/preprocess.py`).

### 2. TF-IDF Vectorization
Unigrams and bigrams are used so that two-word symptom phrases (e.g. "skin rash", "joint pain") are captured, not just single keywords.

### 3. Model Training & Comparison
Two models are trained and automatically compared, with the better-performing one saved for the app:

| Model                   | Accuracy |
| ------------------------ | -------: |
| Logistic Regression      | **95.83%** |
| Multinomial Naive Bayes  | 95.83% |

Logistic Regression was selected as the final model.

### 4. Error Analysis
A confusion matrix (`confusion_matrix.png`) and a dedicated error-analysis script (`src/error_analysis.py`) were used to inspect misclassifications.

**Key finding:** the *drug reaction* class was the weakest (71% F1-score). Inspecting the misclassified examples showed the underlying symptom text contained no medication-related signal (e.g. "fever, dizzy, heart racing, confused") — overlapping heavily with Pneumonia and diabetes symptoms. This was identified as a **data limitation**, not a model or preprocessing issue.

### 5. Semantic Embeddings Experiment
As a further experiment, a sentence-embeddings approach (`all-MiniLM-L6-v2` + Logistic Regression, in `src/train_model_embeddings.py`) was tried to see if semantic understanding would outperform keyword-based TF-IDF.

**Result:** embeddings scored **90.42% accuracy** — lower than TF-IDF (95.83%), and *drug reaction* recall dropped further (30%). This was likely due to the small dataset size (1,200 rows) and the embedding model not being fine-tuned on medical text. TF-IDF's exact keyword matching turned out to be a better fit for this dataset's short, templated symptom phrasing.

This comparison is kept in the repository as a documented, evaluated trade-off rather than a discarded experiment.

## Web Application

The final model is served through a Flask app (`src/app.py`) with a simple form-based UI (`templates/index.html`):

* User enters symptoms in plain English
* Text is cleaned and vectorized using the same TF-IDF pipeline used in training
* The saved Logistic Regression model predicts the most likely disease
* Result is displayed on the page with an educational-use disclaimer

## Project Structure

```text
disease-symptom-predictor/
│
├── src/
│   ├── explore_data.py            # Initial dataset exploration
│   ├── preprocess.py              # Text cleaning function
│   ├── train_model.py             # Trains & compares NB/LR, saves best model
│   ├── train_model_embeddings.py  # Sentence-embeddings experiment
│   ├── error_analysis.py          # Inspects misclassified examples
│   ├── predictor.py               # Loads model, exposes predict_disease()
│   └── app.py                     # Flask web application
│
├── templates/
│   └── index.html                 # Web UI
│
├── data/
│   └── Symptom2Disease.csv        # (excluded from Git)
│
├── model.pkl                      # Trained TF-IDF + Logistic Regression model
├── vectorizer.pkl                 # Fitted TF-IDF vectorizer
├── confusion_matrix.png           # Error analysis visualization
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
* Sentence-Transformers (embeddings experiment)
* Flask
* NLTK
* Matplotlib, Seaborn (confusion matrix)
* Git & GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/aniruddhasutradher07-commits/disease-symptom-predictor.git
cd disease-symptom-predictor
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download **Symptom2Disease.csv** from Kaggle and place it inside:

```text
data/Symptom2Disease.csv
```

## Running the Project

Train the model (regenerates `model.pkl`, `vectorizer.pkl`, `confusion_matrix.png`):

```bash
python src/train_model.py
```

Run the web application:

```bash
python src/app.py
```

Then open `http://127.0.0.1:5000` in your browser.

(Optional) Run the embeddings comparison experiment:

```bash
python src/train_model_embeddings.py
```

(Optional) Inspect misclassified examples:

```bash
python src/error_analysis.py
```

## Current Limitations

* The dataset is relatively small (1,200 samples across 24 classes).
* Some disease classes with overlapping generic symptoms (e.g. drug reaction, pneumonia) are harder to distinguish from text alone.
* Natural-language descriptions very different from the training distribution may produce unreliable predictions.
* The model does not provide clinical reasoning — it is a statistical classifier, not a diagnostic tool.
* A high test accuracy does not imply medical diagnostic accuracy.

## Future Improvements

* Top-3 disease predictions with confidence scores
* Confidence threshold / uncertainty rejection for low-confidence inputs
* Larger, more diverse training dataset
* Live deployment (Render)
* Improved UI/UX for the web app
* Chatbot-style multi-turn symptom clarification

## Learning Outcomes

This project demonstrates practical experience in:

* Natural Language Processing & text preprocessing
* TF-IDF feature engineering (unigrams/bigrams)
* Multi-class classification & model comparison
* Error analysis using confusion matrices
* Evaluating semantic embeddings vs classical NLP methods, and documenting a negative result
* Building and serving a model through a Flask web application
* Git and GitHub workflow

## Disclaimer

This project is intended strictly for educational purposes. Predictions generated by the system should not be interpreted as medical diagnoses, treatment recommendations, or professional medical advice. Users experiencing health concerns should consult a qualified healthcare professional.