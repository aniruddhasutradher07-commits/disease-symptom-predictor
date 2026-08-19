# Disease Symptom Predictor

An NLP-based machine learning project that predicts a possible disease from natural-language symptom descriptions.

## Project Overview

The Disease Symptom Predictor uses Natural Language Processing (NLP) and machine learning to classify symptom descriptions into one of 24 disease categories.

Instead of requiring users to select symptoms from predefined checkboxes, the system accepts symptoms written in natural language, making it more suitable for a future chatbot-style interface.

> **Disclaimer:** This project is developed for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

## Objectives

* Process natural-language symptom descriptions.
* Apply NLP techniques to convert text into numerical features.
* Train machine learning models for disease classification.
* Compare different classification algorithms.
* Develop a foundation for a future web-based symptom prediction application.

## Dataset

The project uses the **Symptom2Disease** dataset containing:

* **1,200 symptom descriptions**
* **24 disease classes**
* Natural-language symptom descriptions
* `text` — symptom description
* `label` — disease category

The dataset is kept locally in the `data/` directory and is excluded from the Git repository using `.gitignore`.

## Diseases Covered

The dataset contains the following 24 disease categories:

* Acne
* Arthritis
* Bronchial Asthma
* Cervical spondylosis
* Chicken pox
* Common Cold
* Dengue
* Dimorphic Hemorrhoids
* Fungal infection
* Hypertension
* Impetigo
* Jaundice
* Malaria
* Migraine
* Pneumonia
* Psoriasis
* Typhoid
* Varicose Veins
* Allergy
* Diabetes
* Drug reaction
* Gastroesophageal reflux disease
* Peptic ulcer disease
* Urinary tract infection

## Methodology

The current machine learning pipeline follows these steps:

```text
Natural-Language Symptoms
          ↓
     Text Cleaning
          ↓
   TF-IDF Vectorization
          ↓
    Train/Test Split
          ↓
 Machine Learning Models
          ↓
 Disease Classification
```

### 1. Text Preprocessing

The symptom descriptions are converted to lowercase and unnecessary characters are removed.

### 2. TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) is used to transform symptom descriptions into numerical feature vectors.

The current configuration uses:

* English stop-word removal
* Unigrams and bigrams
* Maximum 5,000 features

### 3. Train-Test Split

The dataset is divided into:

* **80% training data:** 960 samples
* **20% testing data:** 240 samples

Stratified splitting is used to maintain class distribution.

### 4. Machine Learning Models

Two baseline classification algorithms were evaluated:

* Logistic Regression
* Multinomial Naive Bayes

## Model Performance

Both models achieved **95% accuracy** on the held-out test set.

| Model                   |   Accuracy |
| ----------------------- | ---------: |
| Logistic Regression     | **95.00%** |
| Multinomial Naive Bayes | **95.00%** |

### Classification Performance

Logistic Regression achieved:

* Accuracy: **0.95**
* Macro F1-score: **0.95**
* Weighted F1-score: **0.95**

Multinomial Naive Bayes achieved:

* Accuracy: **0.95**
* Macro F1-score: **0.95**
* Weighted F1-score: **0.95**

Some disease classes showed lower recall than others, demonstrating that further error analysis and model improvement are required.

## Project Structure

```text
disease-symptom-predictor/
│
├── src/
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   └── Symptom2Disease.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

> The `data/` directory is intentionally excluded from GitHub through `.gitignore`.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Flask
* NLTK
* TF-IDF
* Logistic Regression
* Multinomial Naive Bayes
* Git & GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/aniruddhasutradher07-commits/disease-symptom-predictor.git
cd disease-symptom-predictor
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the `Symptom2Disease.csv` file inside:

```text
data/Symptom2Disease.csv
```

The dataset is not included in this repository because the `data/` directory is excluded through `.gitignore`.

## Running the Project

### Explore the dataset

```bash
python src/explore_data.py
```

### Run preprocessing

```bash
python src/preprocess.py
```

### Train and evaluate models

```bash
python src/train.py
```

### Test disease prediction

```bash
python src/predict.py
```

The prediction script accepts a natural-language symptom description and returns a predicted disease along with the model confidence.

## Example

Example input:

```text
I have a headache with nausea and sensitivity to light
```

The model generates a prediction based on patterns learned from the training dataset.

Because the current model can produce low-confidence predictions for unfamiliar symptom descriptions, future versions will include an uncertainty threshold rather than forcing a prediction.

## Current Limitations

* The dataset is relatively small.
* The model is trained only on the diseases represented in the dataset.
* Natural-language descriptions outside the dataset may produce unreliable predictions.
* The model does not provide clinical reasoning.
* A high test accuracy does not imply medical diagnostic accuracy.
* The current system does not yet have a confidence-based rejection mechanism.
* The current version is a command-line application rather than a complete web application.

## Future Improvements

Planned improvements include:

* Top-3 disease predictions
* Confidence threshold and uncertainty detection
* Confusion matrix visualization
* Detailed error analysis
* Improved NLP preprocessing
* Hyperparameter tuning
* Model comparison and cross-validation
* Model serialization using Joblib
* Flask-based web interface
* Chatbot-style symptom input
* User-friendly prediction interface
* Responsible AI and medical safety messaging

## Learning Outcomes

This project demonstrates practical experience in:

* Natural Language Processing
* Text preprocessing
* Feature engineering
* TF-IDF vectorization
* Supervised machine learning
* Multi-class classification
* Model evaluation
* Python programming
* Git and GitHub workflow

## Disclaimer

This project is intended strictly for educational purposes. Predictions generated by the system should not be interpreted as medical diagnoses, treatment recommendations, or professional medical advice. Users experiencing health concerns should consult a qualified healthcare professional.
