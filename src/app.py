from flask import Flask, render_template, request
from predictor import predict_disease
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        symptoms = request.form["symptoms"]
        prediction = predict_disease(symptoms)
    return render_template("index.html", prediction=prediction, symptoms=symptoms)

if __name__ == "__main__":
    app.run(debug=True)