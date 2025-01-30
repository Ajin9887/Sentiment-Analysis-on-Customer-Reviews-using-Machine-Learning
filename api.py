from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
from flask_cors import CORS
import nltk

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download required NLTK data (if not already done)
nltk.download("stopwords")

# Load stopwords
STOPWORDS = set(stopwords.words("english"))

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Load model and associated objects 
    predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))

    try:
        # Check if the request contains a file or text
        if "file" in request.files:
            # Bulk prediction
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")
            return response

        elif "text" in request.form or request.json.get("text"):
            # Single string prediction
            text_input = request.form.get("text") or request.json.get("text")
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})

        else:
            return jsonify({"error": "Invalid input. Provide a file or text."})

    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(predictor, scaler, cv, text_input):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    X_prediction = cv.transform([review]).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict(X_prediction_scl)
    return "Positive" if y_predictions[0] == 1 else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", str(data.iloc[i]["Sentence"]))
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        corpus.append(" ".join(review))

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict(X_prediction_scl)
    data["Predicted Sentiment"] = ["Positive" if pred == 1 else "Negative" for pred in y_predictions]

    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph

def get_distribution_graph(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    tags = data["Predicted Sentiment"].value_counts()
    tags.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
    plt.tight_layout()

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close(fig)
    graph.seek(0)

    return graph

if __name__ == "__main__":
    app.run(port=5000, debug=True)
