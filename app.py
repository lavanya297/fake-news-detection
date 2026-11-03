
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "model.pkl"

FEATURES = [
    "source_credibility",
    "author_credibility",
    "clickbait_score",
    "sentiment_score",
    "article_length",
    "num_quotes",
    "num_images",
    "num_links",
    "social_shares",
    "verified_source"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "new model deployed successfully"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Train the model first"}), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Provide valid JSON"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    try:
        values = {feature: float(data[feature]) for feature in FEATURES}

        if not all(pd.notna(value) for value in values.values()):
            raise ValueError("Features must be finite numbers")

        import math
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Features must be finite numbers")

        if any(value < 0 for value in [
            values["source_credibility"],
            values["author_credibility"],
            values["clickbait_score"],
            values["article_length"],
            values["num_quotes"],
            values["num_images"],
            values["num_links"],
            values["social_shares"],
            values["verified_source"]
        ]):
            raise ValueError("Features cannot be negative")

        if any(values[name] > 1 for name in [
            "source_credibility",
            "author_credibility",
            "clickbait_score",
            "verified_source"
        ]):
            raise ValueError("Credibility, clickbait and verified_source must be between 0 and 1")

        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([values], columns=FEATURES)

        prediction = int(model.predict(input_df)[0])

        return jsonify({
            "prediction": prediction,
            "news_type": "Fake" if prediction == 1 else "Real",
            "message": "Prediction generated successfully",
            "warning": "This is a model prediction, not independent fact-checking."
        })

    except (ValueError, TypeError, OverflowError):
        return jsonify({
            "error": "All features must contain valid numeric values"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
