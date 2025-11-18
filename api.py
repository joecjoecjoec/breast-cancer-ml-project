import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open("model_rf.bin", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # GET: just show a simple help message (for browser)
    if request.method == "GET":
        return (
            "Breast cancer prediction API is running. "
            "Send a POST request with JSON like:<br>"
            "<code>{\"features\": [30 numbers]}</code>",
            200,
        )

    # POST: real prediction
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "JSON must contain 'features' field"}), 400

    features = data["features"]

    # Expect exactly 30 numerical features
    x = np.array(features).reshape(1, -1)
    proba = float(model.predict_proba(x)[0, 1])

    return jsonify({"malignant_probability": proba})


if __name__ == "__main__":
    # 0.0.0.0 so it works inside Docker
    app.run(host="0.0.0.0", port=5000)