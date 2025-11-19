import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open("model_rf.bin", "rb") as f:
    model = pickle.load(f)


# -----------------------
#   Health check route
# -----------------------
@app.route("/", methods=["GET"])
def index():
    return (
        "Breast cancer prediction API is running. "
        "Use POST /predict with JSON: {\"features\": [30 numbers]}",
        200,
    )


# -----------------------
#   Prediction route
# -----------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # GET — show simple help message
    if request.method == "GET":
        return (
            "Prediction endpoint. Send a POST request with JSON like:<br>"
            "<code>{\"features\": [30 numbers]}</code>",
            200,
        )

    # POST — real prediction
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "JSON must contain 'features' field"}), 400

    features = data["features"]

    # Expect exactly 30 features
    try:
        x = np.array(features).reshape(1, -1)
    except:
        return jsonify({"error": "Features must be a list of 30 numerical values"}), 400

    proba = float(model.predict_proba(x)[0, 1])

    return jsonify({"malignant_probability": proba})


# -----------------------
#   Start the server
# -----------------------
if __name__ == "__main__":
    # host=0.0.0.0 is REQUIRED for Docker/Render!
    app.run(host="0.0.0.0", port=5000)