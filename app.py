from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained objects
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return "Football Match Prediction API is running"

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["FTH Goals"], data["FTA Goals"],
        data["H Shots"], data["A Shots"],
        data["H SOT"], data["A SOT"],
        data["H Fouls"], data["A Fouls"],
        data["H Corners"], data["A Corners"],
        data["H Yellow"], data["A Yellow"],
        data["H Red"], data["A Red"]
    ]])

    # Scale features
    features = scaler.transform(features)

    # Prediction
    pred = rf.predict(features)[0]
    result = le.inverse_transform([pred])[0]

    # 🔥 NEW: Prediction confidence
    proba = rf.predict_proba(features)[0]
    confidence = round(max(proba) * 100, 2)

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

