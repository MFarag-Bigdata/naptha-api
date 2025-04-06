from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load model and features
model = joblib.load("model/model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Naptha Flow Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data], columns=feature_names)
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)