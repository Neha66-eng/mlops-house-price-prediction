from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

FEATURES = ["area", "bedrooms", "bathrooms", "age", "location"]
LOCATIONS = ["urban", "suburban", "rural"]


def validate(data):
    """Return (clean_dict, error_message)."""
    try:
        clean = {
            "area": float(data.get("area")),
            "bedrooms": int(data.get("bedrooms")),
            "bathrooms": int(data.get("bathrooms")),
            "age": int(data.get("age")),
            "location": str(data.get("location", "")).lower(),
        }
    except (TypeError, ValueError):
        return None, "All fields are required and must be valid numbers."

    if not 100 <= clean["area"] <= 10000:
        return None, "Area must be between 100 and 10000 sq ft."
    if not 1 <= clean["bedrooms"] <= 10:
        return None, "Bedrooms must be between 1 and 10."
    if not 1 <= clean["bathrooms"] <= 10:
        return None, "Bathrooms must be between 1 and 10."
    if not 0 <= clean["age"] <= 100:
        return None, "Age must be between 0 and 100 years."
    if clean["location"] not in LOCATIONS:
        return None, f"Location must be one of {LOCATIONS}."
    return clean, None


def predict_price(clean):
    df = pd.DataFrame([clean], columns=FEATURES)
    return round(float(model.predict(df)[0]), 2)


# Route 1: home page
@app.route("/")
def home():
    return render_template("index.html", locations=LOCATIONS)


# Route 2: health check
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# Route 3: form prediction (web UI)
@app.route("/predict", methods=["POST"])
def predict():
    clean, error = validate(request.form)
    if error:
        return render_template("index.html", locations=LOCATIONS, error=error,
                               form=request.form), 400
    price = predict_price(clean)
    return render_template("index.html", locations=LOCATIONS, prediction=price,
                           form=request.form)


# Route 4: REST API prediction (JSON)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
    clean, error = validate(data)
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"predicted_price": predict_price(clean), "input": clean})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
