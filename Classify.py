from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle, json
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session management

# Load model & encoders
with open("xgb_balanced_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
print("Allowed Age Groups:", label_encoders["Age Group"].classes_)
# Dropdown options
locations = label_encoders["Location"].classes_
months = label_encoders["Month of Registration"].classes_
branches = label_encoders["Branch"].classes_
age_groups = label_encoders["Age Group"].classes_
counties = label_encoders["County"].classes_

# Load users
with open("users.json", "r") as f:
    users = json.load(f)

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    confidence = None

    if request.method == "POST":
        # Get user inputs
        location = request.form["location"]
        month = request.form["month"]
        branch = request.form["branch"]
        age_group = request.form["age_group"]
        county = request.form["county"]

        # Encode inputs
        encoded = {
            "Location": label_encoders["Location"].transform([location])[0],
            "Month of Registration": label_encoders["Month of Registration"].transform([month])[0],
            "Branch": label_encoders["Branch"].transform([branch])[0],
            "Age Group": label_encoders["Age Group"].transform([age_group])[0],
            "County": label_encoders["County"].transform([county])[0]
        }

        # Create feature vector (dummy with 0s except known encoded values)
        input_vector = np.zeros(model.n_features_in_)
        for feature, val in encoded.items():
            if feature in model.feature_names_in_:
                idx = list(model.feature_names_in_).index(feature)
                input_vector[idx] = val

        # Predict
        prob = model.predict_proba([input_vector])[0][1]
        prediction = "High" if prob >= 0.5 else "Low"
        confidence = f"{prob*100:.2f}%"

    return render_template("dashboard.html",
                           username=session["user"],
                           locations=locations, months=months,
                           branches=branches, age_groups=age_groups,
                           counties=counties,
                           prediction=prediction, confidence=confidence)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ========================
# ✅ REST API Endpoint
# ========================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        input_data = request.get_json()

        # Required features
        required_features = ["Location", "Month of Registration", "Branch", "Age Group", "County"]

        for feature in required_features:
            if feature not in input_data:
                return jsonify({"error": f"Missing required field: {feature}"}), 400

        # Encode features
        encoded = {}
        for feature in required_features:
            try:
                encoded[feature] = label_encoders[feature].transform([input_data[feature]])[0]
            except:
                return jsonify({"error": f"Invalid value '{input_data[feature]}' for feature '{feature}'"}), 400

        # Create feature vector
        input_vector = np.zeros(model.n_features_in_)
        for feature, val in encoded.items():
            if feature in model.feature_names_in_:
                idx = list(model.feature_names_in_).index(feature)
                input_vector[idx] = val

        # Predict
        proba = model.predict_proba([input_vector])[0][1]
        prediction = "High" if proba >= 0.5 else "Low"

        return jsonify({
            "prediction": prediction,
            "probability": round(float(proba), 4)  # ✅ Convert to Python float to fix error
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: check API is alive
@app.route("/api", methods=["GET"])
def api_home():
    return jsonify({"message": "Sales Target API is running."})

if __name__ == "__main__":
    app.run(debug=True)
