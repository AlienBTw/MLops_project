import os
import sys

# Add the parent directory to PYTHONPATH so that the "src" module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
from src.model_pipeline import send_email_notification, prepare_data, save_model, evaluate_model

# Define paths for the saved model and feature columns.
MODEL_PATH = os.path.join(parent_dir, "decision_tree_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(parent_dir, "model_feature_columns.joblib")

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load the trained model.
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please train the model.")
model = joblib.load(MODEL_PATH)
print(f"Flask: Model loaded from {MODEL_PATH}")

# Load feature columns used for training.
if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise RuntimeError(f"Feature columns file not found at {FEATURE_COLUMNS_PATH}. Please retrain the model.")
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
print(f"Flask: Feature columns loaded from {FEATURE_COLUMNS_PATH}")

def preprocess_input(data_dict):
    """
    Preprocess the input dictionary:
      - Converts the dictionary to a one-row DataFrame.
      - Applies one-hot encoding (using pd.get_dummies() with drop_first=True).
      - Reindexes the DataFrame to use the training feature columns (filling missing columns with 0).
    """
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

@app.route("/")
def index():
    return "Flask Interface for the ML Model. Use /predict for predictions and /retrain for retraining."

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON payload with the following fields:
      - State (string)
      - Account_length (integer)
      - Area_code (integer)
      - International_plan (string)
      - Voice_mail_plan (string)
      - Number_vmail_messages (integer)
      - Total_day_minutes (float)
      - Total_day_calls (integer)
      - Total_day_charge (float)
      - Total_eve_minutes (float)
      - Total_eve_calls (integer)
      - Total_eve_charge (float)
      - Total_night_minutes (float)
      - Total_night_calls (integer)
      - Total_night_charge (float)
      - Total_intl_minutes (float)
      - Total_intl_calls (integer)
      - Total_intl_charge (float)
      - Customer_service_calls (integer)
    Returns a descriptive prediction message.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input, JSON payload required"}), 400
    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        if prediction[0] == 1:
            message = "Customer is going to churn"
        else:
            message = "Customer is not going to churn"
        return jsonify({"prediction": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Expects a JSON payload with the following field:
      - max_depth (integer)
    Retrains the model using the provided max_depth, evaluates the new model, saves it along with the new
    feature columns, and sends an email notification with the retraining details.
    """
    data = request.get_json()
    if not data or "max_depth" not in data:
        return jsonify({"error": "Invalid input, JSON payload with 'max_depth' required"}), 400
    try:
        dataset_path = os.path.join(parent_dir, "datasets/churn-bigml-80.csv")
        # Prepare data for retraining.
        X_train, X_test, y_train, y_test = prepare_data(dataset_path)
        from sklearn.tree import DecisionTreeClassifier
        new_model = DecisionTreeClassifier(max_depth=data["max_depth"], random_state=42)
        new_model.fit(X_train, y_train)
        # Save the new model.
        save_model(new_model, MODEL_PATH)
        # Save the updated training feature columns.
        joblib.dump(X_train.columns.tolist(), FEATURE_COLUMNS_PATH)
        print(f"Flask: Training feature columns saved to {FEATURE_COLUMNS_PATH}")
        # Evaluate the new model.
        accuracy = evaluate_model(new_model, X_test, y_test)
        # Send an email with the retraining results.
        subject = "Model Retrained and Evaluated (Flask Interface)"
        message = (
            f"Retraining completed successfully.\n"
            f"New max_depth: {data['max_depth']}\n"
            f"Evaluation Accuracy: {accuracy:.4f}"
        )
        send_email_notification(subject, message)
        # Update global variables.
        global model, feature_columns
        model = new_model
        feature_columns = X_train.columns.tolist()
        return jsonify({"detail": "Model retrained successfully", "max_depth": data["max_depth"], "accuracy": accuracy})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Start the Flask REST API server on 0.0.0.0:5000
    app.run(debug=True, host="0.0.0.0", port=5000)