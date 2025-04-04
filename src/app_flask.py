from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import pandas as pd
import sys
import mlflow
import mlflow.sklearn
from datetime import datetime

# Add the project root to Python path (before the import)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our Elasticsearch logger
from logging_config import logger

# Set MLflow tracking URI from environment variable
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"MLflow tracking URI: {mlflow_tracking_uri}")

# Set experiment name
experiment_name = "Ahmed-Louay-Araour-4DS2-ML"
mlflow.set_experiment(experiment_name)

# Paths to the trained model and feature columns file
MODEL_PATH = "decision_tree_model.joblib"
FEATURE_COLUMNS_PATH = "model_feature_columns.joblib"

app = Flask(__name__)

# Initialize global variables for model and feature columns
model = None
feature_columns = None

def preprocess_input(input_data):
    """
    Preprocess raw input data using the same steps as in training.
    - Convert input data to a DataFrame.
    - Handle column name mapping.
    - Apply pd.get_dummies() to mimic the training preprocessing.
    - Reindex the DataFrame to the training feature columns.
    """
    renamed_dict = {}
    
    field_mapping = {
        'Account_length': 'Account length',
        'Area_code': 'Area code',
        'International_plan': 'International plan',
        'Voice_mail_plan': 'Voice mail plan',
        'Number_vmail_messages': 'Number vmail messages',
        'Total_day_minutes': 'Total day minutes',
        'Total_day_calls': 'Total day calls',
        'Total_day_charge': 'Total day charge',
        'Total_eve_minutes': 'Total eve minutes',
        'Total_eve_calls': 'Total eve calls',
        'Total_eve_charge': 'Total eve charge',
        'Total_night_minutes': 'Total night minutes',
        'Total_night_calls': 'Total night calls',
        'Total_night_charge': 'Total night charge',
        'Total_intl_minutes': 'Total intl minutes',
        'Total_intl_calls': 'Total intl calls',
        'Total_intl_charge': 'Total intl charge',
        'Customer_service_calls': 'Customer service calls'
    }
    
    for key, value in input_data.items():
        if key in field_mapping:
            renamed_dict[field_mapping[key]] = value
        else:
            renamed_dict[key] = value
    
    data = pd.DataFrame([renamed_dict])
    data = pd.get_dummies(data, drop_first=True)
    
    if os.path.exists(FEATURE_COLUMNS_PATH):
        training_columns = joblib.load(FEATURE_COLUMNS_PATH)
        data = data.reindex(columns=training_columns, fill_value=0)
    else:
        return None, "Training feature columns not found. Please train the model first."
    return data, None

def prepare_data_internal(dataset_path):
    """
    Read the CSV dataset, perform one-hot encoding (using pd.get_dummies),
    and split the data into training and testing sets.
    Returns training and testing features/labels and a list of feature columns.
    """
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Columns: {df.columns.tolist()}")
    target_column = "Churn"
    print(f"Using '{target_column}' as target variable")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)
    print(f"Encoded features: {X.columns.tolist()}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def load_resources():
    global model, feature_columns
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}.")
        model = None
    else:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")

    if not os.path.exists(FEATURE_COLUMNS_PATH):
        print(f"Warning: Feature columns file not found at {FEATURE_COLUMNS_PATH}.")
        feature_columns = None
    else:
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        print(f"Training feature columns loaded from {FEATURE_COLUMNS_PATH}")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Make a prediction using the loaded model.
    The raw input is preprocessed to match the training feature format.
    Returns a descriptive message:
      "Customer is going to churn" or "Customer is not going to churn".
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Please train the model first."}), 500
        
        input_data = request.get_json()
        
        required_fields = ["State", "Area_code", "International_plan", "Voice_mail_plan",
                           "Number_vmail_messages", "Total_day_minutes", "Total_day_calls",
                           "Total_day_charge", "Total_eve_minutes", "Total_eve_calls",
                           "Total_eve_charge", "Total_night_minutes", "Total_night_calls",
                           "Total_night_charge", "Total_intl_minutes", "Total_intl_calls",
                           "Total_intl_charge", "Customer_service_calls"]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        processed_data, error = preprocess_input(input_data)
        if error:
            return jsonify({"error": error}), 400
            
        prediction = model.predict(processed_data)
        if prediction[0] == 1 or prediction[0] is True:
            message = "Customer is going to churn"
        else:
            message = "Customer is not going to churn"
        return jsonify({"prediction": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/train-evaluate", methods=["POST"])
def train_and_evaluate():
    """
    Prepares the data, trains the model with a specified max_depth (if provided), 
    evaluates the model performance on the test set, logs metrics to MLflow,
    and sends log events to Elasticsearch.
    Returns training details and evaluation metrics.
    """
    try:
        request_data = request.get_json()
        max_depth = request_data.get('max_depth', None)
        if max_depth is not None and not isinstance(max_depth, int):
            return jsonify({"error": "max_depth must be an integer"}), 400
        
        # Set a default max_depth if not provided
        if max_depth is None:
            max_depth = 5
        
        dataset_path = os.path.join("datasets", "churn-bigml-80.csv")
        if not os.path.exists(dataset_path):
            error_msg = f"Dataset not found at {dataset_path}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 404
        
        # Prepare the data       
        X_train, X_test, y_train, y_test, feature_cols = prepare_data_internal(dataset_path)
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        with mlflow.start_run() as run:
            mlflow.log_param("max_depth", max_depth)
            logger.info(f"Training model with max_depth={max_depth}")
            new_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            new_model.fit(X_train, y_train)
            
            # Save the model and feature columns
            joblib.dump(new_model, MODEL_PATH)
            joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
            logger.info(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")
            
            # Evaluate the model
            y_pred = new_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            mlflow.sklearn.log_model(
                new_model, 
                "model",
                registered_model_name="ChurnPredictionModel"
            )
            
            # Optionally log feature columns as an artifact
            temp_feature_cols_path = "temp_feature_columns.joblib"
            joblib.dump(feature_cols, temp_feature_cols_path)
            mlflow.log_artifact(temp_feature_cols_path)
            os.remove(temp_feature_cols_path)
            
            # Update global variables after training
            global model, feature_columns
            model = new_model
            feature_columns = feature_cols
            
            # Log evaluation results to Elasticsearch via our logger
            logger.info(f"Model evaluation metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            
            return jsonify({
                "detail": "Model trained and evaluated successfully", 
                "max_depth": max_depth, 
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "mlflow_run_id": run.info.run_id,
                "mlflow_model_uri": f"runs:/{run.info.run_id}/model"
            })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error during training and evaluation: {str(e)}\n{error_trace}")
        return jsonify({"error": str(e)}), 400

@app.route("/debug")
def debug_info():
    """Returns debug information about the environment"""
    python_path = sys.path
    current_dir = os.getcwd()
    files_in_current_dir = os.listdir('.')
    src_files = os.listdir('src') if os.path.exists('src') else []
    
    return jsonify({
        "current_directory": current_dir,
        "python_path": python_path,
        "files_in_current_dir": files_in_current_dir,
        "src_files": src_files,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "feature_columns_path_exists": os.path.exists(FEATURE_COLUMNS_PATH),
        "dataset_path_exists": os.path.exists(os.path.join("datasets", "churn-bigml-80.csv"))
    })

@app.route("/")
def root():
    # Serve index.html directly from the same directory as app_flask.py
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    # Manually load resources before starting the server.
    load_resources()
    # Bind to 0.0.0.0 to accept connections from outside the container.
    app.run(debug=True, host="0.0.0.0", port=5001)