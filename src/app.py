from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import sys
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI from environment variable
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"MLflow tracking URI: {mlflow_tracking_uri}")

# Set experiment name
experiment_name = "Ahmed-Louay-Araour-4DS2-ML"
mlflow.set_experiment(experiment_name)

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Paths to the trained model and feature columns file
MODEL_PATH = "decision_tree_model.joblib"
FEATURE_COLUMNS_PATH = "model_feature_columns.joblib"

app = FastAPI(
    title="ML Prediction API",
    description="Exposes /predict and /retrain endpoints using FastAPI."
)

# Pydantic model for the prediction request payload.
# Note: Names match the dataset column names exactly
class PredictionRequest(BaseModel):
    State: str
    Account_length: int = None  # Using "_" instead of space for field names
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

    # Field name aliases to handle the conversion between API field names and dataset column names
    class Config:
        allow_population_by_field_name = True
        fields = {
            'Account_length': {'alias': 'Account length'},
            'Area_code': {'alias': 'Area code'},
            'International_plan': {'alias': 'International plan'},
            'Voice_mail_plan': {'alias': 'Voice mail plan'},
            'Number_vmail_messages': {'alias': 'Number vmail messages'},
            'Total_day_minutes': {'alias': 'Total day minutes'},
            'Total_day_calls': {'alias': 'Total day calls'},
            'Total_day_charge': {'alias': 'Total day charge'},
            'Total_eve_minutes': {'alias': 'Total eve minutes'},
            'Total_eve_calls': {'alias': 'Total eve calls'},
            'Total_eve_charge': {'alias': 'Total eve charge'},
            'Total_night_minutes': {'alias': 'Total night minutes'},
            'Total_night_calls': {'alias': 'Total night calls'},
            'Total_night_charge': {'alias': 'Total night charge'},
            'Total_intl_minutes': {'alias': 'Total intl minutes'},
            'Total_intl_calls': {'alias': 'Total intl calls'},
            'Total_intl_charge': {'alias': 'Total intl charge'},
            'Customer_service_calls': {'alias': 'Customer service calls'}
        }

# Pydantic model for the retraining request payload.
class RetrainRequest(BaseModel):
    max_depth: int

def preprocess_input(input_data: PredictionRequest):
    """
    Preprocess raw input data using the same steps as in training.
    - Convert input data to a DataFrame.
    - Handle column name mapping
    - Apply pd.get_dummies() to mimic the training preprocessing.
    - Reindex the DataFrame to the training feature columns.
    """
    # Convert input to dict and rename fields to match dataset columns with spaces
    input_dict = input_data.dict()
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
    
    for key, value in input_dict.items():
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
        raise HTTPException(status_code=500, detail="Training feature columns not found. Please retrain the model.")
    return data

# Load the trained model and feature columns at startup.
@app.on_event("startup")
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

@app.post("/predict", summary="Make a prediction", response_description="The prediction result")
def predict(input_data: PredictionRequest):
    """
    Make a prediction using the loaded model.
    The raw input is preprocessed to match the training feature format.
    Returns a descriptive message:
      "Customer is going to churn" or "Customer is not going to churn".
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please retrain the model first.")
        
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        if prediction[0] == 1 or prediction[0] == True:
            message = "Customer is going to churn"
        else:
            message = "Customer is not going to churn"
        return {"prediction": message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain", summary="Retrain the model", response_description="The retraining status and evaluation")
def retrain(request: RetrainRequest):
    """
    Retrain the model with a new max_depth hyperparameter.
    Saves the model locally using joblib and logs it to MLflow.
    """
    try:
        # Direct imports to avoid package issues
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Define inline function to read and prepare data
        def prepare_data_internal(dataset_path):
            # Read the dataset
            df = pd.read_csv(dataset_path)
            print(f"Dataset loaded. Columns: {df.columns.tolist()}")
            
            # Use the correct target column name - "Churn" 
            target_column = "Churn"
            print(f"Using '{target_column}' as target variable")
            
            # Get features and target
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            # One-hot encode categorical features
            X = pd.get_dummies(X, drop_first=True)
            
            # Print encoded features
            print(f"Encoded features: {X.columns.tolist()}")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test, X.columns.tolist()
        
        # Path to the dataset
        dataset_path = os.path.join("datasets", "churn-bigml-80.csv")
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found at {dataset_path}")
        
        # Start MLflow run for tracking
        with mlflow.start_run() as run:
            # Log the hyperparameter
            mlflow.log_param("max_depth", request.max_depth)
            
            # Prepare data
            print(f"Reading dataset from {dataset_path}")
            X_train, X_test, y_train, y_test, feature_cols = prepare_data_internal(dataset_path)
            
            # Train model
            print(f"Training model with max_depth={request.max_depth}")
            new_model = DecisionTreeClassifier(max_depth=request.max_depth, random_state=42)
            new_model.fit(X_train, y_train)
            
            # Save model and feature columns locally with joblib
            joblib.dump(new_model, MODEL_PATH)
            joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
            print(f"Model saved to {MODEL_PATH}")
            print(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")
            
            # Evaluate model
            y_pred = new_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                new_model, 
                "model",
                registered_model_name="ChurnPredictionModel"  # Register model in MLflow
            )
            
            # Log feature columns as artifact
            temp_feature_cols_path = "temp_feature_columns.joblib"
            joblib.dump(feature_cols, temp_feature_cols_path)
            mlflow.log_artifact(temp_feature_cols_path)
            os.remove(temp_feature_cols_path)  # Clean up temporary file
            
            # Update global model reference
            global model, feature_columns
            model = new_model
            feature_columns = feature_cols
            
            return {
                "detail": "Model retrained successfully", 
                "max_depth": request.max_depth, 
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "mlflow_run_id": run.info.run_id,
                "mlflow_model_uri": f"runs:/{run.info.run_id}/model"
            }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error during retraining: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=400, detail=str(e))

# Debug route
@app.get("/debug")
def debug_info():
    """Returns debug information about the environment"""
    python_path = sys.path
    current_dir = os.getcwd()
    files_in_current_dir = os.listdir('.')
    src_files = os.listdir('src') if os.path.exists('src') else []
    
    return {
        "current_directory": current_dir,
        "python_path": python_path,
        "files_in_current_dir": files_in_current_dir,
        "src_files": src_files,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "feature_columns_path_exists": os.path.exists(FEATURE_COLUMNS_PATH),
        "dataset_path_exists": os.path.exists(os.path.join("datasets", "churn-bigml-80.csv"))
    }

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "endpoints": {
            "/predict": "Make predictions for customer churn",
            "/retrain": "Retrain the model with a new max_depth parameter",
            "/debug": "Get debug information about the environment",
            "/check-dataset": "Check the dataset structure"
        }
    }