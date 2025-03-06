FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY src/ ./src/
COPY decision_tree_model.joblib* ./
COPY model_feature_columns.joblib* ./

# Expose the port used by app_flask.py (in your file, it runs on port 5001)
EXPOSE 5001

# Set the MLflow tracking URI to point to the host machine (adjust if necessary)
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5000"

# Run the Flask application by launching app_flask.py
CMD ["python", "src/app_flask.py"]
