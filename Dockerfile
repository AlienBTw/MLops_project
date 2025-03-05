FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set Python path to include app directory
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data and model files first
COPY datasets/ ./datasets/
COPY decision_tree_model.joblib* ./
COPY model_feature_columns.joblib* ./

# Copy project files
COPY . .

# Create empty __init__.py in src directory if it doesn't exist
RUN if [ ! -f "src/__init__.py" ]; then touch src/__init__.py; fi

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
