.PHONY: install install-deps prepare train evaluate save load all tests lint format ci clean fastapi mlflow-ui docker-build docker-run docker-push network mlflow-container run-all stop logs docker-clean check-mlflow mlflow-build mlflow-local run-with-local-mlflow flask

# Docker configuration
DOCKER_USERNAME ?= yourname
IMAGE_NAME = ahmed_louay_araour_4ds2_mlops
NETWORK_NAME = ml_network
MLFLOW_CONTAINER = mlflow_server
FASTAPI_CONTAINER = fastapi_mlflow_app
MLFLOW_PORT = 5000
FASTAPI_PORT = 8000

# Install all dependencies from requirements.txt (manual install)
install:
	python3 -m pip install -r requirements.txt

# Target to check and install dependencies.
install-deps:
	@echo "Checking and installing required dependencies..."
	python3 -m pip install -r requirements.txt

# Run the 'prepare' step using the main script in the src folder.
prepare:
	python3 src/main.py --function prepare --dataset datasets/churn-bigml-80.csv

# Run the 'train' step using the main script in the src folder.
train:
	python3 src/main.py --function train --dataset datasets/churn-bigml-80.csv

# Run the 'evaluate' step using the main script in the src folder.
evaluate:
	python3 src/main.py --function evaluate --dataset datasets/churn-bigml-80.csv

# Run the 'save' step using the main script in the src folder.
save:
	python3 src/main.py --function save --dataset datasets/churn-bigml-80.csv --model_filename decision_tree_model.joblib

# Run the 'load' step using the main script in the src folder.
load:
	python3 src/main.py --function load --model_filename decision_tree_model.joblib

# Run the full pipeline (all steps) using the main script in the src folder.
# This target first checks dependencies, then runs the pipeline.
all: install-deps
	python3 src/main.py --function all --dataset datasets/churn-bigml-80.csv --model_filename decision_tree_model.joblib

# Run basic tests (requires pytest).
tests:
	pytest

# Code quality check using flake8.
lint:
	flake8 .

# Code formatting check using black.
format:
	black --check .

# CI target runs lint, format, and tests.
ci: lint format tests

# Clean up generated cache files.
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Create Docker network for services
network:
	@echo "Creating Docker network $(NETWORK_NAME) if it doesn't exist..."
	@docker network inspect $(NETWORK_NAME) >/dev/null 2>&1 || docker network create $(NETWORK_NAME)

# Check if MLflow container is already running
check-mlflow:
	@echo "Checking if MLflow is already running..."
	@if docker ps -a --format '{{.Names}}' | grep -q "$(MLFLOW_CONTAINER)"; then \
		if docker ps --format '{{.Names}}' | grep -q "$(MLFLOW_CONTAINER)"; then \
			echo "MLflow container is already running."; \
			docker network connect $(NETWORK_NAME) $(MLFLOW_CONTAINER) 2>/dev/null || true; \
		else \
			echo "MLflow container exists but is not running. Starting it..."; \
			docker start $(MLFLOW_CONTAINER); \
		fi; \
		exit 0; \
	fi; \
	echo "MLflow container does not exist, will create a new one."

# Build local MLflow image
mlflow-build:
	@echo "Building local MLflow image..."
	@echo 'FROM python:3.9-slim' > MLflowDockerfile
	@echo 'RUN python3 -m pip install mlflow==2.7.1 psycopg2-binary' >> MLflowDockerfile
	@echo 'EXPOSE 5000' >> MLflowDockerfile
	@echo 'WORKDIR /mlflow' >> MLflowDockerfile
	@echo 'CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]' >> MLflowDockerfile
	docker build -t local-mlflow -f MLflowDockerfile .

# Run MLflow container
mlflow-container: network mlflow-build
	@echo "Checking if MLflow container is already running..."
	@if docker ps --format '{{.Names}}' | grep -q "$(MLFLOW_CONTAINER)"; then \
		echo "MLflow container is already running. Skipping container creation."; \
	elif docker ps -a --format '{{.Names}}' | grep -q "$(MLFLOW_CONTAINER)"; then \
		echo "MLflow container exists but is not running. Starting it..."; \
		docker start $(MLFLOW_CONTAINER); \
	else \
		echo "Creating new MLflow container on port $(MLFLOW_PORT)..."; \
		docker run -d --name $(MLFLOW_CONTAINER) \
			--network $(NETWORK_NAME) \
			-p $(MLFLOW_PORT):$(MLFLOW_PORT) \
			-v $(PWD)/mlruns:/mlruns \
			local-mlflow \
			mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) \
			--backend-store-uri sqlite:///mlruns/mlflow.db \
			--default-artifact-root file:///mlruns; \
	fi

# Run MLflow locally
mlflow-local:
	@echo "Starting MLflow server locally on port $(MLFLOW_PORT)..."
	mkdir -p mlruns
	mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root file:///$(shell pwd)/mlruns

# Run the FastAPI application using Uvicorn directly.
fastapi:
	uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Run MLflow UI with SQLite backend directly.
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# Docker Build: Build the Docker image with your name format
docker-build:
	docker build -t $(IMAGE_NAME) .

# Docker Run: Run the FastAPI container with MLflow connection
docker-run: network
	@echo "Stopping and removing any existing container..."
	docker stop $(FASTAPI_CONTAINER) 2>/dev/null || true
	docker rm $(FASTAPI_CONTAINER) 2>/dev/null || true
	@echo "Starting FastAPI container with MLflow connection..."
	docker run -d --name $(FASTAPI_CONTAINER) \
		--network $(NETWORK_NAME) \
		-p $(FASTAPI_PORT):$(FASTAPI_PORT) \
		-e MLFLOW_TRACKING_URI=http://$(MLFLOW_CONTAINER):$(MLFLOW_PORT) \
		$(IMAGE_NAME)

# Run both MLflow and FastAPI containers
run-all: mlflow-container docker-run
	@echo "All services started!"
	@echo "FastAPI available at: http://localhost:$(FASTAPI_PORT)"
	@echo "FastAPI Swagger UI: http://localhost:$(FASTAPI_PORT)/docs"
	@echo "MLflow UI available at: http://localhost:$(MLFLOW_PORT)"

# Update the run-with-local-mlflow target to use host.docker.internal
run-with-local-mlflow: network docker-build
	@echo "Stopping and removing any existing FastAPI container..."
	docker stop $(FASTAPI_CONTAINER) 2>/dev/null || true
	docker rm $(FASTAPI_CONTAINER) 2>/dev/null || true
	@echo "Starting FastAPI container connected to MLflow on the host..."
	docker run -d --name $(FASTAPI_CONTAINER) \
		--network $(NETWORK_NAME) \
		--add-host=host.docker.internal:host-gateway \
		-p $(FASTAPI_PORT):$(FASTAPI_PORT) \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:$(MLFLOW_PORT) \
		$(IMAGE_NAME)
	@echo "FastAPI container started and connected to host MLflow!"
	@echo "FastAPI available at: http://localhost:$(FASTAPI_PORT)"
	@echo "FastAPI Swagger UI: http://localhost:$(FASTAPI_PORT)/docs"
	@echo "MLflow UI should be running at: http://localhost:$(MLFLOW_PORT)"
	
# New target to run the Flask application.
flask:
	@echo "Starting Flask application..."
	python3 src/app_flask.py

# Stop all containers
stop:
	@echo "Stopping containers..."
	docker stop $(FASTAPI_CONTAINER) 2>/dev/null || true
	docker stop $(MLFLOW_CONTAINER) 2>/dev/null || true
	docker rm $(FASTAPI_CONTAINER) 2>/dev/null || true
	docker rm $(MLFLOW_CONTAINER) 2>/dev/null || true

# View logs
logs:
	@echo "FastAPI container logs:"
	docker logs $(FASTAPI_CONTAINER)
	@echo "\nMLflow container logs:"
	docker logs $(MLFLOW_CONTAINER) 2>/dev/null || echo "MLflow container not found"

# Clean Docker resources
docker-clean: stop
	@echo "Removing Docker network $(NETWORK_NAME)..."
	docker network rm $(NETWORK_NAME) 2>/dev/null || true
	@echo "Cleaning up any dangling images..."
	docker image prune -f

# Docker Push: Tag and push the Docker image to Docker Hub
docker-push:
	docker tag $(IMAGE_NAME) $(DOCKER_USERNAME)/$(IMAGE_NAME)
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME)
