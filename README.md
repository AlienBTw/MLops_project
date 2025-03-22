# MLops_project

## Project Description
This project is aimed at implementing Machine Learning Operations (MLOps) to streamline the deployment, monitoring, and management of machine learning models. The project integrates seamlessly with various CI/CD pipelines, allowing for automated testing, deployment, and scaling of ML models.

## Installation Instructions
To set up this project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AlienBTw/MLops_project.git
   cd MLops_project
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions
To use this project, follow these steps:

1. **Run the main script**:
   ```bash
   python main.py
   ```

2. **Run the full pipeline**:
   ```bash
   make all
   ```

3. **Build and push Docker image**:
   ```bash
   make docker-push
   ```

4. **Launch Flask and create Docker container**:
   ```bash
   make flask-docker-push
   ```

5. **Run FastAPI with local MLflow**:
   ```bash
   make run-with-local-mlflow
   ```

6. **Access the web interface** (if applicable):
   Open your web browser and go to `http://localhost:5000`.

7. **Monitor the logs**:
   Logs are generated in the `logs` directory. You can monitor them using:
   ```bash
   tail -f logs/app.log
   ```

### Jenkins Setup
For CI/CD, you can use Jenkins to automate the process. Follow these steps to configure Jenkins:

1. **Install Jenkins**: Follow the official Jenkins installation guide for your platform.
2. **Create a new Jenkins Pipeline**: Use the provided `Jenkinsfile` for your pipeline configuration.
3. **Configure Docker Credentials**: Make sure to set up Docker credentials in Jenkins for pushing images to Docker Hub.
4. **Run the Pipeline**: Trigger the Jenkins pipeline to automate the build, test, and deployment process.

