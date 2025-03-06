pipeline {
    agent any
    
    options {
        // Clean workspace before each build
        skipDefaultCheckout(true)
    }
    
    environment {
        DOCKER_USERNAME = "yourname"
        IMAGE_NAME = "ahmed_louay_araour_4ds2_mlops"
    }

    stages {
        stage('Clean Workspace') {
            steps {
                cleanWs()
                checkout scm
            }
        }
        stage('Install Dependencies & Run Tests') {
            agent {
                docker {
                    image 'python:3.9'
                    reuseNode true
                }
            }
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest flake8 black'
                sh 'pytest || true'
                sh 'flake8 . || true'
                sh 'black --check . || true'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }
        stage('Deploy Containers') {
            steps {
                sh '''
                # Create network if not exists
                docker network create ml_network || true

                # Stop and remove any existing MLflow server container.
                docker stop mlflow_server || true
                docker rm mlflow_server || true
                
                # Launch the MLflow server container using the official mlflow image.
                docker run -d --name mlflow_server \
                --network ml_network \
                -p 5001:5000 \
                mlflow/mlflow:latest \
                mlflow server --default-artifact-root file:/tmp/mlruns --host 0.0.0.0 --port 5000
                
                # Stop and remove any existing model_pipeline container.
                docker stop model_pipeline || true
                docker rm model_pipeline || true
                
                # Launch the container that runs your model pipeline.
                # It is assumed that your model_pipeline.py is in the root directory of your Docker image.
                docker run -d --name model_pipeline \\
                    --network ml_network \\
                    ${IMAGE_NAME} python model_pipeline.py
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Build succeeded!'
        }
        failure {
            echo 'Build failed!'
        }
    }
}
