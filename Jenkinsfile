pipeline {
    agent any

    options {
        // Clean workspace before each build
        skipDefaultCheckout(true)
    }

    environment {
        DOCKER_USERNAME = "yourname"
        IMAGE_NAME = "ahmed_louay_araour_4ds2_mlops"
        // Set MLflow tracking URI to the local instance running on the host
        MLFLOW_TRACKING_URI = "http://localhost:5000"
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
        stage('Deploy Model Pipeline') {
            steps {
                sh '''
                # Create network if needed (optional)
                docker network create ml_network || true

                # Stop and remove any existing model_pipeline container.
                docker stop model_pipeline || true
                docker rm model_pipeline || true

                # Launch the container that runs your model pipeline.
                # It is assumed that your model_pipeline.py is in the image's root directory.
                docker run -d --name model_pipeline \\
                    --network ml_network \\
                    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
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
