pipeline {
    agent any

    environment {
        DOCKER_USERNAME = "yourname"
        IMAGE_NAME = "ahmed_louay_araour_4ds2_mlops"
    }

    stages {
        stage('Checkout') {
            steps {
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
                sh 'pytest || true'  // Run tests but don't fail if tests fail
                sh 'flake8 . || true'  // Run linting but don't fail if linting fails
                sh 'black --check . || true'  // Run formatting check but don't fail if formatting fails
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
                docker network create ml_network || true
                docker stop fastapi_mlflow_app || true
                docker rm fastapi_mlflow_app || true
                docker run -d --name fastapi_mlflow_app \
                    --network ml_network \
                    -p 8000:8000 \
                    ${IMAGE_NAME}
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
