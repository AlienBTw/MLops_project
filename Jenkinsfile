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
        // Rest of your stages...
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
