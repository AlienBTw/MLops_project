pipeline {
    agent any

    environment {
        // Update these variables as needed.
        DOCKER_USERNAME = "yourname"      // Replace with your Docker username if necessary.
        IMAGE_NAME = "ahmed_louay_araour_4ds2_mlops"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Install Dependencies') {
            steps {
                // Installing dependencies using the Makefile target.
                sh 'make install-deps'
            }
        }
        stage('Run Tests') {
            steps {
                sh 'make tests'
            }
        }
        stage('Lint and Format Checks') {
            steps {
                sh 'make lint'
                sh 'make format'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'make docker-build'
            }
        }
        stage('Deploy Containers') {
            steps {
                // For example, to deploy the FastAPI container with MLflow connection.
                sh 'make docker-run'
            }
        }
    }
    
    post {
        always {
            // Clean up the workspace after the build.
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