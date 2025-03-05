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
        stage('Install Dependencies') {
            steps {
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
                sh 'make docker-run'
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
