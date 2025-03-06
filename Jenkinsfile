pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
    }
    
    environment {
        IMAGE_NAME = "ahmed_louay_araour_4ds2_mlops"
        // Your local MLflow server tracking URI from your Makefile logic
        MLFLOW_TRACKING_URI = "http://localhost:5000"
        // Ensure DOCKER_USERNAME and DOCKER_PASSWORD are configured in your Jenkins environment or credentials
        DOCKER_USERNAME = credentials('alienbtw')
        DOCKER_PASSWORD = credentials('9533985tkl')
    }

    stages {
        stage('Clean Workspace and Checkout') {
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
        stage('Run Full Pipeline') {
            agent {
                docker {
                    image 'python:3.9'
                    reuseNode true
                }
            }
            steps {
                // Run the full pipeline as defined by the "all" target in your Makefile.
                sh 'make all'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }
        stage('Deploy Container') {
            steps {
                sh '''
                # Stop and remove any existing container running the model pipeline.
                docker stop model_pipeline || true
                docker rm model_pipeline || true

                # Deploy the container using the built Docker image.
                # It will run the model pipeline code, and connect to the local MLflow server.
                docker run -d --name model_pipeline \\
                    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                    ${IMAGE_NAME} python model_pipeline.py
                '''
            }
        }
        stage('Push to Docker Hub') {
            steps {
                sh '''
                # Login to Docker Hub using the provided credentials.
                echo ${DOCKER_PASSWORD} | docker login -u ${DOCKER_USERNAME} --password-stdin
                # Tag the image for Docker Hub.
                docker tag ${IMAGE_NAME} ${DOCKER_USERNAME}/${IMAGE_NAME}
                # Push the image to Docker Hub.
                docker push ${DOCKER_USERNAME}/${IMAGE_NAME}
                '''
            }
        }
        stage('Check Logs') {
            steps {
                sh '''
                echo "Waiting for container to initialize..."
                sleep 10
                echo "------ Container Logs ------"
                docker logs model_pipeline || true
                '''
            }
        }
    }

    post {
        always {
            echo "Cleaning up workspace..."
            cleanWs()
        }
        success {
            echo "Pipeline run, build, deployment, and push completed successfully."
        }
        failure {
            echo "Pipeline run, build, deployment or push failed. Please check the logs."
        }
    }
}
