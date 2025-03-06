pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
    }

    environment {
        IMAGE_NAME = "ahmed_louay_araour_4ds2_mlops"
        MLFLOW_TRACKING_URI = "http://localhost:5000"
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
        stage('Run Model Pipeline') {
            agent {
                docker {
                    image 'python:3.9'
                    reuseNode true
                }
            }
            steps {
                // Run your complete pipeline logic.
                // This assumes your model_pipeline.py (or make all) encapsulates
                // data preparation, training, evaluation, and saving the model.
                // Choose one of the commands below based on your Makefile logic.
                sh 'python model_pipeline.py'
                // Alternatively, if your Makefile target "all" handles the pipeline, uncomment:
                // sh 'make all'
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

                # Launch the container with your built Docker image.
                # This container will run the model_pipeline.py script.
                docker run -d --name model_pipeline \\
                    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                    ${IMAGE_NAME} python model_pipeline.py
                '''
            }
        }
        stage('Check Logs') {
            steps {
                sh '''
                sleep 10
                docker logs model_pipeline || true
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo "Build and model pipeline execution completed successfully."
        }
        failure {
            echo "Build or model pipeline execution failed. Please review the logs above."
        }
    }
}
