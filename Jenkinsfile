pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
    }

    environment {
        // Name of the Docker image built by the Makefile logic
        IMAGE_NAME = "ahmed_louay_araour_4ds2_mlops"
        // MLflow tracking URI using your locally running MLflow instance, as set in your Makefile target "run-with-local-mlflow"
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
        stage('Build Docker Image') {
            steps {
                // Build the Docker image using the same context as your Makefile's "all" target
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }
        stage('Run Model Pipeline') {
            steps {
                sh '''
                # Stop and remove any existing container running the model pipeline, similar to Makefile cleanup
                docker stop model_pipeline || true
                docker rm model_pipeline || true

                # Run the container using the image built above.
                # This container executes the model_pipeline.py which contains 
                # the logic for preparing data, training, evaluating, and saving the model,
                # while connecting to your local MLflow server as defined in your Makefile.
                docker run -d --name model_pipeline \\
                    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                    ${IMAGE_NAME} python model_pipeline.py
                '''
            }
        }
        // Optional stage to check container logs for debugging purposes
        stage('Check Logs') {
            steps {
                sh '''
                echo "Waiting for model pipeline to initialize..."
                sleep 10
                echo "------ Container Logs for model_pipeline ------"
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
            echo "Build and model pipeline execution completed successfully."
        }
        failure {
            echo "Build or model pipeline execution failed. Please review the logs above."
        }
    }
}
