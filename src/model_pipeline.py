#!/usr/bin/env python3
"""
Module: model_pipeline
Description: This module contains functions for the ML pipeline using a Decision Tree
model with email notifications, MLflow integration, and visualizations for model metrics.
It provides functions to:
    - prepare_data(): Load and preprocess the data.
    - train_model(): Train the model and record training time.
    - evaluate_model(): Evaluate the model's performance and compute metrics (accuracy, precision, recall, f1).
    - save_model(): Save the trained model using joblib.
    - load_model(): Load a saved model.
    - run_full_pipeline(): Run the full pipeline with detailed logging to MLflow and visualizations.
    - send_email_notification(): Send an email notification with the pipeline result.
"""

import os
import time
import joblib
import psutil
import pandas as pd
import smtplib
from email.mime.text import MIMEText
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import visualization functions for additional analysis.
from visualizations import plot_confusion_matrix, plot_metrics, plot_feature_importances

def prepare_data(dataset_path):
    """
    Load and preprocess the data.
    
    Reads a CSV file, drops missing values, renames the target column (using 'Churn' if 'target' is not present),
    separates features and target, and converts non-numeric features into dummy/indicator variables.
    
    Returns:
        X_train, X_test, y_train, y_test: Split training and testing data.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File {dataset_path} does not exist.")
    
    df = pd.read_csv(dataset_path)
    df.dropna(inplace=True)
    
    # Rename target column if necessary.
    if 'target' not in df.columns:
        if 'Churn' in df.columns:
            df.rename(columns={'Churn': 'target'}, inplace=True)
        else:
            raise ValueError("The dataset must contain a 'target' or 'Churn' column.")
    
    target = df['target']
    features = df.drop('target', axis=1)
    
    # Convert categorical features to dummy variables.
    features = pd.get_dummies(features, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    print("Data prepared:")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Decision Tree model and record training time.
    
    Returns:
        model: The trained model object.
        training_time: The duration (in seconds) it took to train the model.
    """
    start_time = time.time()
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print("Model trained.")
    print(f"Training time: {training_time:.4f} seconds")
    return model, training_time

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the model by calculating accuracy, precision, recall, and F1 score.
    
    Returns:
        accuracy (float): Accuracy of the model.
        precision (float): Precision of the model.
        recall (float): Recall of the model.
        f1 (float): F1 score of the model.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Model precision: {precision:.4f}")
    print(f"Model recall: {recall:.4f}")
    print(f"Model F1 score: {f1:.4f}")
    return accuracy, precision, recall, f1

def save_model(model, model_filename):
    """
    Save the trained model using joblib.
    """
    joblib.dump(model, model_filename)
    print(f"Model saved in {model_filename}.")

def load_model(model_filename):
    """
    Load a model saved using joblib.
    """
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"File {model_filename} does not exist.")
    model = joblib.load(model_filename)
    print(f"Model loaded from {model_filename}.")
    return model

def send_email_notification(subject, message, receiver_email="add your reciever email"):
    """
    Send an email notification using Gmail's SMTP server.
    
    Default settings:
        SMTP server: smtp.gmail.com
        SMTP port: 587
        SMTP user: 
        SMTP password: retrieved from SMTP_PASSWORD environment variable or default provided.
    """
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "Add the sender email"  
    smtp_password = os.environ.get("SMTP_PASSWORD", "replace with your password from google passwords")
    sender_email = smtp_user
    
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email notification sent.")
    except Exception as e:
        print(f"Failed to send email notification: {e}")

def run_full_pipeline(dataset_path, model_filename):
    """
    Run the full pipeline:
      - Load and preprocess data.
      - Train the model and measure training time.
      - Evaluate performance (accuracy, precision, recall, F1 score).
      - Save the model.
      - Log parameters, system metrics, and model metrics with MLflow.
      - Generate and save visualizations.
      - Log visualization images as MLflow artifacts.
      - Send email notification.
    """
    # Set tracking URI from environment variable or use default
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
    
    # Set an explicit experiment name.
    mlflow.set_experiment("Ahmed-Louay-Araour-4DS2-ML")
    
    # Record system metrics before pipeline run.
    system_cpu_before = psutil.cpu_percent(interval=1)
    system_mem_before = psutil.virtual_memory().percent
    print(f"System CPU usage before run: {system_cpu_before}%")
    print(f"System memory usage before run: {system_mem_before}%")
    
    with mlflow.start_run():
        mlflow.set_tag("project", "Ahmed-Louay-Araour-4DS2-ML")
        # Log system metrics.
        mlflow.log_param("system_cpu_before", system_cpu_before)
        mlflow.log_param("system_mem_before", system_mem_before)
        
        X_train, X_test, y_train, y_test = prepare_data(dataset_path)

        model, training_time = train_model(X_train, y_train)
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        
        # Save model locally and log info to MLflow.
        save_model(model, model_filename)
        mlflow.log_param("model_filename", model_filename)

        # Log model metrics.
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", training_time)
        
        # Log the model with an input example to infer the signature.
        mlflow.sklearn.log_model(model, "model", input_example=X_test.head(1))
        
        # Save and log training feature columns.
        feature_columns = X_train.columns.tolist()
        feature_columns_file = "model_feature_columns.joblib"
        joblib.dump(feature_columns, feature_columns_file)
        print(f"Training feature columns saved to {feature_columns_file}")
        mlflow.log_artifact(feature_columns_file)
        
        # Record system metrics after run.
        system_cpu_after = psutil.cpu_percent(interval=1)
        system_mem_after = psutil.virtual_memory().percent
        mlflow.log_metric("system_cpu_after", system_cpu_after)
        mlflow.log_metric("system_mem_after", system_mem_after)
        print(f"System CPU usage after run: {system_cpu_after}%")
        print(f"System memory usage after run: {system_mem_after}%")
        
        # Generate and save visualizations.
        predictions = model.predict(X_test)
        # Save confusion matrix plot.
        plot_confusion_matrix(y_test, predictions, labels=[0, 1], save_path="confusion_matrix.png")
        # Save metrics plot.
        metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
        plot_metrics(metrics, save_path="metrics.png")
        # Save feature importances plot (if applicable).
        feature_names = X_train.columns.tolist()
        plot_feature_importances(model, feature_names, save_path="feature_importances.png")
        
        # Log visualization images as MLflow artifacts.
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("metrics.png")
        mlflow.log_artifact("feature_importances.png")
        
        # Send final email notification.
        email_subject = "ML Pipeline Execution Completed"
        email_message = (
            f"Pipeline completed successfully.\n"
            f"Model Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n"
            f"Training Time: {training_time:.4f} seconds\n"
            f"System CPU Before: {system_cpu_before}%, After: {system_cpu_after}%\n"
            f"System Memory Before: {system_mem_before}%, After: {system_mem_after}%\n"
            f"Model saved to: {model_filename}"
        )
        send_email_notification(email_subject, email_message)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        action = sys.argv[1]
        dataset = sys.argv[2] if len(sys.argv) > 2 else "datasets/churn-bigml-80.csv"
        model_file = sys.argv[3] if len(sys.argv) > 3 else "decision_tree_model.joblib"
        if action == "prepare":
            prepare_data(dataset)
        elif action == "train":
            X_train, _, y_train, _ = prepare_data(dataset)
            train_model(X_train, y_train)
        elif action == "evaluate":
            X_train, X_test, y_train, y_test = prepare_data(dataset)
            model, _ = train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)
        elif action == "save":
            X_train, X_test, y_train, _ = prepare_data(dataset)
            model, _ = train_model(X_train, y_train)
            save_model(model, model_file)
        elif action == "load":
            load_model(model_file)
        elif action == "all":
            run_full_pipeline(dataset, model_file)
        else:
            print("Unrecognized action. Choose from: prepare, train, evaluate, save, load, all.")
    else:
        print("No action specified. Please provide an action as an argument.")
