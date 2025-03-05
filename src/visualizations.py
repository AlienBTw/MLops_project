#!/usr/bin/env python3
"""
Module: visualizations
Description:
    This module provides functions to create visualizations for your ML model.
    It currently includes:
      - plot_confusion_matrix: Plots a confusion matrix based on true labels and predictions.
      - plot_metrics: Displays a bar chart for common performance metrics.
      - plot_feature_importances: Plots the feature importances of a Decision Tree model.
      
Usage:
    Import the module in your pipeline or testing scripts and call the desired function.
    Example:
        from visualizations import plot_confusion_matrix, plot_metrics, plot_feature_importances
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, labels=[0, 1], save_path="confusion_matrix.png")
        
        # Plot metrics
        metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
        plot_metrics(metrics, save_path="metrics.png")
        
        # Plot feature importances (if available)
        plot_feature_importances(model, feature_names, save_path="feature_importances.png")
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, cmap="Blues", save_path=None):
    """
    Plot a confusion matrix using Seaborn's heatmap.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of label names. Defaults to None.
        cmap (str, optional): Colormap to use for the heatmap. Defaults to "Blues".
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}.")
    plt.close()

def plot_metrics(metrics, save_path=None):
    """
    Plot a bar chart to display performance metrics.
    
    Args:
        metrics (dict): Dictionary where keys are metric names and values are the corresponding values.
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    plt.figure(figsize=(10, 6))
    names = list(metrics.keys())
    values = list(metrics.values())
    bars = plt.bar(names, values, color="skyblue")
    plt.ylabel("Value")
    plt.title("Model Performance Metrics")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:.2f}", ha="center", va="bottom")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Metrics plot saved to {save_path}.")
    plt.close()

def plot_feature_importances(model, feature_names, save_path=None):
    """
    Plot feature importances for models that support the 'feature_importances_' attribute.
    
    Args:
        model: Trained model (e.g., DecisionTreeClassifier) having feature_importances_ attribute.
        feature_names (list): List of feature names corresponding to the model's input.
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    if not hasattr(model, "feature_importances_"):
        print("The model does not have a 'feature_importances_' attribute.")
        return
    
    importances = model.feature_importances_
    plt.figure(figsize=(12, 8))
    plt.barh(feature_names, importances, color="mediumseagreen")
    plt.xlabel("Feature Importance")
    plt.title("Feature Importances")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Feature importances plot saved to {save_path}.")
    plt.close()