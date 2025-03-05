#!/usr/bin/env python3
"""
Main entry point for the Ahmed-Louay-Araour-4DS2-ML Project.

This script uses the 'model_pipeline' module to run individual pipeline steps,
launch the entire process at once, and send email notifications upon task completion.

Example usage:
    Run the full pipeline:
        python src/main.py --function all --dataset datasets/churn-bigml-80.csv --model_filename decision_tree_model.joblib

    Test only the prepare step:
        python src/main.py --function prepare --dataset datasets/churn-bigml-80.csv
"""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    run_full_pipeline,
    send_email_notification
)

def main():
    parser = argparse.ArgumentParser(
        description="Model pipeline for Ahmed-Louay-Araour-4DS2-ML Project."
    )
    parser.add_argument(
        "--dataset",
        default="datasets/churn-bigml-80.csv",
        help="Path to the CSV file containing the data (must include a 'target' or 'Churn' column)."
    )
    parser.add_argument(
        "--model_filename",
        default="decision_tree_model.joblib",
        help="File name (or path) for saving/loading the model."
    )
    parser.add_argument(
        "--function",
        choices=["prepare", "train", "evaluate", "save", "load", "all"],
        default="all",
        help="The pipeline function to execute."
    )
    args = parser.parse_args()

    if args.function == "all":
        run_full_pipeline(args.dataset, args.model_filename)
    elif args.function == "prepare":
        prepare_data(args.dataset)
        send_email_notification("Task Completed - Prepare", "Data preparation task completed.")
    elif args.function == "train":
        X_train, _, y_train, _ = prepare_data(args.dataset)
        train_model(X_train, y_train)
        send_email_notification("Task Completed - Train", "Model training task completed.")
    elif args.function == "evaluate":
        X_train, X_test, y_train, y_test = prepare_data(args.dataset)
        model, _ = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        send_email_notification("Task Completed - Evaluate", "Model evaluation task completed.")
    elif args.function == "save":
        X_train, _, y_train, _ = prepare_data(args.dataset)
        model, _ = train_model(X_train, y_train)
        save_model(model, args.model_filename)
        send_email_notification("Task Completed - Save", "Model saving task completed.")
    elif args.function == "load":
        load_model(args.model_filename)
        send_email_notification("Task Completed - Load", "Model loading task completed.")
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()