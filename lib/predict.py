import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
import joblib
import json
from pathlib import Path
from datetime import datetime
import argparse
import dotenv
from mlflow.models import infer_signature
from sklearn.metrics import classification_report
from helpers import classifier_level_metrics, class_level_metrics, plot_precision_recall_clinical
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


description = """
Generates predictions from a trained model
and saves them to the output directory.
"""

pallas_model_path_help = """
The path to a pallas model (.joblib file output by train_rf.py).
"""

X_path_help = """
Path to CSV file containing test set features."""

y_path_help = """
Path to CSV file containing test set labels."""

dataset_name_help = """
The name of the dataset being used for inference."""

output_dir_help = """
Directory to save the output."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--pallas_model_path",
        type=str,
        required=True,
        help=pallas_model_path_help,
    )
    parser.add_argument(
        "--X_path",
        type=str,
        required=True,
        help=X_path_help,
    )
    parser.add_argument(
        "--y_path",
        type=str,
        required=True,
        help=y_path_help,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help=dataset_name_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    return parser.parse_args()


def predict(
    model: RandomForestClassifier,
    X_path: str,
    y_path: str,
    dataset_name: str,
    output_dir: str,
):
    # Get the feature names from the model
    feature_names = model.feature_names_in_

    # Load the test data
    X_test = pd.read_csv(X_path, index_col=0)
    y_test = pd.read_csv(y_path, index_col=0)

    # Get just the features that were used to train the model
    X_test = X_test[feature_names]

    # align the indices
    X_test = X_test.loc[y_test.index]

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Convert to 1D numpy array
    y_test = y_test.iloc[:, 0].values

    # Save preds as joblib files
    joblib.dump(y_pred, Path(output_dir) / f"{dataset_name}_y_pred.joblib")
    joblib.dump(y_pred_proba, Path(output_dir) / f"{dataset_name}_y_pred_proba.joblib")

    # Save labels in the output dir for convenience
    joblib.dump(y_test, Path(output_dir) / f"{dataset_name}_y_test.joblib")

    # Save predictions to output dir
    proba_df = pd.DataFrame(y_pred_proba, columns=model.classes_, index=X_test.index)

    # Make first column the true label, then predicted label, then probabilities
    proba_df.insert(0, "true_label", y_test)
    proba_df.insert(1, "predicted_label", y_pred)

    # Save predictions to output dir
    output_path = Path(output_dir) / f"{dataset_name}_predictions.csv"
    proba_df.to_csv(output_path)


def main():
    args = parse_args()

    # Extract model name from model path (one dir above the model file)
    model_name = Path(args.pallas_model_path).parent.name
    dataset_name = args.dataset_name

    # Create subdir in output dir for this dataset
    # It should be output_dir/model_name/dataset_name
    output_dir = Path(args.output_dir) / model_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = joblib.load(Path(args.pallas_model_path))

    predict(
        model=model,
        X_path=args.X_path,
        y_path=args.y_path,
        dataset_name=args.dataset_name,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
