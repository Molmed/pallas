# This script should:
# - load model
# - load and format training data
# - generate predictions
# - save predictions to output dir
# - once script is perfected, upload metrics to mlflow


# +    # Start an MLflow run
# +    if enable_mlflow:
# +        load_dotenv(dotenv_path=mlflow_env_path)
# +
# +        with mlflow.start_run():
# +            # Log the hyperparameters
# +            mlflow.log_params(model.get_params())
# +
# +            # Log the loss metric
# +            mlflow.log_metric("precision", precision_score)
# +            mlflow.log_metric("recall", recall_score)
# +            mlflow.log_metric("f1", f1_score)
# +
# +            mlflow.set_tag("modality", modality)
# +            mlflow.set_tag("hiclass_classifier", hiclass_classifier)
# +            mlflow.set_tag("local_classifier", local_classifier)
# +
# +            # Infer the model signature
# +            signature = infer_signature(X_train, classifier.predict(X_train))
# +
# +            # Log the model
# +            mlflow.sklearn.log_model(
# +                sk_model=classifier,
# +                artifact_path="model",
# +                signature=signature,

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


description = """
Evaluates a model and optionally logs it to MLFlow.
"""

pallas_model_dir_help = """
The path to a pallas model dir (output by train_rf.py)
of the format e.g. rf_DNAm_20260606_193214
"""

X_path_help = """
Path to CSV file containing test set features."""

y_path_help = """
Path to CSV file containing test set labels."""

dataset_name_help = """
The name of the dataset being used to evaluate the model."""

output_dir_help = """
Directory to save the output."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--pallas_model_dir",
        type=str,
        required=True,
        help=pallas_model_dir_help,
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
    pallas_model_dir: str,
    X_path: str,
    y_path: str,
    dataset_name: str,
    output_dir: str,
):
    # Load the model
    model = joblib.load(Path(pallas_model_dir) / "rf_model.joblib")

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

    dataset_name = args.dataset_name
    # Create subdir in output dir for this dataset
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predict(
        pallas_model_dir=args.pallas_model_dir,
        X_path=args.X_path,
        y_path=args.y_path,
        dataset_name=args.dataset_name,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
