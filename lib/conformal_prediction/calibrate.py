import pandas as pd
import os
from conformist import FNRCoP, PredictionDataset
import argparse
import mlflow
import dotenv
import joblib
import sklearn
import numpy as np
import conformist
from mlflow.models import infer_signature


description = """
Calibrates a conformal predictor using
a calibration dataset of PALLAS predictions
for local use, at a specific error rate.
"""

predictions_path_help = """
Path to CSV file containing predictions."""

base_classifier_name_help = """
Name of the base classifier used to generate the predictions."""

output_dir_help = """
Directory to save the output."""

error_rate_alpha_help = """
Error rate alpha for conformal prediction (default: 0.05)."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help=predictions_path_help,
    )
    parser.add_argument(
        "--base_classifier_name",
        type=str,
        required=True,
        help=base_classifier_name_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    parser.add_argument(
        "--error_rate_alpha",
        type=float,
        default=0.05,
        help=error_rate_alpha_help,
    )
    return parser.parse_args()


def calibrate_conformal_prediction(
        predictions_path,
        output_dir,
        base_classifier_name,
        error_rate_alpha=0.05):
    # Create the output directory if it doesn't exist
    # Make conformal predictor output directory
    output_dir = os.path.join(output_dir, "conformal_predictor")
    os.makedirs(output_dir, exist_ok=True)

    # Read in conformal predictions
    preds_cal = pd.read_csv(predictions_path)

    # Read in formatted predictions
    calpd = PredictionDataset(df=preds_cal, dataset_name='cal')

    crc = FNRCoP(calpd, alpha=error_rate_alpha)
    crc.calibrate()

    # Dump crc to joblib for later use
    # Save alpha as string with underscores instead of decimal point
    str_alpha = str(error_rate_alpha).replace('.', '_')
    model_name = f"crc_{base_classifier_name}_alpha{str_alpha}"
    crc_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(crc, crc_path)


def main():
    args = parse_args()
    predictions_path = args.predictions_path
    base_classifier_name = args.base_classifier_name
    output_dir = args.output_dir
    error_rate_alpha = args.error_rate_alpha
    calibrate_conformal_prediction(
        predictions_path,
        output_dir,
        base_classifier_name=base_classifier_name,
        error_rate_alpha=error_rate_alpha)


if __name__ == "__main__":
    main()
