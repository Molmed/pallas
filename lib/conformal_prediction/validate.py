import pandas as pd
import os
from conformist import FNRCoP, PredictionDataset
import argparse


description = """
Validates a conformal prediction model on a test
set using calibration data and predictions.
"""

predictions_path_help = """
Path to CSV file containing predictions."""

output_dir_help = """
Directory to save the output."""

min_samples_per_class_help = """
Minimum number of samples per class to include in the validation trial (default: 10)."""

n_runs_help = """
Number of runs for validation trial (default: 1000)."""

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
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    parser.add_argument(
        "--class_color_tsv_path",
        type=str,
        default=None,
        help="Path to TSV file containing class colors",
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=10,
        help=min_samples_per_class_help,
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1000,
        help=n_runs_help,
    )
    parser.add_argument(
        "--error_rate_alpha",
        type=float,
        default=0.05,
        help=error_rate_alpha_help,
    )
    return parser.parse_args()


def validate_conformal_prediction(
        predictions_path,
        output_dir,
        class_color_tsv_path=None,
        min_samples_per_class=10,
        n_runs=1000,
        error_rate_alpha=0.05):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read in conformal predictions
    preds_cal = pd.read_csv(predictions_path)

    # Get all classes under the threshold
    class_counts = preds_cal['true_label'].value_counts()
    classes_under_threshold = class_counts[class_counts < min_samples_per_class].index.tolist()
    n_classes_under_threshold = len(classes_under_threshold)
    print(f"{n_classes_under_threshold} classes under threshold ({min_samples_per_class} samples): {classes_under_threshold}")

    # Remove them
    preds_cal = preds_cal[~preds_cal['true_label'].isin(classes_under_threshold)]
    # Also remove the columns correspondig to those class names
    class_names = preds_cal['true_label'].unique().tolist()
    keep_col_names = ['true_label', 'predicted_label', 'id'] + class_names
    preds_cal = preds_cal[keep_col_names]

    # Show new class breakdown of set
    # print classes remaining
    print(f"New dataset has {len(preds_cal)} samples and {len(preds_cal['true_label'].unique())} classes.")
    print("New class distribution:")
    print(preds_cal['true_label'].value_counts())

    # Read in formatted predictions
    calpd = PredictionDataset(df=preds_cal, dataset_name='cal')

    mcp = FNRCoP(calpd, alpha=error_rate_alpha)
    trial = mcp.do_validation_trial(n_runs=n_runs)
    trial.run_reports(output_dir, class_color_tsv_path=class_color_tsv_path)


def main():
    args = parse_args()
    predictions_path = args.predictions_path
    output_dir = args.output_dir
    n_runs = args.n_runs
    error_rate_alpha = args.error_rate_alpha
    min_samples_per_class = args.min_samples_per_class
    class_color_tsv_path = args.class_color_tsv_path
    validate_conformal_prediction(
        predictions_path,
        output_dir,
        class_color_tsv_path=class_color_tsv_path,
        min_samples_per_class=min_samples_per_class,
        n_runs=n_runs,
        error_rate_alpha=error_rate_alpha)


if __name__ == "__main__":
    main()
