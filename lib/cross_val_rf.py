import joblib
from pathlib import Path
import argparse
import pandas as pd
from helpers import (cross_val_metrics,
                     plot_precision_recall)


description = """
Cross-validates a model using the training data.
"""

pallas_model_base_dir_help = """
The path to a pallas model base directory
where all trained models are stored. The script will look in
this directory for a subdirectory corresponding to the model being evaluated,
and then a model file named "rf_model.joblib", as well as y.csv
and X_reduced.[n].csv.
"""

pallas_model_name_help = """
The name of the model being evaluated, e.g.
rf_DNAm_20260606_193214."""

output_dir_help = """
Directory to save the output."""

human_readable_title_help = """
A human-readable title for the evaluation,
e.g. "GEX model cross-validated on the training set (5-fold)"
"""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--pallas_model_base_dir",
        type=str,
        required=True,
        help=pallas_model_base_dir_help,
    )
    parser.add_argument(
        "--pallas_model_name",
        type=str,
        required=True,
        help=pallas_model_name_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    parser.add_argument(
        "--human_readable_title",
        type=str,
        required=False,
        help=human_readable_title_help,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.pallas_model_name
    dataset_name = "cross-val"
    # Create subdir in output dir for this dataset
    output_dir = Path(args.output_dir) / model_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = Path(
        args.pallas_model_base_dir) / model_name / "rf_model.joblib"
    model = joblib.load(model_path)

    # Get number of features from model
    n_features = model.n_features_in_

    # Construct name of X and y files based on model name
    # The X file will be of the format X_reduced.[n].csv
    X_file = Path(args.pallas_model_base_dir
                  ) / model_name / f"X_reduced.{n_features}.csv"
    y_file = Path(args.pallas_model_base_dir) / model_name / "y.csv"

    # Load predictions and labels
    X_train = pd.read_csv(X_file, index_col=0)
    y_train = pd.read_csv(y_file, index_col=0).iloc[:, 0].values

    # Generate cross-validation metrics
    classifier_metrics_df, class_metrics_df = cross_val_metrics(
        model, X_train, y_train, model.classes_.tolist())

    print("Classifier-level metrics:")
    print(classifier_metrics_df)
    classifier_metrics_path = Path(output_dir) / "classifier_level_metrics.csv"
    classifier_metrics_df.to_csv(classifier_metrics_path)

    print("\n\nClass-level metrics:")
    print(class_metrics_df)
    class_metrics_path = Path(output_dir) / "class_level_metrics.csv"
    class_metrics_df.to_csv(class_metrics_path,
                            header=True,
                            index=True)

    precision_recall_path = Path(output_dir) / "precision_recall.svg"
    if args.human_readable_title:
        title = args.human_readable_title
    else:
        title = f"Model {model_name} cross-validated on the training set"
    plot_precision_recall(
        class_metrics_df,
        title=title,
        save_path=precision_recall_path
    )


if __name__ == "__main__":
    main()
