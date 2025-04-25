import argparse
import os

MODALITIES = ["GEX", "DNAm"]

REQUIRED_FILES = [
    "pallas.train.X.csv",
    "pallas.train.y.tsv",
    "pallas.test.X.csv",
    "pallas.test.y.tsv",
]

HICLASS_CLASSIFIER_NAMES = [
    "LocalClassifierPerNode",
    "LocalClassifierPerParentNode",
    "LocalClassifierPerLevel",
]

LOCAL_CLASSIFIER_NAMES = [
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
]

description = """
Trains a hiclass classifier on GEX or DNAm data.
"""

modality_help = """Modality of the data."""

input_dir_help = """
Path to directory containing the following files:
pallas.train.X.csv, pallas.train.y.tsv,
pallas.test.X.csv, pallas.test.y.tsv"""

output_dir_help = """
Directory to save the trained models."""

hiclass_classifier_help = """Name of hiclass classifier to use."""

local_classifier_help = f"""
Name of local classifier to use at each node of the hiclass classifier.
Accepted values are: {(',').join(LOCAL_CLASSIFIER_NAMES)}."""

enable_mlflow_help = """
Enable MLflow logging. If set, the model will be logged to MLflow
and the run will be registered in the MLflow tracking server.
The file `mlflow.env` must exist in the 'training' directory."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=MODALITIES,
        help=modality_help,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=input_dir_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    parser.add_argument(
        "--hiclass-classifier",
        type=str,
        required=True,
        choices=HICLASS_CLASSIFIER_NAMES,
        help=hiclass_classifier_help,
    )
    parser.add_argument(
        "--local-classifier",
        type=str,
        required=True,
        choices=LOCAL_CLASSIFIER_NAMES,
        help=local_classifier_help,
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        default=False,
        help=enable_mlflow_help,
    )
    return parser.parse_args()


def train_hiclass_classifier(
        modality,
        input_dir,
        output_dir,
        hiclass_classifier,
        local_classifier,
        enable_mlflow):

    print("Training hiclass with the following arguments:")
    print(f"Modality: {modality}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Hiclass classifier: {hiclass_classifier}")
    print(f"Local classifier: {local_classifier}")
    print(f"Enable MLflow: {enable_mlflow}")

    # Test that all required files exist
    for file_name in REQUIRED_FILES:
        file_path = f"{input_dir}/{file_name}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file {file_path} does not exist.")

    # Check that output directory exists
    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Output directory {output_dir} does not exist.")

    # If mlflow is enabled, check that the mlflow.env file exists
    if enable_mlflow:
        mlflow_env_path = os.path.join(os.path.dirname(__file__), "mlflow.env")
        if not os.path.exists(mlflow_env_path):
            raise FileNotFoundError(
                f"MLflow environment file {mlflow_env_path} does not exist.")


def main():
    args = parse_args()
    train_hiclass_classifier(
        modality=args.modality,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        hiclass_classifier=args.hiclass_classifier,
        local_classifier=args.local_classifier,
        enable_mlflow=args.enable_mlflow,
    )


if __name__ == "__main__":
    main()
