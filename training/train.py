import argparse
import os
import pandas as pd
from dotenv import load_dotenv
import pickle

# MLflow
import mlflow
from mlflow.models import infer_signature

# Classifiers
from hiclass import LocalClassifierPerNode, \
    LocalClassifierPerParentNode, LocalClassifierPerLevel
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Metrics
from hiclass.metrics import precision, recall, f1

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

hiclass_classifier_help = """Name of hiclass classifier to use.
If not specified, will train all available hiclass classifiers."""

local_classifier_help = """
Name of local classifier to use at each node of the hiclass classifier.
If not specified, will train all available local classifiers."""

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
        choices=HICLASS_CLASSIFIER_NAMES,
        help=hiclass_classifier_help,
    )
    parser.add_argument(
        "--local-classifier",
        type=str,
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


def train_hiclass_classifiers(
        modality,
        input_dir,
        output_dir,
        hiclass_classifier,
        local_classifier,
        enable_mlflow):

    # hiclass classifiers to train
    hiclass_classifiers = HICLASS_CLASSIFIER_NAMES
    if hiclass_classifier is not None:
        hiclass_classifiers = [hiclass_classifier]

    # local classifiers to train
    local_classifiers = LOCAL_CLASSIFIER_NAMES
    if local_classifier is not None:
        local_classifiers = [local_classifier]

    print("Training hiclass with the following arguments:")
    print(f"Modality: {modality}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Hiclass classifier(s): {', '.join(hiclass_classifiers)}")
    print(f"Local classifier(s): {', '.join(local_classifiers)}")
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
    mlflow_env_path = None
    if enable_mlflow:
        mlflow_env_path = os.path.join(os.path.dirname(__file__), "mlflow.env")
        if not os.path.exists(mlflow_env_path):
            raise FileNotFoundError(
                f"MLflow environment file {mlflow_env_path} does not exist.")

    X_train = f'{input_dir}/pallas.train.X.csv'
    y_train = f'{input_dir}/pallas.train.y.tsv'
    X_test = f'{input_dir}/pallas.test.X.csv'
    y_test = f'{input_dir}/pallas.test.y.tsv'

    # Read the features and target data
    X_train = pd.read_csv(X_train, header=0, index_col=0)
    y_train = pd.read_csv(y_train, sep="\t", header=0, index_col=0)
    X_test = pd.read_csv(X_test, header=0, index_col=0)
    y_test = pd.read_csv(y_test, sep="\t", header=0, index_col=0)

    # Split the labels into lists
    y_train = y_train['y'].apply(lambda x: x.split(','))
    y_test = y_test['y'].apply(lambda x: x.split(','))

    # Get just the y col
    y_train = y_train.values
    y_test = y_test.values

    for hiclass_classifier in hiclass_classifiers:
        for local_classifier in local_classifiers:
            train_hiclass_classifier(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                modality=modality,
                output_dir=output_dir,
                hiclass_classifier=hiclass_classifier,
                local_classifier=local_classifier,
                mlflow_env_path=mlflow_env_path,
            )


def train_hiclass_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        modality,
        output_dir,
        hiclass_classifier,
        local_classifier,
        mlflow_env_path):

    # Set up model name
    model_name = f"pallas_{modality}_{hiclass_classifier}_{local_classifier}"

    print(f"Training model {model_name}...")

    # Instantiate the local model
    model_class = globals()[local_classifier]
    params = {}

    # Instantiate the local model
    model = model_class(**params)

    hiclass_classifier_class = globals()[hiclass_classifier]
    classifier = hiclass_classifier_class(
        local_classifier=model,
        verbose=True)

    classifier.fit(X_train, y_train, sample_weight=None)

    classifier_pkl = f'{output_dir}/{model_name}.classifier.pkl'
    with open(classifier_pkl, 'wb') as f:
        pickle.dump(classifier, f)

    # Get the metrics
    predictions = classifier.predict(X_test)
    precision_score = precision(y_test, predictions)
    recall_score = recall(y_test, predictions)
    f1_score = f1(y_test, predictions)

    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1: {f1_score}")

    # Start an MLflow run
    if mlflow_env_path is not None:
        load_dotenv(dotenv_path=mlflow_env_path)

        with mlflow.start_run():
            # Log the hyperparameters
            mlflow.log_params(model.get_params())

            # Log the loss metric
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall", recall_score)
            mlflow.log_metric("f1", f1_score)

            mlflow.set_tag("modality", modality)
            mlflow.set_tag("hiclass_classifier", hiclass_classifier)
            mlflow.set_tag("local_classifier", local_classifier)

            # Infer the model signature
            signature = infer_signature(X_train, classifier.predict(X_train))

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=classifier,
                artifact_path="model",
                signature=signature,
                input_example=X_train,
                registered_model_name=model_name,
            )


def main():
    args = parse_args()
    train_hiclass_classifiers(
        modality=args.modality,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        hiclass_classifier=args.hiclass_classifier,
        local_classifier=args.local_classifier,
        enable_mlflow=args.enable_mlflow,
    )


if __name__ == "__main__":
    main()
