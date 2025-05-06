import argparse
import os
import pandas as pd
from dotenv import load_dotenv
import pickle
from sklearn.preprocessing import LabelEncoder

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

FLAT_CLASSIFIER_NAME = "FlatClassifier"

HIERARCHICAL_CLASSIFIER_NAMES = [
    FLAT_CLASSIFIER_NAME,
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
Path to directory containing files output by the split.py script."""

output_dir_help = """
Directory to save the trained models."""

hierarchical_classifier_help = """Name of hierarchical classifier to use.
If not specified, will train all available hierarchical classifiers."""

local_classifier_help = """
Name of local classifier to use at each node of the hiclass classifier.
If not specified, will train all available local classifiers."""

enable_mlflow_help = """
Enable MLflow logging. If set, the model will be logged to MLflow
and the run will be registered in the MLflow tracking server.
The file `mlflow.env` must exist in the 'training' directory."""

force_help = """
Force overwrite of existing models in the output directory."""


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
        "--hierarchical-classifier",
        type=str,
        choices=HIERARCHICAL_CLASSIFIER_NAMES,
        help=hierarchical_classifier_help,
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
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=force_help,
    )
    return parser.parse_args()


def train_hierarchical_classifiers(
        modality,
        input_dir,
        output_dir,
        hierarchical_classifier,
        local_classifier,
        enable_mlflow,
        force):

    # hiclass classifiers to train
    hierarchical_classifiers = HIERARCHICAL_CLASSIFIER_NAMES
    if hierarchical_classifier is not None:
        hierarchical_classifiers = [hierarchical_classifier]

    # local classifiers to train
    local_classifiers = LOCAL_CLASSIFIER_NAMES
    if local_classifier is not None:
        local_classifiers = [local_classifier]

    print("Training hiclass with the following arguments:")
    print(f"Modality: {modality}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Hierarchical classifier(s): {', '.join(hierarchical_classifiers)}")
    print(f"Local classifier(s): {', '.join(local_classifiers)}")
    print(f"Enable MLflow: {enable_mlflow}")
    print(f"Force overwrite: {force}")

    # Test that all required files exist
    features_format = None

    def file_exists(file_name):
        file_path = f"{input_dir}/{file_name}"
        return os.path.exists(file_path)

    REQUIRED_X_FILE_PREFIXES = [
        "pallas.train.X",
        "pallas.test.X",
    ]
    REQUIRED_X_FILE_SUFFIXES = [
        "csv",
        "pkl",
    ]
    REQUIRED_Y_FILES = [
        "pallas.train.y.tsv",
        "pallas.test.y.tsv",
    ]
    all_required_files_exist = True

    for file_name in REQUIRED_Y_FILES:
        if not file_exists(file_name):
            all_required_files_exist = False
            print(f"Required file {file_name} does not exist.")

    for prefix in REQUIRED_X_FILE_PREFIXES:
        x_file_found = False
        for suffix in REQUIRED_X_FILE_SUFFIXES:
            file_name = f"{prefix}.{suffix}"
            if file_exists(file_name):
                x_file_found = True
                features_format = suffix
                break
        if not x_file_found:
            all_required_files_exist = False
            print(f"""Required file
                  {prefix}.[{(','.join(REQUIRED_X_FILE_SUFFIXES))}]
                  does not exist.""")

    if not all_required_files_exist:
        raise FileNotFoundError(
            f"Required files do not exist in {input_dir}.")

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

    X_train = f'{input_dir}/pallas.train.X.{features_format}'
    y_train = f'{input_dir}/pallas.train.y.tsv'
    X_test = f'{input_dir}/pallas.test.X.{features_format}'
    y_test = f'{input_dir}/pallas.test.y.tsv'

    # Read the features and target data
    y_train = pd.read_csv(y_train, sep="\t", header=0, index_col=0)
    y_test = pd.read_csv(y_test, sep="\t", header=0, index_col=0)

    # Check if the features file is in CSV format or pkl format
    if features_format == 'csv':
        X_train = pd.read_csv(X_train, header=0, index_col=0)
        X_test = pd.read_csv(X_test, header=0, index_col=0)
    elif features_format == 'pkl':
        X_train = pd.read_pickle(X_train)
        X_test = pd.read_pickle(X_test)
    else:
        raise ValueError("Features file must be in CSV or pkl format.")

    # Split the labels into lists
    y_train = y_train['y'].apply(lambda x: x.split(','))
    y_test = y_test['y'].apply(lambda x: x.split(','))

    # Get just the y col
    y_train = y_train.values
    y_test = y_test.values

    for hierarchical_classifier in hierarchical_classifiers:
        for local_classifier in local_classifiers:
            # Set up model name and path
            model_name = f"""pallas_{modality}_{hierarchical_classifier}_{
                local_classifier}"""
            model_path = f"{output_dir}/{model_name}.classifier.pkl"

            do_train = True
            if local_classifier == "XGBClassifier" and \
                hierarchical_classifier != "LocalClassifierPerNode" and \
                    hierarchical_classifier != "FlatClassifier":
                print(f"""Skipping {
                    hierarchical_classifier} with {local_classifier}""")
                do_train = False

            if os.path.exists(model_path) and not force:
                print(f"Model {model_name} already exists. Skipping.")
                do_train = False

            if do_train:
                train_hierarchical_classifier(
                    model_name=model_name,
                    model_path=model_path,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    modality=modality,
                    hierarchical_classifier=hierarchical_classifier,
                    local_classifier=local_classifier,
                    mlflow_env_path=mlflow_env_path,
                )


def train_hierarchical_classifier(
        model_name,
        model_path,
        X_train,
        y_train,
        X_test,
        y_test,
        modality,
        hierarchical_classifier,
        local_classifier,
        mlflow_env_path):

    print(f"Training model {model_name} at {model_path}...")

    is_flat_xbg_classifier = False
    if local_classifier == "XGBClassifier" and \
            hierarchical_classifier == FLAT_CLASSIFIER_NAME:
        is_flat_xbg_classifier = True

    # Instantiate the local model
    model_class = globals()[local_classifier]
    params = {}

    if is_flat_xbg_classifier:
        params = {
            "objective": "multi:softmax",
        }

    # Instantiate the local model
    model = model_class(**params)

    def flatten_labels(y_list):
        # return disease (item at index 1) and subtype (last item)
        # joined by underscore
        return ["_".join([y[1], y[-1]]) for y in y_list]

    if hierarchical_classifier == FLAT_CLASSIFIER_NAME:
        classifier = model
        y_train = flatten_labels(y_train)
        y_test = flatten_labels(y_test)
    else:
        hierarchical_classifier_class = globals()[hierarchical_classifier]
        classifier = hierarchical_classifier_class(
            local_classifier=model,
            verbose=True)

    le_path = None
    if is_flat_xbg_classifier:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        le_path = model_path.replace(".classifier.", ".labelencoder.")
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)

    classifier.fit(X_train, y_train, sample_weight=None)

    with open(model_path, 'wb') as f:
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
        timenow = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"{model_name} {timenow}"

        with mlflow.start_run(run_name=run_name):
            # Log the hyperparameters
            mlflow.log_params(model.get_params())

            # Log the loss metric
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall", recall_score)
            mlflow.log_metric("f1", f1_score)

            mlflow.set_tag("modality", modality)
            mlflow.set_tag("hierarchical_classifier", hierarchical_classifier)
            mlflow.set_tag("local_classifier", local_classifier)

            # Infer the model signature
            signature = infer_signature(X_train, classifier.predict(X_train))

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=classifier,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name,
            )

            # Log the label encoder
            if is_flat_xbg_classifier:
                mlflow.log_artifact(
                    local_path=le_path,
                    artifact_path="label_encoder",
                )


def main():
    args = parse_args()
    train_hierarchical_classifiers(
        modality=args.modality,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        hierarchical_classifier=args.hierarchical_classifier,
        local_classifier=args.local_classifier,
        enable_mlflow=args.enable_mlflow,
        force=args.force,
    )


if __name__ == "__main__":
    main()
