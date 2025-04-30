import argparse
import os
import pandas as pd
import pickle

# Generate confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

REQUIRED_FILES = [
    "pallas.test.X.csv",
    "pallas.test.y.tsv",
]

description = """
Generates reports for the trained models.
"""

input_models_dir_help = """
Path to directory containing one or more *.classifier.pkl files."""

input_data_dir_help = """
Path to directory containing the following files:
pallas.test.X.csv, pallas.test.y.tsv"""

output_dir_help = """
Directory to save the reports."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input_models_dir",
        type=str,
        required=True,
        help=input_models_dir_help,
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        required=True,
        help=input_data_dir_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    return parser.parse_args()


def generate_reports(
        input_models_dir,
        input_data_dir,
        output_dir):

    print("Generating reports for the following arguments:")
    print(f"Input models directory: {input_models_dir}")
    print(f"Input data directory: {input_data_dir}")
    print(f"Output directory: {output_dir}")

    # Test that all required files exist
    for file_name in REQUIRED_FILES:
        file_path = f"{input_data_dir}/{file_name}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file {file_path} does not exist.")

    # Check that input models directory exists
    # and contains at least one *.classifier.pkl file
    if not os.path.exists(input_models_dir):
        raise FileNotFoundError(
            f"Input models directory {input_models_dir} does not exist.")
    if not os.path.isdir(input_models_dir):
        raise FileNotFoundError(
            f"Input models directory {input_models_dir} is not a directory.")
    if not os.listdir(input_models_dir):
        raise FileNotFoundError(
            f"Input models directory {input_models_dir} is empty.")
    if not any(file.endswith(".classifier.pkl") for
               file in os.listdir(input_models_dir)):
        raise FileNotFoundError(
            f"""Input models directory {input_models_dir}
            does not contain any *.classifier.pkl files.""")

    # Check that output directory exists
    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Output directory {output_dir} does not exist.")

    X_test = f'{input_data_dir}/pallas.test.X.csv'
    y_test = f'{input_data_dir}/pallas.test.y.tsv'

    # Read the features and target data
    X_test = pd.read_csv(X_test, header=0, index_col=0)
    y_test = pd.read_csv(y_test, sep="\t", header=0, index_col=0)

    # Split the labels into lists
    y_test = y_test['y'].apply(lambda x: x.split(','))

    # Get just the y col
    y_test = y_test.values

    # For each of the pkl files, generate a report
    for file_name in os.listdir(input_models_dir):
        if file_name.endswith(".classifier.pkl"):
            model_path = os.path.join(input_models_dir, file_name)
            generate_report(
                X_test=X_test,
                y_test=y_test,
                model_path=model_path,
                output_dir=output_dir
            )


def generate_report(
    X_test,
    y_test,
    model_path,
    output_dir):

    print(f"Generating report for {model_path}...")
    with open(model_path, 'rb') as f:
        # Get model name from file name
        model_name = os.path.basename(model_path).replace(
            ".classifier.pkl", "")

        classifier = pickle.load(f)
        predictions = classifier.predict(X_test)
        generate_confusion_matrix(
            y_test=y_test,
            predictions=predictions,
            output_dir=output_dir,
            model_name=model_name
        )


def generate_confusion_matrix(
        y_test,
        predictions,
        output_dir,
        model_name):

    predictions_str = []
    for prediction in predictions:
        # Remove empty strings
        prediction = [x for x in prediction if x]
        prediction = f"{prediction[-1]} ({prediction[1]})"
        predictions_str.append(prediction)

    y_str = []
    for y_row in y_test:
        # Remove empty strings
        y_row = [x for x in y_row if x]
        print(y_row)
        y_row = f"{y_row[-1]} ({y_row[1]})"
        y_str.append(y_row)

    # get all unique values in y_str
    unique_y = list(set(y_str))
    print(unique_y)

    cm = confusion_matrix(y_str, predictions_str, labels=unique_y)



    fig, ax = plt.subplots(figsize=(14, 10))  # Increase figure size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_y)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)  # Add colorbar


    # Rotate x-axis labels after disp.plot()
    plt.xticks(rotation=90)
    plt.title("Confusion Matrix")

    # TODO: make this configurable
    plt.tight_layout()

    filename = f"{output_dir}/{model_name}.confusion_matrix.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)  # High resolution
    plt.close(fig)


def main():
    args = parse_args()
    generate_reports(
        input_models_dir=args.input_models_dir,
        input_data_dir=args.input_data_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
