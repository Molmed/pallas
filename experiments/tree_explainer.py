from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import argparse
import os
import pickle


description = """
Runs explainability reports on a tree classifier.
"""

model_path_help = """
Path to the model file (e.g., *.classifier.pkl)."""

output_dir_help = """
Dir to save the reports."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=model_path_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    return parser.parse_args()


def report(
        model_path,
        output_dir):

    print("Generating reports with the following arguments:")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")

    # Check the model is a pkl file
    if not model_path.endswith(".pkl"):
        raise ValueError("Model path must be a pkl file")

    # Check the output directory exists
    if not os.path.exists(output_dir):
        raise ValueError("Output directory does not exist")


    model = None
    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # Check the model is a DecisionTreeClassifier
    if not isinstance(model, DecisionTreeClassifier):
        raise ValueError("Model is not a DecisionTreeClassifier")
    # Check the model has been trained
    if not hasattr(model, "tree_"):
        raise ValueError("Model has not been trained")

    text_representation = tree.export_text(model)

    # filename is model name without extension
    filename = os.path.splitext(os.path.basename(model_path))[0]
    filename = f"{output_dir}/{filename}_tree_report.txt"

    # Save the report to the output directory
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(text_representation)


def main():
    args = parse_args()
    report(
        model_path=args.model_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
