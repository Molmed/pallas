import argparse
import os
import pickle
import pandas as pd

description = """
Runs selected models on a features dataset,
generates predictions, and saves the results.
"""

model_path_help = """
Path to the model file (e.g., *.classifier.pkl)."""

features_path_help = """
Path to the features dataset (.csv/.pkl)."""

output_dir_help = """
Dir to save the predictions."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=model_path_help,
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help=features_path_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    return parser.parse_args()


def predict(
        model_path,
        features_path,
        output_dir):

    print("Generating predictions with the following arguments:")
    print(f"Model path: {model_path}")
    print(f"Features path: {features_path}")
    print(f"Output directory: {output_dir}")

    # Test that model_path is a valid pkl file
    if not os.path.isfile(model_path):
        raise ValueError(f"Model path {model_path} is not a valid file.")
    if not model_path.endswith(".pkl"):
        raise ValueError(f"Model path {model_path} is not a valid pkl file.")

    # Test that features_path is a valid csv or pkl file
    if not os.path.isfile(features_path):
        raise ValueError(f"Features path {features_path} is not a valid file.")
    if not (features_path.endswith(".csv") or features_path.endswith(".pkl")):
        raise ValueError(f"Features path {features_path} is not a valid csv or pkl file.")

    # Check that output_dir is a valid directory
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory {output_dir} is not a valid directory.")

    # Output file name is name of model plus name of features
    # e.g., model_name_features_name.csv
    model_name = os.path.basename(model_path).split(".")[0]
    features_name = os.path.basename(features_path).split(".")[0]
    output_file_name = f"{model_name}_{features_name}.tsv"
    output_file_path = os.path.join(output_dir, output_file_name)

    # Load features dataset
    if features_path.endswith(".csv"):
        features = pd.read_csv(features_path, index_col=0)
    else:
        features = pd.read_pickle(features_path)

    # Load the model from pickl
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        predictions = model.predict(features)

        # Strip out first element of each prediction
        predictions = [pred[1:] for pred in predictions]

        # Join all non empty elements of each prediction
        predictions = [",".join([str(x) for x in pred if x]) for pred in predictions]

        # Join predictions with sample ids
        sample_ids = features.index
        predictions_df = pd.DataFrame(predictions, index=sample_ids, columns=["prediction"])

        # Save predictions to csv
        predictions_df.to_csv(output_file_path, sep="\t", index=True)


def main():
    args = parse_args()
    predict(
        model_path=args.model_path,
        features_path=args.features_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
