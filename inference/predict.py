import argparse
import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font size for matplotlib
plt.rcParams.update({'font.size': 16})

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

target_path_help = """
If targets are known and available, path to the targets tsv file."""

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
    parser.add_argument(
        "--target_path",
        type=str,
        required=False,
        help=target_path_help,
    )
    return parser.parse_args()


def predict(
        model_path,
        features_path,
        output_dir,
        target_path=None):

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

    targets = None
    if target_path:
        # Read tsv file with targets
        if not os.path.isfile(target_path):
            raise ValueError(f"Target path {target_path} is not a valid file.")
        if not target_path.endswith(".tsv"):
            raise ValueError(f"Target path {target_path} is not a valid tsv file.")
        targets = pd.read_csv(target_path, sep="\t", index_col=0)

    # Load the model from pickl
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        predictions = model.predict(features)

        # Join all non empty elements of each prediction
        predictions = [",".join([str(x) for x in pred if x]) for pred in predictions]

        # Join predictions with sample ids
        sample_ids = features.index
        predictions_df = pd.DataFrame(predictions, index=sample_ids, columns=["prediction"])

        # If targets, join the df with predictions
        if targets is not None:
            targets = targets.loc[sample_ids]
            predictions_df = predictions_df.join(targets, how="left")

        # Save predictions to csv
        predictions_df.to_csv(output_file_path, sep="\t", index=True)

        return output_file_path


def pie_chart(output_path, predictions, title="Predictions"):
    # Count the number of occurrences for each class in the 'prediction' column
    class_counts = predictions.value_counts()
    total = class_counts.sum()
    percentages = (class_counts / total * 100).round(1)

    # Convert the Seaborn 'tab20' palette to a Matplotlib colormap
    tab20_palette = sns.color_palette('tab20')

    # Plot a pie chart
    plt.figure(figsize=(16, 16))
    wedges, texts, autotexts = plt.pie(
        class_counts,
        labels=None,  # Turn off pie labels (we'll show everything in legend)
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.9,
        colors=tab20_palette
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(title)

    # Make a legend, putting percentages in the legend
    legend_labels = [
        f"{label} ({count}, {percent}%)"
        for label, count, percent in zip(class_counts.index, class_counts, percentages)
    ]
    plt.legend(wedges, legend_labels, loc="upper right")

    # Make sure the chart fits in the figure
    plt.tight_layout()

    # Save the pie chart
    plt.savefig(output_path)
    plt.close()


def predictions_by_subtype(predictions):
    # Column 0 is node, e.g. "Disease", drop it
    predictions = predictions.drop(columns=[0])

    # Rename the new first column to "disease"
    predictions = predictions.rename(columns={1: "disease"})

    # Column 1 tells us the disease.
    # Get the indices of the remaining columns
    indices = predictions.columns[1:]

    # Of the columns in indices, get the last value for each row that is not None
    # and put it in the "last_value" column
    predictions["last_value"] = predictions[indices].apply(
        lambda x: x[x.notnull()].values[-1] if x.notnull().any() else None, axis=1
    )

    # Join disease and last_value columns and put in "prediction" column for each row
    predictions["prediction"] = predictions.apply(
        lambda x: f"{x['disease']}, {x['last_value']}" if pd.notnull(x['last_value']) else None, axis=1
    )
    return predictions["prediction"]


def reports(csv_path, output_dir):
    """
    Generates reports for the predictions.
    """
    # Read the predictions
    predictions = pd.read_csv(csv_path, sep="\t", index_col=0)

    filename_prefix = os.path.basename(csv_path)

    # Generate reports
    # If y column is present, generate classification report
    if "y" in predictions.columns:
        y_true = predictions["y"]
        y_pred = predictions["prediction"]
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        filename = f"{filename_prefix}.classification_report.csv"
        report_df.to_csv(os.path.join(output_dir, filename), sep="\t")

    # Split predictions by comma
    predictions = predictions['prediction']
    predictions = predictions.str.split(',', expand=True)

    # Disease-level predictions are in column 1
    disease_predictions = predictions[1]
    filename = f"{filename_prefix}.pie_chart.disease.png"
    file_path = os.path.join(output_dir, filename)
    pie_chart(file_path, disease_predictions, "Disease predictions")

    subtype_predictions = predictions_by_subtype(predictions)
    filename = f"{filename_prefix}.pie_chart.subtype.png"
    file_path = os.path.join(output_dir, filename)
    pie_chart(file_path, subtype_predictions, "Subtype predictions")

    # Break down the predictions by disease
    # Column 1 is disease, break separate dataframes for each disease
    diseases = predictions[1].unique()
    for disease in diseases:
        disease_predictions = predictions[predictions[1] == disease]
        subtype_predictions = predictions_by_subtype(disease_predictions)
        filename = f"{filename_prefix}.pie_chart.{disease}.png"
        file_path = os.path.join(output_dir, filename)
        pie_chart(file_path, subtype_predictions, f"Subtype predictions for {disease}")



def main():
    args = parse_args()
    csv_path = predict(
        model_path=args.model_path,
        features_path=args.features_path,
        output_dir=args.output_dir,
        target_path=args.target_path,
    )
    reports(csv_path, args.output_dir)


if __name__ == "__main__":
    main()
