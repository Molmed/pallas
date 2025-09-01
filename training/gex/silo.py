import argparse
import pandas
from sklearn.model_selection import train_test_split


description = """
Silo data into known subtype, misc subtype and unknown subtype."""

features_path_help = """
Path to the features file (.csv or .pkl), indexed by column 'sample'."""

gexml_samplesheet_path_help = """
Path to the samplesheet used as input to the nf-core gexml pipeline,
containing columns 'sample' and 'subtype'."""

output_dir_help = """
Directory to save the siloed sets."""

min_samples_per_class_help = """
Minimum number of samples per class to include in the known subtype set.
Classes not reaching this threshold will be excluded
from training (default: 5)."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help=features_path_help,
    )
    parser.add_argument(
        "--gexml_samplesheet_path",
        type=str,
        required=True,
        help=gexml_samplesheet_path_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=5,
        help=min_samples_per_class_help,
    )
    return parser.parse_args()


def silo_data(features_path,
              gexml_samplesheet_path,
              output_dir,
              min_samples_per_class):

    print("Siloing data with the following parameters:")
    print(f"Features path: {features_path}")
    print(f"Samplesheet path: {gexml_samplesheet_path}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum samples per class: {min_samples_per_class}")

    # Read the target data
    # Check if the target file is in csv format, otherwise raise an error
    if not gexml_samplesheet_path.endswith('.csv'):
        raise ValueError("Target file must be in .csv format.")
    target = pandas.read_csv(
        gexml_samplesheet_path, sep=",", header=0, index_col=0)

    # Rename target index to "id"
    target.index.name = "id"

    # Just keep subtype column in target
    target = target[['subtype']]

    # Get all items where subtype starts with "UNRECOGNIZED"
    unrecognized = target[target['subtype'].str.startswith("UNRECOGNIZED")]

    # Get all subtypes that have less than min_samples_per_class
    rare = target[target['subtype'].map(target['subtype'].value_counts()) < min_samples_per_class]

    # Merge unrecognized and rare
    rare_and_unrecognized = pandas.concat([rare, unrecognized])

    # Get all items where subtype is "B-other" or "AML-other"
    unknown = target[target['subtype'].str.contains("B-other|AML-other")]

    # print all items not in unrecognized or unknown
    known = target[~target.index.isin(
        rare_and_unrecognized.index) & ~target.index.isin(unknown.index)]

    # Check if the features file is in CSV format or pkl format
    features_format = None
    if features_path.endswith('.pkl'):
        features = pandas.read_pickle(features_path)
        features_format = 'pkl'
    elif features_path.endswith('.csv'):
        features = pandas.read_csv(features_path, header=0, index_col=0)
        features_format = 'csv'
    else:
        raise ValueError("Features file must be in CSV or pkl format.")

    # Merge the features and target data on the index
    data = features.join(target)

    # Get all features that are in rare_and_unrecognized
    rare_and_unrecognized = data[
        data['subtype'].isin(
            rare_and_unrecognized['subtype'])].drop(columns=['subtype'])

    # Get all features that are in unknown
    unknown = data[
        data['subtype'].isin(
            unknown['subtype'])].drop(columns=['subtype'])

    # Get all features that are in recognized
    known = data[
        data['subtype'].isin(
            known['subtype'])].drop(columns=['subtype'])

    y = data['subtype']
    y.to_csv(f'{output_dir}/pallas.y.tsv', sep='\t')

    # Save train and test data
    if features_format == 'csv':
        rare_and_unrecognized.to_csv(
            f'{output_dir}/pallas.rare_and_unrecognized.X.csv')
        unknown.to_csv(f'{output_dir}/pallas.unknown.X.csv')
        known.to_csv(f'{output_dir}/pallas.known.X.csv')
    elif features_format == 'pkl':
        rare_and_unrecognized.to_pickle(
            f'{output_dir}/pallas.rare_and_unrecognized.X.pkl')
        unknown.to_pickle(f'{output_dir}/pallas.unknown.X.pkl')
        known.to_pickle(f'{output_dir}/pallas.known.X.pkl')
    else:
        raise ValueError("Unsupported features format.")


def main():
    args = parse_args()
    silo_data(
        features_path=args.features_path,
        gexml_samplesheet_path=args.gexml_samplesheet_path,
        output_dir=args.output_dir,
        min_samples_per_class=args.min_samples_per_class
    )


if __name__ == "__main__":
    main()
