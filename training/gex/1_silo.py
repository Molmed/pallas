import argparse
import pandas
from sklearn.model_selection import train_test_split


description = """
Silo data into labeled diagnostic, labeled labeled_misc, and unlabeled datasets."""

features_path_help = """
Path to the features file (.csv or .pkl), indexed by column 'sample'."""

gexml_samplesheet_path_help = """
Path to the samplesheet used as input to the nf-core gexml pipeline,
containing columns 'sample' and 'subtype'."""

output_dir_help = """
Directory to save the siloed sets."""

min_samples_per_class_help = """
Minimum number of samples per class to include in the "labeled diagnostic" set.
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

    # Rename subtype to y
    target = target.rename(columns={'subtype': 'y'})

    # UNLABELED
    unlabeled = target[target['y'].str.contains("B-other|AML-other|ALLNOS")]

    # LABELED MISC
    # Get all relapse samples. These are sample names that match one of the following patterns:
    # - Start with "SJ" and end in "_R" or "_R" followed by a digit
    # - Start with "ALL" and end in "_2" or "_3"
    relapse = target[target.index.str.match(r'^(SJ.*_R\d*|ALL.*_(2|3))$')]

    # Get replicate samples. These start with "ALL_" and end in "_r"
    replicate = target[target.index.str.match(r'^ALL_.*_r$')]

    # Get all items where subtype starts with "UNRECOGNIZED"
    unrecognized = target[target['y'].str.startswith("UNRECOGNIZED")]

    # Get all subtypes that have less than min_samples_per_class
    rare = target[target['y'].map(target['y'].value_counts()) < min_samples_per_class]

    # Merge relapse, replicate, unrecognized and rare
    labeled_misc = pandas.concat([relapse, replicate, unrecognized, rare])

    # Remove any items that are also in unlabeled
    labeled_misc = labeled_misc[~labeled_misc.index.isin(
        unlabeled.index)]

    # LABELED DIAGNOSTIC
    # print all items not in labeled_misc or unlabeled
    labeled_diagnostic = target[~target.index.isin(
        labeled_misc.index) & ~target.index.isin(unlabeled.index)]

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

    # Get all features that are in labeled_misc
    y_labeled_misc = data.loc[data.index.isin(labeled_misc.index), 'y']
    labeled_misc = data.loc[data.index.isin(labeled_misc.index)].drop(columns=['y'])

    # Get all features that are in unlabeled
    y_unlabeled = data.loc[data.index.isin(unlabeled.index), 'y']
    unlabeled = data.loc[data.index.isin(unlabeled.index)].drop(columns=['y'])

    # Get all features that are in recognized
    y_labeled_diagnostic = data.loc[data.index.isin(labeled_diagnostic.index), 'y']
    labeled_diagnostic = data.loc[data.index.isin(labeled_diagnostic.index)].drop(columns=['y'])

    # Save all the y files
    y_labeled_misc.to_csv(
        f'{output_dir}/pallas.labeled_misc.y.tsv', sep='\t')
    y_unlabeled.to_csv(f'{output_dir}/pallas.unlabeled.y.tsv', sep='\t')
    y_labeled_diagnostic.to_csv(f'{output_dir}/pallas.labeled_diagnostic.y.tsv', sep='\t')

    # Save train and test data
    if features_format == 'csv':
        labeled_misc.to_csv(
            f'{output_dir}/pallas.labeled_misc.X.csv')
        unlabeled.to_csv(f'{output_dir}/pallas.unlabeled.X.csv')
        labeled_diagnostic.to_csv(f'{output_dir}/pallas.labeled_diagnostic.X.csv')
    elif features_format == 'pkl':
        labeled_misc.to_pickle(
            f'{output_dir}/pallas.labeled_misc.X.pkl')
        unlabeled.to_pickle(f'{output_dir}/pallas.unlabeled.X.pkl')
        labeled_diagnostic.to_pickle(f'{output_dir}/pallas.labeled_diagnostic.X.pkl')
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
