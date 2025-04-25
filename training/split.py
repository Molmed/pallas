import argparse
import pandas
from sklearn.model_selection import train_test_split


description = """
Split data into train and test sets."""

features_path_help = """
Path to the features file (CSV), indexed by column 'id'."""

target_path_help = """
Path to the hierarchical target file (TSV), containing columns 'id'
and hierarchical, comma-separated 'y'."""

output_dir_help = """
Directory to save the train and test sets."""

min_samples_per_class_help = """
Minimum number of samples per class to include in the train set.
Classes not reaching this threshold will be excluded
from training (default: 10)."""

test_frac_help = """
Fraction of the data to use for testing (default: 0.2)."""

random_state_help = """
Random state for reproducibility (default: None)."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help=features_path_help,
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help=target_path_help,
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
        default=10,
        help=min_samples_per_class_help,
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help=test_frac_help,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help=random_state_help,
    )
    return parser.parse_args()


def split_data(features_path,
               target_path,
               output_dir,
               min_samples_per_class,
               test_frac,
               random_state):

    print("Splitting data with the following parameters:")
    print(f"Features path: {features_path}")
    print(f"Target path: {target_path}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum samples per class: {min_samples_per_class}")
    print(f"Test fraction: {test_frac}")
    print(f"Random state: {random_state}")

    # Read the features and target data
    features = pandas.read_csv(features_path, header=0, index_col=0)
    target = pandas.read_csv(target_path, sep="\t", header=0, index_col=0)

    # Merge the features and target data on the index
    data = features.join(target)

    # Count rows with NaN y values
    nan_count = data['y'].isna().sum()

    # Filter out rows with NaN y values
    if nan_count > 0:
        print(f"Dropping {nan_count} rows with NaN y values...")
        # Drop rows with NaN y values
        data = data.dropna(subset=['y'])

    X = data.drop(columns=['y'])
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_frac,
        random_state=random_state,
        stratify=y)

    # Save train and test data
    X_train.to_csv(f'{output_dir}/pallas.train.X.csv')
    X_test.to_csv(f'{output_dir}/pallas.test.X.csv')
    y_train.to_csv(f'{output_dir}/pallas.train.y.tsv', sep='\t')
    y_test.to_csv(f'{output_dir}/pallas.test.y.tsv', sep='\t')


def main():
    args = parse_args()
    split_data(
        features_path=args.features_path,
        target_path=args.target_path,
        output_dir=args.output_dir,
        min_samples_per_class=args.min_samples_per_class,
        test_frac=args.test_frac,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
