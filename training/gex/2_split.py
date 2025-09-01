import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


description = """
Split data into train, calibration and test sets."""

features_path_help = """
Path to the features file (.csv or .pkl), indexed by column 'id'."""

target_path_help = """
Path to the target file (.tsv), containing columns 'id' and 'y'."""

output_dir_help = """
Directory to save the train and test sets."""

train_frac_help = """
Fraction of the data to use for training (default: 0.70).
The remaining data will be split equally into calibration and test."""

min_calib_samples_per_class_help = """
Minimum number of samples per class that must be available to further split into
calibration and test sets. Classes not reaching this threshold will be excluded
from the calibration set (default: 10)."""

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
        "--train_frac",
        type=float,
        default=0.70,
        help=train_frac_help,
    )
    parser.add_argument(
        "--min_calib_samples_per_class",
        type=int,
        default=10,
        help=min_calib_samples_per_class_help,
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
               train_frac,
               min_calib_samples_per_class,
               random_state):

    print("Splitting data with the following parameters:")
    print(f"Features path: {features_path}")
    print(f"Target path: {target_path}")
    print(f"Output directory: {output_dir}")
    print(f"Train fraction: {train_frac}")
    print(f"Minimum calibration samples per class: {min_calib_samples_per_class}")
    print(f"Random state: {random_state}")

    # Read the features and target data

    # Check if the features file is in CSV format or pkl format
    features_format = None
    if features_path.endswith('.pkl'):
        features = pd.read_pickle(features_path)
        features_format = 'pkl'
    elif features_path.endswith('.csv'):
        features = pd.read_csv(features_path, header=0, index_col=0)
        features_format = 'csv'
    else:
        raise ValueError("Features file must be in CSV or pkl format.")

    # Check if the target file is in tsv format, otherwise raise an error
    if not target_path.endswith('.tsv'):
        raise ValueError("Target file must be in .tsv format.")
    target = pd.read_csv(target_path, sep="\t", header=0, index_col=0)

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

    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        train_size=train_frac,
        random_state=random_state,
        stratify=y)

    # In the rest split, remove samples totaling
    # less than double the calibration sample minimum
    # Put them into a separate data split
    low_class_indices = y_rest.isin(
        y_rest.value_counts()[
            y_rest.value_counts() < min_calib_samples_per_class].index
    )
    X_low, y_low = X_rest[low_class_indices], y_rest[low_class_indices]
    X_rest = X_rest[~X_rest.index.isin(X_low.index)]
    y_rest = y_rest[~y_rest.index.isin(y_low.index)]

    # Print all classes in this split and the number of samples in each
    print("The following classes will be excluded from the calibration set:")
    for class_label in y_low.unique():
        class_count = y_low.value_counts().get(class_label, 0)
        print(f" - Class {class_label}: {class_count} samples")

    # Split remaining data 50/50
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=0.5,
        random_state=random_state,
        stratify=y_rest
    )

    # Merge low items into test set
    X_test = pd.concat([X_test, X_low])
    y_test = pd.concat([y_test, y_low])

    # Save train and test data
    if features_format == 'csv':
        X_train.to_csv(f'{output_dir}/pallas.train.X.csv')
        X_calib.to_csv(f'{output_dir}/pallas.calib.X.csv')
        X_test.to_csv(f'{output_dir}/pallas.test.X.csv')
    elif features_format == 'pkl':
        X_train.to_pickle(f'{output_dir}/pallas.train.X.pkl')
        X_calib.to_pickle(f'{output_dir}/pallas.calib.X.pkl')
        X_test.to_pickle(f'{output_dir}/pallas.test.X.pkl')
    else:
        raise ValueError("Unsupported features format.")

    y_train.to_csv(f'{output_dir}/pallas.train.y.tsv', sep='\t')
    y_calib.to_csv(f'{output_dir}/pallas.calib.y.tsv', sep='\t')
    y_test.to_csv(f'{output_dir}/pallas.test.y.tsv', sep='\t')


def main():
    args = parse_args()
    split_data(
        features_path=args.features_path,
        target_path=args.target_path,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        min_calib_samples_per_class=args.min_calib_samples_per_class,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
