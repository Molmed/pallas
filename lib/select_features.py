import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import dotenv
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from gene_thesaurus import GeneThesaurus


dotenv.load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_JOBS = int(os.getenv("FEATURE_SELECTION_N_JOBS", 4))
RF_N_ESTIMATORS = int(os.getenv("FEATURE_SELECTION_RF_N_ESTIMATORS", 100))
RF_MAX_DEPTH = int(os.getenv("FEATURE_SELECTION_RF_MAX_DEPTH", 40))
RF_MAX_FEATURES = float(os.getenv("FEATURE_SELECTION_RF_MAX_FEATURES", 0.25))
RF_MIN_SAMPLES_LEAF = int(
    os.getenv("FEATURE_SELECTION_RF_MIN_SAMPLES_LEAF", 1))
RF_CLASS_WEIGHT = os.getenv(
    "FEATURE_SELECTION_RF_CLASS_WEIGHT", "balanced")


description = """
Reduces the feature space of a GEX or DNAm dataset.
"""

X_path_help = """
Path to CSV file containing features."""

y_path_help = """
Path to CSV file containing labels."""

output_dir_help = """
Directory to save the output."""

translate_genes_help = """
If true, will save a copy of the selected features
translated from Ensembl id to gene symbol.
Default: False
"""

scoring_metric_help = """
The metric to be used when evaluating features during cross-validation.
Default: "f1_macro"
"""

min_features_help = """
The minimum number of features to remain after reduction.
Default: 10."""

num_splits_help = """
Number of splits used for cross-validation.
Default: min(min_class_count, 5).
"""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--X_path",
        type=str,
        required=True,
        help=X_path_help,
    )
    parser.add_argument(
        "--y_path",
        type=str,
        required=True,
        help=y_path_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    parser.add_argument(
        "--translate_genes_to_symbols",
        type=bool,
        required=False,
        help=translate_genes_help,
    )
    parser.add_argument(
        "--scoring_metric",
        type=str,
        required=False,
        help=scoring_metric_help,
    )
    parser.add_argument(
        "--min_features",
        type=int,
        required=False,
        help=min_features_help,
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        required=False,
        help=num_splits_help,
    )
    return parser.parse_args()


def run_cv(X,
           y,
           n_splits,
           scoring_metric="f1_macro",
           n_jobs=N_JOBS):
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=1,
        random_state=RANDOM_SEED
    )

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    cv_results = cross_validate(
        rf,
        X,
        y,
        cv=cv,
        scoring=scoring_metric,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True,
        return_estimator=True
    )

    return cv_results


# TODO: make freq_threshold and frac input params
def analyze_feature_importances(output_dir,
                                cv_results,
                                X,
                                translate_genes_to_symbols=False,
                                freq_threshold=1,
                                frac=0.10):
    feature_names = X.columns

    importances = []

    for est in cv_results["estimator"]:
        importances.append(est.feature_importances_)

    importances = np.vstack(importances)

    importance_df = pd.DataFrame(
        importances,
        columns=feature_names
    )

    suffix = len(feature_names)
    output_path = os.path.join(output_dir, f"feature_importances.{suffix}.csv")
    importance_df.to_csv(output_path)

    top_k = int(X.shape[1] * frac)

    selected_per_fold = []

    for row in importances:

        idx = np.argsort(row)[::-1][:top_k]

        selected_per_fold.append(
            feature_names[idx]
        )

    counter = Counter()

    for features in selected_per_fold:
        counter.update(features)

    freq_df = pd.DataFrame(
        counter.items(),
        columns=["feature", "count"]
    )

    freq_df = freq_df.sort_values(
        "count",
        ascending=False
    )
    freq_path=os.path.join(output_dir, f"feature_frequencies.{suffix}.csv")
    freq_df.to_csv(freq_path, index=False)

    # Save all features with freq >= freq_threshold
    selected_features = freq_df[freq_df["count"] >= freq_threshold]["feature"].tolist()
    suffix = len(selected_features)
    selected_features_path = os.path.join(output_dir, f"selected_features.{suffix}.txt")

    with open(selected_features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    # save translated if requested
    if translate_genes_to_symbols:
        gt = GeneThesaurus()
        translated = gt.translate_genes(selected_features,
                                        source="ensembl_id", target="symbol")
        translated_path = os.path.join(
            output_dir, f"selected_features.{suffix}.translated.txt")
        with open(translated_path, 'w') as f:
            for translated_feature in translated.values():
                f.write(f"{translated_feature}\n")

    print(f"Selected {len(selected_features)} features with frequency >= {freq_threshold}. Saved to {selected_features_path}.")

    return selected_features


def reduce(X_path,
           y_path,
           output_dir,
           translate_genes_to_symbols=False,
           scoring_metric=None,
           min_features=None,
           num_splits=None):

    X = pd.read_csv(X_path, index_col=0).astype(np.float32)
    y = pd.read_csv(y_path, index_col=0)

    print("y class distribution:")
    print(y.value_counts())
    min_class_count = y.value_counts().min()
    y = y.iloc[:, 0].values

    # Get initial number of features
    n_features_remaining = len(X.columns)

    # Default min features
    if min_features is None:
        min_features = 10

    # min_class_count or 5, whichever is smaller
    n_splits = min(min_class_count, 5)

    # For the plots
    n_features = []
    train_f1_macros = []
    val_f1_macros = []

    while n_features_remaining > min_features:
        print("\n\n")
        print(f"Cross-validating with {n_features_remaining} features...")
        cv_results = run_cv(X, y, n_splits, scoring_metric)

        train_mean = cv_results['train_score'].mean()
        train_std = cv_results['train_score'].std()
        val_mean = cv_results['test_score'].mean()
        val_std = cv_results['test_score'].std()

        n_features.append(n_features_remaining)
        train_f1_macros.append(train_mean)
        val_f1_macros.append(val_mean)

        print(f"Train F1-macro: {train_mean:.3f} ± {train_std:.3f}")
        print(f"Val   F1-macro: {val_mean:.3f} ± {val_std:.3f}")
        print(f"Gap:            {train_mean - val_mean:.3f}")

        selected_features = analyze_feature_importances(
            output_dir,
            cv_results,
            X,
            translate_genes_to_symbols)

        X = X[selected_features]
        n_features_remaining = len(selected_features)

    # Draw the plot
    plt.plot(n_features, val_f1_macros)
    plt.xlabel("Number of selected features")
    plt.ylabel("Validation F1-macro")

    # Make the plot go down to 0 on the y axis
    plt.ylim(0, 1)

    for index in range(len(n_features)):
        plt.text(
            n_features[index],
            val_f1_macros[index],
            n_features[index],
            size=8)
    plt.savefig(f"{output_dir}/feature_reduction.svg")


def main():
    args = parse_args()

    # Print all args
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Create output dir if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    reduce(
        X_path=args.X_path,
        y_path=args.y_path,
        output_dir=args.output_dir,
        translate_genes_to_symbols=args.translate_genes_to_symbols,
        scoring_metric=args.scoring_metric,
        min_features=args.min_features,
        num_splits=args.num_splits
    )


if __name__ == "__main__":
    main()
