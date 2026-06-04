import os
import pandas as pd
import numpy as np
import argparse
import dotenv
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

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

scoring_metric_help = """
The metric to be used when evaluating features during cross-validation.
Default: "f1_macro"
"""

min_features_help = """
The minimum number of features to remain after reduction.
Default: 10."""

num_splits_help = """
Number of splits used for cross-validation.
Default: max(5, min(min_class_count, 10)).
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
        "--scoring_metric",
        type=str,
        required=False,
        help=scoring_metric_help,
    )
    parser.add_argument(
        "--min-features",
        type=int,
        required=False,
        help=min_features_help,
    )
    parser.add_argument(
        "--num-splits",
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

    train_mean = cv_results['train_score'].mean()
    train_std = cv_results['train_score'].std()
    val_mean = cv_results['test_score'].mean()
    val_std = cv_results['test_score'].std()

    print(f"Train F1-macro: {train_mean:.3f} ± {train_std:.3f}")
    print(f"Val   F1-macro: {val_mean:.3f} ± {val_std:.3f}")
    print(f"Gap:            {train_mean - val_mean:.3f}")

    return cv_results


def analyze_feature_importances(output_dir, cv_results, X, suffix, freq_threshold=1, frac=0.10):
    feature_names = X.columns

    importances = []

    for est in cv_results["estimator"]:
        importances.append(est.feature_importances_)

    importances = np.vstack(importances)

    importance_df = pd.DataFrame(
        importances,
        columns=feature_names
    )

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
    selected_features_path = os.path.join(output_dir, f"selected_features.{suffix}.txt")

    with open(selected_features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    print(f"Selected {len(selected_features)} features with frequency >= {freq_threshold}. Saved to {selected_features_path}.")

    return importance_df, freq_df, selected_features


def reduce(X_path,
           y_path,
           output_dir,
           scoring_metric=None,
           min_features=10,
           num_splits=None):
    return


def main():
    args = parse_args()
    reduce(
        X_path=args.X_path,
        y_path=args.y_path,
        output_dir=args.output_dir,
        scoring_metric=args.scoring_metric,
        min_features=args.min_features,
        num_splits=args.num_splits
    )


if __name__ == "__main__":
    main()
