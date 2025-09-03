import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.feature_selection import VarianceThreshold


description = """
RandomForest hyperparameter tuning and feature selection."""

features_path_help = """
Path to the features file (.csv or .pkl), indexed by column 'id'."""

target_path_help = """
Path to the target file (.tsv), containing columns 'id' and 'y'."""

output_dir_help = """
Directory to save results."""

random_state_help = """
Random state for reproducibility (default: None)."""

n_threads_help = """
Number of threads to use (default: unlimited)."""


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
        "--random_state",
        type=int,
        default=None,
        help=random_state_help,
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=-1,
        help=n_threads_help,
    )
    return parser.parse_args()


def run_grid_search(X,
                    y,
                    pipeline,
                    param_grid,
                    output_dir,
                    prefix,
                    random_state,
                    n_threads):

    cv = StratifiedShuffleSplit(n_splits=5,
                                test_size=0.2,
                                random_state=random_state)

    roc_auc_scorer = make_scorer(
        roc_auc_score,
        needs_proba=True,
        multi_class="ovr",
        average="macro"
    )

    grid = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=cv,
        scoring=roc_auc_scorer,
        n_jobs=n_threads,
        verbose=2,
        random_state=random_state
    )

    grid.fit(X, y)

    print("Best parameters:", grid.best_params_)
    print("Best CV AUC:", grid.best_score_)

    best_pipeline = grid.best_estimator_

    selector = best_pipeline.named_steps["feature_selection"]
    mask = selector.get_support()
    selected_features = X.columns[mask]

    print(f"Number of selected features: {len(selected_features)}")
    print("Top selected features:", selected_features[:20])

    model_final = best_pipeline.named_steps[prefix]
    importances = model_final.feature_importances_
    feature_importances = pd.DataFrame({
        "Gene": selected_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Save
    joblib.dump(best_pipeline, f'{output_dir}/{prefix}.best_pipeline.pkl')
    feature_importances.to_csv(f'{output_dir}/{prefix}.feature_importances.tsv', sep='\t', index=False)

    cv_results = pd.DataFrame(grid.cv_results_)

    # Keep the most informative columns
    cv_results = cv_results[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ]

    # Expand "params" dict into separate columns for readability
    param_cols = cv_results["params"].apply(pd.Series)
    cv_results_expanded = pd.concat([cv_results.drop(columns=["params"]), param_cols], axis=1)

    # Sort by rank
    cv_results_expanded = cv_results_expanded.sort_values("rank_test_score")

    # Save all results
    cv_results_expanded.to_csv(f"{output_dir}/{prefix}.cv_results.tsv", sep="\t", index=False)

    print(f"Saved CV results to {output_dir}/{prefix}.cv_results.tsv")


def tune(features_path,
         target_path,
         output_dir,
         random_state,
         n_threads):

    print("Tuning model with the following parameters:")
    print(f"Features path: {features_path}")
    print(f"Target path: {target_path}")
    print(f"Output directory: {output_dir}")
    print(f"Random state: {random_state}")
    print(f"Number of threads: {n_threads}")

    # Read the features and target data

    # Check if the features file is in CSV format or pkl format
    if features_path.endswith('.pkl'):
        X = pd.read_pickle(features_path)
    elif features_path.endswith('.csv'):
        X = pd.read_csv(features_path, header=0, index_col=0)
    else:
        raise ValueError("Features file must be in CSV or pkl format.")

    # Check if the target file is in tsv format, otherwise raise an error
    if not target_path.endswith('.tsv'):
        raise ValueError("Target file must be in .tsv format.")
    y = pd.read_csv(target_path, sep="\t", header=0, index_col=0)
    y = y["y"]

    # Align X and y just in case
    X = X.loc[y.index]

    print("Hypertuning DecisionTree...")
    # Step 1: Define pipeline
    pipeline = Pipeline([
        ("feature_selection", SelectFromModel(
            DecisionTreeClassifier(
                random_state=random_state
            ),
            threshold="median"
        )),
        ("dt", DecisionTreeClassifier(random_state=random_state))
    ])

    param_grid = {
        # Keep feature selection relatively strict (helps with thousands of genes)
        "feature_selection__threshold": ["median", "1.25*median"],

        # DecisionTree hyperparameters
        "dt__max_depth": [15, 20],          # limit depth to avoid overfitting
        "dt__min_samples_split": [10, 20],  # require more samples to split
        "dt__min_samples_leaf": [5, 10, 20],    # prevent tiny leaves
        "dt__max_features": [0.2, "sqrt"],    # sqrt is often good for high-d data
        "dt__class_weight": ["balanced"]       # handle imbalanced rare classes
    }

    run_grid_search(X=X,
                    y=y,
                    pipeline=pipeline,
                    param_grid=param_grid,
                    output_dir=output_dir,
                    prefix="dt",
                    random_state=random_state,
                    n_threads=n_threads)

    print("Hypertuning RandomForest...")

    # Step 1: Define pipeline
    pipeline = Pipeline([
        ("feature_selection", SelectFromModel(
            RandomForestClassifier(
                n_estimators=100,  # small RF for initial selection
                n_jobs=n_threads,
                random_state=random_state
            ),
            threshold="median"
        )),
        ("rf", RandomForestClassifier(n_jobs=n_threads, random_state=random_state))
    ])

    # Step 2: Define parameter grid for hyperparameter tuning
    param_grid = {
        "feature_selection__threshold": ["0.50*median", "0.25*median"],
        "rf__n_estimators": [200],
        "rf__max_depth": [None, 20],
        "rf__max_features": ["sqrt", "log2"],
        "rf__min_samples_leaf": [1, 5],
        "rf__class_weight": ["balanced"]  # handle rare classes
    }

    run_grid_search(X=X,
                    y=y,
                    pipeline=pipeline,
                    param_grid=param_grid,
                    output_dir=output_dir,
                    prefix="rf",
                    random_state=random_state,
                    n_threads=n_threads)


def main():
    args = parse_args()
    tune(
        features_path=args.features_path,
        target_path=args.target_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_threads=args.n_threads
    )


if __name__ == "__main__":
    main()
