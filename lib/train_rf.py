import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
import joblib
import json
from pathlib import Path
from datetime import datetime
import argparse
import dotenv
from mlflow.models import infer_signature

dotenv.load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_JOBS = int(os.getenv("TRAINING_N_JOBS", 4))
CV_N_ITER = int(os.getenv("TRAINING_CV_N_ITER", 30))
RF_ESTIMATORS = int(os.getenv("TRAINING_RF_ESTIMATORS", 500))

description = """
Tunes hyperparameters and then trains Random Forest
model with the best selected parameters.
"""

X_path_help = """
Path to CSV file containing features."""

y_path_help = """
Path to CSV file containing labels."""

output_dir_help = """
Directory to save the output."""

modality_help = """
Modality of the data, 'GEX' or 'DNAm'"""

selected_features_path_help = """
Path to TXT file containing selected features,
one per line. Default: None
"""

scoring_metric_help = """
The metric to be used when evaluating features during cross-validation.
Default: "f1_macro"
"""

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
        "--modality",
        type=str,
        required=True,
        help=modality_help,
    )
    parser.add_argument(
        "--selected_features_path",
        type=str,
        required=False,
        help=selected_features_path_help,
    )
    parser.add_argument(
        "--scoring_metric",
        type=str,
        required=False,
        help=scoring_metric_help,
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        required=False,
        help=num_splits_help,
    )
    return parser.parse_args()


def get_reduced_X_path(output_dir,
                       n_features):
    return f"{output_dir}/X_reduced.{n_features}.csv"


def reduce_to_selected(X_path,
                       selected_features_path,
                       output_dir):
    if not selected_features_path:
        return X_path

    # Read in selected features
    with open(selected_features_path, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]

    n_features = len(selected_features)
    print(f"Number of selected features: {n_features}")

    reduced_X_path = get_reduced_X_path(output_dir,
                                        n_features)

    # Check if reduced file already exists
    if not Path(reduced_X_path).exists():
        X_train = pd.read_csv(X_path, index_col=0)
        X_train = X_train[selected_features]
        X_train.to_csv(reduced_X_path, index=True)

    return reduced_X_path


def train(X_path,
          y_path,
          output_dir,
          n_splits=None,
          scoring_metric="f1_macro",
          n_jobs=N_JOBS):

    # Read in X_train and y_train
    # Also reduce memory usage by converting to float32
    X_train = pd.read_csv(X_path, index_col=0).astype(np.float32)
    y_train = pd.read_csv(y_path, index_col=0)

    print("y_train class distribution:")
    print(y_train.value_counts())

    # Get least frequently ocurring class count
    min_class_count = y_train.value_counts().min()

    # Set n_splits if not set
    if not n_splits:
        # min_class_count or 5, whichever is smaller
        n_splits = min(min_class_count, 5)

    # Convert to 1D numpy array
    y_train = y_train.iloc[:, 0].values
    y_train

    # parallelism handled at search level
    rf = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    # Read in param grid json
    here = Path(__file__).parent
    params_file = here / "rf_cv_params.json"
    with open(params_file) as f:
        param_grid = json.load(f)

    print(f"Running CV with params: {param_grid}")

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=1,
        random_state=RANDOM_SEED
    )

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=CV_N_ITER,
        cv=cv,
        scoring=scoring_metric,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        verbose=2,
        refit=True
    )

    search.fit(X_train, y_train)

    print("Best hyperparameters:", search.best_params_)
    print("Best CV (f1_macro):",   search.best_score_)

    rf_params = search.best_params_
    rf_params['n_estimators'] = RF_ESTIMATORS
    print("Final RF parameters:", rf_params)

    # Save params
    hyperparams_path = os.path.join(output_dir, "hyperparams.joblib")
    joblib.dump(rf_params, hyperparams_path)

    final_rf = RandomForestClassifier(
        **rf_params,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED
    )

    # Fit on all training data
    final_rf.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(output_dir, "rf_model.joblib")
    joblib.dump(final_rf, model_path)

    # Infer MLFlow signature from training data + predictions
    predictions = final_rf.predict(X_train)
    signature = infer_signature(X_train, predictions)

    # Save mlflow signature
    signature_path = os.path.join(output_dir, "mlflow_signature.joblib")
    joblib.dump(signature, signature_path)

    print(f"Saved hyperparameters to {hyperparams_path}")
    print(f"Saved final RF model to {model_path}")
    print(f"Saved MLFlow signature to {signature_path}")


def main():
    args = parse_args()

    # Create output dir with current timestamp
    output_dir = args.output_dir
    modality = args.modality
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    dir_name = f"rf_{modality}_{timestamp}"
    output_dir = f"{output_dir}/{dir_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_path = reduce_to_selected(
        X_path=args.X_path,
        selected_features_path=args.selected_features_path,
        output_dir=output_dir)

    train(
        X_path=X_path,
        y_path=args.y_path,
        output_dir=output_dir,
        scoring_metric=args.scoring_metric
    )


if __name__ == "__main__":
    main()
