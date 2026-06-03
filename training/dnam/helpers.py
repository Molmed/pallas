import os
import pandas as pd
import numpy as np
import joblib
import dotenv
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

dotenv.load_dotenv()

DATASETS_PATH = os.getenv("DATASETS_PATH")
MODELS_DIR = os.getenv("MODELS_DIR")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_SPLITS = int(os.getenv("N_SPLITS", 5))
N_JOBS = int(os.getenv("N_JOBS", 4))


def run_cv(X, y, n_jobs=N_JOBS):
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=1,
        random_state=RANDOM_SEED
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=40,
        max_features=0.25,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,  # important when parallelizing CV
    )

    cv_results = cross_validate(
        rf,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True,
        return_estimator=True
    )

    train_mean = cv_results['train_score'].mean()
    train_std  = cv_results['train_score'].std()
    val_mean   = cv_results['test_score'].mean()
    val_std    = cv_results['test_score'].std()

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
