import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
import joblib

DATASETS_PATH = "/home/mariya/Data/pallas/mlomix_datasets"

TRAIN_DATASET_PATH = os.path.join(DATASETS_PATH, "train-dnam")
X_train_path = os.path.join(TRAIN_DATASET_PATH, "features.dnam.csv")
y_train_path = os.path.join(TRAIN_DATASET_PATH, "labels.csv")

output_dir = "/home/mariya/Data/pallas/models/dnam"
random_seed = 42
n_jobs = 4
selected_features_path = os.path.join(output_dir, "selected_features.txt")
pipeline_path = os.path.join(output_dir, "pipeline.pkl")
hyperparams_path = os.path.join(output_dir, "hyperparams.pkl")
model_path = os.path.join(output_dir, "rf_model.pkl")

# Print configuration
print(f"X_train_path: {X_train_path}")
print(f"y_train_path: {y_train_path}")
print(f"output_dir: {output_dir}")
print(f"random_seed: {random_seed}")
print(f"n_jobs: {n_jobs}")
print(f"selected_features_path: {selected_features_path}")
print(f"pipeline_path: {pipeline_path}")
print(f"hyperparams_path: {hyperparams_path}")
print(f"model_path: {model_path}")

# Hardcoded variables
FEATURE_SELECTION_PERCENTILE = 10  # KEEP TOP n% OF CPG SITES
FINAL_N_ESTIMATORS = 500

# Read in X_train and y_train
X_train = pd.read_csv(X_train_path, index_col=0)
X_train

y_train = pd.read_csv(y_train_path, index_col=0)
y_train

# Show class distributions in y_train and y_test
print("y_train class distribution:")
print(y_train.value_counts())

y_train = y_train.iloc[:, 0].values  # Convert to 1D numpy array
X_train = X_train.astype(np.float32)

N_REPEATS = 1

rf_params = {
    'max_depth': 40,
    'max_features': 0.25,
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
}

fs_percentile = FEATURE_SELECTION_PERCENTILE
rf_params['n_estimators'] = FINAL_N_ESTIMATORS

print("Final RF parameters:", rf_params)
print("Feature selection percentile:", fs_percentile)

# Build final pipeline
final_pipeline = Pipeline([
    ("feature_selection", SelectPercentile(
        score_func=f_classif,
        percentile=fs_percentile
    )),
    ("rf", RandomForestClassifier(
        **rf_params,
        n_jobs=1,
        random_state=random_seed
    ))
])

# Cross-validate with train score to monitor gap
n_splits = 5
cv = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=N_REPEATS,
    random_state=random_seed
)

cv_results = cross_validate(
    final_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=n_jobs,
    verbose=2,
    return_train_score=True
)

train_mean = cv_results['train_score'].mean()
train_std  = cv_results['train_score'].std()
val_mean   = cv_results['test_score'].mean()
val_std    = cv_results['test_score'].std()

print(f"Train F1-macro: {train_mean:.3f} ± {train_std:.3f}")
print(f"Val   F1-macro: {val_mean:.3f} ± {val_std:.3f}")
print(f"Gap:            {train_mean - val_mean:.3f}")

# Fit on all training data
final_pipeline.fit(X_train, y_train)


# Save final hyperparameters — store rf__ prefixed params plus fs percentile
# so the format matches what was saved from the search step
final_params = {f'rf__{k}': v for k, v in rf_params.items()}
final_params['feature_selection__percentile'] = fs_percentile
joblib.dump(final_params, hyperparams_path)

# Save selected features
selected_mask = final_pipeline.named_steps['feature_selection'].get_support()
selected_features = X_train.columns[selected_mask].tolist()
with open(selected_features_path, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

# Save the whole pipeline
joblib.dump(final_pipeline, pipeline_path)

# Save model separately as well for easy loading without the feature selection step if needed
joblib.dump(final_pipeline.named_steps['rf'], model_path)

print(f"Saved hyperparameters to {hyperparams_path}")
print(f"Saved {len(selected_features)} selected features to {selected_features_path}")
print(f"Saved final pipeline to {pipeline_path}")
print(f"Saved final RF model to {model_path}")
