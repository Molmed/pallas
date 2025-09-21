#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import joblib


# In[ ]:

X_train_path = os.getenv("X_TRAIN_CSV_PATH")
y_train_path = os.getenv("Y_TRAIN_TSV_PATH")
output_dir = os.getenv("OUTPUT_DIR")
random_seed = int(os.getenv("RANDOM_SEED"))
n_jobs = int(os.getenv("N_JOBS", -1))
rf_model_path = os.getenv("RF_MODEL_PATH")
selected_features_path = os.getenv("SELECTED_FEATURES_PATH")
pipeline_path = os.getenv("PIPELINE_PATH")
hyperparams_path = os.getenv("HYPERPARAMS_PATH")

# Print configuration
print(f"X_train_path: {X_train_path}")
print(f"y_train_path: {y_train_path}")
print(f"output_dir: {output_dir}")
print(f"random_seed: {random_seed}")
print(f"n_jobs: {n_jobs}")
print(f"rf_model_path: {rf_model_path}")
print(f"selected_features_path: {selected_features_path}")
print(f"pipeline_path: {pipeline_path}")
print(f"hyperparams_path: {hyperparams_path}")

# Hardcoded variables
FEATURE_SELECTION_PERCENTILE=20  # KEEP TOP n% OF CPG SITES
FINAL_N_ESTIMATORS=500


# In[3]:


# Read in X_train and y_train
X_train = pd.read_csv(X_train_path, index_col=0)
X_train


# In[4]:


y_train = pd.read_csv(y_train_path, sep="\t", index_col=0)
y_train


# In[5]:


# align the indices
X_train = X_train.loc[y_train.index]
X_train


# In[8]:


# Get least frequently ocurring class count
min_class_count = y_train.value_counts().min()
min_class_count


# In[9]:


y_train = y_train.iloc[:, 0].values  # Convert to 1D numpy array

# In[ ]:


N_ITER=30
N_REPEATS=1 # WE WILL VALIDATE FINAL PARAMS USING MORE REPEATS

# ----------------------
# Pipeline: RF feature selection -> RF classifier
# ----------------------
pipeline = Pipeline([
    ("feature_selection", SelectPercentile(percentile=FEATURE_SELECTION_PERCENTILE)),  # keep only top 10% of features
    ("rf", RandomForestClassifier(random_seed, n_jobs=-n_jobs))
])

# ----------------------
# Hyperparameter search space
# ----------------------
param_grid = {
    'rf__n_estimators': [200],  # We will increase this in the final model
    'rf__max_depth': [None, 20, 50],
    'rf__max_features': ['sqrt', 'log2', 0.25],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 5],
    'rf__class_weight': ['balanced']
}

# ----------------------
# Cross-validation + randomized search
# ----------------------
cv = RepeatedStratifiedKFold(n_splits=min_class_count,
                             n_repeats=N_REPEATS,
                             random_state=random_seed)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=N_ITER,
    cv=cv,
    scoring="f1_weighted",   # good for multiclass imbalance
    n_jobs=-n_jobs,
    random_state=random_seed,
    verbose=2
)

search.fit(X_train, y_train)

print("Best hyperparameters:", search.best_params_)
print("Best CV (F1-weighted):", search.best_score_)
best_pipeline = search.best_estimator_

## SAVE EVERYTHING
# Save best hyperparameters
joblib.dump(search.best_params_, hyperparams_path)

# Save features
selected_mask = best_pipeline.named_steps['feature_selection'].get_support()
selected_features = X_train.columns[selected_mask]

joblib.dump(selected_features, selected_features_path)

# Save the whole pipeline
joblib.dump(best_pipeline, pipeline_path)


# In[ ]:


# Cross-validate again with more repeats and larger estimators
N_REPEATS=10

# Use just the selected features
X_train_selected = X_train[selected_features]

# Read in "best_hyperparameters.pkl"
with open(hyperparams_path, "rb") as f:
    best_params = joblib.load(f)

best_params = {k.replace('rf__', ''): v for k, v in best_params.items()}
best_params['n_estimators'] = FINAL_N_ESTIMATORS
print("Final model parameters:", best_params)
final_rf = RandomForestClassifier(**best_params,
                            n_jobs=n_jobs,
                            random_state=random_seed)

# Cross-validate again with more repeats
cv = RepeatedStratifiedKFold(n_splits=min_class_count,
                             n_repeats=N_REPEATS,
                             random_state=random_seed)

cv_scores = cross_val_score(final_rf,
                           X_train_selected,
                           y_train,
                           cv=cv,
                           scoring="f1_weighted",
                           n_jobs=-n_jobs,
                           verbose=1)

print(f"Cross-validation F1-weighted: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train the final model on all training data
final_rf.fit(X_train_selected, y_train)

# Save the final model
joblib.dump(final_rf, rf_model_path)

