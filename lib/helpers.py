import os
import pandas as pd
import numpy as np
import joblib
import dotenv
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    log_loss, brier_score_loss, matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize, LabelEncoder

dotenv.load_dotenv()
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

def classifier_level_metrics(y_true, y_pred, y_proba, class_names):
    present_classes = np.unique(y_true)

    metrics = {
        # threshold-based (need hard labels)
        "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc":          matthews_corrcoef(y_true, y_pred),

        # For multi-class ROC AUC, we need to account for classes not in y_true
        # This is a common approach for calculating macro AUC in imbalanced datasets
        "auroc_macro": np.mean([
            roc_auc_score((y_true == cls).astype(int), y_proba[:, i])
            for i, cls in enumerate(class_names) if cls in present_classes
        ]),

        # For AUPRC, only calculate for classes present in the data
        "auprc_macro": np.mean([
            average_precision_score((y_true == cls).astype(int), y_proba[:, i])
            for i, cls in enumerate(class_names) if cls in present_classes
        ])
    }

    return metrics


def class_level_metrics(y_true, y_pred, y_proba, class_names):
    # Encode y_true/y_pred into 0..n-1 integers
    le = LabelEncoder().fit(class_names)
    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred)

    # Get the unique classes actually present in the data
    present_class_indices = np.unique(np.concatenate([y_true_enc, y_pred_enc]))
    present_classes = [class_names[i] for i in present_class_indices]

    # Use only classes present in data for the classification report
    report = classification_report(y_true_enc, y_pred_enc,
                                  labels=present_class_indices,
                                  target_names=present_classes,
                                  zero_division=0,  # Handle division by zero gracefully
                                  output_dict=True)
    df = pd.DataFrame(report).T

    # Calculate metrics for ALL classes in class_names
    auroc_list = []
    auprc_list = []

    for i, cls in enumerate(class_names):
        y_bin = (y_true_enc == i).astype(int)

        # Skip metrics for classes with no positive examples
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auroc_list.append(np.nan)
            auprc_list.append(np.nan)

        else:
            auroc_list.append(roc_auc_score(y_bin, y_proba[:, i]))
            auprc_list.append(average_precision_score(y_bin, y_proba[:, i]))

    # Create a complete DataFrame with all classes and initialize with NaN values
    full_df = pd.DataFrame(index=class_names, columns=df.columns.tolist() + ["auroc", "auprc"])

    # Copy over values from the classification report for classes that are present
    for cls in present_classes:
        if cls in df.index:
            # Copy columns one by one to avoid issues with column mismatch
            for col in df.columns:
                if col in full_df.columns:
                    full_df.loc[cls, col] = df.loc[cls, col]

    # Add the AUC metrics for all classes
    full_df["auroc"] = auroc_list
    full_df["auprc"] = auprc_list

    # coerce all to float
    full_df = full_df.apply(pd.to_numeric, errors='coerce')

    class_metrics_order = [
        'support',
        'auroc',
        'auprc',
        'recall',
        'precision',
        'f1-score'
    ]
    full_df = full_df.reindex(columns=class_metrics_order)

    return full_df


def cross_val_metrics(clf, X, y, class_names, random_state=RANDOM_SEED):
    # How many splits?
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y)
    n_splits = np.bincount(y_train_encoded).min()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = defaultdict(list)
    per_class_dfs = []  # collect per-class results across folds

    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Processing fold {fold}/{n_splits}...")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

        # y is an array, so we can index it directly without iloc
        y_tr, y_val = y[train_idx], y[val_idx]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)

        # Collect for overall cross-val metrics
        y_true_all.append(y_val)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)

        # --- Classifier-level metrics ---
        metrics = classifier_level_metrics(y_val, y_pred, y_proba, class_names)
        for k, v in metrics.items():
            fold_metrics[k].append(v)

        # --- Class-level metrics ---
        df = class_level_metrics(y_val, y_pred, y_proba, class_names)
        df["fold"] = fold
        per_class_dfs.append(df)

    # Aggregate metrics across folds
    fold_metrics_mean_std = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}

    # Concatenate all validation predictions
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_proba_all = np.vstack(y_proba_all)

    # Overall classifier-level metrics
    overall_metrics = classifier_level_metrics(y_true_all, y_pred_all, y_proba_all, class_names)

    # Overall class-level metrics
    per_class_all = pd.concat(per_class_dfs)

    # Convert all to numeric, errors='coerce' to handle any non-numeric entries
    per_class_all = per_class_all.apply(pd.to_numeric, errors='coerce')
    per_class_summary = (
        per_class_all.groupby(per_class_all.index)
        .mean(numeric_only=True)  # average metrics per class
        .drop(columns=["fold"], errors="ignore")  # fold isn’t meaningful averaged.
        # TODO: maybe drop support as well?
    )

    return (
        overall_metrics,           # dict of overall classifier-level metrics
        per_class_summary          # averaged per-class DataFrame
    )
