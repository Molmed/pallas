import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    log_loss, brier_score_loss, matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from collections import defaultdict

# ==========================================================
# 1. CLASSIFIER-LEVEL METRICS (including safe macro AUPRC)
# ==========================================================

def compute_macro_auprc(y_true, y_proba, classes):
    Y_bin = label_binarize(y_true, classes=classes)  # shape (n_samples, n_classes)
    ap_list = []
    for i in range(len(classes)):
        if Y_bin[:, i].sum() == 0:
            continue  # skip classes with no positives
        ap = average_precision_score(Y_bin[:, i], y_proba[:, i])
        ap_list.append(ap)
    return np.mean(ap_list) if ap_list else 0.0


def classifier_level_metrics(y_true, y_pred, y_proba, class_names, labels=None):
    """
    Compute classifier-level metrics.
    """
    metrics = {}
    # Use labels parameter for log_loss
    metrics["log_loss"] = log_loss(y_true, y_proba, labels=labels if labels else class_names)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # For multi-class ROC AUC, we need to account for classes not in y_true
    # This is a common approach for calculating macro AUC in imbalanced datasets
    present_classes = np.unique(y_true)
    metrics["macro_auroc"] = np.mean([
        roc_auc_score((y_true == cls).astype(int), y_proba[:, i])
        for i, cls in enumerate(class_names) if cls in present_classes
    ])

    # For AUPRC, only calculate for classes present in the data
    metrics["macro_auprc"] = np.mean([
        average_precision_score((y_true == cls).astype(int), y_proba[:, i])
        for i, cls in enumerate(class_names) if cls in present_classes
    ])

    # Add any other metrics that were missing
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")

    return metrics

# ==========================================================
# 2. CROSS-VALIDATION METRICS
# ==========================================================

def cross_val_metrics(clf, X, y, class_names, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = defaultdict(list)
    per_class_dfs = []  # collect per-class results across folds

    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

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


# ==========================================================
# 3. CLASS-LEVEL METRICS
# ==========================================================

from sklearn.preprocessing import LabelEncoder

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
    brier_list = []
    for i, cls in enumerate(class_names):
        y_bin = (y_true_enc == i).astype(int)

        # Skip metrics for classes with no positive examples
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auroc_list.append(np.nan)
            auprc_list.append(np.nan)
            brier_list.append(np.nan)
        else:
            auroc_list.append(roc_auc_score(y_bin, y_proba[:, i]))
            auprc_list.append(average_precision_score(y_bin, y_proba[:, i]))
            brier_list.append(brier_score_loss(y_bin, y_proba[:, i]))

    # Create a complete DataFrame with all classes and initialize with NaN values
    full_df = pd.DataFrame(index=class_names, columns=df.columns.tolist() + ["auroc", "auprc", "brier"])

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
    full_df["brier"] = brier_list

    # coerce all to float
    full_df = full_df.apply(pd.to_numeric, errors='coerce')

    class_metrics_order = [
        'support',
        'auroc',
        'auprc',
        'recall',
        'precision',
        'f1-score',
        'brier'
    ]
    full_df = full_df.reindex(columns=class_metrics_order)

    return full_df


# ==========================================================
# 4. BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================================

def bootstrap_metric(y_true, y_pred, y_proba, metric_fn, n_boot=1000, alpha=0.95):
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = resample(np.arange(n), replace=True, n_samples=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx], y_proba[idx]))
    lower = np.percentile(stats, (1-alpha)/2*100)
    upper = np.percentile(stats, (1+alpha)/2*100)
    return np.mean(stats), (lower, upper)

# ==========================================================
# 5. VISUALIZATIONS
# ==========================================================

def plot_confusion(y_true,
                   y_pred,
                   class_names,
                   normalize=True,
                   save_to_svg=None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt=".1f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True label"); plt.xlabel("Predicted label");
    plt.title("Confusion Matrix")
    # Add tight layout to prevent clipping of tick-labels
    plt.tight_layout()
    if save_to_svg:
        plt.savefig(save_to_svg, format='svg')
    else:
        plt.show()


def plot_calibration(y_true, y_proba, class_idx, class_names, n_bins=10):
    prob_true, prob_pred = calibration_curve((y_true==class_idx).astype(int),
                                             y_proba[:, class_idx], n_bins=n_bins)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--k')
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration: {class_names[class_idx]}")
    plt.show()


def plot_roc_curves(y_true, y_proba, class_names):
    plt.figure(figsize=(7,6))
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        auc_val = roc_auc_score(y_bin, y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.2f})")
    plt.plot([0,1],[0,1],'--k')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.show()


def plot_pr_curves(y_true, y_proba, class_names):
    plt.figure(figsize=(7,6))
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        ap_val = average_precision_score(y_bin, y_proba[:, i])
        plt.plot(rec, prec, label=f"{cls} (AP={ap_val:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend()
    plt.show()

# ==========================================================
# 6. ALL-IN-ONE REPORT
# ==========================================================

def full_evaluation_report(y_true, y_pred, y_proba, class_names):
    print("=== Classifier-level metrics ===")
    clf_metrics = classifier_level_metrics(y_true, y_pred, y_proba, class_names)
    print(pd.Series(clf_metrics))
    print("\n=== Per-class metrics ===")
    per_class = class_level_metrics(y_true, y_pred, y_proba, class_names)
    print(per_class)

    print("\n=== Plots ===")
    plot_confusion(y_true, y_pred, class_names, normalize=False)
    plot_roc_curves(y_true, y_proba, class_names)
    plot_pr_curves(y_true, y_proba, class_names)
    for i, cls in enumerate(class_names):
        plot_calibration(y_true, y_proba, i, class_names)

    return clf_metrics, per_class

# ==========================================================
# 7. USAGE EXAMPLE (pseudo-code)
# ==========================================================
# y_pred = np.argmax(y_proba, axis=1)
# clf_metrics, per_class = full_evaluation_report(y_true, y_pred, y_proba, class_names)
