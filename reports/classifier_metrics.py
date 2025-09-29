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


def classifier_level_metrics(y_true, y_pred, y_proba, class_names):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro")
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["log_loss"] = log_loss(y_true, y_proba)
    metrics["brier"] = np.mean([brier_score_loss((y_true==i).astype(int), y_proba[:, i])
                                for i in range(len(class_names))])
    metrics["macro_auroc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    metrics["macro_auprc"] = compute_macro_auprc(y_true, y_proba, classes=np.arange(len(class_names)))
    return metrics

# ==========================================================
# 2. CROSS-VALIDATION METRICS
# ==========================================================

def cross_val_metrics(clf, X, y, class_names, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = defaultdict(list)

    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)

        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        y_true_idx = np.array([class_to_idx.get(label, -1) for label in y_val])
        y_pred_idx = np.array([class_to_idx.get(label, -1) for label in y_pred])

        # Collect for overall cross-val metrics
        y_true_all.append(y_true_idx)
        y_pred_all.append(y_pred_idx)
        y_proba_all.append(y_proba)

        # Per-fold metrics
        metrics = classifier_level_metrics(y_val, y_pred, y_proba, class_names)
        for k, v in metrics.items():
            fold_metrics[k].append(v)

    # Aggregate metrics across folds
    fold_metrics_mean_std = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}

    # Overall metrics using all concatenated predictions (optional)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_proba_all = np.vstack(y_proba_all)
    overall_metrics = classifier_level_metrics(y_true_all, y_pred_all, y_proba_all, class_names)

    return fold_metrics_mean_std, overall_metrics, y_true_all, y_pred_all, y_proba_all

# ==========================================================
# 3. CLASS-LEVEL METRICS
# ==========================================================

def class_level_metrics(y_true, y_pred, y_proba, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).T
    # add per-class AUROC and AUPRC
    auroc_list, auprc_list = [], []
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        auroc_list.append(roc_auc_score(y_bin, y_proba[:, i]))
        auprc_list.append(average_precision_score(y_bin, y_proba[:, i]))
    df.loc[class_names, "auroc"] = auroc_list
    df.loc[class_names, "auprc"] = auprc_list
    return df

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

def plot_confusion(y_true, y_pred, class_names, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".1f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True label"); plt.xlabel("Predicted label");
    plt.title("Confusion Matrix")
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
