import os
import pandas as pd
import numpy as np
import dotenv
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score, average_precision_score,
    cohen_kappa_score, ConfusionMatrixDisplay
)
from matplotlib.transforms import ScaledTranslation
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.calibration import calibration_curve

dotenv.load_dotenv()
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
ROUND_PLACES = int(os.getenv("ROUND_PLACES", 3))


def classifier_level_metrics(y_true, y_pred, y_proba, class_names):
    present_classes = np.unique(y_true)

    metrics = {
        "f1_macro":     f1_score(y_true,
                                 y_pred,
                                 average="macro",
                                 zero_division=0),
        "f1_weighted":  f1_score(y_true,
                                 y_pred,
                                 average="weighted",
                                 zero_division=0),
        "cohen_kappa":   cohen_kappa_score(y_true, y_pred),

        # For multi-class ROC AUC, we need to account for classes not in y_true
        # This is a common approach for calculating macro AUC in imbalanced datasets
        "auroc_macro": float(np.mean([
            roc_auc_score((y_true == cls).astype(int), y_proba[:, i])
            for i, cls in enumerate(class_names) if cls in present_classes
        ])),

        # For AUPRC, only calculate for classes present in the data
        "auprc_macro": float(np.mean([
            average_precision_score((y_true == cls).astype(int), y_proba[:, i])
            for i, cls in enumerate(class_names) if cls in present_classes
        ]))
    }

    # Return as df, index name should be "metric"
    df = pd.DataFrame([metrics]).round(ROUND_PLACES).T
    df.index.name = "metric"
    # Value should be in a column named "value"
    df.columns = ["value"]
    return df


def class_level_metrics(y_true, y_pred, y_prob, class_names):
    # Encode y_true/y_pred into 0..n-1 integers
    le = LabelEncoder().fit(class_names)
    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred)

    # Get the unique classes actually present in the data
    present_class_indices = np.unique(np.concatenate([y_true_enc, y_pred_enc]))
    present_classes = [class_names[i] for i in present_class_indices]

    # Use only classes present in data for the classification report
    report = classification_report(y_true_enc,
                                   y_pred_enc,
                                   labels=present_class_indices,
                                   target_names=present_classes,
                                   zero_division=0,  # Handle division by zero gracefully
                                   output_dict=True)
    df = pd.DataFrame(report).T

    # Now calculate AUROC and AUPRC for each class present in the data
    # and append it to the classification report df
    for i, cls in enumerate(class_names):
        if cls in present_classes:
            cls_idx = present_classes.index(cls)
            y_bin_true = (y_true == cls).astype(int)
            y_bin_pred_proba = y_prob[:, i]

            # Calculate AUROC and AUPRC only if the class is present in y_true
            if np.any(y_bin_true):
                auroc = roc_auc_score(y_bin_true, y_bin_pred_proba)
                auprc = average_precision_score(y_bin_true, y_bin_pred_proba)
            else:
                auroc = np.nan
                auprc = np.nan

            df.loc[cls, 'auroc'] = auroc
            df.loc[cls, 'auprc'] = auprc
        else:
            # If the class is not present in the data, set metrics to NaN
            df.loc[cls] = [np.nan] * len(df.columns)

    # Order cols as follows: support, precision, recall, f1-score, auroc, auprc
    # And only include rows whose name matches the class names
    df = df.loc[class_names, ['support', 'precision', 'recall', 'f1-score', 'auroc', 'auprc']]

    # Put Non-leukemic class last
    if 'Non-leukemic' in df.index:
        non_leukemic_row = df.loc[['Non-leukemic']]
        df = df.drop(index='Non-leukemic')
        df = pd.concat([df, non_leukemic_row])

    # Name the index "subtype"
    df.index.name = "subtype"
    return df.round(ROUND_PLACES)


def plot_confusion_matrix(y_test, y_pred, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        cmap="YlGnBu",
        values_format="d",
    )

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.tick_params(axis="x", labelsize=10, pad=2)
    ax.tick_params(axis="y", labelsize=10, pad=2)
    ax.set_xlabel(ax.get_xlabel(), fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)

    texts = getattr(display, "text_", None)
    if texts is not None:
        for text in np.ravel(texts):
            text.set_fontsize(8)

    fig.tight_layout(pad=0.6)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.15,
        dpi=300,
    )
    print(f"Saved to {output_path}")


def plot_precision_recall(metrics_df,
                          title="Per-subtype precision and recall",
                          figsize=(8, 6),
                          save_path=None):
    df = metrics_df.copy().reset_index()

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]
    if 'subtype' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'subtype'})

    # Use sensitivity as recall if recall column absent
    if 'recall' not in df.columns and 'sensitivity' in df.columns:
        df['recall'] = df['sensitivity']

    # ----------------------
    # Disease group assignment and sort order
    # ----------------------
    def assign_group(name):
        if name.startswith('AML'):
            return '1_AML'
        elif name.startswith('B-ALL'):
            return '2_B-ALL'
        elif name.startswith('T-ALL'):
            return '3_T-ALL'
        else:
            return '4_Non-leukemic'

    df['group'] = df['subtype'].apply(assign_group)
    df = df.sort_values(['group', 'subtype']).reset_index(drop=True)

    # Short labels — strip disease prefix for readability
    def shorten(name):
        parts = name.split(',')
        if len(parts) > 1:
            return parts[-1].strip()
        return name

    df['label'] = df['subtype'].apply(shorten)

    # ----------------------
    # Color per disease group — matching the reference figure aesthetic
    # ----------------------
    group_colors = {
        '1_AML':   '#667ec8',
        '2_B-ALL':  '#d64a51',
        '3_T-ALL':  '#90c23a',
        '4_Non-leukemic': '#7a8288',
    }
    df['color'] = df['group'].map(group_colors)

    n = len(df)
    x = np.arange(n)

    # ----------------------
    # Figure — single panel
    # ----------------------
    fig, ax = plt.subplots(figsize=figsize)

    markersize = 9

    for i, row in df.iterrows():
        c = row['color']
        prec = row['precision']
        rec  = row['recall']

        # Connecting line between recall and precision
        ax.plot(
            [i, i], [rec, prec],
            color=c, linewidth=1.2, alpha=0.6,
            linestyle='--' if abs(prec - rec) > 0.15 else '-',
            zorder=1
        )

        # Recall — circle
        ax.plot(
            i, rec,
            marker='o', markersize=markersize,
            color=c, markeredgecolor='white',
            markeredgewidth=0.8, zorder=3,
            linestyle='none'
        )

        # Precision — square
        ax.plot(
            i, prec,
            marker='s', markersize=markersize,
            color=c, markeredgecolor='white',
            markeredgewidth=0.8, zorder=3,
            linestyle='none'
        )

    # ----------------------
    # Group boundary lines
    # ----------------------
    group_vals = df['group'].values
    for i in range(1, n):
        if group_vals[i] != group_vals[i - 1]:
            ax.axvline(i - 0.5, color='#d3d1c7', linewidth=0.8,
                      linestyle=':', zorder=0)

    # ----------------------
    # Axes
    # ----------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        df['label'],
        rotation=90, ha='center',
        fontsize=10, color='#444441'
    )

    for tick, color in zip(ax.get_xticklabels(), df['color']):
        tick.set_color(color)

    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='normal', pad=12)

    # Y axis — zoom to data range with small padding
    all_vals = pd.concat([df['precision'], df['recall']]).dropna()
    ymin = max(0.0, all_vals.min() - 0.05)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.2f}')
    )

    ax.set_xlim(-0.6, n - 0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d3d1c7')
    ax.spines['bottom'].set_color('#d3d1c7')
    ax.tick_params(axis='y', colors='#888780', labelsize=9)
    ax.grid(axis='y', linestyle=':', linewidth=0.5,
            color='#d3d1c7', zorder=0)

    # ----------------------
    # Support annotation below x-axis labels
    # ----------------------
    for i, row in df.iterrows():
        ax.annotate(
            f"n={int(row['support'])}",
            xy=(i, 0),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center', fontsize=8,
            color='#888780',
            rotation=90,
        )

    # ----------------------
    # Legend
    # ----------------------
    legend_handles = [
        mlines.Line2D([], [], marker='o', color='#888780',
                      markeredgecolor='white', markersize=9,
                      linestyle='none', label='Recall'),
        mlines.Line2D([], [], marker='s', color='#888780',
                      markeredgecolor='white', markersize=9,
                      linestyle='none', label='Precision'),
    ]
    for group_key, color in group_colors.items():
        group_label = group_key.split('_')[1]
        if (df['group'] == group_key).any():
            legend_handles.append(
                mlines.Line2D([], [], marker='o', color=color,
                              markersize=8, linestyle='none',
                              label=group_label)
            )

    offset = ScaledTranslation(0, 40 / 72, fig.dpi_scale_trans)

    ax.legend(
        handles=legend_handles,
        frameon=True,
        framealpha=0.9,
        edgecolor='#d3d1c7',
        fontsize=9,
        loc='lower left',
        bbox_to_anchor=(0, 0),
        bbox_transform=ax.transAxes + offset,
        ncol=2,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def cross_val_metrics(clf, X, y, class_names, random_state=RANDOM_SEED):
    # How many splits?
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y)
    n_splits = min(5, np.bincount(y_train_encoded).min())

    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=random_state)

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
    )

    return (
        overall_metrics,           # dict of overall classifier-level metrics
        per_class_summary          # averaged per-class DataFrame
    )


def plot_per_class_calibration_curves(
        y_test,
        y_pred_proba,
        classes,
        save_path=None,
        title=None,
        color_tsv_path=None,
        line_color='#667ec8',
        n_bins=10):

    class_colors = {}
    if color_tsv_path is not None:
        color_tsv_path = str(color_tsv_path)
        try:
            color_df = pd.read_csv(color_tsv_path, sep='\t', header=None,
                                  names=['class', 'color'])
            color_df['class'] = color_df['class'].astype(str).str.strip()
            color_df['color'] = color_df['color'].astype(str).str.strip()
            class_colors = dict(zip(color_df['class'], color_df['color']))
        except FileNotFoundError:
            print(f"Warning: color TSV not found at {color_tsv_path}. Falling back to line_color.")
        except Exception as exc:
            print(f"Warning: could not read color TSV at {color_tsv_path}: {exc}. Falling back to line_color.")

    figsize = (8, 10)
    n_classes = len(classes)
    ncols = 5
    nrows = int(np.ceil(n_classes / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, class_label in enumerate(classes):
        # Binarize: current class vs rest
        y_binary = (y_test == class_label).astype(int)
        prob_pos = y_pred_proba[:, i]

        fraction_of_positives, mean_predicted_prob = calibration_curve(
            y_binary, prob_pos, n_bins=n_bins, strategy="uniform"
        )

        class_color = class_colors.get(str(class_label).strip(), line_color)

        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(mean_predicted_prob,
                fraction_of_positives,
                "s-",
                label="Actual calibration for subtype",
                color=class_color)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title(class_label, fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower right", fontsize=10)
    fig.supylabel("Fraction of positives")
    fig.supxlabel("Mean predicted probability")

    fig.suptitle(title, fontsize=10)
    # Add a smaller subtitle if provided
    subtitle = "Per-class calibration plots (one-vs-rest)"
    fig.text(0.5, 0.985, subtitle, ha='center', fontsize=8, color='#444441')
    plt.tight_layout(h_pad=1)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved to {save_path}")
    else:
        plt.show()
