import argparse
import pandas as pd
from gene_thesaurus import GeneThesaurus
import matplotlib.pyplot as plt

description = """
Run reports on the hyperparameter tuning."""

reports_dir_help = """
Path to the dir containing hypertuning reports."""

prefix_help = """
Prefix for report files (e.g. "dt" or "rf")."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--reports_dir",
        type=str,
        required=True,
        help=reports_dir_help,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help=prefix_help,
    )
    return parser.parse_args()


def plot_cv_results(cv_results_df,
                    param_name,
                    output_dir,
                    prefix,
                    score_col="mean_test_score"):
    """
    Plot cross-validation mean score vs. a single hyperparameter.

    Args:
        cv_results_df (pd.DataFrame): Expanded CV results (with params expanded into columns).
        param_name (str): Hyperparameter column to plot (e.g. 'rf__max_depth').
        score_col (str): Score column to plot (default = 'mean_test_score').
    """
    if param_name not in cv_results_df.columns:
        raise ValueError(f"{param_name} not found in cv_results_df columns")

    # Group by parameter value, take mean and std across runs
    grouped = cv_results_df.groupby(param_name)[score_col].agg(["mean", "std"]).reset_index()

    # Sort for plotting
    grouped = grouped.sort_values(param_name)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        grouped[param_name].astype(str),  # x-axis values as strings for categorical params
        grouped["mean"],
        yerr=grouped["std"],
        fmt="o-",
        capsize=5
    )
    plt.xlabel(param_name)
    plt.ylabel("CV AUC")
    plt.title(f"CV AUC vs {param_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}.{param_name}_cv_results.svg")


def report(reports_dir, prefix):
    print("Reporting on hypertuning with the following parameters:")
    print(f"Reports directory: {reports_dir}")
    print(f"Prefix: {prefix}")

    # Read in feature_importances.tsv
    feature_importances = pd.read_csv(f"{reports_dir}/{prefix}.feature_importances.tsv", sep="\t")

    # Translate the Gene column
    gt = GeneThesaurus()
    translated = gt.translate_genes(feature_importances["Gene"].values,
                                    source='ensembl_id',
                                    target='symbol')
    feature_importances["Symbol"] = feature_importances["Gene"].map(translated)
    # Save
    feature_importances.to_csv(f"{reports_dir}/{prefix}.feature_importances_translated.tsv", sep="\t", index=False)


    # Read in cv_results.tsv
    cv_results = pd.read_csv(f"{reports_dir}/{prefix}.cv_results.tsv", sep="\t")

    # Plot AUC vs. max_depth
    plot_cv_results(cv_results, f"{prefix}__max_depth", reports_dir, prefix)

    # Plot AUC vs. min_samples_leaf
    plot_cv_results(cv_results, f"{prefix}__min_samples_leaf", reports_dir, prefix)


def main():
    args = parse_args()
    report(
        reports_dir=args.reports_dir,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
