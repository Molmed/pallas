import os
import pandas as pd
import argparse


description = """
Annotates cpg sites to the nearest gene name.
"""

cpg_sites_path_help = """
Path to txt file containing one cpg site per line."""

annotations_path_help = """
Path to Julia Sulyaeva's small DNAm annotation file
based on Illumina's EPIC annotation file EPIC-8v2-0_A1.csv."""

output_dir_help = """
Directory to save the output."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cpg_sites_path",
        type=str,
        required=True,
        help=cpg_sites_path_help,
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        required=True,
        help=annotations_path_help,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    return parser.parse_args()


def annotate(cpg_sites_path,
             annotations_path,
             output_dir):

    # Read in cpg sites
    with open(cpg_sites_path, "r") as f:
        cpg_sites = [line.strip() for line in f.readlines()]

    # Read in annotations path
    annot = pd.read_csv(annotations_path, index_col=0)

    # Get all the annot records that match the cpg sites
    selected = annot.loc[cpg_sites]

    # Save to csv
    filename = os.path.basename(cpg_sites_path)
    name = os.path.splitext(filename)[0]
    annotated_path = f"{output_dir}/{name}_annotated.csv"
    selected.to_csv(annotated_path)


def main():
    args = parse_args()
    annotate(
        cpg_sites_path=args.cpg_sites_path,
        annotations_path=args.annotations_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
