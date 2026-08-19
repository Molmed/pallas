import os
import pandas as pd
import numpy as np
import argparse
import dotenv
from gene_thesaurus import GeneThesaurus

description = """
Translates a list of gene identifiers from one type to another using the GeneThesaurus class.
"""

ensembl_genes_list_path_help = """
Path to txt file containing a list of Ensembl gene identifiers to be translated.
The txt file should have a single column with no header, where each row contains an Ensembl gene identifier.
"""

output_path_help = """
Path to output CSV file where the translated gene identifiers will be saved.
The output CSV file will have two columns: "Ensembl ID" and "Gene Symbol".
"""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--ensembl_genes_list_path",
        type=str,
        required=True,
        help=ensembl_genes_list_path_help
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=output_path_help
    )
    return parser.parse_args()


def translate_genes(ensembl_genes_list_path, output_path):
    # Load the list of Ensembl gene identifiers
    with open(ensembl_genes_list_path, 'r') as f:
        ensembl_genes = [line.strip() for line in f.readlines()]

    # Initialize the GeneThesaurus
    gene_thesaurus = GeneThesaurus()

    # Translate Ensembl IDs to Gene Symbols
    translated_genes = gene_thesaurus.translate_genes(
        ensembl_genes,
        source="ensembl_id",
        target="symbol")

    # What genes are in ensembl_genes but not in translated_genes?
    # Add empty string for missing genes
    for gene in ensembl_genes:
        if gene not in translated_genes:
            translated_genes[gene] = ""

    # Order the translated_genes dictionary to match the order of ensembl_genes
    translated_genes = {gene: translated_genes[gene] for gene in ensembl_genes}

    # Save the translated genes to a CSV file
    translated_df = pd.DataFrame(list(translated_genes.items()), columns=['Ensembl ID', 'Gene Symbol'])
    translated_df.to_csv(output_path, index=False)


def main():
    args = parse_args()
    translate_genes(args.ensembl_genes_list_path, args.output_path)


if __name__ == "__main__":
    main()
