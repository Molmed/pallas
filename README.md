# PALLAS v0.0.1 &#127963;

## About

PALLAS is a multimodal classifier of molecular subtypes in pediatric acute leukemias.

## Conda environment
[Conda](https://docs.conda.io) must be installed on your system.

You will need to activate the `pallas` conda environment before running any subsequent commands.

Install: `conda env create -f environment.yml`

Activate: `conda activate pallas`

Update (after changes to environment.yml): `conda env update --file environment.yml --prune`

## Data pre-processing

Preprocessing tools are available in the [PALLAS PrePro](https://github.com/Molmed/pallas_prepro) repository. The output files from this pre-processing pipeline are described below.

### Gene expression data

Pre-processed gene expression data will be (optionally) batch corrected, normalized and formatted as in the following example:

| id      | ENSG00000000003 | ENSG00000000005 | ... |
| --------| --------------- | --------------- | --- |
| SAMPLE1 | 3.312           | -1.901          | ... |
| SAMPLE2 | 2.8123          |  0.1201         | ... |
| SAMPLE3 | 3.61            |  1.3749         | ... |

### DNAm data

Work in progress.

### Phenotype files

If you are re-training or validating PALLAS, you can optionally provide a tab-separated phenotype file (TSV) with identifiers matching your GEX/DNAm data, and with the `y` column containing hierarchical disease and subtype information compatible with [hiclass](https://hiclass.readthedocs.io). An example:

| id      | y |
| --------| -------------------------- |
| SAMPLE1 | Disease,AML,KMT2A-r        |
| SAMPLE2 | Disease,AML,RUNX1::RUNX1T1 |
| SAMPLE3 | Disease,B-ALL,PAX5alt      |


## Training

1. Split your dataset by running: `python training/split.py --features_path=/path/to/preprocessed/counts.csv --target_path=/path/to/pheno.tsv --output_dir=/path/to/output_dir`
