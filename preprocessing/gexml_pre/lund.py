import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR

dataset_name = 'EGAS00001001795'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/EGAS00001001795.featurecount.tsv'
pheno_input_file = f'{data_path}/EGAS00001001795.pheno.csv'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# PROCESS PHENO ########
print("Processing phenotype data...")
# Load the data
samplesheet_df = pd.read_csv(pheno_input_file, index_col=0, sep=';')

# Rename subtype column
samplesheet_df = samplesheet_df.rename(columns={'Subtype': 'subtype'})

# Rename index
samplesheet_df.index.name = 'sample'

# Get subtype translation
tax = Taxonomy()
samplesheet_df['subtype'] = tax.translate_subtype_column(samplesheet_df['subtype'])

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set data silo to empty string
samplesheet_df['data_silo'] = ''

# Convert index to string
samplesheet_df.index = samplesheet_df.index.astype(str)

# Prepend each sample id with "case_"
samplesheet_df.index = "case_" + samplesheet_df.index

# PROCESS GEX ########
# Break up the gene expression file
gex_df = pd.read_csv(raw_data_path, index_col=0, delimiter='\t')

# For each column:
# Create a new file with the column name as the file name
# The contents should contain the Gene and the counts for that column
# delimited by a tab. No header, no index.
for col in gex_df.columns:
    output_file = f'{processed_dir}/{col}.txt'
    # Set feature_counts column for this sample in samplesheet_df
    samplesheet_df.loc[col, 'feature_counts'] = output_file
    gex_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)

# Remove all rows from spreadsheet without feature_counts
samplesheet_df = samplesheet_df[samplesheet_df['feature_counts'].notna()]

# Set batch column to blank string
samplesheet_df['batch'] = ''

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
