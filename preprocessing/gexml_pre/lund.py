import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocessing.helpers import silo_unlabeled, silo_remaining

dataset_name = 'EGAS00001001795'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/EGAS00001001795.salmon.merged.gene_counts.tsv'
pheno_input_file = f'{data_path}/EGAS00001001795.pheno.csv'
details_input_file = f'{data_path}/41467_2016_BFncomms11790_MOESM1480_ESM.xlsx'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'
details_file = \
    f'{processed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# Read in details file
# Headers are in the second row, discard first row
details_df = pd.read_excel(details_input_file, index_col=0, header=1)

details_df.index.name = 'sample'
# Preface every sample id with "Case_"
details_df.index = 'Case_' + details_df.index.astype(str)

details_df['dataset'] = dataset_name
details_df = details_df.rename(columns={'Sex': 'sex', 'Age': 'age'})

# Drop all rows where both sex and age are NaN
details_df = details_df.dropna(subset=['sex', 'age'], how='all')

# For sex, replace M with Male and F with Female
details_df['sex'] = details_df['sex'].replace({'M': 'Male', 'F': 'Female'})

details_df = details_df[
    ['dataset', 'sex', 'age']]

details_df.to_csv(details_file)

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

# Set all items whose subtype ends in "B-other" to data_silo "unlabeled"
samplesheet_df = silo_unlabeled(samplesheet_df)

# Silo remaining as labeled_diagnostic
samplesheet_df = silo_remaining(samplesheet_df)

# Convert index to string
samplesheet_df.index = samplesheet_df.index.astype(str)

# Prepend each sample id with "case_"
samplesheet_df.index = "Case_" + samplesheet_df.index

# PROCESS GEX ########
# Break up the gene expression file
gex_df = pd.read_csv(raw_data_path, index_col=0, delimiter='\t')

# Drop second column, it contains gene name in addition to gene id
gex_df = gex_df.drop(columns=[gex_df.columns[0]])

# For each remaining column:
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
