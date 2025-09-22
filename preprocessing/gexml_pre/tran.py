import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocessing.helpers import add_to_silo, silo_unlabeled, \
    silo_remaining

dataset_name = 'GSE181157'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/GSE181157_RAW'
pheno_input_file = f'{data_path}/GSE181157.pheno.csv'
details_input_file = f'{data_path}/GSE181157_SampleMetadata.xlsx'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'
details_file = \
    f'{processed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# Read in details file
details_df = pd.read_excel(details_input_file, index_col=1, header=1)
details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df = details_df.rename(columns={'Age at Dx (years)': 'age', 'Sex': 'sex'})

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
samplesheet_df = samplesheet_df.rename(columns={'Final subtype': 'subtype'})

# Rename index
samplesheet_df.index.name = 'sample'

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''

# Put multiclass items in multiclass silo before translation
multi_subtype_rows = samplesheet_df['subtype'].str.contains(',', na=False)
if multi_subtype_rows.any():
    samplesheet_df = add_to_silo(
        samplesheet_df,
        samplesheet_df[multi_subtype_rows].index, 'multiclass')

# Get subtype translation
tax = Taxonomy()
samplesheet_df['subtype'] = tax.translate_subtype_column(
    samplesheet_df['subtype'], split_delimiter=',')

# UNLABELED SILO
samplesheet_df = silo_unlabeled(samplesheet_df)

# Add remaining to labeled_diagnostic silo
samplesheet_df = silo_remaining(samplesheet_df)

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Create a map of sample names to paths.
sample_to_path_map = {}
for root, dirs, files in os.walk(raw_data_path):
    for filename in files:
        sample_name = filename.split(
            '_', 1)[1].split('.counts.txt.gz')[0].replace('_', '-')
        if sample_name in samplesheet_df.index:
            sample_to_path_map[sample_name] = os.path.join(root, filename)

# Turn the dict into a new column feature_counts
samplesheet_df['feature_counts'] = samplesheet_df.index.map(sample_to_path_map)

# Set batch column to blank string
samplesheet_df['batch'] = ''

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Remove all rows from spreadsheet without feature_counts
samplesheet_df = samplesheet_df[samplesheet_df['feature_counts'].notna()]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
