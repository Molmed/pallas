import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocessing.helpers import silo_unlabeled, silo_remaining

dataset_name = 'GDC_AML'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/feature_counts'
pheno_input_file = f'{data_path}/gdc_gex_aml.subtyped.tsv'
details_input_file = f'{data_path}/gdc_gex_aml.clinical.tsv'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'
details_file = \
    f'{processed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# We only want the subtyped data, use the pre-processed subtypes file as guide
pheno_df = pd.read_csv(pheno_input_file, sep='\t', index_col='case_id')

# Read in details file
# Index should be "cases.submitter_id"
details_df = pd.read_csv(details_input_file,
                         header=0,
                         sep='\t',
                         index_col='cases.submitter_id')

# Only keep cases that are in pheno_df
details_df = details_df[details_df.index.isin(pheno_df.index)]

# Keep the following cols
# diagnoses.age_at_diagnosis
# demographic.gender
details_df = details_df[['diagnoses.age_at_diagnosis', 'demographic.gender']]


details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df = details_df.rename(
    columns={'demographic.gender': 'sex',
             'diagnoses.age_at_diagnosis': 'age'})

# Convert age to numeric, coerce errors to NaN
details_df['age'] = pd.to_numeric(details_df['age'], errors='coerce')

# For age, convert days to years and round to 1 decimal place
details_df['age'] = (details_df['age'] / 365.25).round(1)

# For sex, replace M with Male and F with Female
details_df['sex'] = details_df['sex'].replace({'male': 'Male', 'female': 'Female'})

# Drop duplicate rows
details_df = details_df.drop_duplicates()

details_df = details_df[
    ['dataset', 'sex', 'age']]

details_df.to_csv(details_file)
exit()

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
