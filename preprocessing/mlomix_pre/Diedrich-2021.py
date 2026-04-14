import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from preprocessing.helpers import format_mlomix_columns

dataset_name = 'Diedrich-2021'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/GSE161501_ALL_cell_RNAseq_read_counts.txt'
pheno_input_file = f'{data_path}/SupplementaryInfo.xlsx'

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'
details_file = \
    f'{preprocessed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(preprocessed_dir, exist_ok=True)

# PROCESS PHENO ########
print("Processing phenotype data...")
# Load the data
samplesheet_df = pd.read_excel(pheno_input_file,
                     sheet_name='Supplemental Table 1',
                     header=2,
                     index_col='Sample SJ ID')

details_df = samplesheet_df.copy()

# Keep only index and subtype cols
samplesheet_df = samplesheet_df.iloc[:, :1]

# Rename columns
samplesheet_df = samplesheet_df.rename(columns={'Subtype_name': 'class'})
samplesheet_df.index.name = 'sample'

# In subtype column, strip out everything after the underscore
samplesheet_df['class'] = samplesheet_df['class'].str.split('_').str[0]

# Get subtype translation
tax = Taxonomy()
samplesheet_df['class'] = tax.translate_subtype_column(samplesheet_df['class'])

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set batch column to empty string
samplesheet_df['batch'] = ''

# Set data silo to labeled_diagnostic
samplesheet_df['data_silo'] = 'labeled_diagnostic'

# PROCESS GEX ########
# Break up the gene expression file
gex_df = pd.read_csv(raw_data_path, index_col=0, delimiter='\t')

# For each column:
# Create a new file with the column name as the file name
# The contents should contain the Gene and the counts for that column
# delimited by a tab. No header, no index.
for col in gex_df.columns:
    output_file = f'{preprocessed_dir}/{col}.txt'
    # Set feature_counts column for this sample in samplesheet_df
    samplesheet_df.loc[col, 'gex_feature_counts_file'] = output_file
    gex_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)

# Remove all rows from spreadsheet without feature_counts
samplesheet_df = samplesheet_df[samplesheet_df['gex_feature_counts_file'].notna()]

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)

# Rename index to sample
details_df.index.name = 'sample'
# Only keep the ones that are also in samplesheet_df
details_df = details_df.loc[samplesheet_df.index]
# Rename columns
details_df = details_df.rename(columns={'Sex': 'sex', 'Age': 'age'})
details_df['dataset'] = dataset_name
# Keep just those columns
details_df = details_df[
    ['dataset', 'sex', 'age']]
# Dump to CSV
details_df.to_csv(details_file)
