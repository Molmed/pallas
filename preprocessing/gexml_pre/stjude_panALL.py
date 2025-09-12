import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR

dataset_name = 'SJC-panALL'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/BALL-1988S-HTSeq'
pheno_input_file = f'{data_path}/B-ALL-subtyping.xlsx'
cloud_phenotype_file = f'{RAW_DATA_DIR}/SJC-DS-1009/SAMPLE_INFO.txt'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# PROCESS PHENO ########
print("Processing phenotype data...")
samplesheet_df = pd.read_excel(pheno_input_file,
                               sheet_name='ST1.Cohort',
                               header=0,
                               index_col='patient')
# Get index of first row where patient is NaN
nan_idx = samplesheet_df.index.get_indexer_for(
    samplesheet_df.index[samplesheet_df.index.isna()])[0]

# Drop all rows after the first NaN patient
samplesheet_df = samplesheet_df.iloc[:nan_idx]

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''


# Function to add items to a silo
def add_to_silo(items, silo_name):
    # Only set data_silo if it is currently empty
    mask = samplesheet_df.index.isin(items) & (samplesheet_df['data_silo'] == '')
    samplesheet_df.loc[mask, 'data_silo'] = silo_name


# Compare to the cloud samples and mark all duplicates
cloud_data = pd.read_csv(cloud_phenotype_file, sep='\t', header=0)

# Update sample_name so that it is the characters before first underscore
cloud_data['sample_name'] = cloud_data['sample_name'].str.split('_').str[0]

# Put downs cases into the "down_syndrome" silo
down_syndrome_mask = samplesheet_df['primary subtype'].str.contains(
    'Down', na=False)
add_to_silo(samplesheet_df[down_syndrome_mask].index, 'down_syndrome')
add_to_silo(samplesheet_df[
    samplesheet_df.index.str.startswith('SJDOWN')].index, 'down_syndrome')

# Check "2nd subtype" column,
# put all rows with non-zero values into the "multiclass" silo
multi_subtype_rows = samplesheet_df['2nd subtype'] != 0
if multi_subtype_rows.any():
    add_to_silo(samplesheet_df[multi_subtype_rows].index, 'multiclass')
    # For these rows, append the 2nd subtype to the primary subtype
    for idx in samplesheet_df[multi_subtype_rows].index:
        primary = samplesheet_df.at[idx, 'primary subtype']
        second = samplesheet_df.at[idx, '2nd subtype']
        samplesheet_df.at[idx, 'primary subtype'] = f'{primary},{second}'

# If a sample is in both datasets, put it in the "replicate" silo
samplesheet_df['is_replicate'] = samplesheet_df.index.isin(
    cloud_data['sample_name'])
add_to_silo(
    samplesheet_df[samplesheet_df['is_replicate']].index, 'replicate')
samplesheet_df = samplesheet_df.drop(columns=['is_replicate'])

# Rename index to sample
samplesheet_df.index.name = 'sample'

# Rename subtype col
samplesheet_df = samplesheet_df.rename(columns={'primary subtype': 'subtype'})

# Rename library
samplesheet_df = samplesheet_df.rename(columns={'RNA-seq library': 'batch'})

# In batch, only keep the part of the string that is "total_RNA" or "mRNA"
samplesheet_df['batch'] = samplesheet_df[
    'batch'].str.extract('(total_RNA|mRNA)', expand=False)

# Create a map of sample names to paths.
# For each file in the raw data dir, extract everything
# #before the first dot or underscore and see if it matches
# a row in the samplesheet_df
sample_to_path_map = {}
for root, dirs, files in os.walk(raw_data_path):
    for filename in files:
        sample_name = filename.split('.')[0].split('_')[0]
        if sample_name in samplesheet_df.index:
            sample_to_path_map[sample_name] = os.path.join(root, filename)

# Turn the dict into a new column feature_counts
samplesheet_df['feature_counts'] = samplesheet_df.index.map(sample_to_path_map)

# Keep only the columns we need
samplesheet_df = samplesheet_df[
    ['subtype', 'batch', 'feature_counts', 'data_silo']]

# Get subtype translation
tax = Taxonomy()
samplesheet_df['subtype'] = tax.translate_subtype_column(
    samplesheet_df['subtype'], split_delimiter=',')

# Add unknowns to unlabeled silo
unknown_rows = samplesheet_df['subtype'].str.contains('other', na=False)
if unknown_rows.any():
    add_to_silo(samplesheet_df[unknown_rows].index, 'unlabeled')

# Add UNRECOGNIZED to unrecognized silo
unrecognized_rows = samplesheet_df['subtype'].str.contains(
    'UNRECOGNIZED', na=False)
if unrecognized_rows.any():
    add_to_silo(samplesheet_df[unrecognized_rows].index, 'unrecognized')

# Add EXCLUDED to excluded silo
excluded_rows = samplesheet_df['subtype'].str.contains(
    'EXCLUDED', na=False)
if excluded_rows.any():
    add_to_silo(samplesheet_df[excluded_rows].index, 'excluded')

# Add remaining to labeled_diagnostic silo
add_to_silo(
    samplesheet_df[samplesheet_df['data_silo'] == ''
                   ].index, 'labeled_diagnostic')

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
