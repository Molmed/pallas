import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR, GEX_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, silo_unlabeled, \
    silo_unrecognized, silo_excluded, silo_remaining

dataset_name = 'SJC-panALL'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/BALL-1988S-HTSeq'
pheno_input_file = f'{data_path}/B-ALL-subtyping.xlsx'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'
details_file = \
    f'{processed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# PROCESS PHENO ########
print("Processing phenotype data...")
samplesheet_df = pd.read_excel(pheno_input_file,
                               sheet_name='ST1.Cohort',
                               header=0,
                               index_col='patient')

details_df = samplesheet_df.copy()
details_df.index.name = 'sample'

# Drop rows with NaN index
details_df = details_df[details_df.index.notna()]

# Only keep rows where index starts with 'S' or 'P'
details_df = details_df[details_df.index.str.startswith(('S', 'P'))]

details_df['dataset'] = dataset_name
details_df = details_df.rename(columns={'gender': 'sex'})

details_df = details_df[
    ['dataset', 'sex', 'age']]

details_df.to_csv(details_file)

# Get index of first row where patient is NaN
nan_idx = samplesheet_df.index.get_indexer_for(
    samplesheet_df.index[samplesheet_df.index.isna()])[0]

# Drop all rows after the first NaN patient
samplesheet_df = samplesheet_df.iloc[:nan_idx]

# Rename index to sample
samplesheet_df.index.name = 'sample'

# Rename subtype col
samplesheet_df = samplesheet_df.rename(columns={'primary subtype': 'subtype'})

# Rename library
samplesheet_df = samplesheet_df.rename(columns={'RNA-seq library': 'batch'})

# In batch, only keep the part of the string that is "total_RNA" or "mRNA"
samplesheet_df['batch'] = samplesheet_df[
    'batch'].str.extract('(total_RNA|mRNA)', expand=False)

# For multiclass cases, append the 2nd subtype to the primary subtype
multi_subtype_rows = samplesheet_df['2nd subtype'] != 0
for idx in samplesheet_df[multi_subtype_rows].index:
    primary = samplesheet_df.at[idx, 'subtype']
    second = samplesheet_df.at[idx, '2nd subtype']
    samplesheet_df.at[idx, 'subtype'] = f'{primary},{second}'

# TRANSLATE SUBTYPE
tax = Taxonomy()
samplesheet_df['subtype'] = tax.translate_subtype_column(
    samplesheet_df['subtype'],
    exclude_subtypes=GEX_EXCLUDE_SUBTYPES,
    split_delimiter=',')

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''

# EXCLUDED SILO
samplesheet_df = silo_unrecognized(samplesheet_df)
samplesheet_df = silo_excluded(samplesheet_df)

# Down cases
down_syndrome_mask = samplesheet_df['subtype'].str.contains(
    'Down', na=False)
samplesheet_df = add_to_silo(samplesheet_df,
                             samplesheet_df[down_syndrome_mask].index,
                             'excluded')
down_syndrome_mask = samplesheet_df.index.str.startswith('SJDOWN')
samplesheet_df = add_to_silo(samplesheet_df,
                             samplesheet_df[down_syndrome_mask].index,
                             'excluded')

# Manually add items that seem to have labeling inconsistencies
# e.g. different subtypes specificed in different columns
manual_exclusions = [
    'SJ030210',  # Both KMT2A and NUP98,
    'SJHYPER003',       # Identified as Hypodiploid
    'SJHYPER025',       # Identified as Hypodiploid
    'SJHYPER022044',    # Identified as Hypodiploid
    'SJHYPO121',        # Identified as BCR::ABL1
    'SJHYPO110',        # Identified as BCR::ABL1-like
    'SJHYPO106',        # Identified as ETV6::RUNX1-like
    'SJHYPO115',        # Identified as ETV6::RUNX1-like
    'SJHYPO102',        # Identified as PAX5 P80R
    'SJHYPO103',       # Identified as PAX5 P80R
    'SJHYPO112',       # Identified as PAX5 P80R
    'SJHYPO107',       # Identified as PAX5alt
    'SJHYPO113',       # Identified as PAX5alt
    'SJHYPO147',       # Identified as PAX5alt
    'SJHYPO151',       # Identified as PAX5alt
    'SJHYPO021982',    # Identified as PAX5alt
    'SJHYPO021991',    # Identified as PAX5alt
    'SJHYPO022024',    # Identified as PAX5alt
    'SJETV003804',    # Identified as TCF3::PBX1,
    'SJBALL020113',   # Both HeH and CEBPB-IGH
    ]
samplesheet_df = add_to_silo(samplesheet_df,
                             manual_exclusions,
                             'excluded')

# MULTICLASS SILO
# Check "2nd subtype" column,
# put all rows with non-zero values into the "multiclass" silo
multi_subtype_rows = samplesheet_df['2nd subtype'] != 0
if multi_subtype_rows.any():
    samplesheet_df = add_to_silo(samplesheet_df,
                                 samplesheet_df[multi_subtype_rows].index,
                                 'multiclass')

# UNLABELED SILO
samplesheet_df = silo_unlabeled(samplesheet_df)

# Add remaining to labeled_diagnostic silo
samplesheet_df = silo_remaining(samplesheet_df)

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

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
