import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_excluded, silo_remaining, silo_unrecognized, \
        format_mlomix_columns, summarize_patient_details, \
            deduplicate_patient_details_using_sample_suffix, \
                process_dnam_file

dataset_name = 'Krali-2021'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
beta_matrix_file = f'{data_path}/AML_Krali_Pelle_2021_n=142_beta_matrix_test.tsv'
pvals_file = f'{data_path}/AML_Krali_Pelle_2021_n=142_detection_pvals_test.tsv'
pheno_file = f'{data_path}/PALLAS_FINAL_pheno_combined_n=2479.csv'
details_input_file = f'{data_path}/AML_clinical_n=142.csv'

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
samplesheet_df = pd.read_csv(pheno_file, index_col=0)
details_df = pd.read_csv(details_input_file, index_col=0)

# Drop all rows in samplesheet_df where the index is not in the patient_details_df index, since those are not part of the 142 samples with molecular data
samplesheet_df = samplesheet_df[samplesheet_df.index.isin(details_df.index)]

# Rename details_df index to "sample" and add "dataset" column
details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df['sex'] = details_df['sex'].replace({'male': 'Male', 'female': 'Female'})

# Create new col "unique_patient_id" which is the index but removing
# any potential "_r" suffix.
details_df['unique_patient_id'] = details_df.index.str.split('_r').str[0]
details_df = deduplicate_patient_details_using_sample_suffix('_r', details_df)
summarize_patient_details(details_df)

details_df = details_df[
    ['dataset', 'sex', 'age']]
details_df.to_csv(details_file)

# Format samplesheet
# Rename index
samplesheet_df.index.name = 'sample'

# Rename cols
samplesheet_df = samplesheet_df.rename(
    columns={'subtype': 'class'})

# Initialize columns that will be populated
samplesheet_df['dnam_beta_matrix_file'] = None
samplesheet_df['dnam_pvals_file'] = None
samplesheet_df['gex_feature_counts_file'] = None
samplesheet_df['batch'] = ''

# Get subtype translation
tax = Taxonomy()
samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'],
    exclude_subtypes=DNAM_EXCLUDE_SUBTYPES)

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set data_silo column to empty string
samplesheet_df['data_silo'] = ''

# Silo unlabeled
samplesheet_df = silo_unlabeled(samplesheet_df)

# Everything prefaced with "EXCLUDED" should be set to excluded silo
samplesheet_df = silo_excluded(samplesheet_df)

# Silo unrecognized
samplesheet_df = silo_unrecognized(samplesheet_df)

# If sample name ends in "_r", set data_silo to "relapse"
relapse_mask = samplesheet_df.index.str.endswith('_r')
samplesheet_df = add_to_silo(
    samplesheet_df, samplesheet_df[relapse_mask].index, 'relapse')

# Put the rest in the labeled_diagnostic silo
samplesheet_df = silo_remaining(samplesheet_df)

# Process the dnam data
samplesheet_df = process_dnam_file(samplesheet_df, preprocessed_dir, beta_matrix_file, 'betas', 'dnam_beta_matrix_file')
samplesheet_df = process_dnam_file(samplesheet_df, preprocessed_dir, pvals_file, 'pvals', 'dnam_pvals_file')

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
