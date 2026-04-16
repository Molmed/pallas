import pandas as pd
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from preprocessing.helpers import summarize_patient_details, \
    process_gdc_gex, process_gdc_dnam, \
    summarize_modalities, concatenate_detail_dfs, \
    concatenate_samplesheets


# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/TARGET-AML-TCGA-LAML'
gex_dir = f'{data_path}/feature_counts'
dnam_dir = f'{data_path}/gdc_betas'
gex_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.gex.tsv'
dnam_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.dnam.tsv'
dnam_subtypes_file = f'{data_path}/TARGET-AML-TCGA-LAML.subtypes.dnam.tsv'
clinical_file = f'{data_path}/TCGA-LAML.clinical.tsv'

dataset_name = 'TCGA-LAML'
print(f"Processing {dataset_name}...")

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
gex_output_dir = f'{preprocessed_dir}/gex'
dnam_output_dir = f'{preprocessed_dir}/dnam'

gex_samplesheet_df, gex_details_df = process_gdc_gex(
    dataset_name,
    clinical_file,
    gex_samples_file,
    gex_dir,
    gex_output_dir)

dnam_samplesheet_df, dnam_details_df = process_gdc_dnam(
    dataset_name,
    clinical_file,
    dnam_samples_file,
    dnam_subtypes_file,
    dnam_dir,
    dnam_output_dir)

concatenate_detail_dfs(
    gex_details_df, dnam_details_df, f'{preprocessed_dir}/patient_details.csv')

# Indices are of the format TCGA-AB-2803-03A
# Create new column "target_case_id" which removes the last segment, so TCGA-AB-2803
gex_samplesheet_df['target_case_id'] = gex_samplesheet_df.index.str.rsplit('-', n=1).str[0]
dnam_samplesheet_df['target_case_id'] = dnam_samplesheet_df.index.str.rsplit('-', n=1).str[0]

# Make target_case_id the index
gex_samplesheet_df = gex_samplesheet_df.set_index('target_case_id')
dnam_samplesheet_df = dnam_samplesheet_df.set_index('target_case_id')

concatenate_samplesheets(
    gex_samplesheet_df, dnam_samplesheet_df, f'{preprocessed_dir}/samplesheet.csv')
