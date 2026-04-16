import pandas as pd
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from preprocessing.helpers import summarize_patient_details, \
    process_gdc_gex, process_gdc_dnam, \
    summarize_modalities


# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/TARGET-AML-TCGA-LAML'
gex_dir = f'{data_path}/feature_counts'
dnam_dir = f'{data_path}/betas'
gex_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.gex.tsv'
dnam_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.dnam.tsv'
dnam_subtypes_file = f'{data_path}/TARGET-AML-TCGA-LAML.subtypes.dnam.tsv'

dataset_name = 'TCGA-LAML'
print(f"Processing {dataset_name}...")

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
gex_output_dir = f'{preprocessed_dir}/gex'
dnam_output_dir = f'{preprocessed_dir}/dnam'
clinical_file = f'{data_path}/TCGA-LAML.clinical.tsv'

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

# Concatenate the details dataframes and save
details_df = pd.concat([gex_details_df, dnam_details_df], axis=0)
details_df = details_df[~details_df.index.duplicated(keep='first')]
summarize_patient_details(details_df)
details_output_file = f'{preprocessed_dir}/patient_details.csv'
details_df.to_csv(details_output_file, index=True)

# Indices are of the format TCGA-AB-2803-03A
# Create new column "target_case_id" which removes the last segment, so TCGA-AB-2803
gex_samplesheet_df['target_case_id'] = gex_samplesheet_df.index.str.rsplit('-', n=1).str[0]
dnam_samplesheet_df['target_case_id'] = dnam_samplesheet_df.index.str.rsplit('-', n=1).str[0]

# Make target_case_id the index
gex_samplesheet_df = gex_samplesheet_df.set_index('target_case_id')
dnam_samplesheet_df = dnam_samplesheet_df.set_index('target_case_id')

g = gex_samplesheet_df.replace('', pd.NA)
d = dnam_samplesheet_df.replace('', pd.NA)
samplesheet_df = g.combine_first(d)
samplesheet_output_file = f'{preprocessed_dir}/samplesheet.csv'
samplesheet_df.to_csv(samplesheet_output_file, index=True)
summarize_modalities(samplesheet_df)
