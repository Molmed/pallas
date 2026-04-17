import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, \
        GEX_EXCLUDE_SUBTYPES, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_excluded, silo_remaining, \
        format_mlomix_columns, summarize_patient_details, \
            process_dnam_file, process_gdc_gex, process_gdc_dnam, \
                summarize_modalities, concatenate_detail_dfs, concatenate_samplesheets, process_gdc_clinical

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/TARGET-AML-TCGA-LAML'
pheno_df_GSE124413 = f'{data_path}/GSE124413_series_matrix.txt'
pheno_df_GSE190931 = f'{data_path}/GSE190931_series_matrix.txt'
gdc_gex_dir = f'{data_path}/feature_counts'
gdc_dnam_dir = f'{data_path}/gdc_betas'
idats_dir = f'{data_path}/GSE124413_RAW'
beta_matrix_file = f'{data_path}/GSE190931.beta_matrix.tsv'
pvals_file = f'{data_path}/GSE190931.pvals.tsv'
gex_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.gex.tsv'
dnam_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.dnam.tsv'
dnam_subtypes_file = f'{data_path}/TARGET-AML-TCGA-LAML.subtypes.dnam.tsv'
clinical_file = f'{data_path}/TARGET-AML.clinical.tsv'

# OUTPUT FILES ########
dataset_name = 'TARGET-AML'
print(f"Processing {dataset_name}...")
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
gex_output_dir = f'{preprocessed_dir}/gex'
dnam_output_dir = f'{preprocessed_dir}/dnam'

# First grab all the DNAm samples from GDC
# gex_samplesheet_df, gex_details_df = process_gdc_gex(
#     dataset_name,
#     clinical_file,
#     gex_samples_file,
#     gdc_gex_dir,
#     gex_output_dir)

# dnam_samplesheet_df, dnam_details_df = process_gdc_dnam(
#     dataset_name,
#     clinical_file,
#     dnam_samples_file,
#     dnam_subtypes_file,
#     gdc_dnam_dir,
#     dnam_output_dir)

# concatenate_detail_dfs(
#     gex_details_df, dnam_details_df, f'{preprocessed_dir}/patient_details.csv')

# concatenate_samplesheets(
#     gex_samplesheet_df, dnam_samplesheet_df, f'{preprocessed_dir}/samplesheet.csv')


def process_geo_pheno_file(file_path):
    samples = pd.DataFrame(columns=['id',
                                    'target_case_id',
                                    'class',
                                    'geo_dataset',
                                    'idats_basename'])
    geo_dataset = None
    with open(file_path, "r") as file:
        for line in file:
            # Tokenize line by tab and remove quotes from each token
            tokens = [token.strip('"').strip('"\n') for token in line.split("\t")]
            key = tokens.pop(0)

            if key == 'ID_REF':
                samples['id'] = tokens
            elif key == '!Series_geo_accession':
                accession = tokens[0]
                # If not empty, set geo_dataset to the accession
                if accession:
                    geo_dataset = accession
            elif key == '!Sample_title':
                samples['title'] = tokens
            elif key == '!Sample_supplementary_file':
                # Check if already set
                if 'file_1' in samples.columns:
                    samples['file_2'] = tokens
                else:
                    samples['file_1'] = tokens
            elif key == '!Sample_description':
                # Check if already set
                if 'description' in samples.columns:
                    samples['secondary_name'] = tokens
                else:
                    samples['description'] = tokens
            elif key == '!Sample_characteristics_ch1':
                # Get real key from the first token, before the colon
                real_key = tokens[0].split(':')[0]
                samples[real_key] = [token.split(':')[1].strip() for token in tokens]
            elif key == '!Sample_source_name_ch1':
                samples['source'] = tokens

    # Set geo_dataset column to the value of geo_dataset variable
    samples['geo_dataset'] = geo_dataset

    # Perform secondary processing.
    # First process "title" column
    # For each row, check if title starts with "gDNA from normal human bone marrow"
    # If so, set class to "Control"
    samples['class'] = samples['title'].apply(
        lambda x: 'Control' if x.startswith('gDNA from normal human bone marrow') else '')

    # If title starts with "gDNA from subject", set the last
    # word of the title as the target_case_id
    samples['target_case_id'] = samples['title'].apply(
        lambda x: x.split()[-1] if x.startswith('gDNA from subject') else None)

    # If "secondary_name" is set, set its value before the first underscore as the target_case_id
    if 'secondary_name' in samples.columns:
        samples['target_case_id'] = samples.apply(
            lambda row: row['secondary_name'].split('_')[0] if pd.notna(row['secondary_name']) else row['target_case_id'],
            axis=1)

    # Set id column as index
    samples = samples.set_index('id')

    return samples

# Process the GEO pheno files
df_GSE124413 = process_geo_pheno_file(pheno_df_GSE124413)
df_GSE190931 = process_geo_pheno_file(pheno_df_GSE190931)

# Get details_df
# details_df = process_gdc_clinical(dataset_name,
#                                     clinical_file,
#                                     samples_df['Case ID'])

# Post-process them individually since they are formatted differently
# and have different kinds of input dnam data


def process_idat_path(df, index, row, raw_idats_dir, processed_idats_dir):
    # file_name is just the basename, extract end of url
    file_name = os.path.basename(row['file_1']) if pd.notna(row['file_1']) else None
    file_path = f'{raw_idats_dir}/{file_name}'
    if pd.notna(file_name) and os.path.exists(file_path):
        # ungzip file to processed_idats_dir
        # have the same name but strip the sample name
        # which is before the first underscore
        output_file_path = f"{processed_idats_dir}/{file_name.split('_', 1)[1].replace('.gz', '')}"
        if not os.path.exists(output_file_path):
            print(f"Processing {file_path} to {output_file_path}...")
            os.system(f"gunzip -c {file_path} > {output_file_path}")
        df.at[index, 'idats_basename'] = processed_idats_dir
        # Filename is of the format GSM3532234_200991620105_R05C01_Grn.idat.gz
        # Extract the sentrix id and position from the filename
        sentrix_id = file_name.split('_')[1]
        sentrix_position = file_name.split('_')[2].split('.')[0]
        df.at[index, 'sentrix_id'] = sentrix_id
        df.at[index, 'sentrix_position'] = sentrix_position


# PROCESS GSE124413
raw_idats_dir = f'{data_path}/GSE124413_RAW'
processed_idats_dir = f'{dnam_output_dir}/GSE124413'
# Make if it doesn't exist
os.makedirs(processed_idats_dir, exist_ok=True)

# Rename gender column to sex
df_GSE124413 = df_GSE124413.rename(columns={'gender': 'sex'})

# Loop through each row and assign dnam file values accordingly
for index, row in df_GSE124413.iterrows():
    process_idat_path(df_GSE124413, index, row, raw_idats_dir, processed_idats_dir)


# PROCESS GSE190931
# For these samples, we have beta matrix and pvals files for most (but not all!) samples
# First get those few that have raw idats
raw_idats_dir = f'{data_path}/GSE190931_RAW'
processed_idats_dir = f'{dnam_output_dir}/GSE190931'
for index, row in df_GSE190931.iterrows():
    process_idat_path(df_GSE190931, index, row, raw_idats_dir, processed_idats_dir)

processed_betas_dir = f'{dnam_output_dir}/GSE190931'
df_GSE190931 = process_dnam_file(
    df_GSE190931,
    processed_betas_dir,
    beta_matrix_file,
    'betas',
    'dnam_beta_matrix_file')
df_GSE190931 = process_dnam_file(
    df_GSE190931,
    processed_betas_dir,
    pvals_file,
    'pvals',
    'dnam_pvals_file')

# Copy fusion column to class
df_GSE190931['class'] = df_GSE190931['fusion']

samplesheet_df = pd.concat([df_GSE124413, df_GSE190931], axis=0)

# Create new column target-submitter-id which is e.g. "TARGET-20-PARCAU"
# so preface id with "TARGET-20-"
samplesheet_df['target-submitter-id'] = 'TARGET-20-' + samplesheet_df['target_case_id']

# Read in '/home/mariya/Data/pallas/raw/TARGET-AML.subtypes.tsv'
# then join samplesheet on target-submitter-id
subtypes_df = pd.read_csv('/home/mariya/Data/pallas/raw/TARGET-AML-TCGA-LAML/TARGET-AML.subtypes.tsv', sep=',')
# rename cases.submitter_id to target-submitter-id
subtypes_df = subtypes_df.rename(columns={'cases.submitter_id': 'target-submitter-id'})
samplesheet_df = samplesheet_df.merge(subtypes_df[['target-submitter-id', 'subtype']], on='target-submitter-id', how='left')

print(samplesheet_df)
# dump to csv
tmp_output_file = f'{preprocessed_dir}/dnam_samplesheet.csv'
samplesheet_df.to_csv(tmp_output_file, index=True)
print(samplesheet_df)
