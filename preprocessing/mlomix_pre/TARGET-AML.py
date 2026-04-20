import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, \
    DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_excluded, silo_unrecognized, silo_remaining, \
    format_mlomix_columns, process_dnam_file, process_gdc_gex, \
    process_gdc_dnam, concatenate_detail_dfs, concatenate_samplesheets

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
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
gex_output_dir = f'{preprocessed_dir}/gex'
dnam_output_dir = f'{preprocessed_dir}/dnam'

# First grab all the DNAm samples from GDC
gex_samplesheet_df, gex_details_df = process_gdc_gex(
    dataset_name,
    clinical_file,
    gex_samples_file,
    gdc_gex_dir,
    gex_output_dir)

dnam_samplesheet_df, dnam_details_df = process_gdc_dnam(
    dataset_name,
    clinical_file,
    dnam_samples_file,
    dnam_subtypes_file,
    gdc_dnam_dir,
    dnam_output_dir)

gdc_samplesheet = concatenate_samplesheets(
    gex_samplesheet_df,
    dnam_samplesheet_df,
    report=False)

gdc_details_df = concatenate_detail_dfs(
    gex_details_df, dnam_details_df,
    report=False)


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

# Set "None" values in fusion column to empty string
df_GSE190931['fusion'] = df_GSE190931['fusion'].replace('None', '')

# Copy fusion column to class
df_GSE190931['class'] = df_GSE190931['fusion']

samplesheet_df = pd.concat([df_GSE124413, df_GSE190931], axis=0)

# Copy index to sample_id
samplesheet_df['sample_id'] = samplesheet_df.index

# Merge Sex column into sex, keeping all non-empty values
samplesheet_df['sex'] = samplesheet_df.apply(
    lambda row: row['Sex'] if pd.notna(row['Sex']) else row['sex'], axis=1)

# Create new column target-submitter-id which is e.g. "TARGET-20-PARCAU"
# so preface id with "TARGET-20-"
samplesheet_df['target-submitter-id'] = 'TARGET-20-' + samplesheet_df['target_case_id']

# Fill in subtypes from all possible sources
# First, match to samples in the gdc samplesheet. Copy gdc_samplesheet to new df
gdc_samplesheet_copy = gdc_samplesheet.copy()
# sample column is in format "TARGET-20-PABHET-03A_R1". Discard everything after underscore
gdc_samplesheet_copy['target-submitter-id'] = gdc_samplesheet_copy.index.str.split('_').str[0]
# Now discard everything after the last dash to get the target_case_id
gdc_samplesheet_copy['target-submitter-id'] = gdc_samplesheet_copy['target-submitter-id'].str.split('-').str[:-1].str.join('-')
# Rename class to subtype
gdc_samplesheet_copy.rename(columns={'class': 'subtype_gdc'}, inplace=True)

# Now merge on target-submitter-id to get the subtypes from gdc samplesheet
gdc_samplesheet_copy = gdc_samplesheet_copy[['target-submitter-id', 'subtype_gdc']]

# Drop duplicates
gdc_samplesheet_copy = gdc_samplesheet_copy.drop_duplicates(subset=['target-submitter-id'])

# Combine with samplesheet_df
samplesheet_df = samplesheet_df.merge(gdc_samplesheet_copy, on='target-submitter-id', how='left')

# Replace empty class values with "AML, other"
samplesheet_df['class'] = samplesheet_df['class'].replace('', 'AMLNOS')

# Translate them
# Get subtype translation
tax = Taxonomy()
samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'],
    exclude_subtypes=DNAM_EXCLUDE_SUBTYPES)

# For each row, if class is empty, but subtype_gdc is not empty, copy subtype_gdc to class
samplesheet_df['class'] = samplesheet_df.apply(
    lambda row: row['subtype_gdc'] if row['class'] == 'AML, other' and pd.notna(row['subtype_gdc']) else row['class'],
    axis=1)

# Set dataset_name
samplesheet_df['dataset'] = dataset_name
samplesheet_df['data_silo'] = ''
samplesheet_df['batch'] = ''

# Put multiclass items in multiclass silo
multi_subtype_rows = samplesheet_df['class'].str.contains(';', na=False)
if multi_subtype_rows.any():
    samplesheet_df = add_to_silo(
        samplesheet_df,
        samplesheet_df[multi_subtype_rows].index, 'multiclass')

# Silo unlabeled
samplesheet_df = silo_unlabeled(samplesheet_df)

# Silo excluded and unrecognized
samplesheet_df = silo_unrecognized(samplesheet_df)
samplesheet_df = silo_excluded(samplesheet_df)

# Relapse mask is the samples whose 'timepoint' is 'Relapse'
relapse_mask = samplesheet_df['timepoint'] == 'Relapse'
samplesheet_df = add_to_silo(
    samplesheet_df,
    samplesheet_df[relapse_mask].index,
    'relapse')

samplesheet_df = silo_remaining(samplesheet_df)

# Rename target-submitter-id to sample
samplesheet_df = samplesheet_df.rename(columns={'target-submitter-id': 'sample'})

# Where sample is empty, copy value from sample_id
samplesheet_df['sample'] = samplesheet_df.apply(
    lambda row: row['sample_id'] if pd.isna(row['sample']) else row['sample'],
    axis=1)

# Set sample as the index
samplesheet_df = samplesheet_df.set_index('sample')

# Process age
samplesheet_df['age at diagnosis (days)'] = pd.to_numeric(
    samplesheet_df['age at diagnosis (days)'], errors='coerce')
samplesheet_df['age (years)'] = pd.to_numeric(
    samplesheet_df['age (years)'], errors='coerce')

# Process column "age at diagnosis (days)"
# For each row, if the value is not empty, convert it to years and copy it to age (years)
samplesheet_df['age (years)'] = samplesheet_df.apply(
    lambda row: row['age at diagnosis (days)'] / 365.25 if pd.notna(row['age at diagnosis (days)']) else row['age (years)'],
    axis=1)

# Round age (years) to 1 decimal place
samplesheet_df['age (years)'] = samplesheet_df['age (years)'].round(1)

# Rename it to age
samplesheet_df = samplesheet_df.rename(columns={'age (years)': 'age'})

# Save details
details_df = samplesheet_df[['dataset', 'sex', 'age']].copy()

# Remove rows from details_df starting with "GSM"
details_df = details_df[~details_df.index.str.startswith('GSM')]

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# For both samplesheet_df and gdc_samplesheet, create a new column "target_case_timepoint"
# which is of the format "TARGET-20-PABHET-D" or "TARGET-20-PABHET-R" based on the timepoint column (Primary or Relapse)
# In the case of samplesheet_df, just append the existing index with "R" if data_silo is relapse, else "D"
# For gdc_samplesheet: if data_silo is replicate, leave this new column blank.
# Else drop the suffix after the last dash (e.g. "-04A") and append "D" if data_silo is not relapse, else "R"

samplesheet_df['target_case_timepoint'] = samplesheet_df.apply(
    lambda row: row.name if row['class'] == 'Non-leukemic' else (
        f"{row.name}-R" if row['data_silo'] == 'relapse' else f"{row.name}-D"),
    axis=1)
gdc_samplesheet['target_case_timepoint'] = gdc_samplesheet.apply(
    lambda row: row.name if row['data_silo'] == 'replicate' else f"{row.name.rsplit('-', 1)[0]}-R" if row['data_silo'] == 'relapse' else f"{row.name.rsplit('-', 1)[0]}-D",
    axis=1)

# In both samplesheets, if there is a duplicate in target_case_timepoint,
# append a "2" to the value
samplesheet_dup_mask = (
    samplesheet_df.groupby('target_case_timepoint').cumcount() == 1)
samplesheet_df.loc[samplesheet_dup_mask, 'target_case_timepoint'] = (
    samplesheet_df.loc[samplesheet_dup_mask, 'target_case_timepoint'] + '2')

gdc_dup_mask = (
    gdc_samplesheet.groupby('target_case_timepoint').cumcount() == 1)
gdc_samplesheet.loc[gdc_dup_mask, 'target_case_timepoint'] = (
    gdc_samplesheet.loc[gdc_dup_mask, 'target_case_timepoint'] + '2')

# Check target_case_timepoint for duplicates in both dfs and print
if samplesheet_df['target_case_timepoint'].duplicated().any():
    print("Duplicates in samplesheet_df target_case_timepoint:")
    print(samplesheet_df[samplesheet_df['target_case_timepoint'].duplicated(keep=False)]['target_case_timepoint'])
if gdc_samplesheet['target_case_timepoint'].duplicated().any():
    print("Duplicates in gdc_samplesheet target_case_timepoint:")
    print(gdc_samplesheet[gdc_samplesheet['target_case_timepoint'].duplicated(keep=False)]['target_case_timepoint'])

# Set target_case_timepoint as the index for both samplesheets
samplesheet_df = samplesheet_df.set_index('target_case_timepoint')
gdc_samplesheet = gdc_samplesheet.set_index('target_case_timepoint')

# Merge with gdc samplesheet and gdc details
samplesheet_df = concatenate_samplesheets(
    gdc_samplesheet, samplesheet_df, f'{preprocessed_dir}/samplesheet.csv')

details_df = concatenate_detail_dfs(
    gdc_details_df, details_df, f'{preprocessed_dir}/patient_details.csv')
