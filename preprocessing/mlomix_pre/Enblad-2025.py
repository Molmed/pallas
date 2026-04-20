import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, format_mlomix_columns, \
    silo_unlabeled, silo_remaining, \
    silo_unrecognized, silo_excluded, \
    format_mlomix_columns, summarize_patient_details, summarize_modalities


dataset_name = 'Enblad-2025'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
dnam_beta_matrix_file = f'{data_path}/ALL_FMCA_EPIC2_n=55_beta_matrix.tsv'
dnam_pvals_file = f'{data_path}/ALL_FMCA_EPIC2_n=55_detection_pvals.tsv'
pheno_input_file = f'{data_path}/NOPHO_EPIC_n=55_subtype_info_FINAL.csv'
details_input_file = f'{data_path}/New_DNAm_n_55_with_all_IDs.xlsx'

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
dnam_dir = f'{preprocessed_dir}/dnam'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'
details_file = \
    f'{preprocessed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(dnam_dir, exist_ok=True)

# Process the DNAm data
samplesheet_df = pd.read_csv(pheno_input_file, index_col=0)


# Needs a custom function to map the internal to public ids
def process_dnam_file(filename,
                      suffix,
                      samplesheet_col):
    print(f"Processing {filename} for {samplesheet_col}...")

    # First read in samplesheet, get all sample indices, and see if files already exist for all of them
    # If so, just fill in the relevant column and return to avoid reading in the huge dnam files
    sample_indices = samplesheet_df.index
    all_files_exist = True
    for idx in sample_indices:
        output_file = f'{dnam_dir}/{idx}.{suffix}.txt'

        if os.path.exists(output_file):
            samplesheet_df.at[idx, samplesheet_col] = output_file
        else:
            print(f"{output_file} does not exist.")
            all_files_exist = False
    if all_files_exist:
        print(f"All files for {filename} already exist, skipping processing this file.")
        return

    print(f"Not all files for {filename} exist, processing this file to extract data for each sample...")

    # Create dnam dir if it doesn't exist
    os.makedirs(dnam_dir, exist_ok=True)
    df = pd.read_csv(filename, index_col=0, delimiter='\t')
    for col in df.columns:
        # Same is of the format "NOJ1_99/071_1"
        # Extract the string between the two underscores e.g. "99/071"
        # Then, look it up in samplsheet_df column "sample_id.x"
        # The index of that row will be the id we want to use.
        id = col.split('_')[1]
        matching_rows = samplesheet_df[samplesheet_df['sample_id.x'] == id]
        if matching_rows.empty:
            print(f"Warning: No matching sample found in samplesheet_df for id {id} extracted from column {col} in {filename}")
            continue

        sample_id = matching_rows.index[0]
        output_file = f'{dnam_dir}/{sample_id}.{suffix}.txt'
        samplesheet_df.at[sample_id, samplesheet_col] = output_file

        if os.path.exists(output_file):
            print(f"Warning: {output_file} already exists, skipping writing this file.")
            continue
        df[[col]].to_csv(output_file, sep='\t', header=False, index=True)


# Read in and merge all beta matrix files
process_dnam_file(
    dnam_beta_matrix_file,
    'betas',
    'dnam_beta_matrix_file')

process_dnam_file(
    dnam_pvals_file,
    'pvals',
    'dnam_pvals_file')

# Rename index to sample
samplesheet_df.index.name = 'sample'

# Rename Subtype to class
samplesheet_df = samplesheet_df.rename(columns={'Subtype': 'class'})

# Translate
tax = Taxonomy()
samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'], split_delimiter=',',
    exclude_subtypes=DNAM_EXCLUDE_SUBTYPES)

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''

# Silo unlabeled
samplesheet_df = silo_unlabeled(samplesheet_df)

# Silo excluded and unrecognized
samplesheet_df = silo_excluded(samplesheet_df)

# Silo remaining as labeled_diagnostic
samplesheet_df = silo_remaining(samplesheet_df)

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)

summarize_modalities(samplesheet_df)

# Read in details file
# Headers are in the second row, discard first row
details_df = pd.read_excel(details_input_file,
                           index_col="public_id",
                           header=0)
details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df = details_df.rename(columns={'Sex': 'sex', 'Age': 'age'})
details_df['sex'] = details_df['sex'].str.capitalize()

details_df = details_df[
    ['dataset', 'sex', 'age']]

summarize_patient_details(details_df)

details_df.to_csv(details_file)
