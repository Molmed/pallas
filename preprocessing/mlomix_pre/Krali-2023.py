import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, GEX_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_excluded, silo_remaining, format_mlomix_columns


dataset_name = 'Krali-2023'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/GSE227832_RNAseq_read_counts.txt'
pheno_input_file = f'{data_path}/41698_2023_479_MOESM2_ESM.xlsx'
details_input_file = f'{data_path}/Patients_clinicallist_FINAL.csv'
dnam_beta_matrix_files = [
    f'{data_path}/ALL_Krali_2023_n=384_beta_matrix.tsv',
    f'{data_path}/ALL_Nordlund_Inaki_n=828_beta_matrix.tsv'
]
dnam_pvals_files = [
    f'{data_path}/ALL_Krali_2023_n=384_detection_pvals.tsv',
    f'{data_path}/ALL_Nordlund_Inaki_n=828_detection_pvals.tsv'
]

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
                               sheet_name='Supplementary Data 2',
                               header=1,
                               index_col='public_id')

# REPLICATES DF ###
replicates_df = pd.read_excel(pheno_input_file,
                              sheet_name='Supplementary Data 10',
                              header=1,
                              index_col='public_id')

# Remove all items in replicates_df that are in samplesheet_df
replicates_df = replicates_df[~replicates_df.index.isin(samplesheet_df.index)]

# This column has a newline character that makes it tricky
subtype_col_name = replicates_df.columns[1]
replicates_df = replicates_df[[subtype_col_name,
                               'RNAseq protocol'
                               ]]

# Rename cols
replicates_df = replicates_df.rename(columns={subtype_col_name: 'class',
                                              'RNAseq protocol': 'batch'})
# Rename index
replicates_df.index.name = 'sample'

# MAIN SAMPLESHEET ###
# This column has a newline character that makes it tricky
subtype_col_name = samplesheet_df.columns[1]

samplesheet_df = samplesheet_df[[subtype_col_name,
                                 'GEX dataset',
                                 'DNAm dataset',
                                 'library'
                                 ]]

# Rename cols
samplesheet_df = samplesheet_df.rename(
    columns={subtype_col_name: 'class',
             'library': 'batch'})

# Rename index
samplesheet_df.index.name = 'sample'

# Append replicates_df to samplesheet_df
samplesheet_df = pd.concat([samplesheet_df, replicates_df], ignore_index=False)

# PROCESS DNAm
dnam_dir = f'{preprocessed_dir}/dnam'
os.makedirs(dnam_dir, exist_ok=True)

# Initialize columns that will be populated
samplesheet_df['dnam_beta_matrix_file'] = None
samplesheet_df['dnam_pvals_file'] = None
samplesheet_df['gex_feature_counts_file'] = None


def process_dnam_file(filename, suffix, samplesheet_col):
    print(f"Processing {filename} for {samplesheet_col}...")
    df = pd.read_csv(filename, index_col=0, delimiter='\t')
    for col in df.columns:
        # If there is a dot in the col name, remove it and everything after it
        if '.' in col:
            new_col = col.split('.')[0]
            df = df.rename(columns={col: new_col})
            col = new_col
        print(f"Processing column {col} in {filename}...")
        output_file = f'{dnam_dir}/{col}.{suffix}.txt'
        # Set 'dnam_beta_matrix_file' column for this sample in samplesheet_df
        # First check that the sample is in samplesheet_df
        if col not in samplesheet_df.index:
            print(f"Warning: Sample {col} in {filename} not found in samplesheet_df index")
            continue
        samplesheet_df.loc[col, samplesheet_col] = output_file
        print(f"Saving {col} to {output_file}...")
        beta_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)


# # Read in and merge all beta matrix files
for beta_matrix_file in dnam_beta_matrix_files:
    process_dnam_file(beta_matrix_file, 'betas', 'dnam_beta_matrix_file')

for pvals_file in dnam_pvals_files:
    process_dnam_file(pvals_file, 'pvals', 'dnam_pvals_file')

# PROCESS GEX ########
# Break up the gene expression file
gex_dir = f'{preprocessed_dir}/gex'
os.makedirs(gex_dir, exist_ok=True)
gex_df = pd.read_csv(raw_data_path, index_col=0, delimiter='\t')

# For each column:
# Create a new file with the column name as the file name
# The contents should contain the Gene and the counts for that column
# delimited by a tab. No header, no index.
for col in gex_df.columns:
    output_file = f'{gex_dir}/{col}.txt'
    # Set feature_counts column for this sample in samplesheet_df
    samplesheet_df.loc[col, 'gex_feature_counts_file'] = output_file
    gex_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)

# For items where subtype is null and sample ends in "_2", "_3", or "_r",
# Look for the sample name without that suffix and copy the subtype from there
for suffix in ['_2', '_3', '_r']:
    mask = (samplesheet_df['class'].isna()) & (samplesheet_df.index.str.endswith(suffix))
    for sample in samplesheet_df[mask].index:
        base_sample = sample[:-len(suffix)]
        if base_sample in samplesheet_df.index:
            samplesheet_df.loc[sample, 'class'] = samplesheet_df.loc[base_sample, 'class']

# For remaining samples with null subtype, if it starts with "Constitutional"
# or ends in "_B" or "_T", set subtype to "Control" and batch to "ScriptSeq"
# (The control are ScriptSeq according to https://doi.org/10.1186/s13045-017-0515-y)
mask = (samplesheet_df['class'].isna()) & (
    samplesheet_df.index.str.startswith("Constitutional") |
    samplesheet_df.index.str.endswith(("_B", "_T"))
)
samplesheet_df.loc[mask, 'class'] = "Control"
samplesheet_df.loc[mask, 'batch'] = "ScriptSeq"

# Rename KMT2A's to avoid confusion with AML
samplesheet_df['class'] = samplesheet_df['class'].str.replace(
    'KMT2A-r', 'BALLKMT2A', regex=False)

# Get subtype translation
tax = Taxonomy()
samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'],
    exclude_subtypes=GEX_EXCLUDE_SUBTYPES)

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set data_silo column to empty string
samplesheet_df['data_silo'] = ''

# Silo unlabeled
samplesheet_df = silo_unlabeled(samplesheet_df)

# Everything prefaced with "EXCLUDED" should be set to excluded silo
samplesheet_df = silo_excluded(samplesheet_df)

# If sample name ends in "_2" or "_3", set data_silo to "relapse"
relapse_mask = samplesheet_df.index.str.endswith(("_2", "_3"))
samplesheet_df = add_to_silo(
    samplesheet_df, samplesheet_df[relapse_mask].index, 'relapse')

# If sample name ends in "_r", set data_silo to "replicate"
replicate_mask = samplesheet_df.index.str.endswith("_r")
samplesheet_df = add_to_silo(
    samplesheet_df,
    samplesheet_df[replicate_mask].index,
    'replicate')

# Put the rest in the labeled_diagnostic silo
samplesheet_df = silo_remaining(samplesheet_df)

# remove all rows from spreadsheet missing both gex and dnam data, since there is no point in keeping them
# and report on these samples
has_gex = samplesheet_df['gex_feature_counts_file'].notna()
has_dnam = samplesheet_df['dnam_beta_matrix_file'].notna()
missing_gex_and_dnam = (~has_gex) & (~has_dnam)
if missing_gex_and_dnam.any():
    print("Samples missing both gex and dnam data:")
    print(samplesheet_df[missing_gex_and_dnam].index.to_list())
samplesheet_df = samplesheet_df[~missing_gex_and_dnam]

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)

# Copy df
krali_df = samplesheet_df.copy()
krali_samples = krali_df.index.to_list()

# join on details file to see what we have
details_df = pd.read_csv(details_input_file, index_col=0)
details_df.index.name = 'sample'
# Only keep entries that are in krali_samples
details_df = details_df.loc[details_df.index.isin(krali_samples)]
details_df['dataset'] = dataset_name

# Get list of samples in krali_samples but not in details_df and print them
missing_samples = [s for s in krali_samples if s not in details_df.index]
print(f"Samples in {dataset_name} but not in details file: {missing_samples}")

# Create a new DataFrame for the missing samples
missing_details = pd.DataFrame(index=missing_samples, columns=['dataset', 'sex', 'age'])
missing_details.index.name = 'sample'
missing_details['dataset'] = dataset_name

# Copy details for missing samples
for sample in missing_samples:
    # If sample starts with "ALL" and ends with "_2" or "_r", look up the original sample
    if sample.startswith("ALL") and (sample.endswith("_2") or sample.endswith("_3") or sample.endswith("_r")):
        # Find the base sample name by removing the suffix
        if sample.endswith("_r"):
            base_sample = sample[:-2]  # Remove "_r"
        else:
            base_sample = sample[:-2]  # Remove "_2" or "_3"

        # If the base sample exists in details_df, copy its age and sex
        if base_sample in details_df.index:
            missing_details.loc[sample, 'sex'] = details_df.loc[base_sample, 'Sex']
            missing_details.loc[sample, 'age'] = details_df.loc[base_sample, 'Age']
            print(f"Copied details for {sample} from {base_sample}")
        else:
            print(f"Base sample {base_sample} not found for {sample}")
    # For samples like "Constitutional_*" or "*_B" or "*_T", leave as NaN
    else:
        print(f"No details to copy for {sample}")

# Rename columns in details_df to match desired output format
details_df = details_df.rename(columns={'Sex': 'sex', 'Age': 'age'})

# Combine the original details and the missing details
all_details = pd.concat([details_df[['dataset', 'sex', 'age']], missing_details])
# Save the combined details
all_details.to_csv(details_file)
