import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, \
        GEX_EXCLUDE_SUBTYPES, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_excluded, silo_remaining, \
    format_mlomix_columns, summarize_patient_details, \
    process_dnam_file, concatenate_samplesheets, \
    concatenate_detail_dfs, summarize_modalities


def preprocess_heinaniemi_2023():
    dataset_name = 'Heinaniemi-2023'
    print(f"Processing {dataset_name}...")

    # INPUT FILES ########
    data_path = f'{RAW_DATA_DIR}/{dataset_name}'
    raw_data_path = f'{data_path}/GSE228632_RNAseq_read_counts.txt'
    pheno_input_file = f'{data_path}/GSE228632_meta.csv'
    details_input_file = f'{data_path}/Clinical_data_ALLIUM_FIN_cohort.xlsx'

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
    samplesheet_df = pd.read_csv(pheno_input_file, index_col=0)

    # Drop batch column and rename the rest
    samplesheet_df = samplesheet_df.drop(columns=['batch', 'Subtype'])
    samplesheet_df = samplesheet_df.rename(columns={'Subtype_updated': 'class',
                                                    'batches': 'batch'})

    # Rename index
    samplesheet_df.index.name = 'sample'

    # In the batch column, replace as follows:
    # Batch1 becomes "Globin-Zero"
    # Batch2 becomes "Ribosomal RNA depletion"
    samplesheet_df['batch'] = samplesheet_df['batch'].replace({
        'Batch1': 'globin_zero',
        'Batch2': 'ribosomal_RNA_depletion'
    })

    # Get subtype translation
    tax = Taxonomy()
    samplesheet_df['class'] = tax.translate_subtype_column(samplesheet_df['class'])

    # Set 'dataset' column to dataset name
    samplesheet_df['dataset'] = dataset_name

    # Set data silo to empty string
    samplesheet_df['data_silo'] = ''

    # Silo unlabeled
    samplesheet_df = silo_unlabeled(samplesheet_df)

    # Silo remaining as labeled_diagnostic
    samplesheet_df = silo_remaining(samplesheet_df)

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

    # Read in details from Clinical_data_ALLIUM_FIN_cohort.xlsx
    details_df = pd.read_excel(details_input_file, index_col=0)
    details_df.index.name = 'sample'
    # Only keep the items that are also in samplesheet_df
    details_df = details_df.loc[samplesheet_df.index]
    details_df['dataset'] = dataset_name

    # The age_range column should be converted to a new column "age" that has the midpoint
    def age_range_to_midpoint(age_range):
        if pd.isna(age_range):
            return None
        if isinstance(age_range, str):
            parts = age_range.split('-')
            if len(parts) == 2:
                try:
                    low = int(parts[0])
                    high = int(parts[1])
                    return (low + high) // 2
                except ValueError:
                    return None
        return None

    details_df['age'] = details_df['age_range'].apply(age_range_to_midpoint)
    details_df = details_df[
        ['dataset', 'sex', 'age']]
    return samplesheet_df, details_df


dataset_name = 'Krali-2023'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/GSE227832_RNAseq_read_counts.txt'
pheno_input_file = f'{data_path}/41698_2023_479_MOESM2_ESM.xlsx'
dnam_subtypes_file = f'{data_path}/dnam_subtypes.csv'
clinical_file = f'{data_path}/Patients_clinicallist_FINAL.csv'

dnam_beta_matrix_files = [
    f'{data_path}/ALL_Krali_2023_n=384_beta_matrix.tsv',
    f'{data_path}/ALL_Nordlund_Inaki_n=828_beta_matrix.tsv',
    f'{data_path}/ALL_relapse_Nordlund_Inaki_n=27_beta_matrix.tsv'
]
dnam_pvals_files = [
    f'{data_path}/ALL_Krali_2023_n=384_detection_pvals.tsv',
    f'{data_path}/ALL_Nordlund_Inaki_n=828_detection_pvals.tsv',
    f'{data_path}/ALL_relapse_Nordlund_Inaki_n=27_detection_pvals.tsv'
]

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'
details_file = \
    f'{preprocessed_dir}/patient_details.csv'
dnam_sample_list = f'{preprocessed_dir}/dnam_sample_list.txt'

# Make processed dir if not exists
os.makedirs(preprocessed_dir, exist_ok=True)

# PROCESS DNAm
dnam_dir = f'{preprocessed_dir}/dnam'
os.makedirs(dnam_dir, exist_ok=True)

# If dnam_sample_sheet does not exist,
# Get the first column from all the dnam_beta_matrix_files
# and merge them into a single sample list

# Make it a list, not a set, so duplicates are included
if not os.path.exists(dnam_sample_list):
    dnam_samples = []
    for beta_matrix_file in dnam_beta_matrix_files:
        df = pd.read_csv(beta_matrix_file, sep='\t', index_col=0)
        samples = df.columns.str.split('.').str[0]
        dnam_samples.extend(samples)
    with open(dnam_sample_list, 'w') as f:
        for sample in sorted(set(dnam_samples)):
            f.write(sample + '\n')
else:
    with open(dnam_sample_list, 'r') as f:
        dnam_samples = set(line.strip() for line in f)

# Read in dnam_subtypes_file into samplesheet_dnam
samplesheet_dnam_df = pd.read_csv(dnam_subtypes_file, index_col=0)
samplesheet_dnam_df.index.name = 'sample'
missing_samples = [sample for sample in dnam_samples if sample not in samplesheet_dnam_df.index]

# The missing samples are of the format 'ALL_124r1'
# For each of them, add the sample name to samplesheet_dnam_df
# Then remove the "r[digit]" suffix and find the matching sample in the samplesheet and copy "class" from there
# For sample.type, set "relapse"
for sample in missing_samples:
    samplesheet_dnam_df.loc[sample, 'dataset'] = dataset_name
    samplesheet_dnam_df.loc[sample, 'sample.type'] = 'relapse'
    base_sample = sample.split('r')[0]
    if base_sample in samplesheet_dnam_df.index:
        samplesheet_dnam_df.loc[sample, 'class'] = samplesheet_dnam_df.loc[base_sample, 'class']
    else:
        samplesheet_dnam_df.loc[sample, 'class'] = 'Disease,B-ALL,B-other'

# Read in and merge all beta matrix files
for beta_matrix_file in dnam_beta_matrix_files:
    samplesheet_dnam_df = process_dnam_file(
        samplesheet_dnam_df, dnam_dir,
        beta_matrix_file,
        'betas',
        'dnam_beta_matrix_file')

for pvals_file in dnam_pvals_files:
    samplesheet_dnam_df = process_dnam_file(
        samplesheet_dnam_df,
        dnam_dir,
        pvals_file,
        'pvals',
        'dnam_pvals_file')

# Write to samplesheet_dnam.csv
samplesheet_dnam_df.to_csv(f'{preprocessed_dir}/samplesheet_dnam.csv', index=True)

# Set 'dataset' column to dataset name
samplesheet_dnam_df['dataset'] = dataset_name

# Set data_silo column to empty string
samplesheet_dnam_df['data_silo'] = ''

# Silo unlabeled
samplesheet_dnam_df = silo_unlabeled(samplesheet_dnam_df)

# Everything prefaced with "EXCLUDED" should be set to excluded silo
samplesheet_dnam_df = silo_excluded(samplesheet_dnam_df)

# If "sample.type" is "relapse", set data_silo to "relapse"
relapse_mask = samplesheet_dnam_df['sample.type'] == 'relapse'
samplesheet_dnam_df = add_to_silo(
    samplesheet_dnam_df,
    samplesheet_dnam_df[relapse_mask].index,
    'relapse')

samplesheet_dnam_df = silo_remaining(samplesheet_dnam_df)

# Format cols
samplesheet_dnam_df = format_mlomix_columns(samplesheet_dnam_df)

# Save as csv
samplesheet_dnam_df.to_csv(f'{preprocessed_dir}/samplesheet_dnam.csv', index=True)

print(samplesheet_dnam_df)
exit()

# Read in clinical file
# Report all samples not in it
# clinical_df = pd.read_csv(clinical_file, index_col=0)
# missing_clinical_samples = [sample for sample in dnam_samples if sample not in clinical_df.index]
# if missing_clinical_samples:
#     print(f"Samples in dnam beta matrix files but not in clinical file: {missing_clinical_samples}")


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

# Initialize columns that will be populated
samplesheet_df['dnam_beta_matrix_file'] = None
samplesheet_df['dnam_pvals_file'] = None
samplesheet_df['gex_feature_counts_file'] = None

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
    if not os.path.exists(output_file):
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
# Merge excluded subtypes from GEX and DNAM
all_excluded_subtypes = set(GEX_EXCLUDE_SUBTYPES) | set(DNAM_EXCLUDE_SUBTYPES)

samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'],
    exclude_subtypes=all_excluded_subtypes)

# Set batch to null if there is no gene expression data, since batch is only relevant for gene expression
samplesheet_df.loc[samplesheet_df['gex_feature_counts_file'].isna(), 'batch'] = None

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

# Get heinaniemi dataset
h_samplesheet_df, h_details_df = preprocess_heinaniemi_2023()

# Merge it in
samplesheet_df = concatenate_samplesheets(
    samplesheet_df,
    h_samplesheet_df)

# Merge with the dnam
samplesheet_df = concatenate_samplesheets(
    samplesheet_df,
    samplesheet_dnam_df,
    samplesheet_output_file)

summarize_modalities(samplesheet_df)

# GET ALL PATIENT DETAILS
# Copy df
krali_df = samplesheet_df.copy()
krali_samples = krali_df.index.to_list()

# join on details file to see what we have
details_df = pd.read_csv(details_input_file, index_col=0)
details_df.index.name = 'sample'
# Only keep entries that are in krali_samples
details_df = details_df.loc[details_df.index.isin(krali_samples)]

# Merge in heinaniemi details
details_df = concatenate_detail_dfs(
    details_df,
    h_details_df)

details_df['dataset'] = dataset_name

# Get list of samples in krali_samples but not in details_df and print them
missing_samples = [s for s in krali_samples if s not in details_df.index]
print(f"Samples in {dataset_name} but not in details file: {missing_samples}")

# Rename columns in details_df to match desired output format
details_df = details_df.rename(columns={'Sex': 'sex', 'Age': 'age'})

# Drop all rows with "_r", "_2", or "_3" in the index, since those are replicates or relapses and we don't want to include them in the details file
details_df = details_df[~details_df.index.str.endswith(("_r", "_2", "_3"))]

# Drop all rows ending with "_B" or "_T" in the index, since those are controls and we don't want to include them in the details file
details_df = details_df[~details_df.index.str.endswith(("_B", "_T"))]

# Drop all rows starting with "Constitutional" in the index, since those are controls and we don't want to include them in the details file
details_df = details_df[~details_df.index.str.startswith("Constitutional")]

print(details_df)
exit()

# Report details
summarize_patient_details(details_df)

# Save details file
details_df.to_csv(details_file, index=True)
