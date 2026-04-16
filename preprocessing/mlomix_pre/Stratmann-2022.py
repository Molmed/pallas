import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, GEX_EXCLUDE_SUBTYPES, \
        DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, format_mlomix_columns, \
    silo_unlabeled, silo_remaining, \
    silo_unrecognized, silo_excluded, \
    format_mlomix_columns, summarize_patient_details, \
    deduplicate_patient_details_using_sample_suffix

dataset_name = 'Stratmann-2022'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/AML_RNAseq_Raw_Counts.xlsx'
pheno_input_file = f'{data_path}/stratmann2022_pheno.csv'
details_input_file = f'{data_path}/AML_RNAseq_Metadata_Holmfeldt_220404.xlsx'
dnam_data_path = f'{RAW_DATA_DIR}/{dataset_name}/RE-1840_180726_IDAT'
dnam_raw_samplesheet_input_file = f'{dnam_data_path}/RE-1840_180726_SampleID_SentrixID.txt'

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'
details_file = \
    f'{preprocessed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(preprocessed_dir, exist_ok=True)

# Read in details file
# Headers are in the second row, discard first row
details_df = pd.read_excel(details_input_file,
                           index_col=1, header=0, sheet_name='Metadata')

details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df = details_df.rename(columns={'Sex': 'sex', 'Age at AML onset (Years)': 'age'})

# Drop controls (samples starting with "BM")
details_df = details_df[~details_df.index.str.startswith('BM')]

# For sex, replace M with Male and F with Female
details_df['sex'] = details_df['sex'].replace({'M': 'Male', 'F': 'Female'})

# Create new col "unique_patient_id" which is the index but removing everything after dash
details_df['unique_patient_id'] = details_df.index.str.split('-').str[0]

suffixes_to_dedupe = [
    'R3',
    'R2',
    'R2-P',
    'R1',
    'R1-P',
    'PR'
]

for suffix in suffixes_to_dedupe:
    details_df = deduplicate_patient_details_using_sample_suffix(suffix, details_df)

# Report on all remaining duplicated unique patient ids
duplicate_patients = details_df[details_df.duplicated(subset='unique_patient_id', keep=False)]
if not duplicate_patients.empty:
    print("Warning: There are still duplicate unique patient ids after deduplication:")
    print(duplicate_patients)

details_df = details_df[
    ['dataset', 'sex', 'age']]

summarize_patient_details(details_df)

details_df.to_csv(details_file)

# PROCESS PHENO ########
print("Processing phenotype data...")
# Load the RNA-seq data
samplesheet_df = pd.read_csv(pheno_input_file, index_col=0)

# Read in dnam samplesheet
print("Loading samplesheet...")
dnam_samplesheet_df = pd.read_csv(dnam_raw_samplesheet_input_file, sep='\t', index_col=0)

# Map the raw samplesheet sample name to the sample names in the details file, which are the same as the pheno file
# Raw format: "HOL3_AML072-R2-P.v1_19R06C01" -> extract "AML072-R2-P"
dnam_samplesheet_df['sample'] = dnam_samplesheet_df.index.map(lambda x: x.split('_')[1].split('.')[0])

# Set new column as index
dnam_samplesheet_df = dnam_samplesheet_df.set_index('sample')

# Set 'idats_dir' column to the directory containing the idat files
dnam_samplesheet_df['idats_dir'] = dnam_data_path

# Join the two samplesheets on the sample name, keeping all samples
samplesheet_df = samplesheet_df.join(dnam_samplesheet_df[['Sentrix Barcode', 'Sample Section', 'idats_dir']], how='left')

# Rename subtype column
samplesheet_df = samplesheet_df.rename(columns={'Subtype': 'class'})

# Rename index
samplesheet_df.index.name = 'sample'

samplesheet_df = samplesheet_df.rename(columns={
    'Sentrix Barcode': 'sentrix_id',
    'Sample Section': 'sentrix_position'
})

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''

# Put multiclass items in multiclass silo before translation
multi_subtype_rows = samplesheet_df['class'].str.contains(',', na=False)
if multi_subtype_rows.any():
    samplesheet_df = add_to_silo(
        samplesheet_df,
        samplesheet_df[multi_subtype_rows].index, 'multiclass')

# Get subtype translation
tax = Taxonomy()
# Merge excluded subtypes from GEX and DNAM
all_excluded_subtypes = set(GEX_EXCLUDE_SUBTYPES) | set(DNAM_EXCLUDE_SUBTYPES)
samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'], split_delimiter=',',
    exclude_subtypes=all_excluded_subtypes)

# Silo unlabeled
samplesheet_df = silo_unlabeled(samplesheet_df)

# Silo excluded and unrecognized
samplesheet_df = silo_unrecognized(samplesheet_df)
samplesheet_df = silo_excluded(samplesheet_df)

# Put relapse samples in relapse silo.
# That is all samples that end in "-R" plus a number, or "-PR"
relapse_rows = samplesheet_df.index.str.match(r'.*-R\d*$|.*-PR$')
if relapse_rows.any():
    samplesheet_df = add_to_silo(
        samplesheet_df,
        samplesheet_df[relapse_rows].index, 'relapse')

# Silo remaining as labeled_diagnostic
samplesheet_df = silo_remaining(samplesheet_df)

# PROCESS GEX ########
print("Processing gene expression data...")

# Break up the gene expression file
gex_df = pd.read_excel(raw_data_path, index_col=0, header=8)

# Drop first 6 cols with metadata
gex_df = gex_df.iloc[:, 6:]


# Some col names need to be changed to match the metadata file
def get_meta_col(col):
    if col.startswith('BM') and not col.endswith('I'):
        return f"{col}-I"
    if col == 'AML004-Ref':
        return 'AML004-PR'
    if col == 'AML008-Ref':
        return 'AML008-PR'
    if col == 'AML016-Ref':
        return 'AML016-PR'
    if col == 'AML020-Ref':
        return 'AML020-PR'
    if col == 'AML028-Ref':
        return 'AML028-PR'
    if col == 'AML017-R2':
        return 'AML017-R1'
    if col == 'AML052-R2':
        return 'AML052-R1'
    if col == 'AML056-Ref':
        return 'AML056-R1'
    if col == 'AML093-D-P' or col == 'AML093.D.P':
        return 'AML093-PR'
    return col.replace('.', '-')


# For each remaining column:
# Create a new file with the column name as the file name
# The contents should contain the Gene and the counts for that column
# delimited by a tab. No header, no index.
for col in gex_df.columns:
    meta_col = get_meta_col(col)
    output_file = f'{preprocessed_dir}/{meta_col}.txt'
    # Set feature_counts column for this sample in samplesheet_df
    samplesheet_df.loc[meta_col, 'gex_feature_counts_file'] = output_file
    gex_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)

# Set batch column to blank string
samplesheet_df['batch'] = ''

# If any samples have duplicate sample ids, only keep first entry
samplesheet_df = samplesheet_df[~samplesheet_df.index.duplicated(keep='first')]

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)

# Print report: num total GEX, num total DNAm, num samples with GEX data only, num with DNA data only, and num with both
num_gex_total = samplesheet_df['gex_feature_counts_file'].notna().sum()
num_dnam_total = samplesheet_df['sentrix_id'].notna().sum()
num_gex_only = ((samplesheet_df['gex_feature_counts_file'].notna()) & (samplesheet_df['sentrix_id'].isna())).sum()
num_dnam_only = ((samplesheet_df['gex_feature_counts_file'].isna()) & (samplesheet_df['sentrix_id'].notna())).sum()
num_both = ((samplesheet_df['gex_feature_counts_file'].notna()) & (samplesheet_df['sentrix_id'].notna())).sum()
print(f"Total samples with GEX data: {num_gex_total}")
print(f"Total samples with DNAm data: {num_dnam_total}")
print(f"Number of samples with GEX data only: {num_gex_only}")
print(f"Number of samples with DNAm data only: {num_dnam_only}")
print(f"Number of samples with both GEX and DNAm data: {num_both}")
