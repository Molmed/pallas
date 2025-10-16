import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR, GEX_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_remaining, \
    silo_unrecognized, silo_excluded

dataset_name = 'stratmann2022'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/AML_RNAseq_Raw_Counts.xlsx'
pheno_input_file = f'{data_path}/stratmann2022_pheno.csv'
details_input_file = f'{data_path}/AML_RNAseq_Metadata_Holmfeldt_220404.xlsx'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'
details_file = \
    f'{processed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

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

details_df = details_df[
    ['dataset', 'sex', 'age']]

details_df.to_csv(details_file)

# PROCESS PHENO ########
print("Processing phenotype data...")
# Load the data
samplesheet_df = pd.read_csv(pheno_input_file, index_col=0)

# Rename subtype column
samplesheet_df = samplesheet_df.rename(columns={'Subtype': 'subtype'})

# Rename index
samplesheet_df.index.name = 'sample'

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''

# Put multiclass items in multiclass silo before translation
multi_subtype_rows = samplesheet_df['subtype'].str.contains(',', na=False)
if multi_subtype_rows.any():
    samplesheet_df = add_to_silo(
        samplesheet_df,
        samplesheet_df[multi_subtype_rows].index, 'multiclass')

# Get subtype translation
tax = Taxonomy()
samplesheet_df['subtype'] = tax.translate_subtype_column(
    samplesheet_df['subtype'], split_delimiter=',',
    exclude_subtypes=GEX_EXCLUDE_SUBTYPES)

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
        return f"{col}_I"
    if col.startswith('BM') and col.endswith('-II'):
        return col.replace('-', '_')
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
    output_file = f'{processed_dir}/{meta_col}.txt'
    # Set feature_counts column for this sample in samplesheet_df
    samplesheet_df.loc[meta_col, 'feature_counts'] = output_file
    gex_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)

# Drop all rows with missing feature_counts
samplesheet_df = samplesheet_df.dropna(subset=['feature_counts'])

# Drop all rows with missing subtype
samplesheet_df = samplesheet_df.dropna(subset=['subtype'])

# Set batch column to blank string
samplesheet_df['batch'] = ''

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
