import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocessing.helpers import silo_unlabeled, silo_remaining

dataset_name = 'GSE228632'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/GSE228632_RNAseq_read_counts.txt'
pheno_input_file = f'{data_path}/GSE228632_meta.csv'
details_input_file = f'{data_path}/Clinical_data_ALLIUM_FIN_cohort.xlsx'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'
details_file = \
    f'{processed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# PROCESS PHENO ########
print("Processing phenotype data...")
# Load the data
samplesheet_df = pd.read_csv(pheno_input_file, index_col=0)

# Drop batch column and rename the rest
samplesheet_df = samplesheet_df.drop(columns=['batch', 'Subtype'])
samplesheet_df = samplesheet_df.rename(columns={'Subtype_updated': 'subtype',
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
samplesheet_df['subtype'] = tax.translate_subtype_column(samplesheet_df['subtype'])

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
    output_file = f'{processed_dir}/{col}.txt'
    # Set feature_counts column for this sample in samplesheet_df
    samplesheet_df.loc[col, 'feature_counts'] = output_file
    gex_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)

# Remove all rows from spreadsheet without feature_counts
samplesheet_df = samplesheet_df[samplesheet_df['feature_counts'].notna()]

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)

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
details_df.to_csv(details_file)
