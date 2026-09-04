import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_VALIDATION_DATA_DIR, \
    GEX_EXCLUDE_SUBTYPES, MLOMIX_SAMPLESHEETS_DIR
from preprocessing.helpers import format_mlomix_columns, \
    summarize_patient_details, summarize_modalities

dataset_name = 'ClinicalGenomicsUppsala'

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/validation/{dataset_name}'
raw_data_path_1 = f'{data_path}/GeneCounts.GMS-Barn.table.csv'
raw_data_path_2 = f'{data_path}/GeneCounts2.out'
pheno_input_file = f'{data_path}/cg_phenotype.csv'

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_VALIDATION_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'

# Make processed dir if not exists
os.makedirs(preprocessed_dir, exist_ok=True)

# Read in phenotype file
samplesheet_df = pd.read_csv(pheno_input_file, index_col=0, delimiter=',')

# Get subtype translation
samplesheet_df.fillna('', inplace=True)
tax = Taxonomy()
samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'],
    exclude_subtypes=GEX_EXCLUDE_SUBTYPES)


def process_gex_file(raw_data_path):
    # PROCESS GEX ########
    # Break up the gene expression file
    gex_df = pd.read_csv(raw_data_path, index_col=0, delimiter=',')

    # For each column:
    # Create a new file with the column name as the file name
    # The contents should contain the Gene and the counts for that column
    # delimited by a tab. No header, no index.
    for col in gex_df.columns:
        output_file = f'{preprocessed_dir}/{col}.txt'
        # Set feature_counts column for this sample in samplesheet_df
        samplesheet_df.loc[col, 'gex_feature_counts_file'] = output_file
        gex_df[[col]].to_csv(output_file, sep='\t', header=False, index=True)


process_gex_file(raw_data_path_1)
process_gex_file(raw_data_path_2)

# Create a new column "pheno_id" which is the index id
# but removing everything after a dash or underscore
samplesheet_df['pheno_id'] = samplesheet_df.index.str.split('[-_]').str[0]

# For all rows with 'gex_feature_counts_file' but no class, see if there is another
# row with the same pheno_id that has a class, and if so, copy the class over
for idx, row in samplesheet_df.iterrows():
    if pd.notna(row['gex_feature_counts_file']) and pd.isna(row['class']):
        pheno_id = row['pheno_id']
        matching_rows = samplesheet_df[samplesheet_df['pheno_id'] == pheno_id]
        matching_rows_with_class = matching_rows[matching_rows['class'].notna()]
        if not matching_rows_with_class.empty:
            samplesheet_df.at[idx, 'class'] = matching_rows_with_class.iloc[0]['class']


# Remove all rows from spreadsheet without feature_counts
samplesheet_df = samplesheet_df[samplesheet_df['gex_feature_counts_file'].notna()]

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)

# Copy samplesheet to mlomix samplesheets dir
samplesheet_output_file_mlomix = \
    f'{MLOMIX_SAMPLESHEETS_DIR}/{dataset_name}.csv'
samplesheet_df.to_csv(samplesheet_output_file_mlomix)

summarize_modalities(samplesheet_df)
