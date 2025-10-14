import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocessing.helpers import \
    add_to_silo, silo_remaining

dataset_name = 'GDC_AML'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/feature_counts'
pheno_input_file = f'{data_path}/gdc_gex_aml.subtyped.tsv'
details_input_file = f'{data_path}/gdc_gex_aml.clinical.tsv'
bimodals_input_file = f'{PROCESSED_DATA_DIR}/bimodals/gdc_bimodals.csv'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'
details_file = \
    f'{processed_dir}/patient_details.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

# We only want the subtyped data, use the pre-processed subtypes file as guide
# create multiindex on Sample ID and Replicate
# Load the data
pheno_df = pd.read_csv(pheno_input_file, sep='\t', index_col='Unique Sample ID')

# For now, let's only keep Primary non-replicate samples
pheno_df = pheno_df[(pheno_df['Tumor Descriptor'] == 'Primary') &
                    (pheno_df['Replicate'] == False)]

# Read in details file
# Index should be "cases.submitter_id"
details_df = pd.read_csv(details_input_file,
                         header=0,
                         sep='\t',
                         index_col='cases.submitter_id')

# Only keep cases that are in pheno_df 'Case ID' column
details_df = details_df[details_df.index.isin(pheno_df['Case ID'])]

# Keep the following cols
# diagnoses.age_at_diagnosis
# demographic.gender
details_df = details_df[['diagnoses.age_at_diagnosis', 'demographic.gender']]


details_df.index.name = 'case_id'
details_df['dataset'] = dataset_name
details_df = details_df.rename(
    columns={'demographic.gender': 'sex',
             'diagnoses.age_at_diagnosis': 'age'})

# Convert age to numeric, coerce errors to NaN
details_df['age'] = pd.to_numeric(details_df['age'], errors='coerce')

# For age, convert days to years and round to 1 decimal place
details_df['age'] = (details_df['age'] / 365.25).round(1)

# For sex, replace M with Male and F with Female
details_df['sex'] = details_df['sex'].replace({'male': 'Male', 'female': 'Female'})

details_df = details_df[
    ['dataset', 'sex', 'age']]

# Drop duplicates
details_df = details_df[~details_df.index.duplicated(keep='first')]

details_df.to_csv(details_file)

# PROCESS PHENO ########
print("Processing phenotype data...")
samplesheet_df = pheno_df

# Set Case ID as the index
samplesheet_df = samplesheet_df.set_index('Case ID', drop=True)

# Rename index
samplesheet_df.index.name = 'sample'

# Get subtype translation
tax = Taxonomy()
samplesheet_df['subtype'] = tax.translate_subtype_column(
    samplesheet_df['subtype'],
    split_delimiter=',')

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Set data silo to empty string
samplesheet_df['data_silo'] = ''

# Put multiclass items in multiclass silo
multi_subtype_rows = samplesheet_df['subtype'].str.contains(';', na=False)
if multi_subtype_rows.any():
    samplesheet_df = add_to_silo(
        samplesheet_df,
        samplesheet_df[multi_subtype_rows].index, 'multiclass')

# All that are bimodal should be siloed as 'labeled_diagnostic_bimodal'
bimodals_df = pd.read_csv(bimodals_input_file, index_col='Case ID')
bimodal_samples = bimodals_df.index.tolist()
bimodal_rows = samplesheet_df.index.isin(bimodal_samples)
if bimodal_rows.any():
    samplesheet_df = add_to_silo(
        samplesheet_df,
        samplesheet_df[bimodal_rows].index, 'labeled_diagnostic_bimodal')

# Silo remaining as labeled_diagnostic
samplesheet_df = silo_remaining(samplesheet_df)

# Convert index to string
samplesheet_df.index = samplesheet_df.index.astype(str)

# Rename 'Project Id' to 'batch'
samplesheet_df = samplesheet_df.rename(columns={'Project Id': 'batch'})

# PROCESS GEX ########
# For each row in samplesheet_df, get the file_name
# Open the filename and extract the gene_name and unstranded columns
print("Processing gene expression data...")

for idx, row in samplesheet_df.iterrows():
    # Open the filename and extract the gene_name and unstranded columns
    gex_file = f"{raw_data_path}/{row['File Name']}"

    try:
        gex_df = pd.read_csv(gex_file, sep='\t', header=1, index_col=1)
        # Remove the first four rows which are summary stats
        gex_df = gex_df.iloc[5:, :]

        # Just keep unstranded column
        gex_df = gex_df[['unstranded']]

        # Save to a new file
        output_file = f'{processed_dir}/{idx}.txt'

        row['feature_counts'] = output_file
        gex_df.to_csv(output_file, sep='\t', header=False, index=True)
    except Exception as e:
        print(f"Error reading {gex_file} for {idx}: {e}")
        continue

# Remove all rows from spreadsheet without feature_counts
samplesheet_df = samplesheet_df[samplesheet_df['feature_counts'].notna()]

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
