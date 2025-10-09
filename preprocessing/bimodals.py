import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import silo_unlabeled, \
    silo_remaining, silo_unrecognized, silo_excluded, add_to_silo
import dotenv

dotenv.load_dotenv()

print("Processing GDC/Triche bimodals...")

# INPUT ########
dnam_filtered_pheno_file = f'{RAW_DATA_DIR}/dnam/pheno/PALLAS_FINAL_pheno_combined_n=2479.csv'
triche_meta_file = f'{RAW_DATA_DIR}/dnam/pheno/GSE190931/GSE190931_series_matrix.txt'
gdc_meta_file = f'{RAW_DATA_DIR}/GDC_AML/gdc_sample_sheet.aml.gex.tsv'

# OUTPUT ########
output_dir = f'{PROCESSED_DATA_DIR}/bimodals'
bimodal_file = f'{output_dir}/bimodals.csv'

# Read in dnam filtered pheno
dnam_pheno = pd.read_csv(dnam_filtered_pheno_file, index_col=0)
dnam_pheno = dnam_pheno[dnam_pheno['dataset'] == 'GSE190931']
dnam_filtered_triche_samples = dnam_pheno.index.tolist()

# Format triche meta
with open(triche_meta_file, "r") as file:
    for line in file:
        # Extract !Sample_title
        if line.startswith("!Sample_title"):
            sample_titles = [entry.strip('"') for entry in line.strip().split("\t")[1:]]  # Skip the key and get the values

        # Extract !Sample_geo_accession
        elif line.startswith('!Sample_geo_accession'):
            sample_geo_accessions = [entry.strip('"') for entry in line.strip().split("\t")[1:]]  # Skip the key and get the values

        # Extract !Sample_geo_accession
        elif line.startswith('!Sample_description'):
            sample_description = [entry.strip('"') for entry in line.strip().split("\t")[1:]]  # Skip the key and get the va

        # Extract !Sample_characteristics_ch1
        elif line.startswith("!Sample_characteristics_ch1"):
            sample_characteristics = [entry.strip('"') for entry in line.strip().split("\t")[1:]] # Skip the key and get the values

# Combine the extracted information into a pandas DataFrame
data = {
    "target_sample_id": sample_description,
    "geo_sample_id": sample_geo_accessions,
    "pheno": sample_characteristics,
}
# set first col as index
triche_df = pd.DataFrame(data)
triche_df.set_index("target_sample_id", inplace=True)

# Get only items where geo_sample_id is in dnam_filtered_triche_samples
triche_df = triche_df[triche_df['geo_sample_id'].isin(dnam_filtered_triche_samples)]

# Set target_case_id to the target_sample_id before the underscore
triche_df['target_case_id'] = triche_df.index.str.split('_').str[0]

# If the suffix is Dx set new column Tumor Descriptor to Primary
triche_df['timepoint'] = triche_df.index.str.split('_').str[1]
triche_df['timepoint'] = triche_df['timepoint'].replace({'Dx': 'Primary', 'REL': 'Relapse'})

print(triche_df)

# Read in gdc file
gdc_meta = pd.read_csv(gdc_meta_file, sep='\t')

# Get all items where 'Project ID' is 'TARGET-AML'
gdc_meta = gdc_meta[gdc_meta['Project ID'] == 'TARGET-AML']

# Get all items where 'Tumor Descriptor' is 'Primary'
gdc_meta = gdc_meta[gdc_meta['Tumor Descriptor'] == 'Primary']

# Get all items where 'Tissue Type' is 'Tumor'
gdc_meta = gdc_meta[gdc_meta['Tissue Type'] == 'Tumor']

# Set target_case_id to the string after the last dash of 'Case ID'
gdc_meta['target_case_id'] = gdc_meta['Case ID'].str.split('-').str[-1]
print(gdc_meta)

# Create a new df that joins the two dfs as follows:
# target_case_id from triche_df to target_case_id from gdc_meta
# timepoint from triche_df to Tumor Descriptor from gdc_meta
bimodal_pheno = pd.merge(triche_df, gdc_meta, on='target_case_id', suffixes=('_triche', '_gdc'))
bimodal_pheno = bimodal_pheno[bimodal_pheno['timepoint'] == bimodal_pheno['Tumor Descriptor']]
print(bimodal_pheno)

# Save the bimodal pheno
os.makedirs(output_dir, exist_ok=True)
bimodal_pheno.to_csv(bimodal_file)
