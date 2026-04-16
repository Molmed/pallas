import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, \
        GEX_EXCLUDE_SUBTYPES, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_excluded, silo_remaining, \
        format_mlomix_columns, summarize_patient_details, \
            process_dnam_file, process_gdc_gex


# In this file, we parse all the GDC data (TARGET-AML and TCGA-LAML)
# plus the GEO data (GSE124413 and GSE190931) for TARGET-AML

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/TARGET-AML-TCGA'
geo_pheno_df_files = [
    f'{data_path}/GSE124413_series_matrix.txt',
    f'{data_path}/GSE190931_series_matrix.txt',
]
gex_dir = f'{data_path}/feature_counts'
gex_clinical_file = f'{data_path}/TARGET-AML-TCGA.clinical.tsv'
gex_subtypes_file = f'{data_path}/TARGET-AML-TCGA.subtypes.tsv'

# First do TCGA, which is simpler
dataset_name = 'TCGA-LAML'
print(f"Processing {dataset_name}...")
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'

samplesheet_df, details_df = process_gdc_gex(dataset_name,
                                             gex_clinical_file,
                                             gex_subtypes_file,
                                             gex_dir,
                                             preprocessed_dir)
# Save files to output dir
samplesheet_output_file = f'{preprocessed_dir}/samplesheet.csv'
details_output_file = f'{preprocessed_dir}/patient_details.csv'
samplesheet_df.to_csv(samplesheet_output_file, index=True)
details_df.to_csv(details_output_file, index=True)

exit()

dataset_name = 'TARGET-AML'


def read_geo_pheno_file(file_path):
    samples = pd.DataFrame(columns=['id'])
    with open(file_path, "r") as file:
        for line in file:
            # Tokenize line by tab and remove quotes from each token
            tokens = [token.strip('"').strip('"\n') for token in line.split("\t")]
            key = tokens.pop(0)

            if key == 'ID_REF':
                samples['id'] = tokens
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
    return samples


# Process each pheno file then merge
pheno_dfs = []
for file in geo_pheno_df_files:
    df = read_geo_pheno_file(file)
    pheno_dfs.append(df)
# Merge on target_sample_id
pheno_df = pd.concat(pheno_dfs, axis=0)

# dump to csv
pheno_df.to_csv(samplesheet_output_file, index=True)
print(pheno_df)
