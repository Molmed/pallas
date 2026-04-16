import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, \
        GEX_EXCLUDE_SUBTYPES, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_excluded, silo_remaining, \
        format_mlomix_columns, summarize_patient_details, \
            process_dnam_file, process_gdc_gex, process_gdc_dnam, \
                summarize_modalities

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/TARGET-AML-TCGA-LAML'
geo_pheno_df_files = [
    f'{data_path}/GSE124413_series_matrix.txt',
    f'{data_path}/GSE190931_series_matrix.txt',
]
gex_dir = f'{data_path}/feature_counts'
dnam_dir = f'{data_path}/betas'
gex_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.gex.tsv'
dnam_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.dnam.tsv'
dnam_subtypes_file = f'{data_path}/TARGET-AML-TCGA-LAML.subtypes.dnam.tsv'
clinical_file = f'{data_path}/TARGET-AML.clinical.tsv'

# OUTPUT FILES ########
dataset_name = 'TARGET-AML'
print(f"Processing {dataset_name}...")
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
gex_output_dir = f'{preprocessed_dir}/gex'
dnam_output_dir = f'{preprocessed_dir}/dnam'

samplesheet_df, details_df = process_gdc_gex(
    dataset_name,
    clinical_file,
    gex_samples_file,
    gex_dir,
    gex_output_dir)

# Save files to output dir
samplesheet_output_file = f'{preprocessed_dir}/samplesheet.csv'
details_output_file = f'{preprocessed_dir}/patient_details.csv'
samplesheet_df.to_csv(samplesheet_output_file, index=True)
details_df.to_csv(details_output_file, index=True)


exit()

def process_geo_pheno_file(file_path):
    samples = pd.DataFrame(columns=['id', 'target_sample_id', 'class'])
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

    # Perform secondary processing.
    # First process "title" column
    # For each row, if title starts with "gDNA from normal human bone marrow"
    # samples['is_normal'] = samples['title'].str.startswith("gDNA from normal human bone marrow")

    return samples


# Process each pheno file then merge
pheno_dfs = []
for file in geo_pheno_df_files:
    df = process_geo_pheno_file(file)
    pheno_dfs.append(df)
# Merge on target_sample_id
pheno_df = pd.concat(pheno_dfs, axis=0)

print(pheno_df)

# dump to csv
tmp_output_file = f'{preprocessed_dir}/dnam_samplesheet.csv'
pheno_df.to_csv(tmp_output_file, index=True)
print(pheno_df)
