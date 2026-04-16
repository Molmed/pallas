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


# In this file, we parse all the GDC data (TARGET-AML and TCGA-LAML)
# plus the GEO data (GSE124413 and GSE190931) for TARGET-AML

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/TARGET-AML-TCGA-LAML'
# clinical_file = f'{data_path}/TARGET-AML-TCGA-LAML.clinical.tsv'
geo_pheno_df_files = [
    f'{data_path}/GSE124413_series_matrix.txt',
    f'{data_path}/GSE190931_series_matrix.txt',
]
gex_dir = f'{data_path}/feature_counts'
dnam_dir = f'{data_path}/betas'
gex_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.gex.tsv'
dnam_samples_file = f'{data_path}/TARGET-AML-TCGA-LAML.samples.dnam.tsv'
dnam_subtypes_file = f'{data_path}/TARGET-AML-TCGA-LAML.subtypes.dnam.tsv'

# First do TCGA, which is simpler because there is no GEO data
dataset_name = 'TCGA-LAML'
print(f"Processing {dataset_name}...")
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
gex_output_dir = f'{preprocessed_dir}/gex'
dnam_output_dir = f'{preprocessed_dir}/dnam'
clinical_file = f'{data_path}/TCGA-LAML.clinical.tsv'

gex_samplesheet_df, gex_details_df = process_gdc_gex(dataset_name,
                                             clinical_file,
                                             gex_samples_file,
                                             gex_dir,
                                             gex_output_dir)
dnam_samplesheet_df, dnam_details_df = process_gdc_dnam(dataset_name,
                                                        clinical_file,
                                             dnam_samples_file,
                                             dnam_subtypes_file,
                                             dnam_dir,
                                             dnam_output_dir)

# Concatenate the details dataframes and save
details_df = pd.concat([gex_details_df, dnam_details_df], axis=0)
# Remove duplicates
details_df = details_df[~details_df.index.duplicated(keep='first')]
summarize_patient_details(details_df)
details_output_file = f'{preprocessed_dir}/patient_details.csv'
details_df.to_csv(details_output_file, index=True)

# Indices are of the format TCGA-AB-2803-03A
# Create new column "target_case_id" which removes the last segment, so TCGA-AB-2803
gex_samplesheet_df['target_case_id'] = gex_samplesheet_df.index.str.rsplit('-', n=1).str[0]
dnam_samplesheet_df['target_case_id'] = dnam_samplesheet_df.index.str.rsplit('-', n=1).str[0]

# Make target_case_id the index
gex_samplesheet_df = gex_samplesheet_df.set_index('target_case_id')
dnam_samplesheet_df = dnam_samplesheet_df.set_index('target_case_id')

g = gex_samplesheet_df.replace('', pd.NA)
d = dnam_samplesheet_df.replace('', pd.NA)
samplesheet_df = g.combine_first(d)
samplesheet_output_file = f'{preprocessed_dir}/samplesheet.csv'
samplesheet_df.to_csv(samplesheet_output_file, index=True)
summarize_modalities(samplesheet_df)


exit()



# # Save files to output dir
# samplesheet_output_file = f'{preprocessed_dir}/samplesheet.csv'
# details_output_file = f'{preprocessed_dir}/patient_details.csv'
# samplesheet_df.to_csv(samplesheet_output_file, index=True)
# details_df.to_csv(details_output_file, index=True)

# Now TARGET-AML
dataset_name = 'TARGET-AML'
print(f"Processing {dataset_name}...")
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'

# samplesheet_df, details_df = process_gdc_gex(dataset_name,
#                                              gex_clinical_file,
#                                              gex_samples_file,
#                                              gex_dir,
#                                              preprocessed_dir)
# # Save files to output dir
# samplesheet_output_file = f'{preprocessed_dir}/samplesheet.csv'
# details_output_file = f'{preprocessed_dir}/patient_details.csv'
# samplesheet_df.to_csv(samplesheet_output_file, index=True)
# details_df.to_csv(details_output_file, index=True)


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
