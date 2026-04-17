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
                summarize_modalities, concatenate_detail_dfs, concatenate_samplesheets

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/TARGET-AML-TCGA-LAML'
geo_pheno_df_files = [
    f'{data_path}/GSE124413_series_matrix.txt',
    # f'{data_path}/GSE190931_series_matrix.txt',
]
gex_dir = f'{data_path}/feature_counts'
gdc_dnam_dir = f'{data_path}/gdc_betas'
idats_dir = f'{data_path}/GSE124413_RAW'
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

# First grab all the DNAm samples from GDC
# gex_samplesheet_df, gex_details_df = process_gdc_gex(
#     dataset_name,
#     clinical_file,
#     gex_samples_file,
#     gex_dir,
#     gex_output_dir)

# dnam_samplesheet_df, dnam_details_df = process_gdc_dnam(
#     dataset_name,
#     clinical_file,
#     dnam_samples_file,
#     dnam_subtypes_file,
#     gdc_dnam_dir,
#     dnam_output_dir)

# concatenate_detail_dfs(
#     gex_details_df, dnam_details_df, f'{preprocessed_dir}/patient_details.csv')

# concatenate_samplesheets(
#     gex_samplesheet_df, dnam_samplesheet_df, f'{preprocessed_dir}/samplesheet.csv')


def process_geo_pheno_file(file_path):
    samples = pd.DataFrame(columns=['id',
                                    'target_case_id',
                                    'class',
                                    'geo_dataset',
                                    'idats_basename'])
    geo_dataset = None
    with open(file_path, "r") as file:
        for line in file:
            # Tokenize line by tab and remove quotes from each token
            tokens = [token.strip('"').strip('"\n') for token in line.split("\t")]
            key = tokens.pop(0)

            if key == 'ID_REF':
                samples['id'] = tokens
            elif key == '!Series_geo_accession':
                accession = tokens[0]
                # If not empty, set geo_dataset to the accession
                if accession:
                    geo_dataset = accession
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

    # Set geo_dataset column to the value of geo_dataset variable
    samples['geo_dataset'] = geo_dataset

    # Perform secondary processing.
    # First process "title" column
    # For each row, check if title starts with "gDNA from normal human bone marrow"
    # If so, set class to "Control"
    samples['class'] = samples['title'].apply(
        lambda x: 'Control' if x.startswith('gDNA from normal human bone marrow') else '')

    # If title starts with "gDNA from subject", set the last
    # word of the title as the target_case_id
    samples['target_case_id'] = samples['title'].apply(
        lambda x: x.split()[-1] if x.startswith('gDNA from subject') else None)

    # If "secondary_name" is set, set its value before the first underscore as the target_case_id
    if 'secondary_name' in samples.columns:
        samples['target_case_id'] = samples.apply(
            lambda row: row['secondary_name'].split('_')[0] if pd.notna(row['secondary_name']) else row['target_case_id'],
            axis=1)

    # Process DNAm data differently based on accession
    raw_idats_dir = f'{data_path}/GSE124413_RAW'
    processed_idats_dir = f'{dnam_output_dir}/GSE124413'
    # Make if it doesn't exist
    os.makedirs(processed_idats_dir, exist_ok=True)

    # Loop through each row and assign dnam file values accordingly
    for index, row in samples.iterrows():
        # get geo_dataset value for the row
        row_geo_dataset = row['geo_dataset']
        print(f"Processing row {index} with geo_dataset {row_geo_dataset}...")
        if row_geo_dataset == 'GSE124413':
            # Get file 1
            file_names = [row.get('file_1', None), row.get('file_2', None)]
            for file_name in file_names:
                # file_name is just the basename, extract end of url
                file_name = os.path.basename(file_name) if pd.notna(file_name) else None
                print(f"Checking file {file_name}...")
                file_path = f'{raw_idats_dir}/{file_name}'
                if pd.notna(file_name) and os.path.exists(file_path):
                    # ungzip file to processed_idats_dir
                    # have the same name but strip the sample name
                    # which is before the first underscore
                    output_file_path = f"{processed_idats_dir}/{file_name.split('_', 1)[1].replace('.gz', '')}"
                    if not os.path.exists(output_file_path):
                        print(f"Processing {file_path} to {output_file_path}...")
                        os.system(f"gunzip -c {file_path} > {output_file_path}")
                    samples.at[index, 'idats_basename'] = processed_idats_dir
                    # Filename is of the format GSM3532234_200991620105_R05C01_Grn.idat.gz
                    # Extract the sentrix id and position from the filename
                    sentrix_id = file_name.split('_')[1]
                    sentrix_position = file_name.split('_')[2].split('.')[0]
                    samples.at[index, 'sentrix_id'] = sentrix_id
                    samples.at[index, 'sentrix_position'] = sentrix_position

    # if geo_dataset == 'GSE124413':

        # For GSE124413, we need to point to the raw IDAT files using the file_1 and file_2 columns
        # idats_dir = f'{data_path}/GSE124413_RAW'
        # # Set 'idats_basename' to file_1 minus the "_Grn.idat"
        # samples['idats_basename'] = samples['file_1'].apply(
        #     lambda x: f"{idats_dir}/{x.replace('_Grn.idat', '')}" if pd.notna(x) else None)

    # elif geo_dataset == 'GSE190931':
    #     # For GSE190931, we need to construct the path to the DNAm data file using the id column
    #     samples['dnam_file'] = samples['id'].apply(
    #         lambda x: f'{dnam_dir}/{x}_beta_values.tsv')

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
