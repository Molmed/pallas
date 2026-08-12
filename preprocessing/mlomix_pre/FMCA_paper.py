import pandas as pd
import os
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, PREPROCESSED_INFERENCE_DATA_DIR


dataset_name = 'FMCA_paper'

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/inference/{dataset_name}'
pheno_input_file = f'{data_path}/Copy of B_other_fmca_id_for_Mariya_20260811-JNcomments.xlsx'
krali_samplesheet_file = f'{PREPROCESSED_DATA_DIR}/Krali-2023/samplesheet.csv'
enblad_samplesheet_file = f'{PREPROCESSED_DATA_DIR}/Enblad-2025/samplesheet.csv'

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_INFERENCE_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'
missing_indices_output_file = \
    f'{preprocessed_dir}/missing_FMCA_samples_MLW.txt'

# Read the phenotype data
pheno_df = pd.read_excel(pheno_input_file, sheet_name='Sheet1', index_col=0)
target_indices = pheno_df.index.tolist()

# Read in Krali samplesheet
krali_samplesheet_df = pd.read_csv(krali_samplesheet_file, index_col=0)
enblad_samplesheet_df = pd.read_csv(enblad_samplesheet_file, index_col=0)

# Get all krali indices as list
krali_indices = krali_samplesheet_df.index.tolist()
enblad_indices = enblad_samplesheet_df.index.tolist()

# Which samples are in target_indices but not in krali or enblad samplesheets?
missing_indices = [idx for idx in target_indices if idx not in krali_indices and idx not in enblad_indices]
print(f"Missing indices: {missing_indices}")
print(f"Number of missing indices: {len(missing_indices)}")
# Write missing indices to a text file
with open(missing_indices_output_file, 'w') as f:
    for idx in missing_indices:
        f.write(f"{idx}\n")

# Which samples are in target_indices and in krali or enblad samplesheets?
found_indices = [idx for idx in target_indices if idx in krali_indices or idx in enblad_indices]
print(f"Found indices: {found_indices}")
print(f"Number of found indices: {len(found_indices)}")

# Subset the krali and enblad samplesheets to only include found indices
krali_subset_df = krali_samplesheet_df.loc[krali_samplesheet_df.index.intersection(found_indices)]
enblad_subset_df = enblad_samplesheet_df.loc[enblad_samplesheet_df.index.intersection(found_indices)]
# Merge them
merged_samplesheet_df = pd.concat([krali_subset_df, enblad_subset_df])
# Write the merged samplesheet to a CSV file
os.makedirs(preprocessed_dir, exist_ok=True)
merged_samplesheet_df.to_csv(samplesheet_output_file)
