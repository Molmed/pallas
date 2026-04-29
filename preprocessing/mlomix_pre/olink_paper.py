import runpy
import pandas as pd
import os
import dotenv
from sklearn.model_selection import train_test_split
from preprocessing.constants import PREPROCESSED_DATA_DIR, \
    MLOMIX_SAMPLESHEETS_DIR
from preprocessing.helpers import format_mlomix_columns

# Load environment variables from .env file
dotenv.load_dotenv()

datasets = [
    "SJC-DS-1009",
    "SJC-panALL"
]

samplesheet_dfs = []
# For each dataset, look in the preprocessed data dir / data set and find samplesheet.csv,
# read it in, and combine them into one big samplesheet. do it
for dataset in datasets:
    samplesheet_path = os.path.join(PREPROCESSED_DATA_DIR, dataset, 'samplesheet.csv')
    if os.path.exists(samplesheet_path):
        df = pd.read_csv(samplesheet_path, index_col=0)
        samplesheet_dfs.append(df)
    else:
        print(f"Warning: samplesheet.csv not found for dataset {dataset} at path {samplesheet_path}")

combined_samplesheet_df = pd.concat(samplesheet_dfs)
combined_samplesheet_df = combined_samplesheet_df.sort_index()

# Get all rows where class is either "T-ALL", "B-ALL, high hyperdiploid",
# "B-ALL, ETV6::RUNX1" or starts with "AML"
valid_classes = [
    "T-ALL",
    "B-ALL, high hyperdiploid",
    "B-ALL, ETV6::RUNX1"
]
aml_mask = combined_samplesheet_df['class'].str.startswith('AML')
valid_mask = combined_samplesheet_df['class'].isin(valid_classes) | aml_mask
# Keep all rows with valid mask
combined_samplesheet_df = combined_samplesheet_df[valid_mask]

# Save to MLOMIX_SAMPLESHEETS_DIR as samplesheet_olink.csv
output_path = os.path.join(MLOMIX_SAMPLESHEETS_DIR, 'samplesheet_olink.csv')
combined_samplesheet_df.to_csv(output_path)
print(f"Combined samplesheet saved to {output_path}")
