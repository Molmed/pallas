import runpy
import pandas as pd
import os
from preprocessing.constants import PROCESSED_DATA_DIR

modules = [
    "preprocessing.gexml_pre.diedrich",
    "preprocessing.gexml_pre.heinaniemi",
    "preprocessing.gexml_pre.krali",
    "preprocessing.gexml_pre.stjude_cloud",
    "preprocessing.gexml_pre.stjude_panALL",
    "preprocessing.gexml_pre.tran",
    "preprocessing.gexml_pre.lund",
]

# for mod in modules:
#     print(f"Running {mod}...")
#     runpy.run_module(mod, run_name="__main__")

# Find all samplesheet.csv files and combine them
print("Combining all samplesheets...\n")
samplesheet_dfs = []
for root, dirs, files in os.walk(PROCESSED_DATA_DIR):
    for file in files:
        if file == 'samplesheet.csv':
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, index_col=0)
            samplesheet_dfs.append(df)
combined_samplesheet_df = pd.concat(samplesheet_dfs)
combined_samplesheet_df = combined_samplesheet_df.sort_index()

# Get total number of samples
total_samples = combined_samplesheet_df.shape[0]
print(f"Total number of samples: {total_samples}\n")

# Display stats by dataset
print(combined_samplesheet_df['dataset'].value_counts())

print("\n")

# Display stats by data_silo
print(combined_samplesheet_df['data_silo'].value_counts())

print("\n")

# Display by data_silo, labeled_diagnostic vs unlabeled, and then combine the others under "misc"
combined_samplesheet_df['data_silo_detailed'] = combined_samplesheet_df['data_silo'].apply(
    lambda x: x if x in ['labeled_diagnostic', 'unlabeled'] else 'misc'
)
print(combined_samplesheet_df['data_silo_detailed'].value_counts())

print("\n")

print(combined_samplesheet_df)
