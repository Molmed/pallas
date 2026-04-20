import pandas as pd
import os
import gzip
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from preprocessing.helpers import format_mlomix_columns

dataset_name = 'Paulsson-2023'

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}'

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'

# Make preprocessed dir if not exists
os.makedirs(preprocessed_dir, exist_ok=True)

sample_names = []
# Loop through each file, extract it into a txt file and remove the header
for root, dirs, files in os.walk(raw_data_path):
    for filename in files:
        if filename.endswith(".abundance.gz"):
            sample_name = filename[:-len(".abundance.gz")]
            sample_names.append(sample_name)

            input_file = os.path.join(root, filename)
            output_file = os.path.join(preprocessed_dir, f"{sample_name}.txt")

            with gzip.open(input_file, 'rt', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
                # Skip the header line
                next(infile)
                for line in infile:
                    outfile.write(line)

samplesheet_df = pd.DataFrame(index=sample_names)
samplesheet_df.index.name = 'sample'
samplesheet_df['class'] = 'hypodiploid'
samplesheet_df['batch'] = ''
samplesheet_df['dataset'] = dataset_name
samplesheet_df['data_silo'] = 'labeled_diagnostic'
samplesheet_df['gex_feature_counts_file'] = samplesheet_df.index.map(
    lambda x: os.path.join(preprocessed_dir, f"{x}.txt"))

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
