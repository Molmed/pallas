import pandas as pd
import os
import gzip
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocessing.helpers import add_to_silo, \
    silo_unlabeled, silo_remaining, \
    silo_unrecognized, silo_excluded

dataset_name = 'lund_hypodiploidy'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{processed_dir}/samplesheet.csv'

# Make processed dir if not exists
os.makedirs(processed_dir, exist_ok=True)

sample_names = []
# Loop through each file, extract it into a txt file and remove the header
for root, dirs, files in os.walk(raw_data_path):
    for filename in files:
        if filename.endswith(".abundance.gz"):
            sample_name = filename[:-len(".abundance.gz")]
            sample_names.append(sample_name)

            input_file = os.path.join(root, filename)
            output_file = os.path.join(processed_dir, f"{sample_name}.txt")

            with gzip.open(input_file, 'rt', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
                # Skip the header line
                next(infile)
                for line in infile:
                    outfile.write(line)

samplesheet_df = pd.DataFrame(index=sample_names)
samplesheet_df.index.name = 'sample'
samplesheet_df['subtype'] = 'hypodiploid'
samplesheet_df['batch'] = ''
samplesheet_df['dataset'] = dataset_name
samplesheet_df['data_silo'] = ''
samplesheet_df['feature_counts'] = samplesheet_df.index.map(
    lambda x: os.path.join(processed_dir, f"{x}.txt"))

# Order columns as follows: sample,feature_counts,dataset,batch,subtype
samplesheet_df = samplesheet_df[
    ['feature_counts', 'dataset', 'batch', 'subtype', 'data_silo']]

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)
