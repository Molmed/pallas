import pandas as pd
import os
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
import dotenv

dotenv.load_dotenv()

# First, krali 2021
dataset_name = 'krali2021'
details_input_file = f'{RAW_DATA_DIR}/dnam/pheno/{dataset_name}/AML_clinical_n=142.csv'
details_file = f'{PROCESSED_DATA_DIR}/dnam/pheno/{dataset_name}/patient_details.csv'

print(f"Processing {dataset_name}...")
# Read in details file
details_df = pd.read_csv(details_input_file, index_col=0)
details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df['sex'] = details_df['sex'].replace({'male': 'Male', 'female': 'Female'})
details_df = details_df[
    ['dataset', 'sex', 'age']]
details_df.to_csv(details_file)
