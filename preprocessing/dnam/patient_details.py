import pandas as pd
import os
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
import dotenv

dotenv.load_dotenv()

dataset_name ='enblad2025'
details_input_file = f'{RAW_DATA_DIR}/dnam/pheno/{dataset_name}/New_DNAm_n_55_with_all_IDs.xlsx'
details_file = f'{PROCESSED_DATA_DIR}/dnam/pheno/{dataset_name}/patient_details.csv'

# index col is public_id
details_df = pd.read_excel(details_input_file, index_col='public_id')
details_df.rename(columns={'Sex': 'sex', 'Age': 'age'}, inplace=True)
details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df['sex'] = details_df['sex'].replace({'male': 'Male', 'female': 'Female'})
details_df = details_df[
    ['dataset', 'sex', 'age']]
details_df.to_csv(details_file)

# Krali 2021
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

# Bertrums
dataset_name = 'GSE190931'
details_input_file = f'{RAW_DATA_DIR}/dnam/pheno/{dataset_name}/GSE190931_series_matrix.txt'
details_file = f'{PROCESSED_DATA_DIR}/dnam/pheno/{dataset_name}/patient_details.csv'

# Open and parse the file
with open(details_input_file, "r") as file:
    for line in file:
        # Extract !Sample_title
        if line.startswith("!Sample_title"):
            sample_titles = [entry.strip('"') for entry in line.strip().split("\t")[1:]]  # Skip the key and get the values

        # Extract !Sample_geo_accession
        elif line.startswith('!Sample_geo_accession'):
            sample_geo_accessions = [entry.strip('"') for entry in line.strip().split("\t")[1:]]  # Skip the key and get the values

        # Extract !Sample_geo_accession
        elif line.startswith('!Sample_description'):
            sample_description = [entry.strip('"') for entry in line.strip().split("\t")[1:]]  # Skip the key and get the va


        # Extract !Sample_characteristics_ch1
        elif line.startswith("!Sample_characteristics_ch1"):
            entries = [entry.strip('"') for entry in line.strip().split("\t")[1:]] # Skip the key and get the values
            if "fusion" in line.lower():
                fusions = entries
            elif "sex" in line.lower():
                sexes = entries
            elif "age" in line.lower():
                ages = entries

# Combine the extracted information into a pandas DataFrame
data = {
    "Sample_title": sample_titles,
    "Sample_geo_accession": sample_geo_accessions,
    "Sample_description": sample_description,
    "Fusion": fusions if fusions else [None] * len(sample_titles),  # Handle missing data
    "Sex": sexes if sexes else [None] * len(sample_titles),         # Handle missing data
    "Age": ages if ages else [None] * len(sample_titles)            # Handle missing data
}

details_df = pd.DataFrame(data)
details_df = details_df.rename(columns={'Sample_geo_accession' : 'id', 'Fusion': 'subtype'})
details_df['subtype']= details_df['subtype'].str.replace('fusion: ', '', regex = False)
details_df['Sex']= details_df['Sex'].str.replace('Sex: ', '', regex = False)
details_df['Age']= details_df['Age'].str.replace('age (years):', '', regex = False)

# Rename Sex and Age columns
details_df.rename(columns={'Sex': 'sex', 'Age': 'age'}, inplace=True)

# Strip age of whitespace and convert to numeric, coercing errors to NaN
details_df['age'] = pd.to_numeric(details_df['age'].str.strip(), errors='coerce')

# Set index to id column
details_df.set_index('id', inplace=True)
details_df.index.name = 'sample'
details_df['dataset'] = dataset_name
details_df = details_df[
    ['dataset', 'sex', 'age']]
details_df.to_csv(details_file)



# print(f"Processing {dataset_name}...")
# # Read in details file
# details_df = pd.read_csv(details_input_file, index_col=0)
# details_df.index.name = 'sample'
# details_df['dataset'] = dataset_name
# details_df['sex'] = details_df['sex'].replace({'male': 'Male', 'female': 'Female'})
# details_df = details_df[
#     ['dataset', 'sex', 'age']]
# details_df.to_csv(details_file)
