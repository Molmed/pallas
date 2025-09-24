import pandas as pd
import os
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR
import dotenv

dotenv.load_dotenv()

dataset_name = 'krali2023'
details_input_file = f'{RAW_DATA_DIR}/dnam/pheno/{dataset_name}/Patients_clinicallist_FINAL.csv'
details_file = f'{PROCESSED_DATA_DIR}/dnam/pheno/{dataset_name}/patient_details.csv'
main_pheno_file = f'{RAW_DATA_DIR}/dnam/pheno/PALLAS_FINAL_pheno_combined_n=2479.csv'

# For this dataset, read in main pheno file to get all the sample ids, then see what we can find
main_pheno_df = pd.read_csv(main_pheno_file, index_col=0)

# Get all samples whose "dataset" column starts with "Krali2023"
krali_samples = main_pheno_df[main_pheno_df['dataset'].str.startswith('Krali2023')]

# Copy df
krali_df = krali_samples.copy()
krali_df['dataset'] = dataset_name
krali_df.index.name = 'sample'

# join on details file to see what we have
details_df = pd.read_csv(details_input_file, index_col=0)
details_df.index.name = 'sample'

# Merge details_df into krali_df based on index
details_df = krali_df.merge(details_df, left_index=True, right_index=True, how='left', suffixes=('', '_details'))

# Rename columns
details_df.rename(columns={'Age': 'age', 'Sex': 'sex'}, inplace=True)

# Print all rows where sex is nan and sample starts with "ALL"
missing_sex = details_df[details_df['sex'].isna() & details_df.index.str.startswith("ALL")]

# In these, create a new col original_sample which strips the final r+digit, e.g. r1, r2
missing_sex['original_sample'] = missing_sex.index.str.replace(r'r\d+$', '', regex=True)

# Get the age and sex from the original samples
missing_sex = missing_sex.merge(details_df, left_on='original_sample', right_index=True, how='left', suffixes=('', '_details'))

# Copy them back to details_df
for idx, row in missing_sex.iterrows():
    if pd.isna(details_df.at[idx, 'sex']):
        details_df.at[idx, 'sex'] = row['sex_details']
    if pd.isna(details_df.at[idx, 'age']):
        details_df.at[idx, 'age'] = row['age_details']

details_df = details_df[
    ['dataset', 'sex', 'age']]
details_df.to_csv(details_file)

dataset_name ='enblad2025'
details_input_file = f'{RAW_DATA_DIR}/dnam/pheno/{dataset_name}/New_DNAm_n_55_with_all_IDs.xlsx'
id_mapping_file = f'{RAW_DATA_DIR}/dnam/pheno/{dataset_name}/NOPHO_EPIC_n=55_subtype_info_FINAL.csv'
details_file = f'{PROCESSED_DATA_DIR}/dnam/pheno/{dataset_name}/patient_details.csv'

# For some reason the public ids are not correct on the details_input_file,
# so we need to grab them from the id_mapping_file
id_mapping_df = pd.read_csv(id_mapping_file, index_col=2)

# Set the name of the first column (not index) to public_id
id_mapping_df.rename(columns={id_mapping_df.columns[0]: 'public_id'}, inplace=True)

details_df = pd.read_excel(details_input_file, index_col=1)

# In details_df index, strip out everything after the dot
details_df.index = details_df.index.str.split('.').str[0]

# Drop public_id from details_df
details_df = details_df.drop(columns=['public_id'], errors='ignore')

# Copy public_id from id_mapping_df to details_df based on index
details_df = details_df.merge(id_mapping_df[['public_id']], left_index=True, right_index=True, how='left')

# Set public_id to index
details_df.set_index('public_id', inplace=True)

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

# Only keep samples in details_df that are in the main pheno file
main_pheno_df = pd.read_csv(main_pheno_file, index_col=0)
main_pheno_df.index.name = 'sample'
details_df = details_df[details_df.index.isin(main_pheno_df.index)]

details_df.to_csv(details_file)
