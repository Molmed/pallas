import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

reports_dir = os.getenv('REPORTS_DIR')
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

processed_data_dir = os.getenv('PROCESSED_DATA_DIR')

# Find all patient_details.csv files in processed_data_dir
details_files = []
for root, dirs, files in os.walk(processed_data_dir):
    for file in files:
        if file == 'patient_details.csv':
            details_files.append(os.path.join(root, file))

details_dfs = []
for details_file in details_files:
    # if path contains "dnam" modality is "DNAm", else "GEX"
    if 'dnam' in details_file:
        modality = 'DNAm'
    else:
        modality = 'GEX'

    # Read in details_file
    details = pd.read_csv(details_file, index_col=0)
    details['modality'] = modality
    details_dfs.append(details)

# Concatenate all details_dfs
all_details_df = pd.concat(details_dfs)

# Save to CSV
all_details_df.to_csv(os.path.join(reports_dir, 'all_cohorts_patient_details.csv'))

# Go through all silos and then all the splits in the labeled_diagnostic split
data_silos = [
    'labeled_diagnostic',
    'misc',
    'unlabeled'
]

misc_data_silos = [
    'excluded',
    'relapse',
    'unrecognized'
]

data_splits = [
    'train',
    'calibration',
    'test'
]

modalities = [
    'GEX',
    'DNAm'
]


def gex_samplesheet_to_df(modality, silo, split):
    if split is None:
        filename = f"samplesheet_{silo}.csv"
    else:
        filename = f"samplesheet_{split}.csv"
    filepath = os.path.join(processed_data_dir, filename)
    try:
        df = pd.read_csv(filepath, index_col=0)
        # Set age and sex to none
        df['age'] = None
        df['sex'] = None
        df['modality'] = modality
        df['data_split'] = split

        # Look up age and sex from all_details_df
        # Do not join, copy instead to avoid duplicating rows
        for idx in df.index:
            if idx in all_details_df.index:
                df.at[idx, 'age'] = all_details_df.at[idx, 'age']
                df.at[idx, 'sex'] = all_details_df.at[idx, 'sex']

        print(df.columns)
        df = df[['modality',
                 'dataset',
                 'data_silo',
                 'data_split',
                 'subtype',
                 'batch',
                 'age',
                 'sex']]
    except FileNotFoundError:
        return None


def dnam_samplesheet_to_df(modality, silo, split):
    if split is None:
        filename = f"dnam_{silo}_y.tsv"
    else:
        filename = f"dnam_{split}_y.tsv"
    filepath = os.path.join(processed_data_dir, 'dnam', filename)
    try:
        return pd.read_csv(filepath, sep='\t', index_col=0)
    except FileNotFoundError:
        return None


def get_samplesheet_df(modality, silo, split=None):
    if modality == 'GEX':
        return gex_samplesheet_to_df(modality, silo, split)
    elif modality == 'DNAm':
        return dnam_samplesheet_to_df(modality, silo, split)
    else:
        raise ValueError(f"Unknown modality: {modality}")


dfs = []
for modality in modalities:
    for silo in data_silos + misc_data_silos:
        df = None
        if silo == 'labeled_diagnostic':
            for split in data_splits:
                print(f"Modality: {modality}, Silo: {silo}, Split: {split}")
                df = get_samplesheet_df(modality, silo, split)
        else:
            print(f"Modality: {modality}, Silo: {silo}")
            df = get_samplesheet_df(modality, silo)

        if df is not None:
            dfs.append(df)

print(len(dfs))
