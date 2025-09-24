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
        for idx in df.index:
            if idx in all_details_df.index:
                matching_rows = all_details_df[
                    (all_details_df.index == idx) &
                    (all_details_df['modality'] == modality)
                ]

            if not matching_rows.empty:
                # Use .iloc[0] to get the first (and should be only) matching row
                df.at[idx, 'age'] = matching_rows.iloc[0]['age']
                df.at[idx, 'sex'] = matching_rows.iloc[0]['sex']

        df = df[['modality',
                 'dataset',
                 'data_silo',
                 'data_split',
                 'subtype',
                 'batch',
                 'age',
                 'sex']]
        df = df.reset_index()
        df = df.set_index(['sample', 'modality'])

        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None


def dnam_samplesheet_to_df(modality, silo, split):
    if split is None:
        filename = f"dnam_{silo}_y.tsv"
    else:
        filename = f"dnam_{split}_y.tsv"
    filepath = os.path.join(processed_data_dir, 'dnam', filename)
    try:
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        df['age'] = None
        df['sex'] = None
        df['batch'] = None
        df['modality'] = modality
        df['data_silo'] = silo
        df['data_split'] = split

        # Look up age and sex from all_details_df
        for idx in df.index:
            if idx in all_details_df.index:
                matching_rows = all_details_df[
                    (all_details_df.index == idx) &
                    (all_details_df['modality'] == modality)
                ]

            if not matching_rows.empty:
                # Use .iloc[0] to get the first (and should be only) matching row
                df.at[idx, 'age'] = matching_rows.iloc[0]['age']
                df.at[idx, 'sex'] = matching_rows.iloc[0]['sex']
                df.at[idx, 'dataset'] = matching_rows.iloc[0]['dataset']

        df = df[['modality',
                 'dataset',
                 'data_silo',
                 'data_split',
                 'subtype',
                 'batch',
                 'age',
                 'sex']]
        df.index.name = 'sample'
        df = df.reset_index()
        df = df.set_index(['sample', 'modality'])

        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
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
    def add_df_to_list(df):
        if df is not None:
            print(f"Adding df with modality: {modality}, Silo: {silo}, Split: {split} and shape: {df.shape}")
            dfs.append(df)

    for silo in data_silos + misc_data_silos:
        split = None
        df = None
        if silo == 'labeled_diagnostic':
            for split in data_splits:
                add_df_to_list(get_samplesheet_df(modality, silo, split))
        else:
            add_df_to_list(get_samplesheet_df(modality, silo))


# Concatenate all dfs
all_samples_df = pd.concat(dfs)

# Dump to CSV
all_samples_df.to_csv(os.path.join(reports_dir, 'all_cohorts_samples_report.csv'))
