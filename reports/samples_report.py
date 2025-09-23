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

data_splits = [
    'train',
    'calibration',
    'test'
]

modality_data_paths = {
    'DNAm': os.path.join(processed_data_dir, 'dnam'),
    'GEX': processed_data_dir
}

for modality, data_path in modality_data_paths.items():
    summary = []
    for silo in data_silos:
        if silo == 'labeled_diagnostic':
            for split in data_splits:
                y_path = os.path.join(data_path, f'{split}_y.tsv')
                print(y_path)
        else:
            y_path = os.path.join(data_path, f'{silo}_y.tsv')
        print(y_path)
        # for split in data_splits:
        #     split_path = os.path.join(data_path, silo, split)
    # # Check if path exists
    #         if os.path.exists(split_path):
    #             # Read in samples file
    #             samples_file = os.path.join(split_path, 'samples.txt')
    #             if os.path.exists(samples_file):
    #                 samples = pd.read_csv(samples_file, header=None, names=['sample'])
    #                 num_samples = len(samples)

    #                 # Get patient details for these samples
    #                 details = all_details_df[all_details_df.index.isin(samples['sample'])]
    #                 age_summary = f"{details['age'].mean():.1f} ± ({details['age'].std():.1f})"
    #                summary.append({
    #                    'modality': modality,
    #                    'silo': silo,
    #                    'split': split,
    #                    'num_samples': num_samples,
    #                    'age_summary': age_summary
    #                })
