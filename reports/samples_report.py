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
print(all_details_df)
