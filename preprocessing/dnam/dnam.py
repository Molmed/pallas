import pandas as pd
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import silo_unlabeled, \
    silo_remaining, silo_unrecognized, silo_excluded, add_to_silo

print("Processing DNAm...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/dnam'
raw_data_path = f'{data_path}/concatenated_df_VARIANCE_FILTERED_0_01.pkl'
pheno_input_file = f'{data_path}/PALLAS_FINAL_pheno_combined_n=2479.csv'

# OUTPUT FILES ########
processed_dir = f'{PROCESSED_DATA_DIR}/dnam'
data_output_file = \
    f'{processed_dir}/data.csv'
labels_output_file = \
    f'{processed_dir}/labels.tsv'

# PROCESS PHENO ########
print("Processing phenotype data...")
labels_df = pd.read_csv(pheno_input_file, index_col=0)

# Set data_silo to empty string
labels_df['data_silo'] = ''

# Translate subtypes
tax = Taxonomy()
labels_df['subtype'] = tax.translate_subtype_column(
    labels_df['subtype'],
    exclude_subtypes=DNAM_EXCLUDE_SUBTYPES)

# Silo unrecognized
labels_df = silo_unrecognized(labels_df)

# Silo excluded
labels_df = silo_excluded(labels_df)

# Silo unlabeled
labels_df = silo_unlabeled(labels_df)

# Put all samples with sample.type=="relapse" into relapse silo
relapse_sample_mask = labels_df['sample.type'] == 'relapse'
if relapse_sample_mask.any():
    labels_df = add_to_silo(labels_df,
                            labels_df[relapse_sample_mask].index,
                            'relapse')
# Print all sample names
for(idx, row) in labels_df.iterrows():
    # if data silo is empty
    if(row['data_silo'] == ''):
        print(idx, row['subtype'], row['data_silo'])

labels_df = silo_remaining(labels_df)


# Show class counts of all subtypes with empty data silo
print(labels_df)
exit()

# Show unlabeled silo items, class counts
print("Unlabeled data silos:")
print(labels_df[labels_df['data_silo'] == 'unlabeled']['subtype'].value_counts())
exit()

# Show which items have unrecognized data silo
print("Unrecognized data silos:")
print(labels_df[labels_df['data_silo'] == 'unrecognized'])

# Show unique subtypes in unrecognized
print("Unique subtypes in unrecognized:")
print(labels_df[labels_df['data_silo'] == 'unrecognized']['subtype'].unique())

print(labels_df)
exit()

