import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DNAM_EXCLUDE_SUBTYPES
from preprocessing.helpers import silo_unlabeled, \
    silo_remaining, silo_unrecognized, silo_excluded, add_to_silo
from sklearn.model_selection import train_test_split
import dotenv

dotenv.load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
SPLIT_TRAIN_FRAC = float(os.getenv("SPLIT_TRAIN_FRAC", 0.7))
SPLIT_MIN_CALIBRATION_SAMPLES = int(os.getenv("SPLIT_MIN_CALIBRATION_SAMPLES", 15))

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

labels_df = silo_remaining(labels_df)

## PRINT REPORT
# Get total number of samples
total_samples = labels_df.shape[0]
print(f"Total number of samples: {total_samples}\n")

# Display stats by dataset
print(labels_df['dataset'].value_counts())

print("\n")

# Display stats by data_silo
print(labels_df['data_silo'].value_counts())

print("\n")

# Display by data_silo, labeled_diagnostic vs unlabeled, and then combine the others under "misc"
labels_df['data_silo_detailed'] = labels_df['data_silo'].apply(
    lambda x: x if x in ['labeled_diagnostic', 'unlabeled'] else 'misc'
)
print(labels_df['data_silo_detailed'].value_counts())
labels_df = labels_df.drop(
    columns=['data_silo_detailed'])
print("\n")

# Put labeled_diagnostic into separate dataframe
labeled_diagnostic_df = labels_df[
    labels_df['data_silo'] == 'labeled_diagnostic']

# Display stats by class in labeled_diagnostic only
print("Class distribution in labeled_diagnostic:")
print(labeled_diagnostic_df['subtype'].value_counts())
print("\n")


# SPLIT LABELED_DIAGNOSTIC INTO TRAIN/CAL/TEST
y = labeled_diagnostic_df['subtype']
y_train, y_rest = train_test_split(
        y,
        train_size=SPLIT_TRAIN_FRAC,
        random_state=RANDOM_SEED,
        stratify=y
)
# In the rest split, remove samples totaling less
# than the calibration sample minimum.
# Put them into a separate data split.
low_class_indices = y_rest.isin(
    y_rest.value_counts()[
        y_rest.value_counts() < SPLIT_MIN_CALIBRATION_SAMPLES].index
)
y_low = y_rest[low_class_indices]
y_rest = y_rest[~y_rest.index.isin(y_low.index)]

# Print all classes in this split and the number of samples in each
print("The following classes will be excluded from the calibration set:")
for class_label in y_low.unique():
    class_count = y_low.value_counts().get(class_label, 0)
    print(f" - Class {class_label}: {class_count} samples")
print("\n")

# Split remaining data 50/50
y_calib, y_test = train_test_split(
    y_rest,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=y_rest
)

# Merge low items into test set
y_test = pd.concat([y_test, y_low])
print("Adding classes excluded from calibration into the test set.\n")

# Class distributions in train, cal and test
print("Class distribution in train:")
print(y_train.value_counts())
print("\nClass distribution in calibration:")
print(y_calib.value_counts())
print("\nClass distribution in test:")
print(y_test.value_counts())
print("\n")


# SAVE SAMPLESHEETS
def save_samplesheet(df, filename):
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path)
    print(f"Saved {filename} with {df.shape[0]} samples to {output_path}")


# Save the following samplesheets:
# - samplesheet_labeled_diagnostic.csv (all with data_silo == labeled_diagnostic)
# - samplesheet_unlabeled.csv (all with data_silo == unlabeled)
# - samplesheet_misc.csv (all with data_silo not in labeled_diagnostic or unlabeled)
# - samplesheet_train.csv (all in y_train)
# - samplesheet_calibration.csv (all in y_calib)
# - samplesheet_test.csv (all in y_test)
save_samplesheet(
    labels_df[labels_df['data_silo'] == 'labeled_diagnostic'],
    'dnam_samplesheet_labeled_diagnostic.csv'
)
save_samplesheet(
    labels_df[labels_df['data_silo'] == 'unlabeled'],
    'dnam_samplesheet_unlabeled.csv'
)
save_samplesheet(
    labels_df[~labels_df['data_silo'].isin(['labeled_diagnostic', 'unlabeled'])],
    'dnam_samplesheet_misc.csv'
)
save_samplesheet(
    labels_df.loc[y_train.index],
    'dnam_samplesheet_train.csv'
)
save_samplesheet(
    labels_df.loc[y_calib.index],
    'dnam_samplesheet_calibration.csv'
)
save_samplesheet(
    labels_df.loc[y_test.index],
    'dnam_samplesheet_test.csv'
)

# Print final counts in all sets
print("\nFinal counts in all sets:")
print(f" - TRAIN: {y_train.shape[0]} samples")
print(f" - CALIBRATION: {y_calib.shape[0]} samples")
print(f" - TEST: {y_test.shape[0]} samples")
print(f" - EXCLUDED FROM CALIBRATION: {y_low.shape[0]} samples")
