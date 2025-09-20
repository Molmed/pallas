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

# INPUT ########
data_path = f'{RAW_DATA_DIR}/dnam'
raw_data_path = f'{data_path}/concatenated_df_VARIANCE_FILTERED_0_01_NaN_0_1_imputed.pkl'
pheno_input_file = f'{data_path}/pheno/PALLAS_FINAL_pheno_combined_n=2479.csv'
epic_pheno_input_file = \
    f'{data_path}/pheno/NOPHO_EPIC_n=55_subtype_info_FINAL.csv'

# OUTPUT ########
output_dir = f'{PROCESSED_DATA_DIR}/dnam'

# Ensure output dir exists
os.makedirs(output_dir, exist_ok=True)

# PROCESS DATA ########
# The raw data uses another format of the sample IDs than the pheno file.
# Update the raw data with public sample ids
epic_pheno = pd.read_csv(epic_pheno_input_file, index_col=0)
epic_id_map = epic_pheno['sample_id.x'].to_dict()
epic_id_map = {v: k for k, v in epic_id_map.items()}

data_df = pd.read_pickle(raw_data_path)

# Update index with public sample ids where available
data_df.index = data_df.index.to_series().replace(epic_id_map)

# Get the pheno
labels_df = pd.read_csv(pheno_input_file, index_col=0)

# Check if data_df.index and labels_df.index match
if not data_df.index.isin(labels_df.index).all():
    missing_ids = data_df.index[~data_df.index.isin(labels_df.index)]
    # Drop the missing ids from data_df and print warning
    data_df = data_df[~data_df.index.isin(missing_ids)]
    print(f"Warning: Dropped {len(missing_ids)} samples from data_df")
    print(f"Missing IDs: {missing_ids.tolist()}")

# Sort labels so they match the data
labels_df = labels_df.loc[data_df.index]

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

# PRINT REPORT
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

# Standardize index names
data_df.index.name = 'id'
labels_df.index.name = 'id'


# SAVE DATASETS
def save_dataset(X, y, name):
    X_output_path = os.path.join(output_dir, f"dnam_{name}_X.csv")
    y_output_path = os.path.join(output_dir, f"dnam_{name}_y.tsv")
    X.to_csv(X_output_path, index=True)
    y.to_csv(y_output_path, sep='\t', index=True)
    print(f"Saved {name} with {y.shape[0]} samples to {X_output_path} and {y_output_path}")


def get_silo_dataset(silo_name):
    silo_indices = labels_df[labels_df['data_silo'] == silo_name].index
    X_silo = data_df.loc[silo_indices]
    y_silo = labels_df.loc[silo_indices, 'subtype']
    return X_silo, y_silo


# Save each silo separately
for silo in labels_df['data_silo'].unique():
    X_silo, y_silo = get_silo_dataset(silo)
    save_dataset(X_silo, y_silo, silo)

# SPLIT LABELED_DIAGNOSTIC INTO TRAIN/CAL/TEST
X, y = get_silo_dataset('labeled_diagnostic')

X_train, X_rest, y_train, y_rest = train_test_split(
        X,
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
X_low = X_rest[low_class_indices]
X_rest = X_rest[~X_rest.index.isin(X_low.index)]
y_low = y_rest[low_class_indices]
y_rest = y_rest[~y_rest.index.isin(y_low.index)]

# Print all classes in this split and the number of samples in each
print("The following classes will be excluded from the calibration set:")
for class_label in y_low.unique():
    class_count = y_low.value_counts().get(class_label, 0)
    print(f" - Class {class_label}: {class_count} samples")
print("\n")

# Split remaining data 50/50
X_calib, X_test, y_calib, y_test = train_test_split(
    X_rest,
    y_rest,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=y_rest
)

# Merge low items into test set
y_test = pd.concat([y_test, y_low])
X_test = pd.concat([X_test, X_low])
print("Adding classes excluded from calibration into the test set.\n")

# Save splits
save_dataset(X_train, y_train, 'train')
save_dataset(X_calib, y_calib, 'calibration')
save_dataset(X_test, y_test, 'test')

# Class distributions in train, cal and test
print("Class distribution in train:")
print(y_train.value_counts())
print("\nClass distribution in calibration:")
print(y_calib.value_counts())
print("\nClass distribution in test:")
print(y_test.value_counts())
print("\n")

# Print final counts in all sets
print("\nFinal counts in all sets:")
print(f" - TRAIN: {y_train.shape[0]} samples")
print(f" - CALIBRATION: {y_calib.shape[0]} samples")
print(f" - TEST: {y_test.shape[0]} samples")
print(f" - EXCLUDED FROM CALIBRATION: {y_low.shape[0]} samples")
