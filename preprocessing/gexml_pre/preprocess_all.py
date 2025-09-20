import runpy
import pandas as pd
import os
import dotenv
from sklearn.model_selection import train_test_split
from preprocessing.constants import PROCESSED_DATA_DIR

# Load environment variables from .env file
dotenv.load_dotenv()
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
SPLIT_TRAIN_FRAC = float(os.getenv("SPLIT_TRAIN_FRAC", 0.7))
SPLIT_MIN_TEST_SAMPLES = int(os.getenv("SPLIT_MIN_TEST_SAMPLES", 3))

modules = [
    "preprocessing.gexml_pre.diedrich",
    "preprocessing.gexml_pre.heinaniemi",
    "preprocessing.gexml_pre.krali",
    "preprocessing.gexml_pre.stjude_cloud",
    "preprocessing.gexml_pre.stjude_panALL",
    "preprocessing.gexml_pre.tran",
    "preprocessing.gexml_pre.lund",
]

for mod in modules:
    print(f"Running {mod}...")
    runpy.run_module(mod, run_name="__main__")

# Find all samplesheet.csv files and combine them
print("Combining all samplesheets...\n")
samplesheet_dfs = []
for root, dirs, files in os.walk(PROCESSED_DATA_DIR):
    for file in files:
        if file == 'samplesheet.csv':
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, index_col=0)
            samplesheet_dfs.append(df)
combined_samplesheet_df = pd.concat(samplesheet_dfs)
combined_samplesheet_df = combined_samplesheet_df.sort_index()

# Get total number of samples
total_samples = combined_samplesheet_df.shape[0]
print(f"Total number of samples: {total_samples}\n")

# Display stats by dataset
print(combined_samplesheet_df['dataset'].value_counts())

print("\n")

# Display stats by data_silo
print(combined_samplesheet_df['data_silo'].value_counts())

print("\n")

# Display by data_silo, labeled_diagnostic vs unlabeled, and then combine the others under "misc"
combined_samplesheet_df['data_silo_detailed'] = combined_samplesheet_df['data_silo'].apply(
    lambda x: x if x in ['labeled_diagnostic', 'unlabeled'] else 'misc'
)
print(combined_samplesheet_df['data_silo_detailed'].value_counts())
combined_samplesheet_df = combined_samplesheet_df.drop(
    columns=['data_silo_detailed'])

print("\n")

# Put labeled_diagnostic into separate dataframe
labeled_diagnostic_df = combined_samplesheet_df[
    combined_samplesheet_df['data_silo'] == 'labeled_diagnostic']

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

# Split remaining data 50/50
y_calib, y_test = train_test_split(
    y_rest,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=y_rest
)

# Now see which classes in test are under SPLIT_MIN_TEST_SAMPLES.
# For these classes, we will not support conformal prediction.
# So just move them to the train set.
low_class_indices = y_test.isin(
    y_test.value_counts()[
        y_test.value_counts() < SPLIT_MIN_TEST_SAMPLES].index
)
y_low = y_test[low_class_indices]
y_test = y_test[~y_test.index.isin(y_low.index)]

# Print all classes in this split and the number of samples in each
print("""The following classes will be excluded from conformal prediction
tests, and their test samples will be moved back into the training set:""")
for class_label in y_low.unique():
    class_count = y_low.value_counts().get(class_label, 0)
    print(f" - Class {class_label}: {class_count} test samples")
print("\n")

# Merge low items into train set
y_train = pd.concat([y_train, y_low])

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
    combined_samplesheet_df[combined_samplesheet_df['data_silo'] == 'labeled_diagnostic'],
    'samplesheet_labeled_diagnostic.csv'
)
save_samplesheet(
    combined_samplesheet_df[combined_samplesheet_df['data_silo'] == 'unlabeled'],
    'samplesheet_unlabeled.csv'
)
save_samplesheet(
    combined_samplesheet_df[~combined_samplesheet_df['data_silo'].isin(['labeled_diagnostic', 'unlabeled'])],
    'samplesheet_misc.csv'
)
save_samplesheet(
    combined_samplesheet_df.loc[y_train.index],
    'samplesheet_train.csv'
)
save_samplesheet(
    combined_samplesheet_df.loc[y_calib.index],
    'samplesheet_calibration.csv'
)
save_samplesheet(
    combined_samplesheet_df.loc[y_test.index],
    'samplesheet_test.csv'
)

# Print final counts in all sets
print("\nFinal counts in all sets:")
print(f" - TRAIN: {y_train.shape[0]} samples")
print(f" - CALIBRATION: {y_calib.shape[0]} samples")
print(f" - TEST: {y_test.shape[0]} samples")
print(f" - EXCLUDED FROM CALIBRATION: {y_low.shape[0]} samples")
