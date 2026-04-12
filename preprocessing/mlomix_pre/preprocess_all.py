import runpy
import pandas as pd
import os
import dotenv
from sklearn.model_selection import train_test_split
from preprocessing.constants import PREPROCESSED_DATA_DIR, \
    MLOMIX_SAMPLESHEETS_DIR

# Load environment variables from .env file
dotenv.load_dotenv()
RANDOM_SEED = 42
SPLIT_TRAIN_FRAC = 0.7
SPLIT_CALIB_FRAC = 0.15
SPLIT_TEST_FRAC = 0.15
MIN_TEST_SAMPLES_PER_CLASS = 5

modules = [
    "preprocessing.mlomix_pre.SJC-DS-1009",
    "preprocessing.mlomix_pre.SJC-panALL",
    "preprocessing.mlomix_pre.Krali-2023",
]

for mod in modules:
    print(f"Running {mod}...")
    runpy.run_module(mod, run_name="__main__")

# Find all samplesheet.csv files and combine them
print("Combining all samplesheets...\n")
samplesheet_dfs = []
for root, dirs, files in os.walk(PREPROCESSED_DATA_DIR):
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

# Display by data_silo (labeled_diagnostic, unlabeled, or misc).
combined_samplesheet_df['data_silo_detailed'] = (
    combined_samplesheet_df['data_silo'].apply(
        lambda x: x if x in ['labeled_diagnostic', 'unlabeled'] else 'misc'
    )
)
print(combined_samplesheet_df['data_silo_detailed'].value_counts())
combined_samplesheet_df = combined_samplesheet_df.drop(
    columns=['data_silo_detailed'])

print("\n")

# Put labeled_diagnostic into separate dataframe
labeled_diagnostic_df = combined_samplesheet_df[
    combined_samplesheet_df['data_silo'] == 'labeled_diagnostic']


def _nonempty(series):
    return series.notna() & (series.astype(str).str.strip() != "")


# "Bimodal samples" are labeled_diagnostic samples with non-empty GEX and
# at least one non-empty DNAm-related field.
gex_col = 'gex_feature_counts_file'
dnam_cols = [
    'dnam_beta_matrix_file',
    'dnam_pvals_file',
    'sentrix_id',
    'sentrix_position',
    'idats_dir',
]

for col in [gex_col] + dnam_cols:
    if col not in labeled_diagnostic_df.columns:
        labeled_diagnostic_df[col] = pd.NA

has_gex = _nonempty(labeled_diagnostic_df[gex_col])
has_any_dnam = pd.Series(False, index=labeled_diagnostic_df.index)
for col in dnam_cols:
    has_any_dnam = has_any_dnam | _nonempty(labeled_diagnostic_df[col])

bimodal_df = labeled_diagnostic_df[has_gex & has_any_dnam]

# Display stats by class in labeled_diagnostic only
print("Class distribution in labeled_diagnostic:")
print(labeled_diagnostic_df['class'].value_counts())
print("\n")

n_bimodal = bimodal_df.shape[0]
print(f"Number of bimodal samples: {n_bimodal}\n")

if n_bimodal > 0:
    print("Class distribution in labeled_diagnostic bimodal:")
    print(bimodal_df['class'].value_counts())
    print("\n")

if abs((SPLIT_TRAIN_FRAC + SPLIT_CALIB_FRAC + SPLIT_TEST_FRAC) - 1.0) > 1e-8:
    raise ValueError(
        "SPLIT_TRAIN_FRAC + SPLIT_CALIB_FRAC + SPLIT_TEST_FRAC must equal 1.0"
    )


def safe_train_test_split(series, train_size, random_state, stratify):
    try:
        return train_test_split(
            series,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        print(
            "Warning: stratified split failed "
            f"({exc}). Falling back to unstratified split."
        )
        return train_test_split(
            series,
            train_size=train_size,
            random_state=random_state,
            stratify=None,
        )


# SPLIT LABELED_DIAGNOSTIC INTO TRAIN/CALIB/TEST
y_labeled = labeled_diagnostic_df['class']
y_large_train, y_large_temp = safe_train_test_split(
    y_labeled,
    train_size=SPLIT_TRAIN_FRAC,
    random_state=RANDOM_SEED,
    stratify=y_labeled,
)

calib_frac_of_temp = SPLIT_CALIB_FRAC / (SPLIT_CALIB_FRAC + SPLIT_TEST_FRAC)
y_large_calib, y_large_test = safe_train_test_split(
    y_large_temp,
    train_size=calib_frac_of_temp,
    random_state=RANDOM_SEED,
    stratify=y_large_temp,
)


print("Class distribution in train_large_model:")
print(y_large_train.value_counts())
print("\nClass distribution in calibration_large_model:")
print(y_large_calib.value_counts())
print("\nClass distribution in test_large_model:")
print(y_large_test.value_counts())

# Move classes with too few samples in test back to calibration
test_class_counts = y_large_test.value_counts()
underrepresented_classes = test_class_counts[
    test_class_counts < MIN_TEST_SAMPLES_PER_CLASS
].index.tolist()

if len(underrepresented_classes) > 0:
    min_test_str = str(MIN_TEST_SAMPLES_PER_CLASS)
    print(
        f"\nMoving {len(underrepresented_classes)} underrepresented "
        f"class(es) from test to calibration "
        f"(MIN_TEST_SAMPLES_PER_CLASS={min_test_str}):"
    )
    print(f"  {underrepresented_classes}")

    # Find indices of underrepresented classes in test
    move_to_calib_mask = y_large_test.isin(underrepresented_classes)
    indices_to_move = y_large_test[move_to_calib_mask].index

    # Move from test to calibration
    y_large_calib = y_large_calib._append(y_large_test[move_to_calib_mask])
    y_large_test = y_large_test[~move_to_calib_mask]

    print("\nUpdated class distribution in calibration_large_model:")
    print(y_large_calib.value_counts())
    print("\nUpdated class distribution in test_large_model:")
    print(y_large_test.value_counts())

if n_bimodal > 0:
    # SPLIT BIMODAL INTO TRAIN/TEST
    y_bimodal = bimodal_df['class']
    y_bimodal_train, y_bimodal_test = safe_train_test_split(
        y_bimodal,
        train_size=SPLIT_TRAIN_FRAC,
        random_state=RANDOM_SEED,
        stratify=y_bimodal,
    )

    print("\nClass distribution in train_bimodal_model:")
    print(y_bimodal_train.value_counts())
    print("\nClass distribution in test_bimodal_model:")
    print(y_bimodal_test.value_counts())
    print("\n")


# SAVE SAMPLESHEETS
def save_samplesheet(df, filename):
    os.makedirs(MLOMIX_SAMPLESHEETS_DIR, exist_ok=True)
    output_path = os.path.join(MLOMIX_SAMPLESHEETS_DIR, filename)
    df.to_csv(output_path)
    print(f"Saved {filename} with {df.shape[0]} samples to {output_path}")


# Save the following samplesheets:
save_samplesheet(
    labeled_diagnostic_df,
    'samplesheet_labeled_diagnostic.csv'
)
save_samplesheet(
    bimodal_df,
    'samplesheet_labeled_diagnostic_bimodal.csv'
)
save_samplesheet(
    combined_samplesheet_df[
        combined_samplesheet_df['data_silo'] == 'unlabeled'
    ],
    'samplesheet_unlabeled.csv'
)
save_samplesheet(
    combined_samplesheet_df[
        ~combined_samplesheet_df['data_silo'].isin(
            ['labeled_diagnostic', 'unlabeled']
        )
    ],
    'samplesheet_misc.csv'
)
save_samplesheet(
    labeled_diagnostic_df.loc[y_large_train.index],
    'samplesheet_train_large_model.csv'
)
save_samplesheet(
    labeled_diagnostic_df.loc[y_large_calib.index],
    'samplesheet_calibration_large_model.csv'
)
save_samplesheet(
    labeled_diagnostic_df.loc[y_large_test.index],
    'samplesheet_test_large_model.csv'
)

if n_bimodal > 0:
    save_samplesheet(
        bimodal_df.loc[y_bimodal_train.index],
        'samplesheet_train_bimodal_model.csv'
    )
    save_samplesheet(
        bimodal_df.loc[y_bimodal_test.index],
        'samplesheet_test_bimodal_model.csv'
    )

# Print final counts in all sets
print("\nFinal counts in all sets:")
print(f" - LABELED_DIAGNOSTIC: {labeled_diagnostic_df.shape[0]} samples")
print(f" - LABELED_DIAGNOSTIC_BIMODAL: {bimodal_df.shape[0]} samples")
print(f" - TRAIN_LARGE_MODEL: {y_large_train.shape[0]} samples")
print(f" - CALIBRATION_LARGE_MODEL: {y_large_calib.shape[0]} samples")
print(f" - TEST_LARGE_MODEL: {y_large_test.shape[0]} samples")
if n_bimodal > 0:
    print(f" - TRAIN_BIMODAL_MODEL: {y_bimodal_train.shape[0]} samples")
    print(f" - TEST_BIMODAL_MODEL: {y_bimodal_test.shape[0]} samples")
