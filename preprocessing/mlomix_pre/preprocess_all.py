import runpy
import pandas as pd
import os
import dotenv
from sklearn.model_selection import train_test_split
from preprocessing.constants import PREPROCESSED_DATA_DIR, \
    MLOMIX_SAMPLESHEETS_DIR
from preprocessing.helpers import format_mlomix_columns

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
    "preprocessing.mlomix_pre.TARGET-AML",
    "preprocessing.mlomix_pre.Krali-2023",
    "preprocessing.mlomix_pre.Lilljebjorn-2016",
    "preprocessing.mlomix_pre.Tran-2022",
    "preprocessing.mlomix_pre.Krali-2021",
    "preprocessing.mlomix_pre.Stratmann-2022",
    "preprocessing.mlomix_pre.TCGA-LAML",
    "preprocessing.mlomix_pre.Enblad-2025",
    "preprocessing.mlomix_pre.Diedrich-2021",
    "preprocessing.mlomix_pre.Paulsson-2023",
]

# for mod in modules:
#     print(f"\n\n\n*****RUNNING ML_OMIX_PRE FOR DATASET {mod}...*****\n\n")
#     runpy.run_module(mod, run_name="__main__")

# All the samplesheets we will save:
# - all
# - labeled_diagnostic
# - unlabeled
# - misc
# - labeled_diagnostic_gex
# - labeled_diagnostic_dnam
# - labeled_diagnostic_bimodal
# - train_gex
# - calibrate_gex
# - test_gex
# - train_dnam
# - calibrate_dnam
# - test_dnam
# - train_bimodal
# - test_bimodal
samplesheets_to_save = {}

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

# Samplesheet: all
samplesheets_to_save['all'] = combined_samplesheet_df.copy()

# Create two new columns, has_gex and has_dnam
gex_col = 'gex_feature_counts_file'
dnam_cols = [
    'dnam_beta_matrix_file',
    'dnam_pvals_file',
    'sentrix_id',
    'sentrix_position',
    'idats_basename',
]
# has_gex is true if gex_col is set, otherwise false
# has_dnam is false unless one of dnam_cols is set
combined_samplesheet_df['has_gex'] = combined_samplesheet_df[gex_col].notna()
combined_samplesheet_df['has_dnam'] = combined_samplesheet_df[dnam_cols].notna().any(axis=1)

# Simplify data silo
combined_samplesheet_df['data_silo'] = (
    combined_samplesheet_df['data_silo'].apply(
        lambda x: x if x in ['labeled_diagnostic', 'unlabeled'] else 'misc'
    )
)

# Samplesheets: labeled_diagnostic, unlabeled, misc
samplesheets_to_save['labeled_diagnostic'] = combined_samplesheet_df[
    combined_samplesheet_df['data_silo'] == 'labeled_diagnostic'].copy()
samplesheets_to_save['unlabeled'] = combined_samplesheet_df[
    combined_samplesheet_df['data_silo'] == 'unlabeled'].copy()
samplesheets_to_save['misc'] = combined_samplesheet_df[
    combined_samplesheet_df['data_silo'] == 'misc'].copy()

# Save labeled_diagnostic separately for now since we will split it into train/calib/test and bimodal later
labeled_diagnostic_df = samplesheets_to_save['labeled_diagnostic'].copy()
labeled_diagnostic_bimodal_df = labeled_diagnostic_df[
    labeled_diagnostic_df['has_gex'] & labeled_diagnostic_df['has_dnam']
].copy()
labeled_diagnostic_gex_df = labeled_diagnostic_df[
    labeled_diagnostic_df['has_gex']
].copy()
labeled_diagnostic_dnam_df = labeled_diagnostic_df[
    labeled_diagnostic_df['has_dnam']
].copy()

# Samplesheets: labeled_diagnostic_gex, labeled_diagnostic_dnam, labeled_diagnostic_bimodal
samplesheets_to_save['labeled_diagnostic_gex'] = labeled_diagnostic_gex_df
samplesheets_to_save['labeled_diagnostic_dnam'] = labeled_diagnostic_dnam_df
samplesheets_to_save['labeled_diagnostic_bimodal'] = labeled_diagnostic_bimodal_df


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


sample_splits = {}

# - train_gex
# - calibrate_gex
# - test_gex
# - train_dnam
# - calibrate_dnam
# - test_dnam
# - train_bimodal
# - test_bimodal

for modality in ["gex", "dnam", "bimodal"]:
    # Get the samplesheet for this modality
    modality_samplesheet = samplesheets_to_save[f'labeled_diagnostic_{modality}']

    # Split into train/calib/test
    print(f"\nSplitting {modality} samples into train and calib/test...")
    y = modality_samplesheet['class']
    y_train, y_temp = safe_train_test_split(
        y,
        train_size=SPLIT_TRAIN_FRAC,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # If modality is bimodal, we only do train/test split, no calibration set, since we expect fewer samples and want to maximize training data
    if modality == "bimodal":
        sample_splits['train_bimodal'] = y_train.index
        sample_splits['test_bimodal'] = y_temp.index
        continue

    # For gex and dnam, we do train/calib/test split
    calib_frac_of_temp = SPLIT_CALIB_FRAC / (
        SPLIT_CALIB_FRAC + SPLIT_TEST_FRAC)
    print(
        f"Splitting remaining {1 - SPLIT_TRAIN_FRAC:.0%} of {modality} samples "
        f"into calibration ({calib_frac_of_temp:.0%}) and test "
        f"({1 - calib_frac_of_temp:.0%})..."
    )
    y_calib, y_test = safe_train_test_split(
        y_temp,
        train_size=calib_frac_of_temp,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    # Move classes with too few samples in test back to calibration and report
    test_class_counts = y_test.value_counts()
    underrepresented_classes = test_class_counts[
        test_class_counts < MIN_TEST_SAMPLES_PER_CLASS
    ].index.tolist()
    if len(underrepresented_classes) > 0:
        min_test_str = str(MIN_TEST_SAMPLES_PER_CLASS)
        print(
            f"\nIn modality {modality}, moving {len(underrepresented_classes)} "
            f"underrepresented class(es) from test to calibration "
            f"(MIN_TEST_SAMPLES_PER_CLASS={min_test_str}):"
        )
        print(f"  {underrepresented_classes}")

        # Find indices of underrepresented classes in test
        move_to_calib_mask = y_test.isin(underrepresented_classes)
        indices_to_move = y_test[move_to_calib_mask].index

        # Move from test to calibration
        y_calib = y_calib._append(y_test[move_to_calib_mask])
        y_test = y_test[~move_to_calib_mask]

    # Save the splits
    sample_splits[f'train_{modality}'] = y_train.index
    sample_splits[f'calibrate_{modality}'] = y_calib.index
    sample_splits[f'test_{modality}'] = y_test.index

# For each sample_split, create entry in samplesheets_to_save with the corresponding samplesheet
for split_name, indices in sample_splits.items():
    modality = split_name.split('_')[1]
    samplesheets_to_save[split_name] = samplesheets_to_save[
        f'labeled_diagnostic_{modality}'].loc[indices]

# print(samplesheets_to_save)
exit()


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
    format_mlomix_columns(df).to_csv(output_path)
    print(f"Saved {filename} with {df.shape[0]} samples to {output_path}")


# Save the following samplesheets:
save_samplesheet(
    combined_samplesheet_df,
    'samplesheet_complete.csv'
)

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
print(f" - TOTAL DATA POINTS: {combined_samplesheet_df.shape[0]} samples")
print(f" - LABELED_DIAGNOSTIC: {labeled_diagnostic_df.shape[0]} samples")
print(f" - LABELED_DIAGNOSTIC_BIMODAL: {bimodal_df.shape[0]} samples")
print(f" - TRAIN_LARGE_MODEL: {y_large_train.shape[0]} samples")
print(f" - CALIBRATION_LARGE_MODEL: {y_large_calib.shape[0]} samples")
print(f" - TEST_LARGE_MODEL: {y_large_test.shape[0]} samples")
if n_bimodal > 0:
    print(f" - TRAIN_BIMODAL_MODEL: {y_bimodal_train.shape[0]} samples")
    print(f" - TEST_BIMODAL_MODEL: {y_bimodal_test.shape[0]} samples")
