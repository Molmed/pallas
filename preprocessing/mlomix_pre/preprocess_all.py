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
MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION_SPLIT = 10

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

for mod in modules:
    print(f"\n\n\n*****RUNNING ML_OMIX_PRE FOR DATASET {mod}...*****\n\n")
    runpy.run_module(mod, run_name="__main__")

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

    # Print class distribution before split
    print(f"\nClass distribution in {modality} before splitting:")
    print(modality_samplesheet['class'].value_counts())

    # Split into train/calib/test
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
    # But first see what samples should be excluded from split
    class_counts = y_temp.value_counts()

    print(f"\nClass distribution in {modality} before splitting into calibration/test:")
    print(class_counts)

    classes_to_exclude_from_split = class_counts[
        class_counts < MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION_SPLIT
    ].index.tolist()
    if len(classes_to_exclude_from_split) > 0:
        print(
            f"\nIn modality {modality}, excluding {len(classes_to_exclude_from_split)} "
            f"class(es) from calibration/test split since they have fewer than "
            f"{MIN_SAMPLES_PER_CLASS_FOR_CALIBRATION_SPLIT} samples: "
            f"{classes_to_exclude_from_split}"
        )
        split_mask = ~y_temp.isin(classes_to_exclude_from_split)
        y_temp_for_split = y_temp[split_mask]
    else:
        y_temp_for_split = y_temp

    calib_frac_of_temp = SPLIT_CALIB_FRAC / (
        SPLIT_CALIB_FRAC + SPLIT_TEST_FRAC)
    print(
        f"Splitting remaining {1 - SPLIT_TRAIN_FRAC:.0%} of {modality} samples "
        f"into calibration ({calib_frac_of_temp:.0%}) and test "
        f"({1 - calib_frac_of_temp:.0%})..."
    )
    y_calib, y_test = safe_train_test_split(
        y_temp_for_split,
        train_size=calib_frac_of_temp,
        random_state=RANDOM_SEED,
        stratify=y_temp_for_split,
    )

    # Add back the excluded classes to the calibration set
    if len(classes_to_exclude_from_split) > 0:
        y_calib = y_calib._append(y_temp[~split_mask])

    # Save the splits
    sample_splits[f'train_{modality}'] = y_train.index
    sample_splits[f'calibrate_{modality}'] = y_calib.index
    sample_splits[f'test_{modality}'] = y_test.index

# For each sample_split, create entry in samplesheets_to_save with the corresponding samplesheet
for split_name, indices in sample_splits.items():
    modality = split_name.split('_')[1]
    samplesheets_to_save[split_name] = samplesheets_to_save[
        f'labeled_diagnostic_{modality}'].loc[indices]


# SAVE SAMPLESHEETS
def save_samplesheet(df, filename):
    os.makedirs(MLOMIX_SAMPLESHEETS_DIR, exist_ok=True)
    output_path = os.path.join(MLOMIX_SAMPLESHEETS_DIR, filename)
    format_mlomix_columns(df).to_csv(output_path)
    print(f"Saved {filename} with {df.shape[0]} samples to {output_path}")


# Save each samplesheet and report on its contents
for name, df in samplesheets_to_save.items():
    save_samplesheet(df, f'samplesheet_{name}.csv')
