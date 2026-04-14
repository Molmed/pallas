import pandas as pd

# Function to add items to a silo
def add_to_silo(samplesheet_df, items, silo_name, force=False):
    if force:
        # Set data_silo to silo_name for all items
        mask = samplesheet_df.index.isin(items)
    else:
        # Only set data_silo if it is currently empty
        mask = samplesheet_df.index.isin(items) & (
            samplesheet_df['data_silo'] == '')
    samplesheet_df.loc[mask, 'data_silo'] = silo_name
    return samplesheet_df


def silo_unlabeled(samplesheet_df):
    unknown_rows = samplesheet_df['class'].str.endswith(
        ('other', 'ALLNOS'), na=False)

    if unknown_rows.any():
        samplesheet_df = add_to_silo(samplesheet_df,
                                     samplesheet_df[unknown_rows].index,
                                     'unlabeled')
    return samplesheet_df


def silo_excluded(samplesheet_df):
    excluded_rows = samplesheet_df['class'].str.contains(
        'EXCLUDED', na=False)
    if excluded_rows.any():
        samplesheet_df = add_to_silo(
            samplesheet_df, samplesheet_df[excluded_rows].index, 'excluded')
    return samplesheet_df


def silo_unrecognized(samplesheet_df):
    unrecognized_rows = samplesheet_df['class'].str.contains(
        'UNRECOGNIZED', na=False)
    if unrecognized_rows.any():
        samplesheet_df = add_to_silo(
            samplesheet_df, samplesheet_df[unrecognized_rows].index,
            'unrecognized')
    return samplesheet_df


def silo_remaining(samplesheet_df, silo_name='labeled_diagnostic'):
    # Set all items whose data_silo is still empty to the given silo_name
    mask = samplesheet_df['data_silo'] == ''
    samplesheet_df.loc[mask, 'data_silo'] = silo_name
    return samplesheet_df


def format_mlomix_columns(df):
    # The cols are as follows:
    # data_silo,dataset,batch,class,gex_feature_counts_file,dnam_beta_matrix_file,dnam_pvals_file,sentrix_id,sentrix_position,idats_dir
    # For the df, add empty columns for any that are missing and then put them in the right order
    expected_cols = [
        'data_silo', 'dataset', 'batch', 'class', 'gex_feature_counts_file',
        'dnam_beta_matrix_file', 'dnam_pvals_file', 'sentrix_id',
        'sentrix_position', 'idats_dir'
    ]
    # Ensure that index is named 'sample'
    df.index.name = 'sample'
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ''
    return df[expected_cols]


def summarize_patient_details(df):
    print("Patient details summary:")
    # Num unique patients
    num_patients = df.index.nunique()
    print(f"Number of unique patients: {num_patients}")
    # We want mean age in this format: 19.4 ± (19.0)
    mean_age = df['age'].mean()
    std_age = df['age'].std()
    print(f"Mean age at dx: {mean_age:.1f} ± ({std_age:.1f})")
    # We want Male/Female in this format: 1042/850
    male_count = df['sex'].value_counts().get('Male', 0)
    female_count = df['sex'].value_counts().get('Female', 0)
    print(f"Male/Female: {male_count}/{female_count}")
    # Report the number of patients missing age, sex or both
    missing_age = df['age'].isna().sum()
    missing_sex = df['sex'].isna().sum()
    missing_both = df[df['age'].isna() & df['sex'].isna()].shape[0]
    print(f"Missing age: {missing_age}")
    print(f"Missing sex: {missing_sex}")
    print(f"Missing both: {missing_both}")


def deduplicate_patient_details_using_sample_suffix(suffix, details_df):
    samples = details_df[details_df.index.str.endswith(suffix)]
    print(f"Deduplicating {suffix} samples:")
    for idx, row in samples.iterrows():
        unique_patient_id = row['unique_patient_id']
        # Check if there is another sample with the same unique_patient_id that does not end with this suffix
        other_samples = details_df[
            (details_df['unique_patient_id'] == unique_patient_id) &
            ~details_df.index.str.endswith(suffix)]
        if not other_samples.empty:
            print(f"Warning: Dropping sample {idx} from details_df because there is another sample with the same unique_patient_id {unique_patient_id} that does not end with {suffix}.")
            details_df.drop(idx, inplace=True)
    return details_df


def process_dnam_file(samplesheet_df,
                      dnam_dir,
                      filename,
                      suffix,
                      samplesheet_col):
    print(f"Processing {filename} for {samplesheet_col}...")
    df = pd.read_csv(filename, index_col=0, delimiter='\t')
    for col in df.columns:
        # If there is a dot in the col name, remove it and everything after it
        if '.' in col:
            new_col = col.split('.')[0]
            df = df.rename(columns={col: new_col})
            col = new_col
        output_file = f'{dnam_dir}/{col}.{suffix}.txt'
        # Set 'dnam_beta_matrix_file' column for this sample in samplesheet_df
        # First check that the sample is in samplesheet_df
        if col not in samplesheet_df.index:
            # If sample name starts with "Constitutional" or "cd", add it to the samplesheet_df with class "Control"
            if col.startswith("Constitutional") or col.startswith("cd"):
                print(f"Adding {col} to samplesheet_df with class 'Control'...")
                samplesheet_df.loc[col, :] = pd.NA
                samplesheet_df.at[col, 'class'] = 'Control'
            else:
                print(f"Warning: Sample {col} in {filename} not found in samplesheet_df index")
                continue
        samplesheet_df.loc[col, samplesheet_col] = output_file
        df[[col]].to_csv(output_file, sep='\t', header=False, index=True)
    return samplesheet_df
