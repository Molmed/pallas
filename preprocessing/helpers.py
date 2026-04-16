import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import GEX_EXCLUDE_SUBTYPES

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


def process_gdc_gex(dataset_name,
                    clinical_file,
                    subtypes_file,
                    raw_gex_dir,
                    output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # We only want the subtyped data, use the pre-processed subtypes file as guide
    # create multiindex on Sample ID and Replicate
    # Load the data
    pheno_df = pd.read_csv(subtypes_file, sep='\t', index_col='Unique Sample ID')

    # Only limit to items where 'Project Id' is the given dataset_name
    pheno_df = pheno_df[pheno_df['Project Id'] == dataset_name]

    # For now, let's only keep Primary non-replicate samples
    pheno_df = pheno_df[(pheno_df['Tumor Descriptor'] == 'Primary') &
                        (pheno_df['Replicate'] == False)]

    # Read in details file
    # Index should be "cases.submitter_id"
    details_df = pd.read_csv(clinical_file,
                             header=0,
                             sep='\t',
                             index_col='cases.submitter_id')

    # Only keep cases that are in pheno_df 'Case ID' column
    details_df = details_df[details_df.index.isin(pheno_df['Case ID'])]

    # Keep the following cols
    # diagnoses.age_at_diagnosis
    # demographic.gender
    details_df = details_df[['diagnoses.age_at_diagnosis', 'demographic.gender']]
    details_df.index.name = 'case_id'
    details_df['dataset'] = dataset_name
    details_df = details_df.rename(
        columns={'demographic.gender': 'sex',
                'diagnoses.age_at_diagnosis': 'age'})

    # Convert age to numeric, coerce errors to NaN
    details_df['age'] = pd.to_numeric(details_df['age'], errors='coerce')

    # For age, convert days to years and round to 1 decimal place
    details_df['age'] = (details_df['age'] / 365.25).round(1)

    # For sex, replace M with Male and F with Female
    details_df['sex'] = details_df['sex'].replace({'male': 'Male', 'female': 'Female'})

    details_df = details_df[
        ['dataset', 'sex', 'age']]

    # Drop duplicates
    details_df = details_df[~details_df.index.duplicated(keep='first')]

    # Fix samplesheet_df
    samplesheet_df = pheno_df

    # Set Case ID as the index
    samplesheet_df = samplesheet_df.set_index('Case ID', drop=True)

    # Rename index
    samplesheet_df.index.name = 'sample'

    # Rename columns
    samplesheet_df = samplesheet_df.rename(
        columns={
            'Project Id': 'dataset',
            'subtype': 'class'})

    # Get subtype translation
    tax = Taxonomy()
    samplesheet_df['class'] = tax.translate_subtype_column(
        samplesheet_df['class'],
        split_delimiter=',',
        exclude_subtypes=GEX_EXCLUDE_SUBTYPES)

    # Set data silo to empty string
    samplesheet_df['data_silo'] = ''
    samplesheet_df['batch'] = ''

    # Put multiclass items in multiclass silo
    multi_subtype_rows = samplesheet_df['class'].str.contains(';', na=False)
    if multi_subtype_rows.any():
        samplesheet_df = add_to_silo(
            samplesheet_df,
            samplesheet_df[multi_subtype_rows].index, 'multiclass')

    # Silo excluded and unrecognized
    samplesheet_df = silo_unrecognized(samplesheet_df)
    samplesheet_df = silo_excluded(samplesheet_df)

    # Silo remaining as labeled_diagnostic
    samplesheet_df = silo_remaining(samplesheet_df)

    # Convert index to string
    samplesheet_df.index = samplesheet_df.index.astype(str)

    samplesheet_df['gex_feature_counts_file'] = None
    for idx, row in samplesheet_df.iterrows():
        # Open the filename and extract the gene_name and unstranded columns
        gex_file = f"{raw_gex_dir}/{row['File Name']}"
        output_file = f'{output_dir}/{idx}.txt'

        try:
            # Does output file already exist?
            if os.path.exists(output_file):
                samplesheet_df.at[idx, 'gex_feature_counts_file'] = output_file
                continue

            gex_df = pd.read_csv(gex_file, sep='\t', header=1, index_col=1)
            # Remove the first four rows which are summary stats
            gex_df = gex_df.iloc[5:, :]
            # Just keep unstranded column
            gex_df = gex_df[['unstranded']]

            # Save to a new file
            gex_df.to_csv(output_file, sep='\t', header=False, index=True)

            # Save in samplesheet
            samplesheet_df.at[idx, 'gex_feature_counts_file'] = output_file

        except Exception as e:
            print(f"Error reading {gex_file} for {idx}: {e}")
            continue

    # Remove all rows from spreadsheet without feature_counts
    samplesheet_df = samplesheet_df[samplesheet_df['gex_feature_counts_file'].notna()]

    # Order columns
    samplesheet_df = format_mlomix_columns(samplesheet_df)

    return samplesheet_df, details_df
