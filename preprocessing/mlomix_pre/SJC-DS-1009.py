import pandas as pd
import os
from preprocessing.taxonomy import Taxonomy
from preprocessing.constants import \
    RAW_DATA_DIR, PREPROCESSED_DATA_DIR, GEX_EXCLUDE_SUBTYPES
from preprocessing.helpers import add_to_silo, silo_unlabeled, \
    silo_unrecognized, silo_excluded, silo_remaining, \
        format_mlomix_columns, summarize_patient_details

dataset_name = 'SJC-DS-1009'
print(f"Processing {dataset_name}...")

# INPUT FILES ########
data_path = f'{RAW_DATA_DIR}/{dataset_name}'
raw_data_path = f'{data_path}/feature_counts'
pheno_input_file = f'{data_path}/SAMPLE_INFO.txt'

# OUTPUT FILES ########
preprocessed_dir = f'{PREPROCESSED_DATA_DIR}/{dataset_name}'
samplesheet_output_file = \
    f'{preprocessed_dir}/samplesheet.csv'
details_file = \
    f'{preprocessed_dir}/patient_details.csv'

# Create output directory if it doesn't exist
os.makedirs(preprocessed_dir, exist_ok=True)

# PAN-ALL DATA FOR DE-DUPLICATION ########
# Load the data
panall_data_path = f'{RAW_DATA_DIR}/SJC-panALL'
panall_raw_data_path = f'{panall_data_path}/BALL-1988S-HTSeq'
panall_pheno_input_file = f'{panall_data_path}/B-ALL-subtyping.xlsx'

panall_df = pd.read_excel(panall_pheno_input_file,
                          sheet_name='ST1.Cohort',
                          header=0,
                          index_col='patient')
# Get index of first row where patient is NaN
nan_idx = panall_df.index.get_indexer_for(
    panall_df.index[panall_df.index.isna()])[0]

# Drop all rows after the first NaN patient
panall_df = panall_df.iloc[:nan_idx]

# PROCESS PHENO ########
print("Processing phenotype data...")
samplesheet_df = pd.read_csv(pheno_input_file, sep='\t', header=0)

# Get all rows where sample_name is duplicated
duplicated_sample_names = samplesheet_df[
    samplesheet_df.duplicated(subset='sample_name')]['sample_name']

# Remove second occurrence and print a warning
if not duplicated_sample_names.empty:
    print("Warning: The following sample names are duplicated and secondary occurrences will be removed:")
    print(duplicated_sample_names.tolist())
    # Remove second occurrence
    samplesheet_df = samplesheet_df[~samplesheet_df['sample_name'].duplicated(keep='first')]

# Rename sample_name to sample
samplesheet_df = samplesheet_df.rename(columns={'sample_name': 'sample'})

# Set index to sample
samplesheet_df.set_index('sample', inplace=True)

# Extract the patient details to separate csv
details_df = samplesheet_df.copy()

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''

# Double-check the NOS samples, some of them actually have a subtype!
# Print items where sj_diseas contains "NOS" but attr_diagnosis
# contains the string "_subtype" but does not contain "other"
nos_samples_with_subtype = samplesheet_df[
    samplesheet_df['sj_diseases'].str.contains('NOS', na=False) &
    samplesheet_df['attr_diagnosis'].str.contains('_subtype:', na=False) &
    ~samplesheet_df['attr_diagnosis'].str.contains('other', na=False)
]
if not nos_samples_with_subtype.empty:
    print("Warning: The following NOS samples actually have a subtype:")
    print(nos_samples_with_subtype.index.tolist())

    # Copy attr_diagnosis to sj_diseases for these samples
    for sample in nos_samples_with_subtype.index:
        attr_diag = samplesheet_df.at[sample, 'attr_diagnosis']
        # Extract the subtype from attr_diagnosis
        if '_subtype:' in attr_diag:
            subtype = attr_diag.split('_subtype:')[1].split('_')[0]
            samplesheet_df.at[sample, 'sj_diseases'] = subtype
        else:
            print(f"Error: Could not extract subtype for sample {sample}. Please check the data.")

# Sometimes a sample has a subtype lurking in sj_associated_diagnoses
# For each NOS sample, check if sj_associated_diagnoses contains a known subtype
nos_samples = samplesheet_df[
    samplesheet_df['sj_diseases'].str.contains('NOS', na=False)]
# The subtypes we care about are:
known_subtypes = [
    'BALLHYPO',
    'BALLPAX5',
    'BALLHYPER',
    'AMLNUP98',
    'AMLKMT2A',
    'AMLNPM1',
    'AMLPICALMMLLT10',
    'AMLBCL11B',
    'AMLMECOM',
    'AMLDEKNUP214',
    'AMLUBTF',
    'AMMLKMT2A'
]
for sample in nos_samples.index:
    associated_diagnoses = samplesheet_df.at[sample, 'sj_associated_diagnoses']
    for subtype in known_subtypes:
        # If sj associated diagnoses contains the subtype as substring
        if subtype in associated_diagnoses:
            print(f"Info: Sample {sample} has known subtype {subtype} in sj_associated_diagnoses.")
            # Copy it to sj_diseases
            samplesheet_df.at[sample, 'sj_diseases'] = subtype
            break  # No need to check other subtypes

# Drop all samples where sample_type is not "Diagnostic" or "Relapse"
valid_sample_types = ['Diagnosis', 'Relapse']
invalid_samples = samplesheet_df[
    ~samplesheet_df['sample_type'].isin(valid_sample_types)].index
if not invalid_samples.empty:
    print("Warning: The following samples have invalid sample_type and will be removed:")
    print(invalid_samples.tolist())
    samplesheet_df = samplesheet_df[
        samplesheet_df['sample_type'].isin(valid_sample_types)]

# Rename subtype
samplesheet_df = samplesheet_df.rename(columns={'sj_diseases': 'class'})

# Double-check "other" cases, there might be a biomarker that can be used
# First, find all items whose subtype is "B-other" or "AML-other" or "ALLNOS"
# but whose attr_subtype_biomarkers is not empty
other_samples = samplesheet_df[
    (samplesheet_df['class'].str.contains(
        'B-other|AML-other|ALLNOS|AMLNOS|AMLETS', na=False)) &
    (~samplesheet_df['attr_subtype_biomarkers'].isna()) &
    (samplesheet_df['attr_subtype_biomarkers'] != '') &
    (samplesheet_df['attr_subtype_biomarkers'] != 'Not Available')
]
if not other_samples.empty:
    print("Warning: The following 'other' samples have biomarkers that might help refine the subtype:")
    print(other_samples.index.tolist())
    # Print their biomarkers
    for sample in other_samples.index:
        biomarkers = samplesheet_df.at[sample, 'attr_subtype_biomarkers']
        print(f"Sample {sample} has biomarkers: {biomarkers}")
        samplesheet_df.at[sample, 'class'] = biomarkers

# Translate subtype
tax = Taxonomy()
samplesheet_df['class'] = tax.translate_subtype_column(
    samplesheet_df['class'],
    exclude_subtypes=GEX_EXCLUDE_SUBTYPES,
    split_delimiter=',')

# Set data_silo to empty string
samplesheet_df['data_silo'] = ''

# EXCLUDED SILO
samplesheet_df = silo_unrecognized(samplesheet_df)
samplesheet_df = silo_excluded(samplesheet_df)

# Manually add items that seem to have labeling inconsistencies
# e.g. different subtypes specificed in different columns
manual_exclusions = [
    'SJBALL032645_D1',  # ALL and CML
    'SJAML061502_D1',  # KMT2A and CEBPA,
    'SJAML073691_D1',  # KMT2A and CEBPA,
    'SJBALL032304_D1',  # HeH and PAX5,
    'SJBALL032020_D1',  # HeH and PAX5,
    'SJBALL032489_D1',  # HeH and PAX5,
    'SJBALL032630_D1',  # HeH and PAX5,
    'SJBALL032755_D1',  # HeH and PAX5,
    'SJBALL031979_D1',  # HeH and PAX5,
    'SJAML030330_D1',  # Identified as B-ALL
    'SJBALL015199_D1',  # Identified as T-ALL
    'SJBALL022672_D1',  # Identified as T-ALL
    'SJAML032072_D1',   # Identified as T-ALL,
    'SJAML030210_D2',   # KMT2A and NUP98
    'SJCBF100_E',       # what is this suffix,
    'SJBALL020113_D1',   # Both HeH and CEBPB-IGH,
    'SJBALL021790_D1',   # Both HeH and CEBPB-IGH,
    'SJBALL032135_D1'    # ETV6-RUNX1 and PAX5,
    'SJTALL031944_D1'   # T-ALL and MLLT10-PICALM,
    'SJTALL030880_D1'   # T-ALL and MLLT10-PICALM,
    'SJTALL030907_D1'   # T-ALL and MLLT10-PICALM,
    'SJTALL031798_D1'   # T-ALL and MLLT10-PICALM,
    'SJTALL031662_D1'   # T-ALL and MLLT10-PICALM,
    ]
samplesheet_df = add_to_silo(samplesheet_df,
                             manual_exclusions,
                             'excluded')

# If attr_diagnosis says the same is in relapse or remission remove it
# it is unclear if this was the status when the sample was taken
ambiguous_samples = samplesheet_df[
    samplesheet_df['attr_diagnosis'].str.contains('in relapse', na=False) |
    samplesheet_df['attr_diagnosis'].str.contains('in remission', na=False)
].index
if not ambiguous_samples.empty:
    print("Warning: The following samples have ambiguous status (relapse or remission) and will be excluded:")
    print(ambiguous_samples.tolist())
    samplesheet_df = add_to_silo(
        samplesheet_df, ambiguous_samples, 'excluded')

# Detect down syndrome samples
down_syndrome_samples = samplesheet_df[
    samplesheet_df['attr_diagnosis'].str.contains('DOWN SYNDROME', na=False) |
    samplesheet_df['attr_diagnosis'].str.contains('_subtype:Down', na=False)
].index

# Next, get all sample names starting with "SJDOWN"
down_syndrome_samples_2 = samplesheet_df[
    samplesheet_df.index.str.startswith('SJDOWN')].index

# Combine both sets of down syndrome samples
down_syndrome_samples = down_syndrome_samples.union(down_syndrome_samples_2)

# Put all down syndrome samples into the 'excluded' silo
samplesheet_df = add_to_silo(
    samplesheet_df, down_syndrome_samples, 'excluded')

# Check sj_associated_diagnoses to see if there are samples
# with multiple cancers that should be excluded
keywords = ['Ganglioneuroblastoma',
            'Ganglioneuroma',
            'Papillary Thyroid Cancer',
            'TMDS',
            'Ewings',
            'NORM',
            'THFO',
            'Neuroblastoma',
            'ALAL',
            'MBLWNT',
            'THPA',
            'TAML',
            'SCCNOS',
            'Epithelioid Sarcoma',
            'THERAPY-RELATED']
excluded_rows = samplesheet_df[
    samplesheet_df['sj_associated_diagnoses'].str.contains(
        '|'.join(keywords), na=False)]
if not excluded_rows.empty:  # Check if the DataFrame is not empty
    samplesheet_df = add_to_silo(samplesheet_df,
                                 excluded_rows.index,
                                 'excluded')
    # And copy the sj_associated_diagnoses value to subtype
    samplesheet_df.loc[excluded_rows.index, 'class'] = \
        samplesheet_df.loc[excluded_rows.index, 'sj_associated_diagnoses']

# Also some keywords in 'attr_diagnosis' indicate exclusion
keywords = ['THERAPY-RELATED']
excluded_rows = samplesheet_df[
    samplesheet_df['attr_diagnosis'].str.contains(
        '|'.join(keywords), na=False)]
if not excluded_rows.empty:  # Check if the DataFrame is not empty
    samplesheet_df = add_to_silo(samplesheet_df,
                                 excluded_rows.index,
                                 'excluded')
    # And copy the attr_diagnosis value to subtype
    samplesheet_df.loc[excluded_rows.index, 'class'] = \
        samplesheet_df.loc[excluded_rows.index, 'attr_diagnosis']

# Finally, if there are any samples where 'sj_associated_diagnoses' contains
# both "BALL" and "AML", then exclude them
multi_subtype_rows = samplesheet_df[
    samplesheet_df['sj_associated_diagnoses'].str.contains('BALL', na=False) &
    samplesheet_df['sj_associated_diagnoses'].str.contains('AML', na=False)
]
if not multi_subtype_rows.empty:
    samplesheet_df = add_to_silo(samplesheet_df,
                                 multi_subtype_rows.index,
                                 'excluded')
    # And copy the sj_associated_diagnoses value to subtype
    samplesheet_df.loc[multi_subtype_rows.index, 'class'] = \
        samplesheet_df.loc[multi_subtype_rows.index, 'sj_associated_diagnoses']

# MULTICLASS SILO
# If "secondary" in attr_diagnosis, then it's multiclass
# Put them into the "multiclass" silo
multi_subtype_rows = samplesheet_df[
    'attr_diagnosis'].str.contains('Secondary', na=False)
if multi_subtype_rows.any():
    samplesheet_df = add_to_silo(samplesheet_df,
                                 samplesheet_df[multi_subtype_rows].index,
                                 'multiclass')
    # And copy the attr_diagnosis value to subtype
    samplesheet_df.loc[multi_subtype_rows, 'class'] = \
        samplesheet_df.loc[multi_subtype_rows, 'attr_diagnosis']

# UNLABELED SILO
samplesheet_df = silo_unlabeled(samplesheet_df)

# RELAPSE SILO
# Set data_silo to "relapse" if any of the following is true:
# - sample_type contains "Relapse"
# - sample name ends with "_R"
relapse_mask = samplesheet_df[
    'sample_type'].str.contains('Relapse', na=False) | \
    samplesheet_df.index.str.endswith('_R')
relapse_items = samplesheet_df[relapse_mask].index
samplesheet_df = add_to_silo(samplesheet_df,
                             relapse_items,
                             'relapse')

# DE-DUPE PANALL DATA
# Make a copy of the samplesheet_df index for panALL items,
# only keeping the string before the first underscore
samplesheet_df_index_short = samplesheet_df.index.str.split('_').str[0]
samplesheet_df['panALL_index'] = samplesheet_df_index_short

# Get all items whose panALL_index is in panall_df index
panall_items = samplesheet_df[
    samplesheet_df['panALL_index'].isin(panall_df.index)].index

# Put them in the "replicate" silo
samplesheet_df = add_to_silo(samplesheet_df,
                             panall_items,
                             'replicate')

# Find all items in samplesheet_df whose sample name ends in a
# D followed by a digit that isnt 1
digit_suffix_items = samplesheet_df[
    samplesheet_df.index.str.match(r'.*D\d$') &  # Ends with a digit
    ~samplesheet_df.index.str.endswith('D1')  # But not D1
].index

# Get short names for these items and for D1 items
short_names_digit_suffix = samplesheet_df.loc[digit_suffix_items, 'panALL_index']
d1_items = samplesheet_df[samplesheet_df.index.str.endswith('D1')].index
short_names_d1 = samplesheet_df.loc[d1_items, 'panALL_index']

# Find items with digit suffix that have a corresponding D1 item
digit_suffix_items_with_d1 = digit_suffix_items[
    short_names_digit_suffix.isin(short_names_d1)]

# Find the items that do not yet have a silo
digit_suffix_items_no_silo = digit_suffix_items_with_d1[
    samplesheet_df.loc[digit_suffix_items_with_d1, 'data_silo'] == '']

# Put these into the "replicate" silo
samplesheet_df = add_to_silo(samplesheet_df,
                             digit_suffix_items_no_silo,
                             'replicate')

# Add remaining to labeled_diagnostic silo
samplesheet_df = silo_remaining(samplesheet_df)

# Rename library
samplesheet_df = samplesheet_df.rename(
    columns={'attr_library_selection_protocol': 'batch'})

# Rename file_path
samplesheet_df = samplesheet_df.rename(columns={'file_path': 'gex_feature_counts_file'})

# In each file path, replace "/FEATURE_COUNTS" with raw_data_path
samplesheet_df['gex_feature_counts_file'] = samplesheet_df[
    'gex_feature_counts_file'].str.replace('/FEATURE_COUNTS', raw_data_path)


# Copy to new temp df
temp_df = samplesheet_df.copy()
# Keep the following cols: sample, sample_type, sj_long_disease_name, attr_diagnosis, attr_subtype_biomarkers, sj_associated_diagnoses, subtype, sj_datasets, sj_disease_hierarchy_code, sj_disease_hierarchy_name, data_silo
temp_df = temp_df[['sample_type', 'sj_long_disease_name',
                   'attr_diagnosis', 'attr_subtype_biomarkers',
                   'sj_associated_diagnoses', 'class', 'sj_datasets',
                   'sj_disease_hierarchy_code', 'sj_disease_hierarchy_name',
                   'data_silo']]
# Reorder the columns so sample, sample_type, subtype and data_silo are first
temp_df = temp_df[['sample_type', 'class', 'data_silo',
                   'sj_long_disease_name', 'attr_diagnosis',
                   'attr_subtype_biomarkers', 'sj_associated_diagnoses',
                   'sj_datasets', 'sj_disease_hierarchy_code',
                   'sj_disease_hierarchy_name']]

# Only keep labeled_diagnostic silo
temp_df = temp_df[temp_df['data_silo'] == 'labeled_diagnostic']

# Keep only the columns we need
samplesheet_df = samplesheet_df[
    ['class', 'batch', 'gex_feature_counts_file', 'data_silo']]

# Set 'dataset' column to dataset name
samplesheet_df['dataset'] = dataset_name

# Order columns
samplesheet_df = format_mlomix_columns(samplesheet_df)

# Save the samplesheet
samplesheet_df.to_csv(samplesheet_output_file)

# Finish processing details
details_df['dataset'] = dataset_name
details_df = details_df.rename(
    columns={'attr_sex': 'sex',
             'attr_age_at_diagnosis': 'age'})

# Coerce age to numeric, forcing errors to NaN
details_df['age'] = pd.to_numeric(details_df['age'], errors='coerce')

# Only keep details_df rows that are in samplesheet_df
details_df = details_df.loc[details_df.index.intersection(samplesheet_df.index)]

# Replace sex "Not Available" with NaN
details_df['sex'] = details_df['sex'].replace('Not Available', pd.NA)

# Create new col "unique_patient_id" which is the index but removing everything after underscore
details_df['unique_patient_id'] = details_df.index.str.split('_').str[0]


def deduplicate_using_sample_suffix(suffix):
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


# The largest diagnostic value we have is D8 and largest relapse is R4
diagnostic_sampling_number = 8
for i in range(diagnostic_sampling_number, 1, -1):
    suffix = f'_D{i}'
    deduplicate_using_sample_suffix(suffix)

relapse_sampling_number = 4
for i in range(relapse_sampling_number, 0, -1):
    suffix = f'_R{i}'
    deduplicate_using_sample_suffix(suffix)

# Find all items ending in any letter except "D" and dedupe them
other_suffix_items = details_df[
    details_df.index.str.match(r'.*[A-CE-Z]$')]

# Get all unique suffixes from these items
other_suffixes = other_suffix_items.index.str.extract(r'.*(_[A-CE-Z])$')[0].unique()
for suffix in other_suffixes:
    deduplicate_using_sample_suffix(suffix)

# Check for duplicate unique_patient_id, report if any left
duplicate_patients = details_df[details_df.duplicated(subset='unique_patient_id', keep=False)]
if not duplicate_patients.empty:
    print("Warning: The following unique_patient_id values are duplicated in details_df after deduplication:")
    print(duplicate_patients['unique_patient_id'].unique())

details_df = details_df[
    ['dataset', 'sex', 'age']]

summarize_patient_details(details_df)

details_df.to_csv(details_file)
