import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

reports_dir = os.getenv('REPORTS_DIR')
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

processed_data_dir = os.getenv('PROCESSED_DATA_DIR')
sample_report_path = os.getenv('SAMPLE_REPORT_PATH')

# Read in sample report
all_samples_df = pd.read_csv(sample_report_path, index_col=0)

# Get all items in labeled_diagnostic
labeled_diagnostic_df = all_samples_df[
    all_samples_df['data_silo'] == 'labeled_diagnostic']

# Get all unique subtypes
unique_subtypes = labeled_diagnostic_df['subtype'].unique()

# Create a new df with columns: DNAm samples, GEX samples, Total samples, Mean age (sd), Male/Female
report_rows = []
for subtype in unique_subtypes:
    subtype_df = labeled_diagnostic_df[labeled_diagnostic_df['subtype'] == subtype]
    dnam_samples = subtype_df[subtype_df['modality'] == 'DNAm'].shape[0]
    gex_samples = subtype_df[subtype_df['modality'] == 'GEX'].shape[0]
    total_samples = subtype_df.shape[0]
    mean_age = subtype_df['age'].mean()
    sd_age = subtype_df['age'].std()
    male_female_ratio = subtype_df['sex'].value_counts().to_dict()

    report_rows.append({
        'subtype': subtype,
        'DNAm samples': dnam_samples,
        'GEX samples': gex_samples,
        'Total samples': total_samples,
        'Mean age (sd)': f"{mean_age:.1f} ({sd_age:.1f})",
        'Male/Female': f"{male_female_ratio.get('Male', 0)}/{male_female_ratio.get('Female', 0)}"
    })
report_df = pd.DataFrame(report_rows)

# Sort rows so that all subtypes starting with "AML" and "B-ALL" are grouped together.
# Within those groups, sort by Total samples descending

# First, create helper columns for sorting
report_df['sort_prefix'] = report_df['subtype'].apply(lambda x:
    '1' if x.startswith('AML') else
    '2' if x.startswith('B-ALL') else '3')

report_df['total_neg'] = -report_df['Total samples']  # Negative to sort descending

# Now sort by these columns
report_df = report_df.sort_values(by=['sort_prefix', 'total_neg', 'subtype'])

# Remove the helper columns
report_df = report_df.drop(columns=['sort_prefix', 'total_neg'])

# Remove non-leukemic row
report_df = report_df[report_df['subtype'] != 'Non-leukemic']

# Save as CSV
report_df.to_csv(os.path.join(reports_dir, 'labeled_diagnostic_subtypes_report.csv'))
