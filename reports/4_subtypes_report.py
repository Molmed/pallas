import pandas as pd
import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()

base_dir = Path.cwd()
REPORTS_PATH = base_dir / "reports" / "output"
sample_report_path = REPORTS_PATH / "sample_report.csv"
private_reports_dir = Path(os.getenv("PRIVATE_REPORTS_DIR"))
patient_details_report_path = private_reports_dir / "cohorts_report" / "patient_summaries.csv"
output_report_path = REPORTS_PATH / "subtypes_report.csv"

# Read in sample report
all_samples_df = pd.read_csv(sample_report_path, index_col=0)

# For samples of the format "TARGET-20-PANXAF-D", remove trailing "-D"
all_samples_df.index = all_samples_df.index.str.replace(r'-D$', '', regex=True)

# Read in patient details report
patient_details_df = pd.read_csv(patient_details_report_path, index_col=0)

# Join them on index
all_samples_df = all_samples_df.join(patient_details_df, how='left', rsuffix='_patient')

# Get all items in labeled_diagnostic
labeled_diagnostic_df = all_samples_df[
    all_samples_df['Data silo'] == 'labeled_diagnostic']

# Get all unique subtypes
unique_subtypes = labeled_diagnostic_df['Subtype'].unique()

# Create a new df with columns: DNAm samples, GEX samples, Total samples, Mean age (sd), Male/Female
report_rows = []
for subtype in unique_subtypes:
    subtype_df = labeled_diagnostic_df[labeled_diagnostic_df['Subtype'] == subtype]
    # All samples where 'DNAm data split' is not empty
    dnam_samples = subtype_df[subtype_df['DNAm data split'].notna()].shape[0]
    gex_samples = subtype_df[subtype_df['GEX data split'].notna()].shape[0]
    bimodal_samples = subtype_df[subtype_df['Bimodal data split'].notna()].shape[0]
    total_samples = subtype_df.shape[0]
    mean_age = subtype_df['age'].mean()
    sd_age = subtype_df['age'].std()
    male_female_ratio = subtype_df['sex'].value_counts().to_dict()

    report_rows.append({
        'Subtype': subtype,
        'Total samples': total_samples,
        'GEX samples': gex_samples,
        'DNAm samples': dnam_samples,
        'Bimodal samples': bimodal_samples,
        'Mean age (sd)': f"{mean_age:.1f} ({sd_age:.1f})",
        'Male/Female': f"{male_female_ratio.get('Male', 0)}/{male_female_ratio.get('Female', 0)}"
    })
report_df = pd.DataFrame(report_rows)

# Sort rows so that all subtypes starting with "AML" and "B-ALL" are grouped together.
# Within those groups, sort by Total samples descending

# First, create helper columns for sorting
report_df['sort_prefix'] = report_df['Subtype'].apply(lambda x:
    '1' if x.startswith('AML') else
    '2' if x.startswith('B-ALL') else '3')

report_df['total_neg'] = -report_df['Total samples']  # Negative to sort descending

# Now sort by these columns
report_df = report_df.sort_values(by=['sort_prefix', 'total_neg', 'Subtype'])

# Remove the helper columns
report_df = report_df.drop(columns=['sort_prefix', 'total_neg'])

# Remove non-leukemic row
# report_df = report_df[report_df['Subtype'] != 'Non-leukemic']

# For non-leukemic row, set age and sex to n/a
report_df.loc[report_df['Subtype'] == 'Non-leukemic', ['Mean age (sd)', 'Male/Female']] = 'n/a'

# The last row should have "All subtypes" as the subtype and the sum of all samples, mean age and sex
total_samples = report_df['Total samples'].sum()
total_dnam_samples = report_df['DNAm samples'].sum()
total_gex_samples = report_df['GEX samples'].sum()
total_bimodal_samples = report_df['Bimodal samples'].sum()
mean_age = labeled_diagnostic_df['age'].mean()
sd_age = labeled_diagnostic_df['age'].std()
male_female_ratio = labeled_diagnostic_df['sex'].value_counts().to_dict()
report_df = pd.concat([report_df, pd.DataFrame([{
    'Subtype': 'TOTALS',
    'Total samples': total_samples,
    'DNAm samples': total_dnam_samples,
    'GEX samples': total_gex_samples,
    'Bimodal samples': total_bimodal_samples,
    'Mean age (sd)': f"{mean_age:.1f} ({sd_age:.1f})",
    'Male/Female': f"{male_female_ratio.get('Male', 0)}/{male_female_ratio.get('Female', 0)}"
}])], ignore_index=True)

# Save as CSV
report_df.to_csv(
    output_report_path,
    index=False)
