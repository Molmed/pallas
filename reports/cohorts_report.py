import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

reports_dir = os.getenv('REPORTS_DIR')
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

# Generate age and sex statistics for each cohort
gex_cohorts = [
    'SJC-DS-1009',
    'SJC-panALL',
    'GSE227832',
    'EGAS00001001795',
    'GSE181157',
    'GSE228632',
    'GSE161501'
]

gex_summary = []

for cohort in gex_cohorts:
    print(f"Processing cohort: {cohort}")
    details_path = os.path.join(os.getenv('PROCESSED_DATA_DIR'), cohort, 'patient_details.csv')
    details = pd.read_csv(details_path)

    age_summary = f"{details['age'].mean():.1f} ± ({details['age'].std():.1f})"
    male_female_summary = f"{details['sex'].value_counts().get('Male', 0)} / {details['sex'].value_counts().get('Female', 0)}"
    cohort_summary = {
        'cohort': cohort,
        'num_patients': len(details),
        'mean_age': age_summary,
        'sex_summary': male_female_summary
    }
    gex_summary.append(cohort_summary)

# Convert to df
gex_summary = pd.DataFrame(gex_summary)

# Rename cols
gex_summary.rename(columns={'cohort': 'Cohort', 'num_patients': 'Number of samples',
                    'mean_age': 'Mean age (SD)', 'sex_summary': 'Male/Female'}, inplace=True)

gex_summary_path = os.path.join(reports_dir, 'gex_cohorts_summary.csv')
gex_summary.to_csv(gex_summary_path, index=False)
print(f"GEX cohorts summary saved to {gex_summary_path}")
