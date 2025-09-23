import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

reports_dir = os.getenv('REPORTS_DIR')
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)


def path_to_df(details_path):
    details = pd.read_csv(details_path)

    age_summary = f"{details['age'].mean():.1f} ± ({details['age'].std():.1f})"
    male_female_summary = f"{details['sex'].value_counts().get('Male', 0)} / {details['sex'].value_counts().get('Female', 0)}"
    cohort_summary = {
        'cohort': cohort,
        'num_patients': len(details),
        'mean_age': age_summary,
        'sex_summary': male_female_summary
    }
    return cohort_summary


def save_dfs_to_csv(dfs, file_path):
    dfs = pd.DataFrame(dfs)

    # Rename cols
    dfs.rename(columns={'cohort': 'Cohort', 'num_patients': 'Number of samples',
                        'mean_age': 'Mean age (SD)', 'sex_summary': 'Male/Female'}, inplace=True)
    dfs.to_csv(file_path, index=False)


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
    details_path = os.path.join(
        os.getenv('PROCESSED_DATA_DIR'),
        cohort,
        'patient_details.csv')
    details = pd.read_csv(details_path)
    gex_summary.append(path_to_df(details_path))

save_dfs_to_csv(gex_summary, os.path.join(reports_dir, 'gex_cohorts_details.csv'))

dnam_cohorts = [
    'krali2021',
    'GSE190931',
    'krali2023',
    'enblad2025'
    ]
dnam_summary = []
for cohort in dnam_cohorts:
    details_path = os.path.join(
        os.getenv('PROCESSED_DATA_DIR'),
        'dnam',
        'pheno',
        cohort,
        'patient_details.csv')
    details = pd.read_csv(details_path)
    dnam_summary.append(path_to_df(details_path))
save_dfs_to_csv(dnam_summary, os.path.join(reports_dir, 'dnam_cohorts_details.csv'))
