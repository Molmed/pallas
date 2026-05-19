import pandas as pd
import os
import dotenv
from pathlib import Path
from preprocessing.helpers import summarize_patient_details

dotenv.load_dotenv()

data_dir = Path(os.getenv("MLOMIX_PRE_DATA_DIR"))
private_reports_dir = Path(os.getenv("PRIVATE_REPORTS_DIR"))

# Create reports dir if it doesn't exist
reports_dir = private_reports_dir / "cohorts_report"
reports_dir.mkdir(parents=True, exist_ok=True)


# Find all files named patient_details.csv in data dir
# The cohort name is the parent directory of patient_details.csv
patient_summaries = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file == 'patient_details.csv':
            cohort_name = os.path.basename(root)
            details = pd.read_csv(os.path.join(root, file))
            patient_summaries.append(details)

# Merge patient summaries
patient_summaries = pd.concat(patient_summaries, ignore_index=True)

# Print summary stats across all cohorts
summarize_patient_details(patient_summaries)

# Save patient summaries to reports dir
patient_summaries.to_csv(reports_dir / "patient_summaries.csv", index=False)
