import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

reports_dir = os.getenv('REPORTS_DIR')
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

processed_data_dir = os.getenv('PROCESSED_DATA_DIR')
sample_report_path = os.getenv('SAMPLE_REPORT_PATH')
data_silos_report_path = os.getenv('DATA_SILOS_REPORT_PATH')

# Read in sample report
all_samples_df = pd.read_csv(sample_report_path, index_col=0)

TOTAL_SAMPLES_LABEL = 'Total samples'
DNAM_SAMPLES_LABEL = 'DNAm samples'
GEX_SAMPLES_LABEL = 'GEX samples'
FULL_DATASET_LABEL = 'Full dataset'
UNLABELED_SAMPLES_LABEL = 'Unlabeled samples ("unlabeled")'
MISC_SAMPLES_LABEL = 'Misc samples ("misc")'
LABELED_DIAGNOSTIC_SAMPLES_LABEL = 'Labeled diagnostic samples ("labeled_diagnostic")'
TRAINING_SAMPLES_LABEL = 'Feature selection, training & cross-validation samples ("training")'
CALIBRATION_SAMPLES_LABEL = 'Conformal calibration & model testing samples ("calibration")'
TEST_SAMPLES_LABEL = 'Conformal prediction testing samples ("test")'
summary_df = pd.DataFrame(columns=[
    TOTAL_SAMPLES_LABEL, DNAM_SAMPLES_LABEL, GEX_SAMPLES_LABEL], index=[
    FULL_DATASET_LABEL, UNLABELED_SAMPLES_LABEL,
    MISC_SAMPLES_LABEL, LABELED_DIAGNOSTIC_SAMPLES_LABEL,
    TRAINING_SAMPLES_LABEL,
    CALIBRATION_SAMPLES_LABEL,
    TEST_SAMPLES_LABEL])
# Fill in the dataframe
summary_df.loc[FULL_DATASET_LABEL, TOTAL_SAMPLES_LABEL] = all_samples_df.shape[0]
summary_df.loc[FULL_DATASET_LABEL, DNAM_SAMPLES_LABEL] = all_samples_df[all_samples_df['modality'] == 'DNAm'].shape[0]
summary_df.loc[FULL_DATASET_LABEL, GEX_SAMPLES_LABEL] = all_samples_df[all_samples_df['modality'] == 'GEX'].shape[0]
summary_df.loc[UNLABELED_SAMPLES_LABEL, TOTAL_SAMPLES_LABEL] = all_samples_df[all_samples_df['data_silo'] == 'unlabeled'].shape[0]
summary_df.loc[UNLABELED_SAMPLES_LABEL, DNAM_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_silo'] == 'unlabeled') & (all_samples_df['modality'] == 'DNAm')].shape[0]
summary_df.loc[UNLABELED_SAMPLES_LABEL, GEX_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_silo'] == 'unlabeled') & (all_samples_df['modality'] == 'GEX')].shape[0]

# Misc samples are all that are not unlabeled or labeled_diagnostic
misc_mask = ~all_samples_df['data_silo'].isin(['unlabeled', 'labeled_diagnostic'])
summary_df.loc[MISC_SAMPLES_LABEL, TOTAL_SAMPLES_LABEL] = all_samples_df[misc_mask].shape[0]
summary_df.loc[MISC_SAMPLES_LABEL, DNAM_SAMPLES_LABEL] = all_samples_df[misc_mask & (all_samples_df['modality'] == 'DNAm')].shape[0]
summary_df.loc[MISC_SAMPLES_LABEL, GEX_SAMPLES_LABEL] = all_samples_df[misc_mask & (all_samples_df['modality'] == 'GEX')].shape[0]

summary_df.loc[LABELED_DIAGNOSTIC_SAMPLES_LABEL, TOTAL_SAMPLES_LABEL] = all_samples_df[all_samples_df['data_silo'] == 'labeled_diagnostic'].shape[0]
summary_df.loc[LABELED_DIAGNOSTIC_SAMPLES_LABEL, DNAM_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_silo'] == 'labeled_diagnostic') & (all_samples_df['modality'] == 'DNAm')].shape[0]
summary_df.loc[LABELED_DIAGNOSTIC_SAMPLES_LABEL, GEX_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_silo'] == 'labeled_diagnostic') & (all_samples_df['modality'] == 'GEX')].shape[0]

summary_df.loc[TRAINING_SAMPLES_LABEL, TOTAL_SAMPLES_LABEL] = all_samples_df[all_samples_df['data_split'] == 'train'].shape[0]
summary_df.loc[TRAINING_SAMPLES_LABEL, DNAM_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_split'] == 'train') & (all_samples_df['modality'] == 'DNAm')].shape[0]
summary_df.loc[TRAINING_SAMPLES_LABEL, GEX_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_split'] == 'train') & (all_samples_df['modality'] == 'GEX')].shape[0]
summary_df.loc[CALIBRATION_SAMPLES_LABEL, TOTAL_SAMPLES_LABEL] = all_samples_df[all_samples_df['data_split'] == 'calibration'].shape[0]
summary_df.loc[CALIBRATION_SAMPLES_LABEL, DNAM_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_split'] == 'calibration') & (all_samples_df['modality'] == 'DNAm')].shape[0]
summary_df.loc[CALIBRATION_SAMPLES_LABEL, GEX_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_split'] == 'calibration') & (all_samples_df['modality'] == 'GEX')].shape[0]
summary_df.loc[TEST_SAMPLES_LABEL, TOTAL_SAMPLES_LABEL] = all_samples_df[all_samples_df['data_split'] == 'test'].shape[0]
summary_df.loc[TEST_SAMPLES_LABEL, DNAM_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_split'] == 'test') & (all_samples_df['modality'] == 'DNAm')].shape[0]
summary_df.loc[TEST_SAMPLES_LABEL, GEX_SAMPLES_LABEL] = all_samples_df[(all_samples_df['data_split'] == 'test') & (all_samples_df['modality'] == 'GEX')].shape[0]

print(summary_df)
# Save as CSV
summary_df.to_csv(
    data_silos_report_path,
    index=True)
print(f"Saved data silos report to {data_silos_report_path}")
