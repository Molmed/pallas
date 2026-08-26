import pandas as pd
import os
from conformist import FNRCoP, PredictionDataset

OUTPUT_DIR = "/home/mariya/Development/pallas/reports/output/cp/rf_GEX_20260606_190416"
# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load test data
X_TEST_CSV_PATH = "/home/mariya/Data/pallas/mlomix_datasets/test-gex/features.gex.csv"
Y_TEST_TSV_PATH = "/home/mariya/Data/pallas/mlomix_datasets/test-gex/labels.csv"
X_test = pd.read_csv(X_TEST_CSV_PATH, index_col=0)
y_test = pd.read_csv(Y_TEST_TSV_PATH, index_col=0, sep="\t").squeeze()
X_test.shape, y_test.shape

PREDS_CAL_CSV_PATH = "/home/mariya/Development/pallas/reports/output/predictions/rf_GEX_20260606_190416/cal-gex/cal-gex_predictions.csv"
# Load calibration predictions
preds_cal = pd.read_csv(PREDS_CAL_CSV_PATH)
preds_cal

PREDS_TEST_CSV_PATH = "/home/mariya/Development/pallas/reports/output/predictions/rf_GEX_20260606_190416/test-gex/test-gex_predictions.csv"
# Load test predictions
preds_test = pd.read_csv(PREDS_TEST_CSV_PATH)
preds_test


# What classes differ between calibration and test sets?
cal_classes = set(preds_cal['true_label'].unique())
test_classes = set(preds_test['true_label'].unique())
non_overlapping_classes = cal_classes - test_classes
if non_overlapping_classes:
    print("Classes in calibration set but not in test set:", non_overlapping_classes)


print(f"Calibration set has {len(preds_cal)} samples and {len(cal_classes)} classes.")
# Remove all items with non-overlapping classes from the calibration set
preds_cal = preds_cal[~preds_cal['true_label'].isin(non_overlapping_classes)]
print(f"After removing non-overlapping classes, calibration set has {len(preds_cal)} samples and {len(preds_cal['true_label'].unique())} classes.")

# Show new class breakdown of calibration set
print("New class distribution in calibration data:")
print(preds_cal['true_label'].value_counts())

# Get just class names as list
class_names = preds_cal['true_label'].unique().tolist()
keep_col_names = ['true_label', 'predicted_label', 'id'] + class_names

preds_cal = preds_cal[keep_col_names]
preds_test = preds_test[keep_col_names]

# Read in formatted predictions
calpd = PredictionDataset(df=preds_cal, dataset_name='cal')
testpd = PredictionDataset(df=preds_test, dataset_name='test')


ALPHA = 0.05  # Try a smaller alpha (was 0.1)
N_RUNS=10 # TODO: increase


mcp = FNRCoP(calpd, alpha=ALPHA)
mcp.calibrate()


# Validation trial and reports
mcp = FNRCoP(calpd, alpha=ALPHA)
trial = mcp.do_validation_trial(n_runs=N_RUNS)
trial.run_reports(OUTPUT_DIR)


formatted_predictions = mcp.predict(testpd,
                                    OUTPUT_DIR)
