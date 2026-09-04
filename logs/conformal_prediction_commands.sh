# GEX calibration
python lib/predict.py \
    --pallas_model_base_dir=/home/mariya/Data/pallas/models \
    --pallas_model_name=rf_GEX_20260606_190416 \
    --X_path=/home/mariya/Data/pallas/mlomix_datasets/cal-gex/features.gex.csv \
    --y_path=/home/mariya/Data/pallas/mlomix_datasets/cal-gex/labels.csv \
    --dataset_name="cal-gex" \
    --output_dir=/home/mariya/Development/pallas/reports/output/predictions

# Validate GEX conformal prediction
python lib/validate_cp.py --predictions_path=/home/mariya/Development/pallas/reports/output/predictions/rf_GEX_20260606_190416/cal-and-test-gex/cal-and-test-gex_predictions.csv --output_dir=/home/mariya/Development/pallas/reports/output/cp/rf_GEX_20260606_190416 --class_color_tsv_path=/home/mariya/Development/pallas/configs/class_colors.tsv > logs/validate_cp_gex.log

# Validate DNAm conformal prediction
python lib/validate_cp.py --predictions_path=/home/mariya/Development/pallas/reports/output/predictions/rf_DNAm_20260606_193214/cal-and-test-dnam/cal-and-test-dnam_predictions.csv --output_dir=/home/mariya/Development/pallas/reports/output/cp/rf_DNAm_20260606_193214 --class_color_tsv_path=/home/mariya/Development/pallas/configs/class_colors.tsv > logs/validate_cp_dnam.log

# Calibrate GEX
python lib/conformal_prediction/calibrate.py --predictions_path=/home/mariya/Development/pallas/reports/output/predictions/rf_GEX_20260606_190416/cal-and-test-gex/cal-and-test-gex_predictions.csv --output_dir=/home/mariya/Development/pallas/reports/output/cp/rf_GEX_20260606_190416 --base_classifier_name=rf_GEX_20260606_190416

# Calibrate DNAm
python lib/conformal_prediction/calibrate.py --predictions_path=/home/mariya/Development/pallas/reports/output/predictions/rf_DNAm_20260606_193214/cal-dnam/cal-dnam_predictions.csv --output_dir=/home/mariya/Development/pallas/reports/output/cp/rf_DNAm_20260606_193214 --base_classifier_name=rf_DNAm_20260606_193214

# Deploy CP
python lib/conformal_prediction/deploy.py --predictions_path=/home/mariya/Development/pallas/reports/output/predictions/rf_GEX_20260606_190416/cal-and-test-gex/cal-and-test-gex_predictions.csv --base_classifier_name=rf_GEX_20260606_190416 --base_classifier_path=/home/mariya/Data/pallas/models/rf_GEX_20260606_190416/rf_model.joblib
