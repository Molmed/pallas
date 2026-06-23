# DNAm cal and test
python lib/predict.py \
    --pallas_model_path=/home/mariya/Data/pallas/models/rf_DNAm_20260606_193214/rf_model.joblib \
    --X_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-dnam/features.dnam.csv \
    --y_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-dnam/labels.csv \
    --dataset_name="cal-and-test-dnam" \
    --output_dir=/home/mariya/Development/pallas/reports/output/predictions

# GEX cal and test
python lib/predict.py \
    --pallas_model_path=/home/mariya/Data/pallas/models/rf_GEX_20260606_190416/rf_model.joblib \
    --X_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-gex/features.gex.csv \
    --y_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-gex/labels.csv \
    --dataset_name="cal-and-test-gex" \
    --output_dir=/home/mariya/Development/pallas/reports/output/predictions
