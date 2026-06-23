# GEX cal and test
python lib/predict.py \
    --pallas_model_base_dir=/home/mariya/Data/pallas/models \
    --pallas_model_name=rf_GEX_20260606_190416 \
    --X_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-gex/features.gex.csv \
    --y_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-gex/labels.csv \
    --dataset_name="cal-and-test-gex" \
    --output_dir=/home/mariya/Development/pallas/reports/output/predictions

python lib/evaluate_rf.py \
    --pallas_model_base_dir=/home/mariya/Data/pallas/models \
    --pallas_model_name=rf_GEX_20260606_190416 \
    --predictions_base_dir=/home/mariya/Development/pallas/reports/output/predictions \
    --dataset_name="cal-and-test-gex" \
    --output_dir=/home/mariya/Development/pallas/reports/output/evaluation

# DNAm cal and test
python lib/predict.py \
    --pallas_model_base_dir=/home/mariya/Data/pallas/models \
    --pallas_model_name=rf_DNAm_20260606_193214 \
    --X_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-dnam/features.dnam.csv \
    --y_path=/home/mariya/Data/pallas/mlomix_datasets/cal-and-test-dnam/labels.csv \
    --dataset_name="cal-and-test-dnam" \
    --output_dir=/home/mariya/Development/pallas/reports/output/predictions
