# GEX calibration
python lib/predict.py \
    --pallas_model_base_dir=/home/mariya/Data/pallas/models \
    --pallas_model_name=rf_GEX_20260606_190416 \
    --X_path=/home/mariya/Data/pallas/mlomix_datasets/cal-gex/features.gex.csv \
    --y_path=/home/mariya/Data/pallas/mlomix_datasets/cal-gex/labels.csv \
    --dataset_name="cal-gex" \
    --output_dir=/home/mariya/Development/pallas/reports/output/predictions
