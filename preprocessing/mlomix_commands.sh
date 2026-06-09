# GEX train
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-train-gex --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_train_gex.csv --gex_intermediate_umaps true

# GEX test and cal
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-cal-and-test-gex --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_calibrate_and_test_gex.csv --gex_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/class_names.gex.txt --gex_genes_file /home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt --tsne --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv --gex_norm_factors_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/gex_norm_factors.rds

# DNam train
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-train-dnam --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_train_dnam.csv --skip_umaps true

# VALIDATION
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/clinical-genomics-uppsala-gex \
    --input /home/mariya/Data/pallas/mlomix_pre_validation/ClinicalGenomicsUppsala/samplesheet.csv \
    --gex_genes_file /home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt \
    --gex_intermediate_umaps=True \
    --gex_min_samples_per_class=1 \
    --tsne \
    --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv \
    --gex_norm_factors_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/gex_norm_factors.rds
