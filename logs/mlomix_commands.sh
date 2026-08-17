# GEX train
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-train-gex \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_train_gex.csv \
    --gex_intermediate_umaps true

# GEX test and cal
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-cal-and-test-gex \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_calibrate_and_test_gex.csv \
    --gex_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/class_names.gex.txt \
    --gex_genes_file /home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt \
    --tsne \
    --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv \
    --gex_norm_factors_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/gex_norm_factors.rds

# GEX calibration
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-cal-gex \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_calibrate_gex.csv \
    --gex_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/class_names.gex.txt \
    --gex_genes_file /home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt \
    --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv \
    --gex_norm_factors_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/gex_norm_factors.rds \
    --skip_umaps

# GEX test
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-test-gex \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_test_gex.csv \
    --gex_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/class_names.gex.txt \
    --gex_genes_file /home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt \
    --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv \
    --gex_norm_factors_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/gex_norm_factors.rds \
    --skip_umaps


# DNam train
nextflow run . -c conf/dev.config -resume -profile dev,conda \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-train-dnam \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_train_dnam.csv \
    --skip_umaps true

# DNAm test and cal
nextflow run . -c conf/dev.config -profile dev,conda -resume \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-cal-and-test-dnam \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_calibrate_and_test_dnam.csv \
    --dnam_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-dnam/class_names.dnam.txt \
    --dnam_probes_file=/home/mariya/Data/pallas/mlomix_datasets/train-dnam/feature_names.dnam.txt \
    --skip_umaps true

# DNAm cal
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-cal-dnam \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_calibrate_dnam.csv \
    --dnam_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-dnam/class_names.dnam.txt \
    --dnam_probes_file=/home/mariya/Development/pallas/reports/output/feature_selection/dnam_feature_selection/selected_features.1186.txt \
    --skip_umaps true

# DNAm test
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-test-dnam \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_test_dnam.csv \
    --dnam_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-dnam/class_names.dnam.txt \
    --dnam_probes_file=/home/mariya/Development/pallas/reports/output/feature_selection/dnam_feature_selection/selected_features.1186.txt \
    --skip_umaps true

# Validation
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/clinical-genomics-uppsala-gex \
    --input /home/mariya/Data/pallas/mlomix_pre_validation/ClinicalGenomicsUppsala/samplesheet.csv \
    --gex_genes_file /home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt \
    --gex_intermediate_umaps=True \
    --gex_min_samples_per_class=1 \
    --tsne \
    --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv \
    --gex_norm_factors_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/gex_norm_factors.rds

# Bimodal unlabeled
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-unlabeled-bimodal-reduced \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_unlabeled_bimodal.csv \
    --dnam_probes_file=/home/mariya/Development/pallas/reports/output/feature_selection/dnam_feature_selection/selected_features.1186.txt \
    --gex_genes_file=/home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt \
    --tsne \
    --gex_norm_factors_file=/home/mariya/Data/pallas/mlomix_datasets/labeled-diagnostic-bimodal-reduced/gex_norm_factors.rds

# Bimodal cross-val
nextflow run . -c conf/dev.config -profile dev,conda -resume --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/labeled-diagnostic-bimodal-reduced \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_labeled_diagnostic_bimodal.csv \
    --dnam_probes_file=/home/mariya/Development/pallas/reports/output/feature_selection/dnam_feature_selection/selected_features.1186.txt \
    --gex_genes_file=/home/mariya/Development/pallas/reports/output/feature_selection/gex_feature_selection/selected_features.1121.txt \
    --tsne \
    --dnam_min_samples_per_class=7 \
    --gex_min_samples_per_class=7

# TSNE GENERATION ####
# GEX labeled diagnostic (for TSNE)
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-labeled-diagnostic-gex-all-features \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_labeled_diagnostic_gex.csv \
    --gex_intermediate_umaps true \
    --tsne \
    --gex_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-gex/class_names.gex.txt \
    --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv

# DNAm labeled diagnostic (for TSNE)
nextflow run . -c conf/dev.config -resume -profile dev,conda \
    --outdir /home/mariya/Data/pallas/nf-core-mlomix-labeled-diagnostic-dnam-all-features \
    --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_labeled_diagnostic_dnam.csv \
    --tsne \
    --dnam_classes_file=/home/mariya/Data/pallas/mlomix_datasets/train-dnam/class_names.dnam.txt \
    --class_colors_file=/home/mariya/Development/pallas/configs/class_colors.tsv
