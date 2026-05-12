# Bimodal all data
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-bimodal --dnam_variance_keep_top_sites 5000 --gex_variance_keep_top_sites 5000 --gex_min_samples_per_class 4 --dnam_min_samples_per_class 4 --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_labeled_diagnostic_bimodal.csv

# DNam train
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-train-dnam --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_train_dnam.csv --dnam_variance_keep_top_sites 5000

# DNAm test
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-test-dnam --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_test_dnam.csv --dnam_classes_file=/home/mariya/Data/pallas/nf-core-mlomix-train-dnam/finalize/class_names.dnam.txt --dnam_probes_file=/home/mariya/Data/pallas/nf-core-mlomix-train-dnam/finalize/feature_names.dnam.txt
