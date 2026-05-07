# Bimodal train data
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-train-bimodal --dnam_variance_keep_top_sites 5000 --gex_variance_keep_top_sites 5000 --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_train_bimodal.csv

# Bimodal test data
nextflow run . -c conf/dev.config -resume -profile dev,conda --annotation_version 115 --outdir /home/mariya/Data/pallas/nf-core-mlomix-test-bimodal --gex_class_file=/home/mariya/Data/pallas/nf-core-mlomix-train-bimodal/finalize/class_names.gex.txt --dnam_class_file=/home/mariya/Data/pallas/nf-core-mlomix-train-bimodal/finalize/class_names.dnam.txt --dnam_probes_file=/home/mariya/Data/pallas/nf-core-mlomix-train-bimodal/finalize/feature_names.dnam.txt --input /home/mariya/Development/pallas/preprocessing/mlomix_samplesheets/samplesheet_test_bimodal.csv
