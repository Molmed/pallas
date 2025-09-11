#!/bin/bash
# Usage from main directory: bash preprocessing/gexml_pre/preprocess_all.sh

source preprocessing/.env

echo "Reading from ${RAW_DATA_DIR}"
echo "Writing to ${PROCESSED_DATA_DIR}"

python -m preprocessing.gexml_pre.diedrich
python -m preprocessing.gexml_pre.heinaniemi
python -m preprocessing.gexml_pre.krali
python -m preprocessing.gexml_pre.stjude_cloud
python -m preprocessing.gexml_pre.stjude_panALL
python -m preprocessing.gexml_pre.tran

# This script finds all 'samplesheet.csv' files in 'processed/',
# concatenates them into one CSV while keeping only the first header,
# and saves only rows that contain 'labeled_diagnostic'
# to 'nf-core-gexml-test-data/samplesheet.csv'.

output="${PROCESSED_DATA_DIR}/samplesheet.csv"

# Ensure output directory exists
mkdir -p "$(dirname "$output")"

# Temporary file
tmpfile=$(mktemp)

# Find and sort files
find ${PROCESSED_DATA_DIR} -type f -name "samplesheet.csv" -print0 | sort -z | while IFS= read -r -d '' file; do
    if [ ! -s "$tmpfile" ]; then
        # First file: keep header + matching rows
        { head -n 1 "$file"; grep "labeled_diagnostic" "$file"; } >> "$tmpfile"
    else
        # Subsequent files: skip header, only append matching rows
        tail -n +2 "$file" | grep "labeled_diagnostic" >> "$tmpfile"
    fi
done

# Move final concatenated file to output location
mv "$tmpfile" "$output"

echo "Combined samplesheet saved to $output"
