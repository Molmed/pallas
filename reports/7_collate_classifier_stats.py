import pandas as pd
from pathlib import Path


def get_dataset_name_from_path(path):
    if "cal-and-test" in path.parent.name:
        return "calibration + test"
    elif "cross-val" in path.parent.name:
        return "train (5-fold cv)"
    else:
        return path.parent.name


def get_modality_from_path(path):
    if "DNAm" in path.parent.parent.name:
        return "DNAm"
    elif "GEX" in path.parent.parent.name:
        return "GEX"
    else:
        return path.parent.name


base_dir = Path(__file__).resolve().parent.parent / \
    "reports" / "output" / "evaluation"

output_dir = base_dir / "collated"

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Find all paths under base_dir that end in classifier_level_metris.csv
csv_paths = list(base_dir.rglob("classifier_level_metrics.csv"))

# For each csv path, read it into a dataframe.
# Rename value column to the name of the parent directory of the csv file and its parent directory
dfs = []
for csv_path in csv_paths:
    df = pd.read_csv(csv_path)

    dataset_name = get_dataset_name_from_path(csv_path)
    modality_name = get_modality_from_path(csv_path)
    df = df.rename(columns={"value": modality_name + "_" + dataset_name})
    dfs.append(df)

# Concatenate all dataframes into one dataframe,
# keeping only a single "metric" column and the renamed value columns.
df = pd.concat(dfs, axis=1)
df = df.loc[:, ~df.columns.duplicated()]


ordered_cols = ["metric",
                "GEX_train (5-fold cv)",
                "GEX_calibration + test",
                "DNAm_train (5-fold cv)",
                "DNAm_calibration + test"]
df = df[ordered_cols]

# Save the concatenated dataframe to a csv file
df.to_csv(output_dir / "collated_classifier_stats.csv", index=False)

print(df)
