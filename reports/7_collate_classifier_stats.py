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


def collate_classifier_level_metrics():
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


def collate_class_level_metrics():
    csv_paths = list(base_dir.rglob("class_level_metrics.csv"))

    dfs = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        dataset_name = get_dataset_name_from_path(csv_path)
        modality_name = get_modality_from_path(csv_path)

        dfs[modality_name + "_" + dataset_name] = df

    # All dfs should have the same rows, fill with NaN if a row is missing in a df.
    for name, df in dfs.items():
        for other_name, other_df in dfs.items():
            if name != other_name:
                missing_rows = set(other_df["subtype"]) - set(df["subtype"])
                if missing_rows:
                    missing_df = pd.DataFrame(
                        {"subtype": list(missing_rows)})
                    df = pd.concat([df, missing_df], ignore_index=True)
                    df = df.sort_values(by="subtype").reset_index(drop=True)
                    dfs[name] = df

    order = ["GEX_train (5-fold cv)",
             "GEX_calibration + test",
             "DNAm_train (5-fold cv)",
             "DNAm_calibration + test"]

    # Order dictionary by order list
    dfs = {k: dfs[k] for k in order if k in dfs}

    # Conatenate dfs. Each column should be prepended by the modality and dataset name.
    result = pd.concat(dfs.values(), axis=1, keys=dfs.keys())

    # Round everything to 3 decimal places, except for the subtype column
    for col in result.columns:
        if col[1] != "subtype":
            result[col] = result[col].round(3)

    # Save the concatenated dataframe to a csv file
    result.to_csv(output_dir / "collated_class_stats.csv", index=False)


collate_classifier_level_metrics()
collate_class_level_metrics()
