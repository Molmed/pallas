import os
import pandas as pd
import numpy as np
import joblib
import dotenv

dotenv.load_dotenv()

DATASETS_PATH = os.getenv("DATASETS_PATH")
MODELS_DIR = os.getenv("MODELS_DIR")

def load_dataset(modality, split):
    dataset_path = os.path.join(DATASETS_PATH, f"{split}-{modality}")
    X_path = os.path.join(dataset_path, f"features.{modality}.csv")
    y_path = os.path.join(dataset_path, "labels.csv")

    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)

    return X, y


def load_train_dataset(modality):
    return load_dataset(modality, "train")


def load_test_dataset(modality, split):
    X, y = load_dataset(modality, split)
    y = y.iloc[:, 0].values  # Convert to 1D numpy array
    return X, y


def load_merged_test_datasets(modality):
    X_test, y_test = load_test_dataset(modality, "test")
    X_cal, y_cal = load_test_dataset(modality, "calibrate")

    # Append X_cal to X_test and y_cal to y_test
    X_merged = pd.concat([X_cal, X_test], axis=0)
    y_merged = np.concatenate([y_cal, y_test], axis=0)

    return X_merged, y_merged

def load_model(modality):
    model_path = os.path.join(MODELS_DIR, modality, f"final_model.pkl")
    return joblib.load(model_path)
