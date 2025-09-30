import pandas as pd
import os
import joblib
import dotenv
import numpy as np
from .classifier_metrics import cross_val_metrics, \
    classifier_level_metrics, class_level_metrics

dotenv.load_dotenv()

reports_dir = os.path.join(os.getenv('REPORTS_DIR'), 'classifier')
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
MODALITIES = ['DNAm', 'GEX']
DATA_SPLITS = ['train', 'calibration', 'test']

classifier_level_reports = {}
class_level_reports = {}
for modality in MODALITIES:
    classifier_level_reports[modality] = {}
    class_level_reports[modality] = {}
    classifier_path = os.path.join(os.getenv('MODELS_DIR'), f'{modality.lower()}.model.rf.pkl')
    selected_features_path = os.path.join(os.getenv('MODELS_DIR'), f'{modality.lower()}.features.pkl')
    classifier = joblib.load(classifier_path)
    selected_features = joblib.load(selected_features_path)
    class_names = classifier.classes_.tolist()

    for split in DATA_SPLITS:
        X_path = os.path.join(os.getenv('DATA_DIR'), modality, f'{modality.lower()}_{split}_X.csv')
        y_path = os.path.join(os.getenv('DATA_DIR'), modality, f'{modality.lower()}_{split}_y.tsv')
        print(f'Loading {X_path} and {y_path}...')

        # Read in x and y
        X = pd.read_csv(X_path, index_col=0)
        y = pd.read_csv(y_path, sep='\t', index_col=0)

        # Flatten y
        y = y['class'] if modality == 'GEX' else y['subtype']

        # Align indices of x and y
        X = X.loc[y.index]

        # Get only selected features
        X = X[selected_features]

        # If data split is train, use cross-validation to compute metrics
        if split == 'train':
            # Get smallest sample size in order to calculate number of folds
            min_class_size = y.value_counts().min()
            n_splits = min(5, min_class_size) if min_class_size >= 2 else 2
            print(f'Computing cross-validation metrics for {modality} with {n_splits} folds...')
            classifier_metrics, class_metrics = \
                cross_val_metrics(classifier,
                                  X,
                                  y,
                                  class_names,
                                  n_splits=n_splits,
                                  random_state=RANDOM_SEED)
            key = f'train ({n_splits}-fold cv)'
            classifier_level_reports[modality][key] = classifier_metrics
            class_level_reports[modality][key] = class_metrics
        else:
            print(f'Computing metrics on {modality} {split} set...')
            y_pred = classifier.predict(X)
            y_proba = classifier.predict_proba(X)

            metrics = classifier_level_metrics(y, y_pred, y_proba, class_names)
            classifier_level_reports[modality][split] = metrics

            class_metrics = class_level_metrics(y, y_pred, y_proba, class_names)
            class_level_reports[modality][split] = class_metrics


# Create dataframe from classifier_level_reports
df_classifier_reports = pd.concat({(modality, split): pd.Series(metrics)
                                   for modality, splits in classifier_level_reports.items()
                                   for split, metrics in splits.items()}, axis=1)
df_classifier_reports.columns.names = ['modality', 'data_split']
df_classifier_reports = df_classifier_reports.sort_index()

# Round all numbers to 3 decimal places
df_classifier_reports = df_classifier_reports.round(3)

df_classifier_reports.to_csv(os.path.join(reports_dir, 'classifier_level_reports.csv'))

# Create dataframe from class level reports
df_class_reports = pd.concat({(modality, split): metrics
                              for modality, splits in class_level_reports.items()
                              for split, metrics in splits.items()}, axis=1)
# df_class_reports.columns.names = ['modality', 'data_split']
df_class_reports = df_class_reports.sort_index(axis=1)
df_class_reports = df_class_reports.apply(pd.to_numeric, errors='ignore')
df_class_reports = df_class_reports.round(3)

# Sort alphabetically but with Non-leukemic at the bottom
if 'Non-leukemic' in df_class_reports.index:
    non_leukemic_row = df_class_reports.loc['Non-leukemic']
    df_class_reports = df_class_reports.drop(index='Non-leukemic')
    df_class_reports = df_class_reports.sort_index()
    df_class_reports = pd.concat([df_class_reports, pd.DataFrame([non_leukemic_row], index=['Non-leukemic'])])
else:
    df_class_reports = df_class_reports.sort_index()


df_class_reports.to_csv(os.path.join(reports_dir, 'class_level_reports.csv'))

print(f'Saved classifier-level reports to {os.path.join(reports_dir, "classifier_level_reports.csv")}')
print(f'Saved class-level reports to {os.path.join(reports_dir, "class_level_reports.csv")}')
