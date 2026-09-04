import pandas as pd
import os
import argparse
import mlflow
import dotenv
import joblib
import sklearn
import numpy as np
from mlflow.models import infer_signature


description = """
Deploys conformal predictor framework
and calibration dataset of PALLAS predictions
to MLFlow.
"""

predictions_path_help = """
Path to CSV file containing predictions."""

base_classifier_name_help = """
Name of the base classifier used to generate the predictions."""

base_classifier_path_help = """
Path to the base classifier model file."""


def parse_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help=predictions_path_help,
    )
    parser.add_argument(
        "--base_classifier_name",
        type=str,
        required=True,
        help=base_classifier_name_help,
    )
    parser.add_argument(
        "--base_classifier_path",
        type=str,
        required=True,
        help=base_classifier_path_help,
    )
    return parser.parse_args()


def deploy_conformal_prediction(
        predictions_path,
        base_classifier_name,
        base_classifier_path):

    print("\n\nLogging to MLFlow...")

    path_to_dot_env = os.path.join(os.path.dirname(__file__), '../.env')
    dotenv.load_dotenv(path_to_dot_env)
    model = joblib.load(base_classifier_path)

    model_name = f"crc_{base_classifier_name}"
    run_name = f"deploy_crc_{base_classifier_name}"
    pip_requirements = [
        f"scikit-learn=={sklearn.__version__}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"joblib=={joblib.__version__}",
        "git+https://github.com/Molmed/conformist.git@pallas#egg=conformist",
    ]
    path_to_wrapper = os.path.join(os.path.dirname(__file__), 'crc_model.py')

    feature_names = model.feature_names_in_

    X_example = pd.DataFrame(
        np.zeros((1, len(feature_names)), dtype=np.float32),
        columns=feature_names,
    )

    signature = infer_signature(
        X_example,
        [["example"]],
        params={"error_rate_alpha": 0.05},
    )

    with mlflow.start_run(run_name=run_name):
        base_model_info = mlflow.sklearn.log_model(
            model,
            name=base_classifier_name)

        mlflow.log_param("base_classifier_name", base_classifier_name)
        mlflow.log_param("base_classifier_path", base_classifier_path)
        mlflow.log_param("predictions_path", predictions_path)

        # Log number of samples in calibration dataset
        calibration_df = pd.read_csv(predictions_path)
        mlflow.log_param("calibration_dataset_size", len(calibration_df))

        mlflow.pyfunc.log_model(
            name=model_name,
            python_model=path_to_wrapper,
            artifacts={
                "base_model": base_model_info.model_uri,
                "calibration_data": predictions_path,
            },
            signature=signature,
            pip_requirements=pip_requirements,
        )


def main():
    args = parse_args()
    predictions_path = args.predictions_path
    base_classifier_name = args.base_classifier_name
    base_classifier_path = args.base_classifier_path
    deploy_conformal_prediction(
        predictions_path,
        base_classifier_name=base_classifier_name,
        base_classifier_path=base_classifier_path)


if __name__ == "__main__":
    main()
