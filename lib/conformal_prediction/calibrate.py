import pandas as pd
import os
from conformist import FNRCoP, PredictionDataset
import argparse
import mlflow
import dotenv
import joblib
import sklearn
import numpy as np
import conformist
from mlflow.models import infer_signature


description = """
Calibrates a conformal predictor using
a calibration dataset of PALLAS predictions.
"""

predictions_path_help = """
Path to CSV file containing predictions."""

base_classifier_name_help = """
Name of the base classifier used to generate the predictions."""

base_classifier_path_help = """
Path to the base classifier model file."""

output_dir_help = """
Directory to save the output."""

error_rate_alpha_help = """
Error rate alpha for conformal prediction (default: 0.05)."""

enable_mlflow_help = """
Whether to enable MLFlow logging.
If enabled, the script will log the conformal predictor model
and metadata to MLFlow. Default is False.
MLFlow environment variables must be set for this to work.
"""


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
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=output_dir_help,
    )
    parser.add_argument(
        "--error_rate_alpha",
        type=float,
        default=0.05,
        help=error_rate_alpha_help,
    )
    parser.add_argument(
        "--enable_mlflow",
        type=bool,
        required=False,
        default=False,
        help=enable_mlflow_help,
    )
    return parser.parse_args()


def calibrate_conformal_prediction(
        predictions_path,
        output_dir,
        base_classifier_name,
        base_classifier_path,
        error_rate_alpha=0.05,
        enable_mlflow=False):
    # Create the output directory if it doesn't exist
    # Make conformal predictor output directory
    output_dir = os.path.join(output_dir, "conformal_predictor")
    os.makedirs(output_dir, exist_ok=True)

    # Read in conformal predictions
    preds_cal = pd.read_csv(predictions_path)

    # Read in formatted predictions
    calpd = PredictionDataset(df=preds_cal, dataset_name='cal')

    crc = FNRCoP(calpd, alpha=error_rate_alpha)
    crc.calibrate()

    # Dump crc to joblib for later use
    # Save alpha as string with underscores instead of decimal point
    str_alpha = str(error_rate_alpha).replace('.', '_')
    model_name = f"crc_{base_classifier_name}_alpha{str_alpha}"
    crc_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(crc, crc_path)

    if not enable_mlflow:
        return

    print("\n\nLogging to MLFlow...")

    path_to_dot_env = os.path.join(os.path.dirname(__file__), '../.env')
    dotenv.load_dotenv(path_to_dot_env)
    model = joblib.load(base_classifier_path)

    model_name = f"crc_{base_classifier_name}_alpha{str_alpha}"
    run_name = f"calibrate_{base_classifier_name}_alpha{str_alpha}"
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
        mlflow.log_param("error_rate_alpha", error_rate_alpha)
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
    output_dir = args.output_dir
    error_rate_alpha = args.error_rate_alpha
    enable_mlflow = args.enable_mlflow
    calibrate_conformal_prediction(
        predictions_path,
        output_dir,
        base_classifier_name=base_classifier_name,
        base_classifier_path=base_classifier_path,
        error_rate_alpha=error_rate_alpha,
        enable_mlflow=enable_mlflow)


if __name__ == "__main__":
    main()
