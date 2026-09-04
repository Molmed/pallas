import mlflow
import pandas as pd
from conformist import FNRCoP, PredictionDataset


class CRCWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = mlflow.sklearn.load_model(context.artifacts["base_model"])
        self.calibration_data = pd.read_csv(context.artifacts["calibration_data"])

    def predict(self, context, model_input, params=None):
        label_names = set(self.calibration_data["true_label"].astype(str))

        probability_columns = [
            column
            for column in self.calibration_data.columns
            if column in label_names
        ]

        calibration_data = self.calibration_data[
            ["true_label", "predicted_label", *probability_columns]
        ].copy()

        alpha = float((params or {}).get("error_rate_alpha", 0.05))
        calibration_dataset = PredictionDataset(
            df=calibration_data,
            dataset_name="cal",
        )

        conformal_predictor = FNRCoP(calibration_dataset, alpha=alpha)
        conformal_predictor.calibrate()

        probabilities = self.model.predict_proba(model_input)
        prediction_sets = probabilities >= conformal_predictor.softmax_threshold

        return [
            [cls for cls, included in zip(
                calibration_dataset.class_names(), row) if included]
            for row in prediction_sets
        ]


mlflow.models.set_model(CRCWrapper())
