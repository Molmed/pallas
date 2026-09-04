import mlflow
import pandas as pd
from conformist import FNRCoP, PredictionDataset


class CRCWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = mlflow.sklearn.load_model(context.artifacts["base_model"])
        self.calibration_data = pd.read_csv(context.artifacts["calibration_data"])

    def predict(self, context, model_input, params=None):
        model_input = model_input[self.model.feature_names_in_]

        alpha = float((params or {}).get("error_rate_alpha", 0.05))
        calibration_dataset = PredictionDataset(
            df=self.calibration_data.copy(),
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
