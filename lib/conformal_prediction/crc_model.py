import mlflow
import pandas as pd
from conformist import FNRCoP, PredictionDataset
import skops.io as sio
from mlflow.models import Model
import os


class CRCWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        local_path = context.artifacts["base_model"]
        flavor = Model.load(local_path).flavors["sklearn"]
        model_file = os.path.join(local_path, flavor["pickled_model"])
        self.model = sio.load(model_file, trusted=["sklearn.tree._tree.Tree"])
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
