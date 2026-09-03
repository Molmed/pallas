import mlflow
import json


class CRCWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = mlflow.sklearn.load_model(context.artifacts["base_model"])

        with open(context.artifacts["lambdas"]) as file:
            self.lambdas = json.load(file)

        with open(context.artifacts["classes"]) as file:
            self.classes = json.load(file)

    def predict(self, context, model_input, params=None):
        alpha = str((params or {}).get("error_rate_alpha", 0.05))
        lam = self.lambdas[alpha]

        probabilities = self.model.predict_proba(model_input)
        prediction_sets = probabilities >= lam

        return [
            [cls for cls, included in zip(self.classes, row) if included]
            for row in prediction_sets
        ]


mlflow.models.set_model(CRCWrapper())
