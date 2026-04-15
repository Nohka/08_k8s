from __future__ import annotations

import os

import mlflow
import mlflow.pyfunc


def resolve_model_uri() -> str:
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return model_uri

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "nyc-green-taxi-regression-v2"
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.rmse ASC"],
        max_results=10,
    )

    for run in runs:
        model_uri = run.data.tags.get("run_model_uri")
        if model_uri:
            return model_uri

    raise ValueError("No finished run with a model URI tag was found.")


def load_model_from_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = resolve_model_uri()
    model = mlflow.pyfunc.load_model(model_uri)

    return model, model_uri
