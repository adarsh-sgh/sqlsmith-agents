"""Router versioning through the MLflow model registry: publish a trained router, promote by alias, load by alias."""

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from sqlsmith.router import BACKENDS, cross_val_accuracy, train_router, training_data

MODEL_NAME = "sqlsmith-router"
DEFAULT_ALIAS = "champion"


class RouterModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so both backends (sklearn pipeline, torch linear) version through one registry entry."""

    def __init__(self, router):
        self.router = router

    def predict(self, context, model_input, params=None):
        qs = model_input["question"] if isinstance(model_input, pd.DataFrame) else model_input
        return self.router.predict_many(list(qs))


def publish_router(backend="sklearn", alias=DEFAULT_ALIAS, min_cv_acc=0.8, tracking_uri="file:./mlruns", experiment="sqlsmith"):
    """Train, CV-score, register a new model version; move `alias` to it only if CV accuracy clears the gate."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    acc = cross_val_accuracy(backend)
    router = train_router(backend)
    with mlflow.start_run(run_name=f"router-{backend}") as run:
        mlflow.log_params({"backend": backend, "n_train": len(training_data()[0])})
        mlflow.log_metric("cv_accuracy", acc)
        info = mlflow.pyfunc.log_model(
            name="router",
            python_model=RouterModel(router),
            registered_model_name=MODEL_NAME,
            input_example=pd.DataFrame({"question": ["How many customers are there?"]}),
        )
    version = str(info.registered_model_version)
    promoted = acc >= min_cv_acc
    if promoted:
        MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)
    return {"version": version, "backend": backend, "cv_accuracy": acc, "alias": alias if promoted else None, "run_id": run.info.run_id}


def load_router(alias=DEFAULT_ALIAS, tracking_uri="file:./mlruns"):
    """Resolve `models:/<name>@<alias>` and return (native router, version string)."""
    mlflow.set_tracking_uri(tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
    loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
    return loaded.unwrap_python_model().router, str(mv.version)


def make_router(name, tracking_uri="file:./mlruns"):
    """'sklearn' | 'torch' train in-process; 'registry' or 'registry:<alias>' load a published version."""
    if name in BACKENDS:
        return train_router(name), {"source": "trained", "backend": name}
    if name.startswith("registry"):
        alias = name.split(":", 1)[1] if ":" in name else DEFAULT_ALIAS
        router, version = load_router(alias, tracking_uri)
        return router, {"source": "registry", "name": MODEL_NAME, "alias": alias, "version": version, "backend": type(router).__name__}
    raise ValueError(f"unknown router '{name}' (use sklearn|torch|registry[:alias])")
