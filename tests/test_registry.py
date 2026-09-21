"""Model registry flow: publish versions, gate promotion on CV accuracy, load the champion into the service."""

import mlflow
import pytest
from mlflow import MlflowClient

from sqlsmith.registry import MODEL_NAME, load_router, make_router, publish_router


def test_publish_promote_and_load_by_alias(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    v1 = publish_router("sklearn", tracking_uri=uri)
    v2 = publish_router("torch", tracking_uri=uri)
    assert (v1["version"], v2["version"]) == ("1", "2") and v2["alias"] == "champion"
    assert mlflow.get_run(v2["run_id"]).data.metrics["cv_accuracy"] == pytest.approx(v2["cv_accuracy"])

    router, version = load_router(tracking_uri=uri)
    assert version == "2" and type(router).__name__ == "TorchRouter"
    assert router.predict_many(["How many products are there?", "Which customers spent more than the average per country?"]) == ["easy", "complex"]

    # a version that fails the gate is still recorded but the alias stays where it was
    v3 = publish_router("sklearn", min_cv_acc=0.99, tracking_uri=uri)
    assert v3["version"] == "3" and v3["alias"] is None
    assert str(MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version) == "2"

    _, info = make_router("registry", tracking_uri=uri)
    assert info == {"source": "registry", "name": MODEL_NAME, "alias": "champion", "version": "2", "backend": "TorchRouter"}
    with pytest.raises(ValueError):
        make_router("bogus")
