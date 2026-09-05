import pytest

from sqlsmith.router import LABELS, cross_val_accuracy, train_router, training_data


@pytest.mark.parametrize("backend", ["sklearn", "torch"])
def test_router_learns_easy_vs_complex(backend):
    qs, ys = training_data()
    assert set(ys) == set(LABELS) and len(qs) > 100
    r = train_router(backend)
    assert r.predict("How many products are there?") == "easy"
    assert r.predict("Which customers spent more than the average customer in each country?") == "complex"
    assert cross_val_accuracy(backend) > 0.8  # majority baseline is 0.66
