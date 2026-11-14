import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import apply_label
from ml.model import train_model, compute_model_metrics, inference


def test_apply_labels():
    """Checks predictions become the correct salary labels."""
    assert apply_label(np.array([0])) == "<=50K"
    assert apply_label(np.array([1])) == ">50K"


def test_train_model():
    """Checks training returns a fitted random forest classifier."""
    X_train = np.array([[0], [1], [2], [3], [4], [5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = train_model(X_train, y_train)
    predictions = inference(model, X_train)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_features_in_ == 1
    assert predictions.shape == y_train.shape
    assert set(predictions) == {0, 1}


def test_compute_model_metrics():
    """Checks the precision, recall, and F1 against known results."""
    actual = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])

    precision, recall, f1 = compute_model_metrics(actual, predictions)

    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(0.8)
