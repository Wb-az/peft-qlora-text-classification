import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.eval_metrics import bootstrap_metrics


EXPECTED_KEYS = {"accuracy", "precision", "recall", "f1", "matthews_correlation"}


def make_mock_classification_data(n_samples: int = 2000, n_classes: int = 6,
                                  accuracy: float = 0.72, seed: int = 1234):
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0. Got {n_samples}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2. Got {n_classes}")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1]. Got {accuracy}")

    rng = np.random.default_rng(seed=seed)
    labels = rng.integers(0, n_classes, size=n_samples, dtype=np.int64)
    predictions = labels.copy()

    wrong_mask = rng.random(n_samples) > accuracy
    n_wrong = int(wrong_mask.sum())
    if n_wrong:
        wrong = rng.integers(0, n_classes, size=n_wrong, dtype=np.int64)
        true_wrong = labels[wrong_mask]
        wrong = np.where(wrong == true_wrong, (wrong + 1) % n_classes, wrong)
        predictions[wrong_mask] = wrong

    return labels, predictions


@pytest.fixture(scope="module")
def mock_dataset():
    return make_mock_classification_data(
        n_samples=75000,
        n_classes=6,
        accuracy=0.7,
        seed=42,
    )


def test_make_mock_classification_data(mock_dataset):
    labels, predictions = mock_dataset
    assert labels.shape == predictions.shape
    assert labels.ndim == 1
    assert labels.min() >= 0
    assert predictions.min() >= 0


def test_bootstrap_metrics_with_numpy(mock_dataset):
    labels, predictions = mock_dataset
    out = bootstrap_metrics(labels, predictions, rounds=1000, seed=7)

    assert set(out.keys()) == EXPECTED_KEYS
    assert all(len(v) == 1000 for v in out.values())
    assert all(np.isfinite(v).all() for v in out.values())


def test_bootstrap_metrics_with_torch_cpu(mock_dataset):
    torch = pytest.importorskip("torch")
    labels, predictions = mock_dataset
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    preds_t = torch.as_tensor(predictions, dtype=torch.long)

    out = bootstrap_metrics(labels_t, preds_t, rounds=1000, seed=11)
    assert set(out.keys()) == EXPECTED_KEYS
    assert all(len(v) == 1000 for v in out.values())


def test_bootstrap_metrics_with_torch_cuda_if_available(mock_dataset):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    labels, predictions = mock_dataset
    labels_t = torch.as_tensor(labels, dtype=torch.long, device="cuda")
    preds_t = torch.as_tensor(predictions, dtype=torch.long, device="cuda")

    out = bootstrap_metrics(labels_t, preds_t, rounds=1000, seed=11)
    assert set(out.keys()) == EXPECTED_KEYS
    assert all(len(v) == 10 for v in out.values())
