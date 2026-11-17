import pytest

torch = pytest.importorskip("torch")

from scripts import train_models


def test_training_script_can_build_its_loss_function():
    assert train_models.nn is torch.nn


def test_training_script_defaults_to_bundled_dataset(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_models.py"])

    args = train_models.parse_args()

    assert args.dataset == train_models.DATASET_PATH
