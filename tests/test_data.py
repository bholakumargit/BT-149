import pytest
import pandas as pd

from malware_classifier.data import (
    build_sample_groups,
    get_feature_columns,
    grouped_train_validation_test_indices,
    validate_feature_dataset,
)


def test_validate_feature_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_feature_dataset(tmp_path / "missing.csv")


def test_feature_columns_exclude_targets_and_sample_identifier():
    assert get_feature_columns(["feature_a", "Class", "Filename", "Category"]) == [
        "feature_a"
    ]


def test_groups_malware_dumps_by_sample_and_keeps_benign_rows_separate():
    frame = pd.DataFrame(
        {
            "Class": ["Benign", "Benign", "Malware", "Malware"],
            "Filename": [
                "Benign",
                "Benign",
                "Ransomware-Maze-hash-1.raw",
                "Ransomware-Maze-hash-10.raw",
            ],
        }
    )

    groups = build_sample_groups(frame)

    assert groups[0] != groups[1]
    assert groups[2] == groups[3]


def test_stratified_group_splits_keep_groups_disjoint_and_classes_present():
    targets = []
    groups = []
    for label in ("Benign", "Malware"):
        for group_number in range(10):
            for _ in range(2):
                targets.append(label)
                groups.append(f"{label}-{group_number}")

    train, validation, test = grouped_train_validation_test_indices(
        targets, pd.Series(groups).to_numpy()
    )
    splits = [set(pd.Series(groups).iloc[indices]) for indices in (train, validation, test)]

    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    assert len(train) / len(targets) == pytest.approx(0.8, abs=0.03)
    assert len(validation) / len(targets) == pytest.approx(0.1, abs=0.03)
    assert len(test) / len(targets) == pytest.approx(0.1, abs=0.03)
    for indices in (train, validation, test):
        assert set(pd.Series(targets).iloc[indices]) == {"Benign", "Malware"}
