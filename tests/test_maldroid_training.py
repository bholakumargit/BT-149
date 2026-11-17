import numpy as np
import pandas as pd
import pytest

from scripts.train_maldroid_models import (
    canonicalize_targets,
    load_feature_table,
    remove_conflicting_duplicate_rows,
)


def test_load_feature_table_excludes_class_and_canonicalizes_numeric_labels(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {"feature_a": [0, 1], "feature_b": [2, 3], "Class": ["1.00E+00", "2"]}
    ).to_csv(path, index=False)

    names, features, targets = load_feature_table(path)

    assert names == ["feature_a", "feature_b"]
    np.testing.assert_array_equal(features, [[0, 2], [1, 3]])
    np.testing.assert_array_equal(targets, ["1", "2"])


def test_load_feature_table_rejects_missing_target_and_invalid_features(tmp_path):
    missing_target = tmp_path / "missing_target.csv"
    pd.DataFrame({"feature_a": [1]}).to_csv(missing_target, index=False)
    with pytest.raises(ValueError, match="Class"):
        load_feature_table(missing_target)

    invalid_feature = tmp_path / "invalid_feature.csv"
    pd.DataFrame({"feature_a": ["not-numeric"], "Class": [1]}).to_csv(
        invalid_feature, index=False
    )
    with pytest.raises(ValueError, match="non-numeric"):
        load_feature_table(invalid_feature)


def test_conflicting_duplicate_feature_rows_are_excluded_and_same_label_rows_grouped():
    features = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1], [2, 0]], dtype=np.float32
    )
    targets = np.asarray(["1", "1", "2", "3", "2"])

    filtered_features, filtered_targets, groups, conflict_groups, removed_rows = (
        remove_conflicting_duplicate_rows(features, targets)
    )

    assert conflict_groups == 1
    assert removed_rows == 2
    assert len(filtered_features) == len(filtered_targets) == len(groups) == 3
    assert groups[0] == groups[1]
    assert set(filtered_targets) == {"1", "2"}


def test_canonicalize_targets_rejects_empty_labels():
    with pytest.raises(ValueError, match="missing or empty"):
        canonicalize_targets(pd.Series(["1", " "]))
