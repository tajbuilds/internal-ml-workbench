import pandas as pd

from app.core.ml import train_and_compare
from app.core.planning import feature_presets, recommended_target_transform


def test_train_and_compare_regression_with_log_target_and_selected_features(
    sample_regression_df: pd.DataFrame,
) -> None:
    shifted = sample_regression_df.copy()
    shifted["target"] = shifted["target"] - shifted["target"].min() + 1.0

    artifacts = train_and_compare(
        shifted,
        target="target",
        task_type="regression",
        feature_columns=["feature_1", "feature_2"],
        target_transform="log1p",
    )

    assert artifacts.task_type == "regression"
    assert artifacts.setup_df.iloc[0]["target_transform"] == "log1p"
    assert artifacts.setup_df.iloc[0]["feature_count"] == 2
    assert "rmse" in artifacts.evaluation_df.columns


def test_feature_presets_and_recommended_transform_for_regression(
    sample_regression_df: pd.DataFrame,
) -> None:
    shifted = sample_regression_df.copy()
    shifted["target"] = shifted["target"] - shifted["target"].min() + 1.0

    presets = feature_presets(shifted, "target", "regression")

    assert "All Features" in presets
    assert "Numeric Only" in presets
    assert "Top Correlated" in presets
    assert recommended_target_transform(shifted, "target", "regression") in {"none", "log1p"}
