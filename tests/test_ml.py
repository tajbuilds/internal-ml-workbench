import pandas as pd

from app.core.ml import detect_task_type, train_and_compare


def test_detect_task_type_classification(sample_classification_df: pd.DataFrame) -> None:
    assert detect_task_type(sample_classification_df, "segment") == "classification"


def test_detect_task_type_regression(sample_regression_df: pd.DataFrame) -> None:
    assert detect_task_type(sample_regression_df, "target") == "regression"


def test_train_and_compare_classification(sample_classification_df: pd.DataFrame) -> None:
    artifacts = train_and_compare(
        sample_classification_df,
        target="segment",
        task_type="classification",
    )

    assert artifacts.task_type == "classification"
    assert not artifacts.leaderboard_df.empty
    assert "accuracy" in artifacts.leaderboard_df.columns
    assert "accuracy" in artifacts.evaluation_df.columns
    assert len(artifacts.model_bytes) > 0


def test_train_and_compare_regression(sample_regression_df: pd.DataFrame) -> None:
    artifacts = train_and_compare(
        sample_regression_df,
        target="target",
        task_type="regression",
    )

    assert artifacts.task_type == "regression"
    assert not artifacts.leaderboard_df.empty
    assert "r2" in artifacts.leaderboard_df.columns
    assert "r2" in artifacts.evaluation_df.columns
    assert len(artifacts.model_bytes) > 0


def test_model_predicts(sample_regression_df: pd.DataFrame) -> None:
    artifacts = train_and_compare(
        sample_regression_df,
        target="target",
        task_type="regression",
    )
    x = sample_regression_df.drop(columns=["target"]).head(5)
    preds = artifacts.best_model.predict(x)
    assert len(preds) == 5
