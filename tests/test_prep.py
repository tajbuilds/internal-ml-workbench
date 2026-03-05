import pandas as pd

from app.core.prep import apply_preparation, audit_dataset


def test_audit_dataset_finds_missing_and_type_candidates() -> None:
    df = pd.DataFrame(
        {
            "num_text": ["1", "2", "3", None],
            "dates": ["2024-01-01", "2024-02-01", None, "2024-03-01"],
            "value": [1.0, None, 3.0, 4.0],
        }
    )

    audit = audit_dataset(df)

    assert "value" in audit["missing"]["column"].tolist()
    assert "num_text" in audit["coercion_candidates"]["column"].tolist()
    assert "dates" in audit["coercion_candidates"]["column"].tolist()


def test_apply_preparation_coerces_imputes_and_expands_dates() -> None:
    df = pd.DataFrame(
        {
            "amount": ["10", "20", None, "40"],
            "event_date": ["2024-01-01", "2024-02-01", None, "2024-03-05"],
            "category": ["a", None, "b", "b"],
            "dup": [1, 1, 2, 2],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    result = apply_preparation(
        df,
        drop_duplicates=True,
        coerce_numeric=True,
        numeric_threshold=0.75,
        expand_dates=True,
        datetime_threshold=0.75,
        max_missing_pct=0.0,
        impute_numeric=True,
        impute_categorical=True,
        clip_outliers=False,
    )

    assert result.report["duplicate_rows_removed"] == 1
    assert "amount" in result.report["columns_coerced_numeric"]
    assert "event_date" in result.report["date_columns_expanded"]
    assert "event_date" not in result.df.columns
    assert "event_date_year" in result.df.columns
    assert result.df["amount"].isna().sum() == 0
    assert result.df["category"].isna().sum() == 0
