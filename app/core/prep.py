from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PreparationResult:
    df: pd.DataFrame
    report: dict[str, Any]


def _numeric_parse_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    parsed = pd.to_numeric(non_null, errors="coerce")
    return float(parsed.notna().mean())


def _datetime_parse_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def audit_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": [int(df[c].isna().sum()) for c in df.columns],
                "missing_pct": [round(float(df[c].isna().mean()) * 100.0, 2) for c in df.columns],
            }
        )
        .query("missing_count > 0")
        .sort_values(["missing_pct", "missing_count"], ascending=False)
        .reset_index(drop=True)
    )

    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    coercion_rows: list[dict[str, Any]] = []
    for col in object_cols:
        coercion_rows.append(
            {
                "column": col,
                "numeric_parse_pct": round(_numeric_parse_ratio(df[col]) * 100.0, 2),
                "datetime_parse_pct": round(_datetime_parse_ratio(df[col]) * 100.0, 2),
                "sample_dtype": str(df[col].dtype),
            }
        )
    coercion_candidates = (
        pd.DataFrame(coercion_rows)
        .sort_values(["numeric_parse_pct", "datetime_parse_pct"], ascending=False)
        .reset_index(drop=True)
        if coercion_rows
        else pd.DataFrame(
            columns=["column", "numeric_parse_pct", "datetime_parse_pct", "sample_dtype"]
        )
    )

    outlier_rows: list[dict[str, Any]] = []
    for col in df.select_dtypes(include=["number", "bool"]).columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            outliers = 0
        else:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outliers = int(((series < lo) | (series > hi)).sum())
        outlier_rows.append(
            {
                "column": col,
                "outliers": outliers,
                "outlier_pct": round((outliers / len(series)) * 100.0, 2),
            }
        )
    outliers = (
        pd.DataFrame(outlier_rows)
        .sort_values(["outlier_pct", "outliers"], ascending=False)
        .reset_index(drop=True)
        if outlier_rows
        else pd.DataFrame(columns=["column", "outliers", "outlier_pct"])
    )

    return {
        "missing": missing,
        "coercion_candidates": coercion_candidates,
        "outliers": outliers,
    }


def apply_preparation(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool,
    coerce_numeric: bool,
    numeric_threshold: float,
    expand_dates: bool,
    datetime_threshold: float,
    max_missing_pct: float,
    impute_numeric: bool,
    impute_categorical: bool,
    clip_outliers: bool,
) -> PreparationResult:
    prepared = df.copy()
    report: dict[str, Any] = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "duplicate_rows_removed": 0,
        "columns_coerced_numeric": [],
        "date_columns_expanded": [],
        "columns_dropped_for_missing": [],
        "numeric_columns_imputed": [],
        "categorical_columns_imputed": [],
        "outlier_columns_clipped": [],
    }

    if drop_duplicates:
        before = len(prepared)
        prepared = prepared.drop_duplicates().reset_index(drop=True)
        report["duplicate_rows_removed"] = before - len(prepared)

    if coerce_numeric:
        for col in prepared.select_dtypes(include=["object", "string"]).columns.tolist():
            ratio = _numeric_parse_ratio(prepared[col])
            if ratio >= numeric_threshold:
                prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
                report["columns_coerced_numeric"].append(col)

    if expand_dates:
        candidate_cols = prepared.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in candidate_cols:
            ratio = _datetime_parse_ratio(prepared[col])
            if ratio >= datetime_threshold:
                parsed = pd.to_datetime(prepared[col], errors="coerce", format="mixed")
                prepared[f"{col}_year"] = parsed.dt.year
                prepared[f"{col}_month"] = parsed.dt.month
                prepared[f"{col}_day"] = parsed.dt.day
                prepared = prepared.drop(columns=[col])
                report["date_columns_expanded"].append(col)

    if max_missing_pct > 0:
        drop_cols = [
            col
            for col in prepared.columns
            if float(prepared[col].isna().mean()) * 100.0 > max_missing_pct
        ]
        if drop_cols:
            prepared = prepared.drop(columns=drop_cols)
            report["columns_dropped_for_missing"] = drop_cols

    if impute_numeric:
        for col in prepared.select_dtypes(include=["number", "bool"]).columns:
            if prepared[col].isna().any():
                prepared[col] = prepared[col].fillna(prepared[col].median())
                report["numeric_columns_imputed"].append(col)

    if impute_categorical:
        for col in prepared.columns:
            if col in prepared.select_dtypes(include=["number", "bool"]).columns:
                continue
            if prepared[col].isna().any():
                mode = prepared[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "missing"
                prepared[col] = prepared[col].fillna(fill_value)
                report["categorical_columns_imputed"].append(col)

    if clip_outliers:
        for col in prepared.select_dtypes(include=["number", "bool"]).columns:
            series = pd.to_numeric(prepared[col], errors="coerce")
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            clipped = series.clip(lower=lo, upper=hi)
            if not clipped.equals(series):
                prepared[col] = clipped
                report["outlier_columns_clipped"].append(col)

    report["final_rows"] = len(prepared)
    report["final_cols"] = len(prepared.columns)
    return PreparationResult(df=prepared, report=report)
