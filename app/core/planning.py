from __future__ import annotations

import pandas as pd


def recommended_target_transform(df: pd.DataFrame, target_col: str, task_type: str) -> str:
    if task_type != "regression" or target_col not in df.columns:
        return "none"

    target = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if target.empty or bool((target < 0).any()):
        return "none"

    return "log1p" if abs(float(target.skew())) >= 1.0 else "none"


def feature_presets(df: pd.DataFrame, target_col: str, task_type: str) -> dict[str, list[str]]:
    available = [col for col in df.columns if col != target_col]
    if not available:
        return {"All Features": []}

    numeric = [
        col
        for col in df.select_dtypes(include=["number", "bool"]).columns.tolist()
        if col != target_col
    ]
    categorical = [col for col in available if col not in numeric]

    presets: dict[str, list[str]] = {"All Features": available}

    if numeric:
        presets["Numeric Only"] = numeric

    location_cols = [
        col
        for col in available
        if col.lower() in {"lat", "long", "longitude", "latitude", "zip"}
    ]
    if location_cols:
        location_preset = location_cols[:]
        for col in numeric:
            if col not in location_preset:
                location_preset.append(col)
            if len(location_preset) >= min(5, len(available)):
                break
        presets["Location Focus"] = location_preset

    if task_type == "regression" and numeric:
        corr = df[numeric + [target_col]].corr(numeric_only=True)[target_col].drop(
            labels=[target_col], errors="ignore"
        )
        top_numeric = corr.abs().sort_values(ascending=False).head(min(5, len(corr))).index.tolist()
        if top_numeric:
            presets["Top Correlated"] = top_numeric
            enriched = top_numeric[:]
            for col in categorical[:2]:
                if col not in enriched:
                    enriched.append(col)
            presets["Enriched Mix"] = enriched
    else:
        baseline = available[: min(5, len(available))]
        presets["Baseline"] = baseline

    deduped: dict[str, list[str]] = {}
    for name, cols in presets.items():
        cleaned: list[str] = []
        for col in cols:
            if col in available and col not in cleaned:
                cleaned.append(col)
        deduped[name] = cleaned
    return deduped


def preparation_recommendations(audit: dict[str, pd.DataFrame]) -> list[str]:
    notes: list[str] = []

    missing = audit.get("missing")
    if missing is not None and not missing.empty:
        worst = missing.iloc[0]
        notes.append(
            "Missing data exists. "
            f"Highest missing column: {worst['column']} ({worst['missing_pct']}%)."
        )

    coercion = audit.get("coercion_candidates")
    if coercion is not None and not coercion.empty:
        numeric_like = coercion[coercion["numeric_parse_pct"] >= 80]
        datetime_like = coercion[coercion["datetime_parse_pct"] >= 80]
        if not numeric_like.empty:
            notes.append(
                "Some text columns look numeric and can be coerced safely."
            )
        if not datetime_like.empty:
            notes.append(
                "Some text columns look date-like and can be expanded into year/month/day features."
            )

    outliers = audit.get("outliers")
    if outliers is not None and not outliers.empty:
        flagged = outliers[outliers["outlier_pct"] >= 5]
        if not flagged.empty:
            notes.append(
                "Several numeric columns have material outliers; "
                "review boxplots or apply IQR clipping."
            )

    if not notes:
        notes.append(
            "No obvious preparation issues detected. "
            "You can move straight to EDA or modeling."
        )
    return notes
