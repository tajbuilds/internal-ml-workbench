import io
import pickle
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainingArtifacts:
    task_type: str
    best_model_name: str
    best_model: Pipeline
    leaderboard_df: pd.DataFrame
    setup_df: pd.DataFrame
    evaluation_df: pd.DataFrame
    evaluation_payload: dict[str, Any]
    model_bytes: bytes


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    series = df[target_col]
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return "classification"
    unique_count = series.nunique(dropna=True)
    if unique_count <= 20:
        return "classification"
    return "regression"


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def classification_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def regression_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def get_cv_folds(y: pd.Series, task_type: str) -> int:
    if task_type == "classification":
        min_class = int(y.value_counts(dropna=True).min()) if not y.empty else 2
        return max(2, min(5, min_class))
    return max(2, min(5, len(y) // 5))


def scoring_for_task(task_type: str) -> tuple[dict[str, str], str]:
    if task_type == "classification":
        return ({"accuracy": "accuracy", "f1_weighted": "f1_weighted"}, "test_accuracy")
    return (
        {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        "test_r2",
    )


def build_evaluation(
    y_test: pd.Series,
    y_pred: pd.Series,
    task_type: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if task_type == "classification":
        eval_df = pd.DataFrame(
            [
                {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_weighted": precision_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    "recall_weighted": recall_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    "f1_weighted": f1_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                }
            ]
        )
        payload = {
            "y_true": pd.Series(y_test).astype(str).tolist(),
            "y_pred": pd.Series(y_pred).astype(str).tolist(),
            "labels": sorted(pd.Series(y_test).astype(str).unique().tolist()),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }
        return eval_df, payload

    rmse = root_mean_squared_error(y_test, y_pred)
    eval_df = pd.DataFrame([
        {
            "r2": r2_score(y_test, y_pred),
            "rmse": rmse,
            "mae": mean_absolute_error(y_test, y_pred),
        }
    ])
    payload = {
        "y_true": pd.Series(y_test).tolist(),
        "y_pred": pd.Series(y_pred).tolist(),
    }
    return eval_df, payload


def train_and_compare(df: pd.DataFrame, target: str, task_type: str) -> TrainingArtifacts:
    train_df = df.dropna(subset=[target]).copy()
    if train_df.empty:
        raise ValueError("Target column contains only missing values.")

    x = train_df.drop(columns=[target])
    y = train_df[target]
    preprocessor = build_preprocessor(x)

    if task_type == "classification" and y.nunique(dropna=True) < 2:
        raise ValueError("Classification requires at least 2 target classes.")

    stratify = y if task_type == "classification" and y.nunique(dropna=True) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    models = classification_models() if task_type == "classification" else regression_models()
    scoring, sort_key = scoring_for_task(task_type)
    cv = get_cv_folds(y_train, task_type)

    leaderboard_rows: list[dict[str, Any]] = []
    best_model_name = ""
    best_score = float("-inf")
    best_pipeline: Pipeline | None = None

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            error_score="raise",
            n_jobs=-1,
        )

        row: dict[str, Any] = {"Model": model_name, "cv_folds": cv}
        for key in scoring.keys():
            metric_key = f"test_{key}"
            mean_score = float(scores[metric_key].mean())
            if key in {"rmse", "mae"}:
                mean_score = -mean_score
            row[key] = mean_score
        leaderboard_rows.append(row)

        current_score = float(scores[sort_key].mean())
        if current_score > best_score:
            best_score = current_score
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model could be trained.")

    best_pipeline.fit(x_train, y_train)
    y_pred = best_pipeline.predict(x_test)

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by="accuracy" if task_type == "classification" else "r2",
        ascending=False,
    )
    setup_df = pd.DataFrame(
        [
            {
                "task_type": task_type,
                "rows_used": len(train_df),
                "features": x.shape[1],
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "cv_folds": cv,
                "best_model": best_model_name,
            }
        ]
    )
    evaluation_df, evaluation_payload = build_evaluation(y_test, y_pred, task_type)

    buffer = io.BytesIO()
    pickle.dump(best_pipeline, buffer)
    buffer.seek(0)

    return TrainingArtifacts(
        task_type=task_type,
        best_model_name=best_model_name,
        best_model=best_pipeline,
        leaderboard_df=leaderboard_df,
        setup_df=setup_df,
        evaluation_df=evaluation_df,
        evaluation_payload=evaluation_payload,
        model_bytes=buffer.getvalue(),
    )
