from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import APP_DATA_DIR, DATA_FILE
from app.core.ml import TrainingArtifacts, train_and_compare

DATASETS_DIR = APP_DATA_DIR / "datasets"
REGISTRY_FILE = DATASETS_DIR / "registry.json"
REPORTS_DIR = APP_DATA_DIR / "reports"
ARTIFACTS_DIR = APP_DATA_DIR / "artifacts"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "dataset"


def _read_registry() -> dict[str, Any]:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        return {"datasets": []}
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def _write_registry(data: dict[str, Any]) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_datasets() -> list[dict[str, Any]]:
    return _read_registry().get("datasets", [])


def get_dataset_meta(dataset_id: str) -> dict[str, Any] | None:
    for item in list_datasets():
        if item.get("id") == dataset_id:
            return item
    return None


def _dataset_csv_path(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id / "data.csv"


def load_dataset_by_id(dataset_id: str) -> pd.DataFrame:
    path = _dataset_csv_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, index_col=None)


def save_uploaded_dataset(
    uploaded_file,
    display_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(uploaded_file, index_col=None)

    source_name = display_name or getattr(uploaded_file, "name", "dataset")
    stem = Path(source_name).stem
    dataset_id = f"{_slugify(stem)}-{uuid.uuid4().hex[:8]}"

    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    registry = _read_registry()
    datasets = registry.get("datasets", [])
    datasets.insert(
        0,
        {
            "id": dataset_id,
            "name": stem,
            "file": str(csv_path),
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    registry["datasets"] = datasets
    _write_registry(registry)

    return df, dataset_id


def delete_dataset(dataset_id: str) -> None:
    registry = _read_registry()
    datasets = [d for d in registry.get("datasets", []) if d.get("id") != dataset_id]
    registry["datasets"] = datasets
    _write_registry(registry)

    dataset_dir = DATASETS_DIR / dataset_id
    if not dataset_dir.exists():
        return

    for item in sorted(dataset_dir.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink(missing_ok=True)
        else:
            item.rmdir()
    dataset_dir.rmdir()


def migrate_legacy_dataset_if_present() -> str | None:
    if not DATA_FILE.exists():
        return None

    existing = list_datasets()
    for item in existing:
        if item.get("name") == "legacy-sourcedata":
            return item.get("id")

    df = pd.read_csv(DATA_FILE, index_col=None)
    dataset_id = f"legacy-sourcedata-{uuid.uuid4().hex[:8]}"
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    registry = _read_registry()
    datasets = registry.get("datasets", [])
    datasets.insert(
        0,
        {
            "id": dataset_id,
            "name": "legacy-sourcedata",
            "file": str(csv_path),
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    registry["datasets"] = datasets
    _write_registry(registry)

    return dataset_id


def _dataset_reports_dir(dataset_id: str) -> Path:
    return REPORTS_DIR / dataset_id


def _dataset_artifacts_dir(dataset_id: str) -> Path:
    return ARTIFACTS_DIR / dataset_id


def _coerce_datetime_candidates(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    object_cols = coerced.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        series = coerced[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        parsed = pd.to_datetime(non_null, errors="coerce")
        parse_ratio = float(parsed.notna().mean())
        if parse_ratio >= 0.8:
            coerced[col] = pd.to_datetime(coerced[col], errors="coerce")

    return coerced


def generate_eda_report(
    df: pd.DataFrame,
    dataset_id: str,
    full_report: bool = True,
) -> Path:
    from ydata_profiling import ProfileReport

    out_dir = _dataset_reports_dir(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "eda_report.html"

    prof_df = _coerce_datetime_candidates(df)
    report = ProfileReport(prof_df, minimal=not full_report, explorative=True)
    report.to_file(report_file)
    return report_file


def run_training(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    *,
    feature_columns: list[str] | None = None,
    target_transform: str = "none",
) -> TrainingArtifacts:
    return train_and_compare(
        df,
        target,
        task_type,
        feature_columns=feature_columns,
        target_transform=target_transform,
    )


def persist_training_artifacts(
    artifacts: TrainingArtifacts,
    dataset_id: str,
) -> dict[str, Path]:
    out_dir = _dataset_artifacts_dir(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = out_dir / "best_model.pkl"
    leaderboard_file = out_dir / "leaderboard.csv"
    setup_file = out_dir / "setup.csv"
    evaluation_file = out_dir / "evaluation.csv"

    model_file.write_bytes(artifacts.model_bytes)
    artifacts.leaderboard_df.to_csv(leaderboard_file, index=False)
    artifacts.setup_df.to_csv(setup_file, index=False)
    artifacts.evaluation_df.to_csv(evaluation_file, index=False)

    return {
        "model": model_file,
        "leaderboard": leaderboard_file,
        "setup": setup_file,
        "evaluation": evaluation_file,
    }


def dataset_kpis(df: pd.DataFrame, target_col: str | None) -> dict[str, str]:
    rows = len(df)
    cols = len(df.columns)
    total_cells = max(1, rows * cols)
    missing_pct = (float(df.isna().sum().sum()) / total_cells) * 100.0
    duplicates = int(df.duplicated().sum())
    target_type = "-"
    if target_col and target_col in df.columns:
        target_type = str(df[target_col].dtype)

    return {
        "rows": f"{rows:,}",
        "cols": f"{cols:,}",
        "missing_pct": f"{missing_pct:.2f}%",
        "duplicates": f"{duplicates:,}",
        "target_type": target_type,
    }
