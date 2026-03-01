from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.config import APP_DATA_DIR, DATA_FILE
from app.core.ml import TrainingArtifacts, train_and_compare

REPORTS_DIR = APP_DATA_DIR / "reports"
ARTIFACTS_DIR = APP_DATA_DIR / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "best_model.pkl"
LEADERBOARD_FILE = ARTIFACTS_DIR / "leaderboard.csv"
SETUP_FILE = ARTIFACTS_DIR / "setup.csv"
EVALUATION_FILE = ARTIFACTS_DIR / "evaluation.csv"
EDA_REPORT_FILE = REPORTS_DIR / "eda_report.html"


def save_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, index_col=None)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    return df


def load_persisted_dataframe() -> pd.DataFrame | None:
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE, index_col=None)
    return None


def clear_persisted_dataframe() -> None:
    if DATA_FILE.exists():
        DATA_FILE.unlink(missing_ok=True)


def generate_eda_report(df: pd.DataFrame) -> Path:
    from ydata_profiling import ProfileReport

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = ProfileReport(df, minimal=True, explorative=True)
    report.to_file(EDA_REPORT_FILE)
    return EDA_REPORT_FILE


def run_training(df: pd.DataFrame, target: str, task_type: str) -> TrainingArtifacts:
    return train_and_compare(df, target, task_type)


def persist_training_artifacts(artifacts: TrainingArtifacts) -> dict[str, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_FILE.write_bytes(artifacts.model_bytes)
    artifacts.leaderboard_df.to_csv(LEADERBOARD_FILE, index=False)
    artifacts.setup_df.to_csv(SETUP_FILE, index=False)
    artifacts.evaluation_df.to_csv(EVALUATION_FILE, index=False)

    return {
        "model": MODEL_FILE,
        "leaderboard": LEADERBOARD_FILE,
        "setup": SETUP_FILE,
        "evaluation": EVALUATION_FILE,
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
