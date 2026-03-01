import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    path = ROOT / "data" / "samples" / "classification_sample.csv"
    return pd.read_csv(path)


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    path = ROOT / "data" / "samples" / "regression_sample.csv"
    return pd.read_csv(path)
