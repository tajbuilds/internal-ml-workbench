from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    path = Path("data/samples/classification_sample.csv")
    return pd.read_csv(path)


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    path = Path("data/samples/regression_sample.csv")
    return pd.read_csv(path)
