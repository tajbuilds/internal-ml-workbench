import os
from pathlib import Path

APP_DATA_DIR = Path(os.getenv("APP_DATA_DIR", "."))
DATA_FILE = APP_DATA_DIR / "sourcedata.csv"

APP_IMAGE_URL = (
    "https://github.com/elvis-darko/AUTO-MACHINE-LEARNING-WEB-APP-USING-"
    "STREAMLIT-AND-PYCARET/raw/main/images/AUTOML.jpg"
)
DEVELOPER_IMAGE_URL = (
    "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/"
    "Assets/images/developer.png"
)
