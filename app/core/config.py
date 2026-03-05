import os
from pathlib import Path


def _default_app_data_dir() -> Path:
    configured = os.getenv("APP_DATA_DIR")
    if configured:
        return Path(configured)

    container_default = Path("/data")
    try:
        container_default.mkdir(parents=True, exist_ok=True)
        return container_default
    except PermissionError:
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "data" / "workbench"


APP_DATA_DIR = _default_app_data_dir()
DATA_FILE = APP_DATA_DIR / "sourcedata.csv"

ASSET_DIR = Path("app") / "assets"
HERO_CANDIDATES = (
    ASSET_DIR / "hero.jpg",
    ASSET_DIR / "hero.jpeg",
    ASSET_DIR / "hero.png",
    ASSET_DIR / "hero.webp",
)


def resolve_hero_image() -> Path | None:
    for candidate in HERO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None
