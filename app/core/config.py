import os
from pathlib import Path

APP_DATA_DIR = Path(os.getenv("APP_DATA_DIR", "/data"))
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
