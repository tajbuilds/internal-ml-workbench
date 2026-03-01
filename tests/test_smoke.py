from app.main import main


def test_smoke_import_main() -> None:
    assert callable(main)
