# AutoML Workbench

Self-hosted Streamlit workbench for tabular ML workflows (CSV upload, automated EDA, model training, and evaluation).

## Features
- CSV upload with persistent storage under `/data`
- Automated EDA with `ydata-profiling`
- Classification and regression model comparison (scikit-learn)
- Holdout evaluation report (metrics + confusion matrix/scatter)
- Download trained model pipeline as `.pkl`
- GHCR image publishing workflow and production compose deployment

## Quick Start (Docker Compose)
1. Clone and enter repo.
```bash
git clone https://github.com/tajbuilds/internal-ml-workbench.git
cd internal-ml-workbench
```
2. Create env file.
```bash
cp .env.example .env
```
3. Start service.
```bash
docker compose up -d
```

Open: `http://localhost:${APP_PORT}` (default `8501`).

## Configuration
Configured via `.env`:
- `IMW_VERSION` image tag (for example `latest` or `v1.0.0`)
- `APP_PORT` host port
- `WORKBENCH_DATA_LOCATION` host path mounted to container `/data`

## Data and Artifacts
- Container path: `/data`
- Persisted host path: `${WORKBENCH_DATA_LOCATION}`
- Uploaded dataset file: `/data/sourcedata.csv`

## Security Notes
- This is intended as an internal/self-hosted tool.
- Do not expose directly to the internet without reverse proxy auth/TLS.
- Keep `.env` and any secrets out of git.

## Architecture
- `app/main.py`: Streamlit entrypoint/routing
- `app/pages/main_pages.py`: UI pages
- `app/core/ml.py`: model training and evaluation engine
- `app/core/state.py`: Streamlit session/data state
- `app/core/config.py`: environment-driven configuration
- `tests/`: smoke/unit coverage

## Roadmap
- Add user authentication option for shared deployments
- Add model metadata/versioning for exported artifacts
- Add optional experiment tracking

## Credits
- This project was inspired by the original Auto ML Streamlit work by Elvis Darko.
- It has been modernized and extended for internal self-hosted operation, including dependency upgrades, Docker hardening, GHCR publishing, production compose layout, and CI checks.
- Any remaining issues are mine.

## License
MIT. See `LICENSE`.
