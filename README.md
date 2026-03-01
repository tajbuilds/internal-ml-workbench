# AutoML Workbench

Self-hosted Streamlit workbench for tabular ML workflows (CSV upload, EDA, model training, and evaluation).

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

## Data Handling
- Uploaded datasets are stored on disk at `/data/sourcedata.csv` (mapped to `${WORKBENCH_DATA_LOCATION}` on the host).
- Model artifacts are generated in-memory for download and are not persisted unless you save them externally.
- Do not upload sensitive production data unless your host storage, backups, and access controls meet your organization security requirements.
## Security Notes
- Intended for internal/self-hosted use.
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
- 2026-Q2: Add authentication and basic role-based access (admin/user) for shared internal deployments.
- 2026-Q2: Add model artifact metadata (`model_card.json`) with schema, metrics, train timestamp, and app version.
- 2026-Q3: Add optional experiment tracking (MLflow) with run history and model comparison in UI.

## Credits
- Based on the original Auto ML Streamlit work by Elvis Darko.
- I modernized it for self-hosted use: updated dependencies, modularized app code, added tests/CI, containerized runtime, GHCR publishing, and production compose.
- Remaining bugs and operational issues are my responsibility.

## License
MIT. See `LICENSE`.



