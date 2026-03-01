# AutoML Workbench (Streamlit + scikit-learn)

A modular local-first Streamlit app that:
- uploads a CSV dataset,
- generates automated profiling (ydata-profiling),
- compares multiple scikit-learn models with cross-validation,
- provides a holdout evaluation report,
- downloads the best fitted pipeline.

## Quick Start (Docker Compose)

### 1) Clone
```bash
git clone <your-repo-url>
cd internal-ml-workbench
```

### 2) Create env file
```bash
cp .env.example .env
```

### 3) Run
```bash
docker compose up -d --build
```

Open: `http://localhost:8501` (or your `APP_PORT` in `.env`).

## Update flow
When the repo updates:
```bash
git pull
docker compose up -d --build
```

## Compose model
- Compose builds local image `internal-ml-workbench:latest` by default.
- Data persistence is through volume `workbench-data` mounted to `/data`.
- Uploaded app data is stored as `/data/sourcedata.csv` in container.

## Runtime target
- Python 3.11

## Project layout
- `streamlit_app.py`: thin launcher
- `workbench/app.py`: routing + page selection
- `workbench/pages.py`: Streamlit pages
- `workbench/ml.py`: training, CV ranking, evaluation, model packaging
- `workbench/state.py`: Streamlit session state helpers
- `tests/`: unit tests for ML flow
- `data/samples/`: sample regression/classification datasets
- `.github/workflows/ci.yml`: lint + tests + smoke CI

## Local Python setup (optional, non-Docker)
```bat
call C:\Users\Taj\anaconda3\Scripts\activate.bat C:\Users\Taj\anaconda3
conda create -y -n imw311 python=3.11
conda activate imw311
cd /d C:\dev\internal-ml-workbench
pip install -r requirements.txt
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

## Quality checks
```bash
ruff check .
pytest
```
