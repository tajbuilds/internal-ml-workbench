# Upstream Attribution and Change Summary

## Original Project
This repository is based on the original work:
- https://github.com/elvis-darko/AUTO-MACHINE-LEARNING-WEB-APP-USING-STREAMLIT-AND-PYCARET
- Author: Elvis Darko

## What Changed in This Repository

### 1) ML Stack Migration
- Replaced PyCaret-based training flow with native scikit-learn pipelines.
- Added explicit preprocessing for numeric and categorical features.
- Added model comparison with cross-validation for both classification and regression tasks.

### 2) App Structure and Maintainability
- Refactored from a single-file Streamlit app into a modular package structure:
  - `app/main.py`
  - `app/pages/main_pages.py`
  - `app/core/*`
- Added clearer separation of UI, state, configuration, and ML logic.

### 3) Evaluation and UX Improvements
- Added holdout evaluation reporting.
- Added confusion matrix and classification report for classification tasks.
- Added predicted-vs-actual visualization for regression tasks.

### 4) Deployment and Operations
- Added containerized runtime with a production-focused Dockerfile.
- Added environment-driven Docker Compose deployment.
- Standardized runtime data path (`APP_DATA_DIR=/data`) for persistent storage.

### 5) CI/CD and Distribution
- Added CI workflow for linting, tests, and Streamlit smoke checks.
- Added GHCR image publishing workflow:
  - `latest` on `main` pushes
  - version tag (e.g. `v1.0.0`) on release tag pushes

### 6) Documentation and Security
- Rewrote README for Docker-first self-hosted usage.
- Added data-handling notes.
- Added `SECURITY.md` policy for vulnerability reporting.

## Responsibility
The upstream project credit remains with the original author. Any bugs, regressions, deployment issues, or operational decisions introduced in this repository are owned by the current maintainer.
