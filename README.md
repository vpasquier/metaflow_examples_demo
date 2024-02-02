# Prolaio DS Demo

## Basic tooling: Google Collab running on top of Bigquery and GCS

The goal is to show how to run a python notebook in Google Collab on top of Google Bigquery and Google Storage:

- Using the Google Python Client
- Using custom libraries like `prolaiotoolkit`

Please see this notebook:

## Advanced tooling: Metaflow

The goal is to show how to use Metaflow to:
- Run in the cloud very easily a notebook
- Train/Evaluate/Serve models
- Save models and promote them in production

To run locally:
- `poetry install` to install the project
- `poetry run python -m demo_metaflow.py run`

To run on k8s with the metaflow UI:
- `poetry run python -m demo_metaflow.py argo-workflows create`
- `poetry run python -m demo_metaflow.py argo-workflows trigger`