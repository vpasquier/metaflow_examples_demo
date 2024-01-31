# Prolaio DS Demo

## Example of Google Collab running on top of Bigquery and GCS

- Please copy and paste the content of [demo_collab.ipynb](demo_collab.ipynb) in Google Collab
- Run the python notebook in collab to see the results and the different cells executions

## Example of Metaflow running two model training with same dataset

To run locally:
- `poetry install` to install the project
- `poetry run python -m demo_metaflow.py run`

To run on k8s with the metaflow UI:
- `poetry run python -m demo_metaflow.py argo-workflows create`
- `poetry run python -m demo_metaflow.py argo-workflows trigger`