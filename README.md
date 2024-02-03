# Prolaio DS Demo

## Google Collab running on top of Bigquery and GCS

The goal is to show how to run a python notebook in Google Collab on top of Google Bigquery and Google Storage.

Please see this [notebook](https://console.cloud.google.com/vertex-ai/colab/notebooks?authuser=4&hl=en&project=prolaio-data-testing&activeNb=projects%2Fprolaio-data-testing%2Flocations%2Fus-east1%2Frepositories%2Fd74ee767-34f2-480f-8c86-569787605ab1)

Here a [local copy](demo_collab.ipynb)

## Metaflow

To run locally:
- `poetry install` to install the project
- `poetry run python -m demo_metaflow_parallel.py run`

## Google Vertex Registry

Here is an [example](saving_model_vertex.py) of Google Model Registry