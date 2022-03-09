import yaml
from pathlib import Path
from google.cloud import storage
import os


def get_config(filename: str = None) -> dict:
    filename = (
        filename or "pipeline_config.yaml"
    )  # deploy config is called during the running of each py file
    for path in Path(os.getcwd()).rglob(filename):
        config_path = path.absolute()
    assert os.path.isfile(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def upload_file(project_id, bucket, destination_filename, source_filename):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket)
    blob = bucket.blob(destination_filename)
    blob.upload_from_filename(source_filename)
