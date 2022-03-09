from kfp.v2.dsl import component, OutputPath, Output, Dataset
from typing import NamedTuple
from google_cloud_pipeline_components.experimental.custom_job import utils


@component(
    packages_to_install=["sklearn", "joblib"], base_image="python:3.9",
)
def iris_data_pull(iris_data_path: OutputPath()):
    from sklearn.datasets import load_iris
    import joblib

    x, y = load_iris(return_X_y=True)

    joblib.dump(x, f"{iris_data_path}.pkl")


iris_data_op = utils.create_custom_training_job_op_from_component(
    component_spec=iris_data_pull, machine_type="n1-standard-4", replica_count=1,
)

