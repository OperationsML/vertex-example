# Vertex and Kubeflow
from kfp.v2 import dsl
from kfp.v2 import compiler
from google_cloud_pipeline_components.experimental.custom_job import utils
from google.cloud import aiplatform
import google.cloud.aiplatform as aip
from datetime import datetime
from utils.helpers import get_config
import os

from data.iris_data import iris_data_op
from data.nlp_data import nlp_data_op
from model.model_component import nlp_train_task


# Globals
config = get_config()
PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = config.get("BUCKET_NAME")
REGION = config.get("REGION")
PIPELINE_NAME = config.get("PIPELINE_NAME")
PIPELINE_ROOT = "{}/pipeline".format(BUCKET_NAME)
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME = "test_" + TIMESTAMP

# Initialize ai platform
aip.init(project=PROJECT_ID, staging_bucket=BUCKET_NAME)


def run_pipeline():
    @dsl.pipeline(
        pipeline_root=PIPELINE_ROOT, name="pipeline-v2",
    )
    def pipeline():
        from google_cloud_pipeline_components.v1.model import ModelUploadOp

        iris_op = iris_data_op(project=PROJECT_ID, location=REGION)

        nlp_op = nlp_data_op(
            project=PROJECT_ID,
            location=REGION,
            input_file_path=iris_op.outputs["iris_data_path"],
        )

        train_op = nlp_train_task(
            project=PROJECT_ID,
            location=REGION,
            x_train_path=nlp_op.outputs["x_train_path"],
            y_train_path=nlp_op.outputs["y_train_path"],
            x_val_path=nlp_op.outputs["x_val_path"],
            y_val_path=nlp_op.outputs["y_val_path"],
        )

    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="data_pipeline.json",
    )

    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="data_pipeline.json".replace(" ", "_"),
        job_id=f"test-pipeline-{TIMESTAMP}",
        pipeline_root=f"gs://{PIPELINE_ROOT}",
    )

    job.submit()


if __name__ == "__main__":
    run_pipeline()
