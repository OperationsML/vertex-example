import joblib
from kfp.v2.dsl import component, Input, OutputPath, Dataset, InputPath
from typing import NamedTuple
from google_cloud_pipeline_components.experimental.custom_job import utils


@component(
    packages_to_install=["joblib"],
    base_image="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-6:latest",
)
def nlp_data_pull(
    input_file_path: InputPath(),
    x_train_path: OutputPath(),
    y_train_path: OutputPath(),
    x_val_path: OutputPath(),
    y_val_path: OutputPath(),
):

    import tensorflow as tf
    from tensorflow import keras
    import joblib

    print("trying to load input path")
    try:
        print(input_file_path)
    except Exception as e:
        pass

    print("load nlp data")
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=vocab_size
    )
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    print("save data")
    joblib.dump(x_train, f"{x_train_path}.pkl")
    joblib.dump(x_val, f"{x_val_path}.pkl")
    joblib.dump(y_train, f"{y_train_path}.pkl")
    joblib.dump(y_val, f"{y_val_path}.pkl")
    print("job complete")


nlp_data_op = utils.create_custom_training_job_op_from_component(
    component_spec=nlp_data_pull, machine_type="e2-standard-8", replica_count=1,
)
