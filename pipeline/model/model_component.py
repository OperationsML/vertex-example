import joblib
from kfp.v2.dsl import component, InputPath, OutputPath
from google_cloud_pipeline_components.experimental.custom_job import utils


@component(
    packages_to_install=["joblib"],
    base_image="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-6:latest",
)
def train_task(
    x_train_path: InputPath(),
    y_train_path: InputPath(),
    x_val_path: InputPath(),
    y_val_path: InputPath(),
    model_path: OutputPath(),
):
    import logging
    import sys

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import joblib

    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review

    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
            super(TransformerBlock, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.rate = rate
            self.att = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embed_dim
            )
            self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(self.ff_dim, activation="relu"),
                    tf.keras.layers.Dense(self.embed_dim),
                ]
            )
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(self.rate)
            self.dropout2 = tf.keras.layers.Dropout(self.rate)

        def get_config(self):
            config = super().get_config().copy()
            config.update(
                {
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim,
                    "rate": self.rate,
                    "att": self.att,
                    "ffn": self.ffn,
                    "layernorm1": self.layernorm1,
                    "layernorm2": self.layernorm2,
                    "dropout1": self.dropout1,
                    "dropout2": self.dropout2,
                }
            )
            return config

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    class TokenAndPositionEmbedding(tf.keras.layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
            super(TokenAndPositionEmbedding, self).__init__()
            self.maxlen = maxlen
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.token_emb = tf.keras.layers.Embedding(
                input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero=True
            )
            self.pos_emb = tf.keras.layers.Embedding(
                input_dim=self.maxlen, output_dim=self.embed_dim, mask_zero=True
            )

        def get_config(self):
            config = super().get_config().copy()
            config.update(
                {
                    "maxlen": self.maxlen,
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                    "token_emb": self.token_emb,
                    "pos_emb": self.pos_emb,
                }
            )
            return config

        def call(self, x):
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions

    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{model_path}.keras", save_best_only=True)
    ]

    # Load data
    x_train = joblib.load(f"{x_train_path}.pkl")
    y_train = joblib.load(f"{y_train_path}.pkl")
    x_val = joblib.load(f"{x_val_path}.pkl")
    y_val = joblib.load(f"{y_val_path}.pkl")

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        validation_data=(x_val, y_val),
        epochs=10,
        callbacks=callbacks,
    )


nlp_train_task = utils.create_custom_training_job_op_from_component(
    component_spec=train_task,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    replica_count=1,
)
