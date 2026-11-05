from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


def build_cnn_gru_model(
    sequence_length,
    feature_count,
    class_count,
    conv_filters=64,
    kernel_size=3,
    gru_units=64,
    dense_units=64,
    dropout_rate=0.3,
    learning_rate=1e-3,
    l2_strength=1e-4,
):
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    if feature_count < 1:
        raise ValueError("feature_count must be positive.")
    if class_count < 2:
        raise ValueError("class_count must be at least 2.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    inputs = layers.Input(
        shape=(sequence_length, feature_count),
        name="network_traffic_sequence",
    )

    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength),
        name="cnn_feature_extraction",
    )(inputs)

    x = layers.BatchNormalization(name="cnn_batch_normalization")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="temporal_pooling")(x)
    x = layers.Dropout(dropout_rate, name="cnn_dropout")(x)

    x = layers.GRU(
        units=gru_units,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=0.0,
        kernel_regularizer=regularizers.l2(l2_strength),
        name="gru_temporal_learning",
    )(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength),
        name="dense_representation",
    )(x)

    x = layers.Dropout(dropout_rate, name="dense_dropout")(x)

    if class_count == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)
        loss = "binary_crossentropy"
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    else:
        outputs = layers.Dense(
            class_count,
            activation="softmax",
            name="multiclass_output",
        )(x)
        loss = "sparse_categorical_crossentropy"
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]

    model = Model(inputs=inputs, outputs=outputs, name="GWO_XCGIDS_CNN_GRU")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


def create_training_callbacks(
    model_path,
    patience=10,
    monitor="val_loss",
):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode="min" if "loss" in monitor else "max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode="min" if "loss" in monitor else "max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
            mode="min" if "loss" in monitor else "max",
            verbose=1,
        ),
    ]


if __name__ == "__main__":
    from pathlib import Path

    model = build_cnn_gru_model(
        sequence_length=10,
        feature_count=16,
        class_count=2,
    )
    model.summary()
