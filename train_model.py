import json
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from config import DATASETS, RANDOM_SEED, PROJECT_ROOT
    from cnn_gru_model import build_cnn_gru_model, create_training_callbacks
except ImportError:
    from .config import DATASETS, RANDOM_SEED, PROJECT_ROOT
    from .cnn_gru_model import build_cnn_gru_model, create_training_callbacks


PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "esp32" / "models"
EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 12
LEARNING_RATE = 1e-3


def set_random_seeds(seed=RANDOM_SEED):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_preprocessed_dataset(dataset_key):
    if dataset_key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Available datasets: {', '.join(DATASETS)}"
        )

    dataset_dir = PROCESSED_DATA_DIR / dataset_key
    data_path = dataset_dir / "preprocessed_data.npz"
    metadata_path = dataset_dir / "metadata.json"

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_path}. "
            "Run data_preprocessing.py first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    with np.load(data_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return arrays, metadata


def load_sequence_data(dataset_key, sequence_length=10):
    from sequence_generation import generate_split_sequences

    arrays, metadata = load_preprocessed_dataset(dataset_key)

    sequences = generate_split_sequences(
        X_train=arrays["X_train"],
        y_train=arrays["y_train"],
        X_validation=arrays["X_validation"],
        y_validation=arrays["y_validation"],
        X_test=arrays["X_test"],
        y_test=arrays["y_test"],
        sequence_length=sequence_length,
        label_position="last",
    )

    return sequences, metadata


def train_dataset(
    dataset_key,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    sequence_length=10,
    learning_rate=LEARNING_RATE,
):
    set_random_seeds()

    sequences, metadata = load_sequence_data(
        dataset_key,
        sequence_length=sequence_length,
    )

    class_count = len(metadata["class_names"])
    feature_count = sequences["feature_count"]

    model = build_cnn_gru_model(
        sequence_length=sequence_length,
        feature_count=feature_count,
        class_count=class_count,
        learning_rate=learning_rate,
    )

    dataset_model_dir = MODEL_OUTPUT_DIR / dataset_key
    dataset_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = dataset_model_dir / "cnn_gru_best.keras"

    callbacks = create_training_callbacks(
        model_path=model_path,
        patience=EARLY_STOPPING_PATIENCE,
    )

    history = model.fit(
        sequences["X_train"],
        sequences["y_train"],
        validation_data=(
            sequences["X_validation"],
            sequences["y_validation"],
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(model_path)
    test_metrics = best_model.evaluate(
        sequences["X_test"],
        sequences["y_test"],
        batch_size=batch_size,
        return_dict=True,
        verbose=1,
    )

    training_summary = {
        "dataset_key": dataset_key,
        "dataset_name": metadata["dataset_name"],
        "sequence_length": sequence_length,
        "feature_count": feature_count,
        "class_count": class_count,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "test_metrics": {
            key: float(value) for key, value in test_metrics.items()
        },
        "model_path": str(model_path),
    }

    summary_path = dataset_model_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=4)

    return best_model, history, training_summary


def main():
    for dataset_key in DATASETS:
        _, _, summary = train_dataset(dataset_key)
        print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
