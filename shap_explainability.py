import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import shap

try:
    from config import DATASETS, PROJECT_ROOT, RANDOM_SEED
    from train_model import load_sequence_data, MODEL_OUTPUT_DIR
except ImportError:
    from .config import DATASETS, PROJECT_ROOT, RANDOM_SEED
    from .train_model import load_sequence_data, MODEL_OUTPUT_DIR


EXPLANATION_OUTPUT_DIR = PROJECT_ROOT / "explainability"
BACKGROUND_SAMPLES = 50
EXPLANATION_SAMPLES = 100
KERNEL_NSAMPLES = 100


def flatten_sequences(sequences):
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError("Expected sequence data with shape (samples, time_steps, features).")
    return sequences.reshape(sequences.shape[0], -1)


def build_prediction_function(model, sequence_length, feature_count, batch_size=64):
    def predict(flattened_data):
        flattened_data = np.asarray(flattened_data, dtype=np.float32)
        shaped_data = flattened_data.reshape(
            -1,
            sequence_length,
            feature_count,
        )
        return model.predict(shaped_data, batch_size=batch_size, verbose=0)

    return predict


def calculate_feature_importance(shap_values, sequence_length, feature_count):
    values = np.asarray(shap_values)

    if values.ndim == 3:
        values = np.mean(np.abs(values), axis=2)

    if values.ndim != 2:
        raise ValueError(
            f"Unsupported SHAP output shape: {values.shape}"
        )

    values = values.reshape(
        values.shape[0],
        sequence_length,
        feature_count,
    )

    importance = np.mean(np.abs(values), axis=(0, 1))
    return importance


def explain_dataset(
    dataset_key,
    sequence_length=10,
    background_samples=BACKGROUND_SAMPLES,
    explanation_samples=EXPLANATION_SAMPLES,
    kernel_nsamples=KERNEL_NSAMPLES,
):
    if dataset_key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Available datasets: {', '.join(DATASETS)}"
        )

    model_path = MODEL_OUTPUT_DIR / dataset_key / "cnn_gru_best.keras"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run train_model.py first."
        )

    sequences, metadata = load_sequence_data(
        dataset_key,
        sequence_length=sequence_length,
    )

    X_train = np.asarray(sequences["X_train"], dtype=np.float32)
    X_test = np.asarray(sequences["X_test"], dtype=np.float32)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training and test sequence sets must both be non-empty.")

    rng = np.random.default_rng(RANDOM_SEED)
    background_count = min(background_samples, len(X_train))
    explanation_count = min(explanation_samples, len(X_test))

    background_indices = rng.choice(
        len(X_train),
        size=background_count,
        replace=False,
    )
    explanation_indices = rng.choice(
        len(X_test),
        size=explanation_count,
        replace=False,
    )

    background = flatten_sequences(X_train[background_indices])
    samples = flatten_sequences(X_test[explanation_indices])

    model = tf.keras.models.load_model(model_path)
    feature_count = X_train.shape[2]

    prediction_function = build_prediction_function(
        model,
        sequence_length,
        feature_count,
    )

    explainer = shap.KernelExplainer(
        prediction_function,
        background,
    )

    raw_values = explainer.shap_values(
        samples,
        nsamples=kernel_nsamples,
    )

    if isinstance(raw_values, list):
        shap_array = np.stack(
            [np.asarray(value) for value in raw_values],
            axis=-1,
        )
    else:
        shap_array = np.asarray(raw_values)

    if shap_array.ndim == 3:
        aggregated_values = np.mean(np.abs(shap_array), axis=2)
    else:
        aggregated_values = shap_array

    aggregated_values = aggregated_values.reshape(
        explanation_count,
        sequence_length,
        feature_count,
    )

    feature_importance = np.mean(
        np.abs(aggregated_values),
        axis=(0, 1),
    )

    feature_names = metadata.get("feature_names")
    if not feature_names or len(feature_names) != feature_count:
        feature_names = [
            f"feature_{index + 1}"
            for index in range(feature_count)
        ]

    ranked_features = sorted(
        [
            {
                "feature": str(feature_names[index]),
                "mean_absolute_shap": float(feature_importance[index]),
            }
            for index in range(feature_count)
        ],
        key=lambda item: item["mean_absolute_shap"],
        reverse=True,
    )

    output_dir = EXPLANATION_OUTPUT_DIR / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_key": dataset_key,
        "dataset_name": metadata["dataset_name"],
        "method": "SHAP KernelExplainer",
        "background_samples": background_count,
        "explained_samples": explanation_count,
        "sequence_length": sequence_length,
        "feature_count": feature_count,
        "ranked_feature_importance": ranked_features,
    }

    summary_path = output_dir / "shap_feature_importance.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    np.savez_compressed(
        output_dir / "shap_values.npz",
        shap_values=shap_array,
        explained_samples=samples,
        background_samples=background,
    )

    return summary


def main():
    for dataset_key in DATASETS:
        summary = explain_dataset(dataset_key)
        print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
