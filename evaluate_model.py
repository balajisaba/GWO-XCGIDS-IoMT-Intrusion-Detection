import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from config import DATASETS, PROJECT_ROOT
    from train_model import load_sequence_data, MODEL_OUTPUT_DIR
except ImportError:
    from .config import DATASETS, PROJECT_ROOT
    from .train_model import load_sequence_data, MODEL_OUTPUT_DIR


EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "evaluation"


def calculate_metrics(y_true, probabilities, class_names=None):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 1:
        predictions = (probabilities >= 0.5).astype(int)
        class_count = 2
    elif probabilities.ndim == 2:
        predictions = np.argmax(probabilities, axis=1)
        class_count = probabilities.shape[1]
    else:
        raise ValueError("Predictions must be a one- or two-dimensional array.")

    labels = np.arange(class_count)
    target_names = None

    if class_names is not None and len(class_names) == class_count:
        target_names = [str(name) for name in class_names]

    average = "binary" if class_count == 2 else "weighted"

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                average=average,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                average=average,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                average=average,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    try:
        if class_count == 2:
            positive_scores = (
                probabilities[:, 1]
                if probabilities.ndim == 2
                else probabilities
            )
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, positive_scores)
            )
        else:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=labels,
                )
            )
    except ValueError:
        metrics["roc_auc"] = None

    return predictions, metrics


def evaluate_dataset(dataset_key, sequence_length=10, batch_size=64):
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

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(
        sequences["X_test"],
        batch_size=batch_size,
        verbose=1,
    )

    class_names = metadata.get("class_names")
    predictions, metrics = calculate_metrics(
        sequences["y_test"],
        probabilities,
        class_names=class_names,
    )

    output_dir = EVALUATION_OUTPUT_DIR / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset_key": dataset_key,
        "dataset_name": metadata["dataset_name"],
        "test_samples": int(len(sequences["y_test"])),
        "sequence_length": sequence_length,
        "class_names": class_names,
        "metrics": metrics,
    }

    result_path = output_dir / "evaluation_metrics.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    np.savez_compressed(
        output_dir / "predictions.npz",
        y_true=sequences["y_test"],
        y_pred=predictions,
        y_prob=probabilities,
    )

    return result


def main():
    for dataset_key in DATASETS:
        result = evaluate_dataset(dataset_key)
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
