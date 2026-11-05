import json
import random
from pathlib import Path

import numpy as np


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def ensure_directory(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, output_path, indent=4):
    path = Path(output_path)
    ensure_directory(path.parent)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    return path


def load_json(input_path):
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_numpy_arrays(output_path, **arrays):
    path = Path(output_path)
    ensure_directory(path.parent)
    np.savez_compressed(path, **arrays)
    return path


def load_numpy_arrays(input_path):
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(f"NumPy archive not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_classification_arrays(y_true, y_pred):
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Classification labels must be one-dimensional.")

    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("True and predicted labels must have equal lengths.")

    if true_values.size == 0:
        raise ValueError("Classification arrays must not be empty.")

    return true_values, predicted_values


def get_class_distribution(labels):
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    total = int(counts.sum())

    return {
        str(value): {
            "count": int(count),
            "percentage": float(count / total * 100),
        }
        for value, count in zip(values, counts)
    }


def validate_feature_matrix(features):
    values = np.asarray(features)

    if values.ndim != 2:
        raise ValueError("Feature matrix must have shape (samples, features).")

    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Feature matrix must contain samples and features.")

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Feature matrix must contain numeric values.")

    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    return values


def summarize_array(values):
    array = np.asarray(values)

    if array.size == 0:
        raise ValueError("Cannot summarize an empty array.")

    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }

    if np.issubdtype(array.dtype, np.number):
        finite_values = array[np.isfinite(array)]
        if finite_values.size:
            summary.update(
                {
                    "minimum": float(np.min(finite_values)),
                    "maximum": float(np.max(finite_values)),
                    "mean": float(np.mean(finite_values)),
                    "standard_deviation": float(np.std(finite_values)),
                }
            )

    return summary
