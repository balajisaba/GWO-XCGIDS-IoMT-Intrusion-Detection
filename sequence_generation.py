import numpy as np


def create_sequences(features, labels, sequence_length=10, label_position="last"):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")
    if len(features) != len(labels):
        raise ValueError("features and labels must contain the same number of samples.")
    if not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer.")
    if label_position not in {"last", "majority"}:
        raise ValueError("label_position must be either 'last' or 'majority'.")
    if len(features) < sequence_length:
        raise ValueError(
            f"At least {sequence_length} samples are required to create one sequence; "
            f"received {len(features)}."
        )

    sequence_count = len(features) - sequence_length + 1
    sequences = np.empty(
        (sequence_count, sequence_length, features.shape[1]),
        dtype=np.float32,
    )
    sequence_labels = np.empty(sequence_count, dtype=labels.dtype)

    for start_index in range(sequence_count):
        end_index = start_index + sequence_length
        sequences[start_index] = features[start_index:end_index]

        window_labels = labels[start_index:end_index]
        if label_position == "last":
            sequence_labels[start_index] = window_labels[-1]
        else:
            values, counts = np.unique(window_labels, return_counts=True)
            sequence_labels[start_index] = values[np.argmax(counts)]

    return sequences, sequence_labels


def generate_split_sequences(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    sequence_length=10,
    label_position="last",
):
    X_train_seq, y_train_seq = create_sequences(
        X_train, y_train, sequence_length, label_position
    )
    X_validation_seq, y_validation_seq = create_sequences(
        X_validation, y_validation, sequence_length, label_position
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test, y_test, sequence_length, label_position
    )

    return {
        "X_train": X_train_seq,
        "y_train": y_train_seq,
        "X_validation": X_validation_seq,
        "y_validation": y_validation_seq,
        "X_test": X_test_seq,
        "y_test": y_test_seq,
        "sequence_length": sequence_length,
        "feature_count": X_train_seq.shape[2],
    }


def generate_sequences_from_preprocessed(preprocessed_data, sequence_length=10):
    required_keys = {
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
        "X_test",
        "y_test",
    }
    missing_keys = required_keys.difference(preprocessed_data)
    if missing_keys:
        raise KeyError(
            f"Preprocessed data is missing required keys: {sorted(missing_keys)}"
        )

    return generate_split_sequences(
        X_train=preprocessed_data["X_train"],
        y_train=preprocessed_data["y_train"],
        X_validation=preprocessed_data["X_validation"],
        y_validation=preprocessed_data["y_validation"],
        X_test=preprocessed_data["X_test"],
        y_test=preprocessed_data["y_test"],
        sequence_length=sequence_length,
        label_position="last",
    )


if __name__ == "__main__":
    raise SystemExit(
        "Import generate_sequences_from_preprocessed() from the training pipeline."
    )
