from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from config import DATASETS, RANDOM_SEED, PROJECT_ROOT
except ImportError:
    from .config import DATASETS, RANDOM_SEED, PROJECT_ROOT


LABEL_CANDIDATES = (
    "label",
    "class",
    "target",
    "attack",
    "attack_label",
    "attack_type",
    "category",
    "traffic_type",
    "is_attack",
    "binary_label",
)

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
OUTPUT_DIR = PROJECT_ROOT / "processed_data"


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def load_csv(dataset_key):
    if dataset_key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Available datasets: {', '.join(DATASETS)}"
        )

    dataset_path = Path(DATASETS[dataset_key]["path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, low_memory=False)
    if frame.empty:
        raise ValueError(f"The dataset is empty: {dataset_path}")

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")

    return frame


def detect_label_column(frame, dataset_key, label_column=None):
    if label_column is not None:
        if label_column not in frame.columns:
            raise ValueError(
                f"Configured label column '{label_column}' was not found. "
                f"Available columns: {frame.columns.tolist()}"
            )
        return label_column

    normalized = {
        normalize_column_name(column): column for column in frame.columns
    }

    dataset_label_columns = DATASETS[dataset_key].get("label_column")
    if dataset_label_columns:
        if dataset_label_columns not in frame.columns:
            raise ValueError(
                f"Configured label column '{dataset_label_columns}' was not found "
                f"in dataset '{dataset_key}'."
            )
        return dataset_label_columns

    matches = [
        normalized[candidate]
        for candidate in LABEL_CANDIDATES
        if candidate in normalized
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple possible label columns were found: {matches}. "
            "Specify label_column explicitly when calling preprocess_dataset()."
        )

    raise ValueError(
        f"No label column could be identified in '{dataset_key}'. "
        f"Available columns: {frame.columns.tolist()}. "
        "Specify the correct label_column explicitly."
    )


def clean_frame(frame, label_column):
    data = frame.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.drop_duplicates().reset_index(drop=True)

    data[label_column] = data[label_column].astype("string").str.strip()
    invalid_labels = data[label_column].isna() | data[label_column].eq("")
    data = data.loc[~invalid_labels].reset_index(drop=True)

    if data.empty:
        raise ValueError("No samples remain after removing rows with missing labels.")

    feature_columns = [column for column in data.columns if column != label_column]
    for column in feature_columns:
        if data[column].dtype == object or str(data[column].dtype) == "string":
            converted = pd.to_numeric(data[column], errors="coerce")
            non_missing_original = data[column].notna().sum()
            non_missing_converted = converted.notna().sum()
            if non_missing_original and non_missing_converted / non_missing_original >= 0.95:
                data[column] = converted

    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    return data


def prepare_feature_matrices(features, labels, test_size, validation_size, random_seed):
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two target classes are required for classification.")

    if class_counts.min() < 3:
        raise ValueError(
            "Each target class must contain at least three samples for "
            "stratified train/validation/test splitting."
        )

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    train_val_counts = y_train_val.value_counts()
    if train_val_counts.min() < 2:
        raise ValueError(
            "There are too few samples in at least one class to create a "
            "stratified validation split. Adjust the split proportions."
        )

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_seed,
        stratify=y_train_val,
    )

    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_validation = x_validation.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)

    numeric_columns = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in x_train.columns if column not in numeric_columns
    ]

    numeric_medians = x_train[numeric_columns].median()
    x_train.loc[:, numeric_columns] = x_train[numeric_columns].fillna(numeric_medians)
    x_validation.loc[:, numeric_columns] = x_validation[numeric_columns].fillna(numeric_medians)
    x_test.loc[:, numeric_columns] = x_test[numeric_columns].fillna(numeric_medians)

    categorical_modes = {}
    for column in categorical_columns:
        modes = x_train[column].mode(dropna=True)
        fill_value = modes.iloc[0] if not modes.empty else "__missing__"
        categorical_modes[column] = fill_value
        x_train.loc[:, column] = x_train[column].fillna(fill_value).astype(str)
        x_validation.loc[:, column] = x_validation[column].fillna(fill_value).astype(str)
        x_test.loc[:, column] = x_test[column].fillna(fill_value).astype(str)

    x_train = pd.get_dummies(x_train, columns=categorical_columns, dtype=float)
    x_validation = pd.get_dummies(x_validation, columns=categorical_columns, dtype=float)
    x_test = pd.get_dummies(x_test, columns=categorical_columns, dtype=float)

    x_validation = x_validation.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_validation_scaled = scaler.transform(x_validation).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    return (
        x_train_scaled,
        x_validation_scaled,
        x_test_scaled,
        y_train.to_numpy(),
        y_validation.to_numpy(),
        y_test.to_numpy(),
        x_train.columns.tolist(),
        scaler,
    )


def preprocess_dataset(
    dataset_key,
    label_column=None,
    test_size=TEST_SIZE,
    validation_size=VALIDATION_SIZE,
    random_seed=RANDOM_SEED,
    save=True,
):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    frame = load_csv(dataset_key)
    selected_label_column = detect_label_column(
        frame,
        dataset_key,
        label_column=label_column,
    )
    frame = clean_frame(frame, selected_label_column)

    features = frame.drop(columns=[selected_label_column])
    labels = frame[selected_label_column].astype(str)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
        feature_names,
        scaler,
    ) = prepare_feature_matrices(
        features,
        pd.Series(encoded_labels, index=features.index, name=selected_label_column),
        test_size,
        validation_size,
        random_seed,
    )

    result = {
        "dataset_key": dataset_key,
        "dataset_name": DATASETS[dataset_key]["name"],
        "label_column": selected_label_column,
        "X_train": x_train,
        "X_validation": x_validation,
        "X_test": x_test,
        "y_train": y_train.astype(np.int64),
        "y_validation": y_validation.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "feature_names": feature_names,
        "class_names": label_encoder.classes_.tolist(),
        "label_encoder": label_encoder,
        "scaler": scaler,
        "input_feature_count": len(feature_names),
        "class_count": len(label_encoder.classes_),
    }

    if save:
        save_preprocessed_dataset(result)

    return result


def save_preprocessed_dataset(result):
    dataset_output_dir = OUTPUT_DIR / result["dataset_key"]
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        dataset_output_dir / "preprocessed_data.npz",
        X_train=result["X_train"],
        X_validation=result["X_validation"],
        X_test=result["X_test"],
        y_train=result["y_train"],
        y_validation=result["y_validation"],
        y_test=result["y_test"],
    )

    metadata = {
        "dataset_key": result["dataset_key"],
        "dataset_name": result["dataset_name"],
        "label_column": result["label_column"],
        "feature_names": result["feature_names"],
        "class_names": result["class_names"],
        "input_feature_count": result["input_feature_count"],
        "class_count": result["class_count"],
        "sequence_length": 10,
        "split": {
            "training": 0.64,
            "validation": 0.16,
            "testing": 0.20,
        },
    }

    with (dataset_output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def main():
    for dataset_key in DATASETS:
        result = preprocess_dataset(dataset_key)
        print(
            f"{result['dataset_name']}: "
            f"train={len(result['y_train'])}, "
            f"validation={len(result['y_validation'])}, "
            f"test={len(result['y_test'])}, "
            f"features={result['input_feature_count']}, "
            f"classes={result['class_count']}"
        )


if __name__ == "__main__":
    main()
