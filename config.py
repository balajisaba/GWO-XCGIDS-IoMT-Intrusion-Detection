from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
SOURCE_DIR = PROJECT_ROOT / "src"
ESP32_DIR = PROJECT_ROOT / "esp32"
ESP32_FIRMWARE_DIR = ESP32_DIR / "firmware"
ESP32_MODELS_DIR = ESP32_DIR / "models"

DATASETS = {
    "wustl_ehms_2020": {
        "name": "WUSTL-EHMS-2020",
        "path": DATASET_DIR / "WUSTL-EHMS-2020.csv",
        "original_feature_count": 44,
        "selected_feature_count": 16,
    },
    "medsec_25": {
        "name": "MedSec-25",
        "path": DATASET_DIR / "MedSec-25.csv",
        "original_feature_count": 77,
        "selected_feature_count": 27,
    },
    "iomt_ind_2026": {
        "name": "IoMT-IND2026",
        "path": DATASET_DIR / "IoMT-IND2026.csv",
        "original_feature_count": 12,
        "selected_feature_count": 5,
    },
}

RANDOM_SEED = 42
SEQUENCE_LENGTH = 10

GWO = {
    "population_size": 20,
    "max_iterations": 50,
    "alpha_weight": 0.5,
    "beta_weight": 0.3,
    "delta_weight": 0.2,
    "position_min": 0.0,
    "position_max": 1.0,
}

MODEL = {
    "architecture": "CNN-GRU",
    "framework": "tensorflow",
}

EXPLAINABILITY = {
    "method": "SHAP",
}

EMBEDDED_TARGET = {
    "platform": "ESP32-WROOM-32E",
    "firmware_dir": ESP32_FIRMWARE_DIR,
    "model_dir": ESP32_MODELS_DIR,
}

for directory in (DATASET_DIR, SOURCE_DIR, ESP32_FIRMWARE_DIR, ESP32_MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def get_dataset_path(dataset_key):
    if dataset_key not in DATASETS:
        valid_keys = ", ".join(DATASETS)
        raise KeyError(f"Unknown dataset '{dataset_key}'. Available datasets: {valid_keys}")
    return DATASETS[dataset_key]["path"]


def validate_dataset_files():
    missing_files = [
        str(dataset["path"])
        for dataset in DATASETS.values()
        if not dataset["path"].is_file()
    ]
    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(f"The following dataset files were not found:\n{missing}")
    return True
