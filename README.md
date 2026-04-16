# EECS 836 Project — MIMIC-III Clinical Prediction Baselines

Baseline training and evaluation scripts for three clinical prediction tasks on the MIMIC-III dataset, built with [PyHealth](https://pyhealth.readthedocs.io/).

## Scripts

All three scripts follow the same structure: they read settings from `config.json`, load and split the dataset once, then loop over all models and training-epoch checkpoints defined in the config, logging validation metrics at each checkpoint and test metrics at the end.

### `drug_recommendation_mimic3.py` — Task 1: Drug Recommendation (Multi-label)

Predicts the set of medications for a given hospital admission using diagnosis and procedure history. Evaluates with sample-wise AUPRC, Precision@10, Precision@20, Recall@10, and Recall@20.

### `mortality_prediction_mimic3.py` — Task 2: In-Hospital Mortality (Binary)

Predicts whether a patient will die during the hospital stay. Evaluates with AUPRC and AUROC.

### `readmission_prediction_mimic3.py` — Task 3: Hospital Readmission (Binary)

Predicts whether a patient will be readmitted within 30 days of discharge. Evaluates with AUPRC and AUROC.

## Configuration

All three scripts read their settings from `config.json` in the working directory. The file has a top-level section for shared paths and one section per task:

```json
{
    "DATA_DIR": "/path/to/mimic-iii-clinical-database-1.4",
    "LOG_DIR": "/path/to/logs",
    "CACHE_DIR": "/path/to/cache",
    "DRUG_RECOMMENDATION": {
        "MODELS": ["RNN", "RETAIN", "Transformer"],
        "TABLES": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        "DEV_MODE": false,
        "CHECK_POINTS": [20, 30, 40, 50, 75, 100],
        "SPLITS": { "TRAIN": 0.8, "VAL": 0.1, "TEST": 0.1 }
    },
    "MORTALITY_PREDICTION": {
        "MODELS": ["RNN", "RETAIN", "Transformer"],
        "TABLES": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        "DEV_MODE": true,
        "CHECK_POINTS": [20, 30, 40, 50, 75, 100],
        "SPLITS": { "TRAIN": 0.8, "VAL": 0.1, "TEST": 0.1 }
    },
    "READMISSION_PREDICTION": {
        "MODELS": ["RNN", "RETAIN", "Transformer"],
        "TABLES": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        "DEV_MODE": true,
        "CHECK_POINTS": [20, 30, 40, 50, 75, 100],
        "SPLITS": { "TRAIN": 0.8, "VAL": 0.1, "TEST": 0.1 }
    }
}
```

### Top-level keys

| Key | Description |
|---|---|
| `DATA_DIR` | Absolute path to the root of the MIMIC-III v1.4 dataset (the directory containing the CSV files). |
| `LOG_DIR` | Absolute path to the directory where timestamped log files are written. |
| `CACHE_DIR` | Absolute path to a directory where PyHealth caches the parsed dataset. Reusing the same cache directory across scripts avoids re-parsing the raw CSVs on every run. |

### Per-task keys (same structure for all three tasks)

| Key | Description |
|---|---|
| `MODELS` | List of models to train. Supported values: `"RNN"`, `"RETAIN"`, `"Transformer"`. |
| `TABLES` | MIMIC-III tables passed to `MIMIC3Dataset`. |
| `DEV_MODE` | If `true`, PyHealth loads only a small subset of the data (useful for development). Set to `false` for full runs. |
| `CHECK_POINTS` | List of epoch counts at which to evaluate on the validation set. Each checkpoint re-trains from scratch and the final checkpoint's model is used for test evaluation. |
| `SPLITS` | Patient-level train/val/test fractions. Must sum to 1.0. |

## Running

Activate the environment that has PyHealth installed, then run any script from the `Python/` directory:

```bash
python drug_recommendation_mimic3.py
python mortality_prediction_mimic3.py
python readmission_prediction_mimic3.py
```

Each script writes a timestamped log file to `LOG_DIR` (e.g., `drug_recommendation_20260416_120000.log`).

## Common Pipeline

All three scripts follow the same pipeline:

1. **Initialize** — Load `config.json`, configure logging to a timestamped file in `LOG_DIR`, set random seed to 42 via `pyhealth.utils.set_seed`.
2. **Load data** — `MIMIC3Dataset` using the tables and paths from the task's config section.
3. **Set task** — Apply the task-specific sample function to the base dataset.
4. **Split** — Patient-level train/val/test split using the fractions in `SPLITS`.
5. **Train and evaluate** — For each model in `MODELS`, iterate over each epoch count in `CHECK_POINTS`, training from scratch each time and logging validation metrics. After the final checkpoint, evaluate on the held-out test set and log the results.

## Data Access

MIMIC-III is credentialed data. Each user must obtain their own access through [PhysioNet](https://physionet.org/content/mimiciii/). Do not commit the dataset or `config.json` (which contains local paths) to a public repository.
