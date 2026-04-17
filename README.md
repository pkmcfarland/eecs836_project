# EECS 836 Project — MIMIC-III Clinical Prediction Baselines

Baseline training and evaluation scripts for three clinical prediction tasks on the MIMIC-III dataset, built with [PyHealth](https://pyhealth.readthedocs.io/).

## File overview

| File | Purpose |
|---|---|
| [eecs836_project_helpers.py](eecs836_project_helpers.py) | Shared pipeline: `MODEL_DICT`, evaluation helpers, and the four pipeline functions used by all task scripts. |
| [drug_recommendation_mimic3.py](drug_recommendation_mimic3.py) | Task 1 — Drug Recommendation (multi-label). |
| [mortality_prediction_mimic3.py](mortality_prediction_mimic3.py) | Task 2 — In-Hospital Mortality (binary). |
| [readmission_prediction_mimic3.py](readmission_prediction_mimic3.py) | Task 3 — Hospital Readmission (binary). |
| [config.json](config.json) | All paths and hyperparameters. Edit this file; do not edit the scripts. |

## Task scripts

Each task script is responsible only for what is unique to that task: instantiating the correct PyHealth task object, choosing the right metrics, and any task-specific post-processing. Everything else is delegated to the helpers library.

### `drug_recommendation_mimic3.py` — Task 1: Drug Recommendation (Multi-label)

Predicts the set of medications for a given hospital admission using diagnosis and procedure history. Evaluates with sample-wise AUPRC, Precision@10, Precision@20, Recall@10, and Recall@20.

### `mortality_prediction_mimic3.py` — Task 2: In-Hospital Mortality (Binary)

Predicts whether a patient will die during the hospital stay. Evaluates with AUPRC and AUROC.

### `readmission_prediction_mimic3.py` — Task 3: Hospital Readmission (Binary)

Predicts whether a patient will be readmitted within 30 days of discharge. Evaluates with AUPRC and AUROC.

## Helpers library (`eecs836_project_helpers.py`)

Contains everything shared across the three task scripts:

| Symbol | Type | Description |
|---|---|---|
| `MODEL_DICT` | dict | Maps model name strings (`"RNN"`, `"RETAIN"`, `"Transformer"`) to their PyHealth classes. |
| `CheckpointingTrainer` | class | `Trainer` subclass that adds a per-epoch callback mechanism. See below. |
| `precision_at_k(y_true, y_prob, k)` | function | Sample-averaged Precision@k for multi-label prediction. |
| `recall_at_k(y_true, y_prob, k)` | function | Sample-averaged Recall@k for multi-label prediction. |
| `setup_logging(task_name, config)` | function | Creates `LOG_DIR`, configures root logging to a timestamped file, returns a named logger. |
| `load_task_config(config, task_key, logger)` | function | Extracts the task block from config and returns a flat dict of pipeline parameters. |
| `load_and_split_dataset(...)` | function | Loads `MIMIC3Dataset`, applies a task, splits by patient, and returns three dataloaders. |
| `run_training_loop(...)` | function | Trains each model once to the longest checkpoint, capturing validation metrics at intermediate checkpoints via callback. Returns `(results, raw_preds)`. |
| `save_results(results, task_name, log_dir, logger)` | function | Serialises the results dict to a timestamped JSON file in `LOG_DIR`. Converts numpy scalars to plain Python types before writing. |

### `CheckpointingTrainer`

PyHealth's `Trainer` has no callback mechanism, so `CheckpointingTrainer` subclasses it and overrides `train()` to add two extra keyword arguments:

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_epochs` | `set[int]` | 1-based epoch numbers at which to fire the callback. |
| `on_checkpoint` | `Callable[[int, dict], None]` | Called as `on_checkpoint(epoch, val_scores)` immediately after validation completes at each checkpoint epoch. `epoch` is the 1-based count; `val_scores` is the dict returned by `self.evaluate(val_dataloader)`. |

All other parameters and behaviour — optimizer setup, gradient clipping, best-model tracking, early stopping, checkpoint saving — are identical to the parent `Trainer`.

### `run_training_loop`

Each model is trained **once** to `max(CHECK_POINTS)` epochs using a `CheckpointingTrainer`. An `on_checkpoint` closure fires at the end of every epoch whose 1-based count appears in `CHECK_POINTS`, storing that epoch's validation scores without restarting training. After the full run completes, the model is evaluated on the test set.

This replaces the previous approach of re-instantiating a fresh model and trainer for each checkpoint value, which was redundant and discarded all learned weights between checkpoints.

`run_training_loop` returns two values:
- `results` — dict keyed by model name, containing `val_by_epoch` (one entry per checkpoint epoch) and `test` metric dicts.
- `raw_preds` — dict keyed by model name, containing `(y_true, y_prob)` arrays from test-set inference. Used by the drug recommendation script to compute Precision@k and Recall@k; ignored by the binary tasks.

## Configuration (`config.json`)

All paths and hyperparameters live in `config.json`. The file has shared top-level keys and one section per task:

```json
{
    "DATA_DIR": "/path/to/mimic-iii-clinical-database-1.4",
    "LOG_DIR": "/path/to/logs",
    "CACHE_DIR": "/path/to/cache",
    "SEED": 42,
    "DRUG_RECOMMENDATION": {
        "MODELS": ["RNN", "RETAIN", "Transformer"],
        "TABLES": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        "DEV_MODE": false,
        "CHECK_POINTS": [20, 30, 40, 50, 75, 100],
        "SEED": 42,
        "BATCH_SIZE": 32,
        "SPLITS": { "TRAIN": 0.8, "VAL": 0.1, "TEST": 0.1 }
    },
    "MORTALITY_PREDICTION": {
        "MODELS": ["RNN", "RETAIN", "Transformer"],
        "TABLES": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        "DEV_MODE": true,
        "CHECK_POINTS": [20, 30, 40, 50, 75, 100],
        "SEED": 42,
        "BATCH_SIZE": 32,
        "SPLITS": { "TRAIN": 0.8, "VAL": 0.1, "TEST": 0.1 }
    },
    "READMISSION_PREDICTION": {
        "MODELS": ["RNN", "RETAIN", "Transformer"],
        "TABLES": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        "DEV_MODE": true,
        "CHECK_POINTS": [20, 30, 40, 50, 75, 100],
        "SEED": 42,
        "BATCH_SIZE": 32,
        "SPLITS": { "TRAIN": 0.8, "VAL": 0.1, "TEST": 0.1 }
    }
}
```

### Top-level keys

| Key | Description |
|---|---|
| `DATA_DIR` | Absolute path to the root of the MIMIC-III v1.4 dataset (the directory containing the CSV files). |
| `LOG_DIR` | Absolute path to the directory where timestamped log files are written. Created automatically if it does not exist. |
| `CACHE_DIR` | Absolute path to a directory where PyHealth caches the parsed dataset. Reusing the same cache directory across scripts avoids re-parsing the raw CSVs on every run. |
| `SEED` | Top-level seed (reserved; not currently read by the task scripts, which each use their own per-task `SEED`). |

### Per-task keys (same structure for all three tasks)

| Key | Description |
|---|---|
| `MODELS` | List of models to train. Supported values: `"RNN"`, `"RETAIN"`, `"Transformer"`. |
| `TABLES` | MIMIC-III tables passed to `MIMIC3Dataset`. |
| `DEV_MODE` | If `true`, PyHealth loads only a small subset of the data (useful for development). Set to `false` for full runs. |
| `CHECK_POINTS` | List of epoch counts at which to record validation metrics. A single model is trained once to the largest value; validation scores are captured at every listed epoch via callback. The model is then evaluated on the test set. |
| `SEED` | Random seed passed to `pyhealth.utils.set_seed` at the start of the run. |
| `BATCH_SIZE` | Batch size used for all three dataloaders. |
| `SPLITS` | Patient-level train/val/test fractions. Must sum to 1.0. |

## Running

Activate the environment that has PyHealth installed, then run any script from the `Python/` directory:

```bash
python drug_recommendation_mimic3.py
python mortality_prediction_mimic3.py
python readmission_prediction_mimic3.py
```

Each script writes two timestamped output files to `LOG_DIR`:
- A log file (e.g., `drug_recommendation_20260416_120000.log`) with progress and per-checkpoint metrics.
- A JSON results file (e.g., `drug_recommendation_results_20260416_120000.json`) with the full results dict, including all validation snapshots and final test metrics.

## Pipeline

All three scripts follow the same pipeline, implemented in `eecs836_project_helpers.py`:

1. **`setup_logging`** — Create `LOG_DIR`, open a timestamped log file, return a logger.
2. **`load_task_config`** — Read the task's config block; call `set_seed`.
3. **`load_and_split_dataset`** — Load `MIMIC3Dataset`, apply the task, split 80/10/10 by patient, return dataloaders.
4. **`run_training_loop`** — For each model in `MODELS`, create one `CheckpointingTrainer` and train to `max(CHECK_POINTS)` epochs. Validation metrics are captured at every epoch in `CHECK_POINTS` via the `on_checkpoint` callback; no model is re-instantiated between checkpoints. The final model is then evaluated on the test set.
5. **`save_results`** — Serialise the completed results dict to a timestamped JSON file in `LOG_DIR`.

## Data Access

MIMIC-III is credentialed data. Each user must obtain their own access through [PhysioNet](https://physionet.org/content/mimiciii/). Do not commit the dataset or `config.json` (which contains local paths) to a public repository.
