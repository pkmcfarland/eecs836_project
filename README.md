# EECS 836 Project — MIMIC-III Clinical Prediction Baselines

Baseline and focal-loss training/evaluation scripts for three clinical prediction tasks on the MIMIC-III dataset, built with [PyHealth](https://pyhealth.readthedocs.io/).

## File overview

| File | Purpose |
|---|---|
| [eecs836_project_helpers.py](eecs836_project_helpers.py) | Shared library: `MODEL_DICT`, `CheckpointingTrainer`, focal-loss classes, focal-aware task subclasses, evaluation helpers, and the pipeline functions used by all task scripts. |
| [drug_recommendation_mimic3.py](drug_recommendation_mimic3.py) | Task 1 — Drug Recommendation (multi-label), default BCE loss. |
| [mortality_prediction_mimic3.py](mortality_prediction_mimic3.py) | Task 2 — In-Hospital Mortality (binary), default BCE loss. |
| [readmission_prediction_mimic3.py](readmission_prediction_mimic3.py) | Task 3 — Hospital Readmission (binary), default BCE loss. |
| [drug_recommendation_focal_mimic3.py](drug_recommendation_focal_mimic3.py) | Task 1 variant trained with multilabel focal loss. |
| [mortality_prediction_focal_mimic3.py](mortality_prediction_focal_mimic3.py) | Task 2 variant trained with binary focal loss. |
| [readmission_prediction_focal_mimic3.py](readmission_prediction_focal_mimic3.py) | Task 3 variant trained with binary focal loss. |
| [config.json](config.json) | All paths and hyperparameters. Edit this file; do not edit the scripts. Not committed. |
| [config_template.json](config_template.json) | Reference template showing every required key. Copy to `config.json` and fill in local paths. |

## Task scripts

Each task script is responsible only for what is unique to that task: instantiating the correct PyHealth task object, choosing the right metrics, and any task-specific post-processing. Everything else is delegated to the helpers library.

The three baseline scripts (`*_mimic3.py`) train with PyHealth's default loss (binary cross-entropy with logits). The three focal-loss scripts (`*_focal_mimic3.py`) are otherwise identical pipelines, but swap the model's loss for `BinaryFocalLoss` or `MultilabelFocalLoss` at training time.

### `drug_recommendation_mimic3.py` / `drug_recommendation_focal_mimic3.py` — Task 1: Drug Recommendation (Multi-label)

Predicts the set of medications for a given hospital admission using diagnosis and procedure history. Evaluates with sample-wise AUPRC, Precision@10, Precision@20, Recall@10, and Recall@20. The focal variant uses `DrugRecommendationFocalMIMIC3` (a subclass of the standard task) and `run_multilabel_focal_training_loop` with per-label alpha derived from training-set positive rates.

### `mortality_prediction_mimic3.py` / `mortality_prediction_focal_mimic3.py` — Task 2: In-Hospital Mortality (Binary)

Predicts whether a patient will die during the hospital stay. Evaluates with AUPRC and AUROC. The focal variant uses `MortalityPredictionFocalMIMIC3` and `run_focal_training_loop` with a scalar alpha (default `0.25`).

### `readmission_prediction_mimic3.py` / `readmission_prediction_focal_mimic3.py` — Task 3: Hospital Readmission (Binary)

Predicts whether a patient will be readmitted within 30 days of discharge. Evaluates with AUPRC and AUROC. The focal variant calls the standard `ReadmissionPredictionMIMIC3` task and runs `run_focal_training_loop` with a scalar alpha (default `0.25`).

## Helpers library (`eecs836_project_helpers.py`)

Contains everything shared across the six task scripts.

### Constants

| Symbol | Type | Description |
|---|---|---|
| `MODEL_DICT` | dict | Maps model name strings (`"RNN"`, `"RETAIN"`, `"Transformer"`) to their PyHealth classes. |

### Trainer

| Symbol | Type | Description |
|---|---|---|
| `CheckpointingTrainer` | class | `Trainer` subclass that adds a per-epoch callback mechanism. See below. |

### Focal-loss classes and task subclasses

| Symbol | Type | Description |
|---|---|---|
| `BinaryFocalLoss(alpha=0.25, gamma=2.0)` | `nn.Module` | Binary focal loss `FL = -alpha_t * (1 - p_t)^gamma * log(p_t)`. Operates on raw logits and matches `F.binary_cross_entropy_with_logits`'s call signature so it drops in wherever PyHealth uses BCE-with-logits. |
| `MultilabelFocalLoss(alpha=0.25, gamma=2.0)` | `nn.Module` | Per-cell focal loss on `(B, L)` logits/targets. `alpha` may be a scalar (uniform across labels) or a 1D tensor of length L (per-label positive weight). Reduces by mean over all `(B, L)` cells, matching PyHealth's default multilabel BCE reduction. |
| `MortalityPredictionFocalMIMIC3` | class | Subclass of `MortalityPredictionMIMIC3` with `task_name = "MortalityPredictionFocalMIMIC3"`. Sample generation is identical to the parent (per-patient processing, no cross-patient state, so subclassing introduces no data leakage). The actual loss substitution happens in `run_focal_training_loop`. |
| `DrugRecommendationFocalMIMIC3` | class | Subclass of `DrugRecommendationMIMIC3` with `task_name = "DrugRecommendationFocalMIMIC3"`. Same rationale as the mortality variant; loss is swapped in `run_multilabel_focal_training_loop`. |

### Evaluation helpers

| Symbol | Type | Description |
|---|---|---|
| `precision_at_k(y_true, y_prob, k)` | function | Sample-averaged Precision@k for multi-label prediction. |
| `recall_at_k(y_true, y_prob, k)` | function | Sample-averaged Recall@k for multi-label prediction. |

### Pipeline helpers

| Symbol | Type | Description |
|---|---|---|
| `setup_logging(task_name, config)` | function | Creates `LOG_DIR`, configures root logging to a timestamped file, returns a named logger. |
| `load_task_config(config, task_key, logger)` | function | Extracts the task block from config and returns a flat dict of pipeline parameters. |
| `load_and_split_dataset(...)` | function | Loads `MIMIC3Dataset`, applies a task, splits by patient, and returns three dataloaders. |
| `run_training_loop(...)` | function | Trains each model once to the longest checkpoint with the model's default loss, capturing validation metrics at intermediate checkpoints via callback. Returns `(results, raw_preds)`. |
| `compute_positive_rate(train_dl, label_key, logger)` | function | Iterates the **training** dataloader only and returns the binary positive-class rate. Used to derive the focal-loss `alpha` without leaking val/test labels. |
| `run_focal_training_loop(...)` | function | Same shape as `run_training_loop`, but trains with `BinaryFocalLoss`. If `alpha` is `None`, sets `alpha = 1 - train_positive_rate` so the rare class is up-weighted. |
| `compute_label_positive_rates(train_dl, label_key, logger)` | function | Per-label positive rates from the **training** dataloader only. Returns a 1D tensor of length L (label vocabulary size), clamped to `[eps, 1 - eps]`. Used to derive a per-label `alpha` for `MultilabelFocalLoss`. |
| `run_multilabel_focal_training_loop(...)` | function | Same shape as `run_training_loop`, but trains with `MultilabelFocalLoss`. If `alpha` is `None`, sets `alpha = 1 - train_positive_rate_per_label`. Accepts either a scalar, a 1D tensor of length L, or `None`. |
| `save_results(results, task_name, log_dir, logger)` | function | Serialises the results dict to a timestamped JSON file in `LOG_DIR`. Converts numpy scalars to plain Python types before writing. |

### `CheckpointingTrainer`

PyHealth's `Trainer` has no callback mechanism, so `CheckpointingTrainer` subclasses it and overrides `train()` to add two extra keyword arguments:

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_epochs` | `set[int]` | 1-based epoch numbers at which to fire the callback. |
| `on_checkpoint` | `Callable[[int, dict], None]` | Called as `on_checkpoint(epoch, val_scores)` immediately after validation completes at each checkpoint epoch. `epoch` is the 1-based count; `val_scores` is the dict returned by `self.evaluate(val_dataloader)`. |

All other parameters and behaviour — optimizer setup, gradient clipping, best-model tracking, early stopping, checkpoint saving — are identical to the parent `Trainer`.

### Training loops

`run_training_loop`, `run_focal_training_loop`, and `run_multilabel_focal_training_loop` share the same structure. For each model in `MODELS`:

1. Instantiate the model from `MODEL_DICT`.
2. (Focal variants only) Construct a `BinaryFocalLoss` or `MultilabelFocalLoss` and override `model.get_loss_function` on the instance so PyHealth's existing forward path (`self.get_loss_function()(logits, y_true)`) picks up focal loss without any changes to the model class.
3. Create a `CheckpointingTrainer` and train **once** to `max(CHECK_POINTS)` epochs. The `on_checkpoint` callback fires at the end of every epoch whose 1-based count appears in `CHECK_POINTS`, recording validation scores without restarting training.
4. After training, run inference on the test set and store `(y_true, y_prob)` plus the test metrics.

Each of these functions returns:
- `results` — dict keyed by model name, containing `val_by_epoch` (one entry per checkpoint epoch) and `test` metric dicts.
- `raw_preds` — dict keyed by model name, containing `(y_true, y_prob)` arrays from test-set inference. Used by the drug-recommendation scripts to compute Precision@k and Recall@k; ignored by the binary tasks.

## Configuration (`config.json`)

All paths and hyperparameters live in `config.json`. Use [config_template.json](config_template.json) as a starting point — copy it to `config.json` and fill in your local paths. The same config is consumed by both the baseline and focal-loss script for each task (e.g. `mortality_prediction_mimic3.py` and `mortality_prediction_focal_mimic3.py` both read the `MORTALITY_PREDICTION` block).

```json
{
    "DATA_DIR": "/path/to/mimic/data",
    "LOG_DIR": "/path/to/mimic/logs",
    "CACHE_DIR": "/path/to/mimic/cache",
    "DRUG_RECOMMENDATION": {
        "MODELS": ["RNN", "RETAIN"],
        "TABLES": ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        "DEV_MODE": false,
        "PATIENCE": 20,
        "CHECK_POINTS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        "SEED": 42,
        "BATCH_SIZE": 32,
        "SPLITS": { "TRAIN": 0.8, "VAL": 0.1, "TEST": 0.1 }
    },
    "MORTALITY_PREDICTION": { "...same shape...": "..." },
    "READMISSION_PREDICTION": { "...same shape...": "..." }
}
```

### Top-level keys

| Key | Description |
|---|---|
| `DATA_DIR` | Absolute path to the root of the MIMIC-III v1.4 dataset (the directory containing the CSV files). |
| `LOG_DIR` | Absolute path to the directory where timestamped log files and result JSONs are written. Created automatically if it does not exist. |
| `CACHE_DIR` | Absolute path to a directory where PyHealth caches the parsed dataset. Reusing the same cache directory across scripts avoids re-parsing the raw CSVs on every run. |

### Per-task keys (same structure for all three tasks)

| Key | Description |
|---|---|
| `MODELS` | List of models to train. Supported values: `"RNN"`, `"RETAIN"`, `"Transformer"`. |
| `TABLES` | MIMIC-III tables passed to `MIMIC3Dataset`. |
| `DEV_MODE` | If `true`, PyHealth loads only a small subset of the data (useful for development). Set to `false` for full runs. |
| `PATIENCE` | Early-stopping patience in epochs. Passed through to `CheckpointingTrainer.train`. |
| `CHECK_POINTS` | List of epoch counts at which to record validation metrics. A single model is trained once to the largest value; validation scores are captured at every listed epoch via callback. The model is then evaluated on the test set. |
| `SEED` | Random seed passed to `pyhealth.utils.set_seed` at the start of the run. |
| `BATCH_SIZE` | Batch size used for all three dataloaders. |
| `SPLITS` | Patient-level train/val/test fractions. Must sum to 1.0. |

## Running

Activate the environment that has PyHealth installed, then run any script from the `Python/` directory. Baseline scripts:

```bash
python drug_recommendation_mimic3.py
python mortality_prediction_mimic3.py
python readmission_prediction_mimic3.py
```

Focal-loss scripts:

```bash
python drug_recommendation_focal_mimic3.py
python mortality_prediction_focal_mimic3.py
python readmission_prediction_focal_mimic3.py
```

Each script writes two timestamped output files to `LOG_DIR`:
- A log file (e.g., `drug_recommendation_20260506_120000.log` or `drug_recommendation_focal_20260506_120000.log`) with progress and per-checkpoint metrics.
- A JSON results file (e.g., `drug_recommendation_results_20260506_120000.json`) with the full results dict, including all validation snapshots and final test metrics.

The focal scripts use distinct task-name stems (`drug_recommendation_focal`, `mortality_prediction_focal`, `readmission_prediction_focal`), so their output files do not collide with the baseline runs in the same `LOG_DIR`.

## Pipeline

All six scripts follow the same pipeline, implemented in `eecs836_project_helpers.py`:

1. **`setup_logging`** — Create `LOG_DIR`, open a timestamped log file, return a logger.
2. **`load_task_config`** — Read the task's config block; call `set_seed`.
3. **`load_and_split_dataset`** — Load `MIMIC3Dataset`, apply the task, split by patient using `SPLITS`, return dataloaders.
4. **Training loop** — `run_training_loop` (baseline), `run_focal_training_loop` (binary focal), or `run_multilabel_focal_training_loop` (multilabel focal). For each model in `MODELS`, create one `CheckpointingTrainer` and train to `max(CHECK_POINTS)` epochs. Validation metrics are captured at every epoch in `CHECK_POINTS` via the `on_checkpoint` callback; no model is re-instantiated between checkpoints. The final model is then evaluated on the test set.
5. **`save_results`** — Serialise the completed results dict to a timestamped JSON file in `LOG_DIR`.

## Data Access

MIMIC-III is credentialed data. Each user must obtain their own access through [PhysioNet](https://physionet.org/content/mimiciii/). Do not commit the dataset or `config.json` (which contains local paths) to a public repository. `config_template.json` is safe to commit and exists for that purpose.
