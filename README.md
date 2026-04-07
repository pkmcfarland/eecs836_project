# EECS 836 Project — MIMIC-III Clinical Prediction Baselines

Baseline training and evaluation scripts for three clinical prediction tasks on the MIMIC-III dataset, built with [PyHealth](https://pyhealth.readthedocs.io/).

## Scripts

### `drug_recommendation_mimic3.py` — Task 1: Drug Recommendation (Multi-label)

Predicts the set of medications for a given hospital admission using diagnosis and procedure history. Uses a **Transformer** model and evaluates with sample-wise Jaccard, F1, and AUPRC.

### `mortality_prediction_mimic3.py` — Task 2: In-Hospital Mortality (Binary)

Predicts whether a patient will die during the next hospital stay based on clinical codes from the current visit. Uses an **RNN** (GRU) model and evaluates with AUPRC and AUROC.

### `readmission_prediction_mimic3.py` — Task 3: Hospital Readmission (Binary)

Predicts whether a patient will be readmitted within 30 days of discharge. Uses an **RNN** (GRU) model and evaluates with AUPRC and AUROC.

## Configuration

All three scripts read their settings from a `config.json` file in the same directory. Create it with the following structure:

```json
{
    "MIMIC3": {
        "data_path": "/path/to/mimic-iii-clinical-database-1.4"
    },
    "CACHE_DIR": "/path/to/cache/directory"
}
```

| Key | Description |
|---|---|
| `MIMIC3.data_path` | Absolute path to the root of the MIMIC-III v1.4 dataset (the directory containing the CSV files). |
| `CACH_DIR` | Absolute path to a directory where PyHealth will cache the parsed dataset. Reusing the same cache directory across scripts avoids re-parsing the raw CSVs on every run. |

## Running

Activate the environment that has PyHealth installed, then run any script from this directory:

```bash
python drug_recommendation_mimic3.py
python mortality_prediction_mimic3.py
python readmission_prediction_mimic3.py
```

Each script logs progress and final test metrics to a corresponding `.log` file (e.g., `drug_recommendation_mimic3.log`).

## Common Pipeline

All three scripts follow the same five-step pipeline:

1. **Load data** — `MIMIC3Dataset` with tables `DIAGNOSES_ICD`, `PROCEDURES_ICD`, `PRESCRIPTIONS` (dev mode enabled).
2. **Set task** — Apply the task-specific sample function to the base dataset.
3. **Split** — 80/10/10 patient-level train/val/test split.
4. **Train** — 1 epoch, batch size 32, monitored on the primary metric.
5. **Evaluate** — Report metrics on the held-out test set.

## Data Access

MIMIC-III is credentialed data. Each user must obtain their own access through [PhysioNet](https://physionet.org/content/mimiciii/). Do not commit the dataset or `config.json` (which contains local paths) to a public repository.
