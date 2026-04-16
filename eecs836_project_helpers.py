import datetime
import json
import logging
from pathlib import Path

import numpy as np

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import RNN, RETAIN, Transformer
from pyhealth.trainer import Trainer

### CONSTANTS ###
MODEL_DICT = {
    "RNN": RNN,
    "RETAIN": RETAIN,
    "Transformer": Transformer,
}


### EVALUATION HELPERS ###

def precision_at_k(y_true, y_prob, k):
    """For each sample, take top-k predicted drugs and compute precision."""
    precisions = []
    for true, prob in zip(y_true, y_prob):
        top_k_indices = np.argsort(prob)[-k:]
        hits = sum(true[i] == 1 for i in top_k_indices)
        precisions.append(hits / k)
    return np.mean(precisions)


def recall_at_k(y_true, y_prob, k):
    """For each sample, take top-k predicted drugs and compute recall."""
    recalls = []
    for true, prob in zip(y_true, y_prob):
        top_k_indices = np.argsort(prob)[-k:]
        hits = sum(true[i] == 1 for i in top_k_indices)
        total_positive = sum(true)
        if total_positive > 0:
            recalls.append(hits / total_positive)
    return np.mean(recalls)


### PIPELINE HELPERS ###

def setup_logging(task_name, config):
    """Create the log directory, configure root logging, and return a named logger.

    Args:
        task_name: Used as the log filename stem and the logger name
                   (e.g. "drug_recommendation").
        config:    The parsed config.json dict. Must contain "LOG_DIR".

    Returns:
        A logging.Logger instance.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        Path(config["LOG_DIR"]).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=Path(config["LOG_DIR"]) / f"{task_name}_{now}.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        return logging.getLogger(task_name)
    except KeyError as e:
        print(f"Key not found in config: {e}")
        raise


def load_task_config(config, task_key, logger):
    """Extract task-specific settings from config and return them as a dict.

    Args:
        config:   The parsed config.json dict.
        task_key: Top-level key for the task block, e.g. "DRUG_RECOMMENDATION".
        logger:   Logger to use for error messages.

    Returns:
        Dict with keys: data_path, cache_dir, tables, check_points, dev_mode,
        splits, models, batch_size, random_seed.
    """
    try:
        task_cfg = config[task_key]
        return {
            "data_path":   config["DATA_DIR"],
            "cache_dir":   config["CACHE_DIR"],
            "tables":      task_cfg["TABLES"],
            "check_points": task_cfg["CHECK_POINTS"],
            "dev_mode":    task_cfg["DEV_MODE"],
            "splits": [
                task_cfg["SPLITS"]["TRAIN"],
                task_cfg["SPLITS"]["VAL"],
                task_cfg["SPLITS"]["TEST"],
            ],
            "models":      task_cfg["MODELS"],
            "batch_size":  task_cfg["BATCH_SIZE"],
            "random_seed": task_cfg["SEED"],
        }
    except KeyError as e:
        logger.error("Key not found in config: %s", e)
        raise


def load_and_split_dataset(data_path, tables, cache_dir, dev_mode,
                           task, splits, batch_size, logger):
    """Load MIMIC-III, apply a task, split by patient, and return dataloaders.

    Args:
        data_path:  Root directory of the MIMIC-III CSV files.
        tables:     List of MIMIC-III table names to load.
        cache_dir:  Directory for PyHealth's parsed-dataset cache.
        dev_mode:   If True, load only a small subset of the data.
        task:       An instantiated PyHealth task object.
        splits:     [train_frac, val_frac, test_frac] — must sum to 1.0.
        batch_size: Batch size for all three dataloaders.
        logger:     Logger for progress messages.

    Returns:
        (sample_dataset, train_dataloader, val_dataloader, test_dataloader)
    """
    logger.info("Loading MIMIC3 dataset from %s", data_path)
    base_dataset = MIMIC3Dataset(
        root=data_path,
        tables=tables,
        cache_dir=cache_dir,
        dev=dev_mode,
    )
    base_dataset.stats()

    logger.info("Setting task: %s", type(task).__name__)
    sample_dataset = base_dataset.set_task(task)

    logger.info("Splitting dataset by patient %s", splits)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, splits
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    train_dl = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl   = get_dataloader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_dl  = get_dataloader(test_dataset,  batch_size=batch_size, shuffle=False)

    return sample_dataset, train_dl, val_dl, test_dl


def run_training_loop(models, check_points, sample_dataset,
                      train_dl, val_dl, test_dl,
                      metrics, monitor, logger):
    """Train each model at each checkpoint, then evaluate on the test set.

    For each model, each value in check_points is treated as a total epoch
    count: a fresh model is trained from scratch for that many epochs. This
    produces one validation snapshot per checkpoint. The model trained to the
    final checkpoint is then evaluated on the test set.

    Args:
        models:         List of model name strings (must be keys in MODEL_DICT).
        check_points:   List of epoch counts, e.g. [20, 30, 50, 100].
        sample_dataset: The PyHealth SampleDataset (used to initialise models).
        train_dl:       Training dataloader.
        val_dl:         Validation dataloader.
        test_dl:        Test dataloader.
        metrics:        List of PyHealth metric name strings passed to Trainer.
        monitor:        Metric name to monitor for best-model selection during
                        training.
        logger:         Logger for progress and result messages.

    Returns:
        results:   Dict keyed by model name, each containing
                   {"val_by_epoch": {epoch: metrics_dict}, "test": metrics_dict}.
        raw_preds: Dict keyed by model name, each containing
                   (y_true, y_prob) arrays from inference on the test set.
                   Useful for computing additional metrics (e.g. Precision@k).
    """
    results = {}
    raw_preds = {}

    for model_name in models:
        results[model_name] = {"val_by_epoch": {}, "test": {}}
        logger.info("Running model: %s", model_name)

        for i, epoch in enumerate(check_points):
            logger.info("Training %s for %d epochs", model_name, epoch)

            try:
                model = MODEL_DICT[model_name](dataset=sample_dataset)
            except KeyError:
                logger.error("Model %s not found in MODEL_DICT", model_name)
                raise

            trainer = Trainer(model=model, metrics=metrics)
            trainer.train(
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                epochs=epoch,
                monitor=monitor,
            )
            results[model_name]["val_by_epoch"][epoch] = trainer.evaluate(val_dl)
            logger.info(
                "Validation results at epoch %d: %s",
                epoch, results[model_name]["val_by_epoch"][epoch],
            )

            # keep only the last trainer/model to save memory
            if i < len(check_points) - 1:
                del model, trainer

        y_true, y_prob, _ = trainer.inference(test_dl)  # type: ignore
        results[model_name]["test"] = trainer.evaluate(test_dl)  # type: ignore
        raw_preds[model_name] = (y_true, y_prob)
        logger.info("Test results: %s", results[model_name])

    return results, raw_preds


def save_results(results, task_name, log_dir, logger):
    """Serialise the results dict to a timestamped JSON file in log_dir.

    Args:
        results:   The results dict returned by run_training_loop (optionally
                   augmented with additional metrics such as Precision@k).
        task_name: Used as the filename stem, e.g. "drug_recommendation".
        log_dir:   Directory to write the file into (will be created if absent).
        logger:    Logger for the confirmation message.

    Returns:
        The Path of the written file.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}_results_{now}.json"

    # json.dump requires plain Python types; convert numpy scalars to float
    def _to_serialisable(obj):
        if isinstance(obj, dict):
            return {str(k): _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serialisable(v) for v in obj]
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(_to_serialisable(results), f, indent=2)

    logger.info("Results written to %s", out_path)
    return out_path
