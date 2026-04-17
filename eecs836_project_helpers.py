import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Type

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import RNN, RETAIN, Transformer
from pyhealth.trainer import Trainer, is_best

### CONSTANTS ###
MODEL_DICT = {
    "RNN": RNN,
    "RETAIN": RETAIN,
    "Transformer": Transformer,
}


### TRAINER SUBCLASS ###

class CheckpointingTrainer(Trainer):
    """Extends Trainer with a per-epoch callback fired at specified checkpoints.

    Identical to the parent in every respect except that train() accepts two
    extra keyword arguments:

        checkpoint_epochs : set[int]
            1-based epoch numbers at which to fire the callback.
        on_checkpoint : Callable[[int, dict], None]
            Called as on_checkpoint(epoch, val_scores) immediately after
            validation completes at each checkpoint epoch. The epoch argument
            is the 1-based epoch count; val_scores is the dict returned by
            self.evaluate(val_dataloader).
    """

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        steps_per_epoch: Optional[int] = None,
        evaluation_steps: int = 1,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        patience=None,
        checkpoint_epochs: Optional[Set[int]] = None,
        on_checkpoint: Optional[Callable[[int, dict], None]] = None,
    ):
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # --- identical to Trainer.train() from here --- #

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.bias", "LayerNorm.weight"])
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.bias", "LayerNorm.weight"])
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0
        patience_counter = 0

        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                output = self.model(**data)
                loss = output["loss"]
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1

            if self.exp_path is not None:
                self.save_ckpt(
                    __import__("os").path.join(self.exp_path, "last.ckpt")
                )

            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)

                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        best_score = score
                        patience_counter = 0
                        if self.exp_path is not None:
                            self.save_ckpt(
                                __import__("os").path.join(self.exp_path, "best.ckpt")
                            )
                    else:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            break

                # fire the callback at checkpoint epochs (1-based)
                if on_checkpoint is not None and checkpoint_epochs is not None:
                    if (epoch + 1) in checkpoint_epochs:
                        on_checkpoint(epoch + 1, scores)

        if load_best_model_at_last and self.exp_path is not None:
            best_path = __import__("os").path.join(self.exp_path, "best.ckpt")
            if __import__("os").path.isfile(best_path):
                self.load_ckpt(best_path)

        if test_dataloader is not None:
            self.evaluate(test_dataloader)


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
    """Train each model once to the longest checkpoint, capturing validation
    metrics at every intermediate checkpoint via a callback, then evaluate on
    the test set.

    A single CheckpointingTrainer is created per model and trained for
    max(check_points) epochs. The on_checkpoint callback fires at the end of
    each epoch whose 1-based count appears in check_points, storing that
    epoch's validation scores without restarting training.

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
    checkpoint_epochs = set(check_points)

    for model_name in models:
        results[model_name] = {"val_by_epoch": {}, "test": {}}
        logger.info("Running model: %s", model_name)

        try:
            model = MODEL_DICT[model_name](dataset=sample_dataset)
        except KeyError:
            logger.error("Model %s not found in MODEL_DICT", model_name)
            raise

        def on_checkpoint(epoch, scores, model_name=model_name):
            results[model_name]["val_by_epoch"][epoch] = scores
            logger.info("Validation results at epoch %d: %s", epoch, scores)

        trainer = CheckpointingTrainer(model=model, metrics=metrics)
        logger.info("Training %s for %d epochs", model_name, max(check_points))
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            epochs=max(check_points),
            monitor=monitor,
            checkpoint_epochs=checkpoint_epochs,
            on_checkpoint=on_checkpoint,
        )

        y_true, y_prob, _ = trainer.inference(test_dl)
        results[model_name]["test"] = trainer.evaluate(test_dl)
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
