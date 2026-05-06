import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import RNN, RETAIN, Transformer
from pyhealth.tasks import DrugRecommendationMIMIC3, MortalityPredictionMIMIC3
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


### FOCAL LOSS ###

class BinaryFocalLoss(nn.Module):
    """Binary focal loss matching F.binary_cross_entropy_with_logits's call signature.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where alpha_t = alpha if y == 1 else (1 - alpha), and p_t is the predicted
    probability of the true class. Operates on raw logits so it can be dropped
    in wherever F.binary_cross_entropy_with_logits is used.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float().view_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


class MortalityPredictionFocalMIMIC3(MortalityPredictionMIMIC3):
    """Mortality prediction task variant intended to be trained with focal loss.

    Sample generation is identical to MortalityPredictionMIMIC3 (which processes
    each patient independently with no cross-patient state, so subclassing
    introduces no data leakage). The actual loss substitution happens at
    training time in run_focal_training_loop, since PyHealth selects the loss
    inside the model based on output_schema rather than in the task.
    """

    task_name: str = "MortalityPredictionFocalMIMIC3"


class MultilabelFocalLoss(nn.Module):
    """Multilabel focal loss matching F.binary_cross_entropy_with_logits's call
    signature on (B, L) logits/targets.

    For each (sample, label) cell:
        FL = -alpha_l_t * (1 - p_t)^gamma * log(p_t)
    where p_t = sigmoid(logit) if y == 1 else 1 - sigmoid(logit), and alpha_l_t
    is alpha for positives and (1 - alpha) for negatives. alpha may be a scalar
    (uniform over labels) or a 1D tensor of length L (per-label positive
    weight). Final reduction is mean over all (B, L) cells, matching how the
    PyHealth model's default multilabel BCE reduces.
    """

    def __init__(self, alpha: "float | torch.Tensor" = 0.25, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        if isinstance(alpha, torch.Tensor):
            self.alpha_vec: Optional[torch.Tensor] = alpha.float().detach()
            self.alpha_scalar: Optional[float] = None
        else:
            self.alpha_vec = None
            self.alpha_scalar = float(alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        if self.alpha_vec is not None:
            a = self.alpha_vec.to(logits.device)  # (L,)
            alpha_t = a * target + (1.0 - a) * (1.0 - target)
        else:
            a_s: float = self.alpha_scalar  # type: ignore[assignment]
            alpha_t = a_s * target + (1.0 - a_s) * (1.0 - target)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


class DrugRecommendationFocalMIMIC3(DrugRecommendationMIMIC3):
    """Drug recommendation task variant intended to be trained with focal loss.

    Sample generation is identical to DrugRecommendationMIMIC3 (per-patient
    processing, no cross-patient state, so subclassing introduces no data
    leakage). The actual loss substitution happens at training time in
    run_multilabel_focal_training_loop, since PyHealth selects the loss inside
    the model based on output_schema rather than in the task.
    """

    task_name: str = "DrugRecommendationFocalMIMIC3"


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
        patience, splits, models, batch_size, random_seed.
    """
    try:
        task_cfg = config[task_key]
        return {
            "data_path":   config["DATA_DIR"],
            "cache_dir":   config["CACHE_DIR"],
            "tables":      task_cfg["TABLES"],
            "check_points": task_cfg["CHECK_POINTS"],
            "dev_mode":    task_cfg["DEV_MODE"],
            "patience":    task_cfg["PATIENCE"],
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
                      metrics, monitor, logger, patience=None):
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
        patience:       Early-stopping patience in epochs. None disables it.

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
        logger.info(
            "Training %s for %d epochs (patience=%s)",
            model_name, max(check_points), patience,
        )
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            epochs=max(check_points),
            monitor=monitor,
            patience=patience,
            checkpoint_epochs=checkpoint_epochs,
            on_checkpoint=on_checkpoint,
        )

        y_true, y_prob, _ = trainer.inference(test_dl)
        results[model_name]["test"] = trainer.evaluate(test_dl)
        raw_preds[model_name] = (y_true, y_prob)
        logger.info("Test results: %s", results[model_name])

    return results, raw_preds


def compute_positive_rate(train_dl, label_key, logger):
    """Compute the positive-class rate from the TRAINING dataloader only.

    Used to derive the focal-loss alpha. Iterating only train_dl (never val/test)
    is what keeps focal-loss class balancing free of label leakage from the
    held-out splits.
    """
    pos = 0
    total = 0
    for batch in train_dl:
        labels = batch[label_key]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        flat = labels.float().view(-1)
        total += int(flat.numel())
        pos += int(flat.sum().item())
    rate = (pos / total) if total > 0 else 0.5
    logger.info(
        "Train positive rate: %.4f (%d / %d)", rate, pos, total
    )
    return rate


def run_focal_training_loop(models, check_points, sample_dataset,
                            train_dl, val_dl, test_dl,
                            metrics, monitor, label_key, logger,
                            gamma=2.0, alpha=None, patience=None):
    """Same as run_training_loop, but trains with BinaryFocalLoss instead of
    the model's default binary_cross_entropy_with_logits.

    The loss swap is performed by overriding model.get_loss_function on the
    instance after construction, so the model's existing forward path
    (`self.get_loss_function()(logits, y_true)`) picks up focal loss without
    any further changes.

    Args:
        label_key: Output schema key (e.g. "mortality"). Used to look up labels
                   in batches when computing the positive-class rate.
        gamma:     Focusing parameter for focal loss. Default 2.0.
        alpha:     Positive-class weight in [0, 1]. If None, alpha is set to
                   (1 - train_positive_rate) so the rare class is up-weighted.
                   train_positive_rate is computed only from train_dl to avoid
                   leaking val/test labels into the loss.
        patience:  Early-stopping patience in epochs. None disables it.
    """
    if alpha is None:
        pos_rate = compute_positive_rate(train_dl, label_key, logger)
        alpha = 1.0 - pos_rate
        logger.info(
            "Focal loss alpha auto-set to %.4f (1 - train positive rate)", alpha
        )
    logger.info("Focal loss params: alpha=%.4f, gamma=%.2f", alpha, gamma)

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

        focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        # Override the loss selector on this model instance only.
        model.get_loss_function = lambda focal=focal: focal

        def on_checkpoint(epoch, scores, model_name=model_name):
            results[model_name]["val_by_epoch"][epoch] = scores
            logger.info("Validation results at epoch %d: %s", epoch, scores)

        trainer = CheckpointingTrainer(model=model, metrics=metrics)
        logger.info(
            "Training %s for %d epochs (patience=%s)",
            model_name, max(check_points), patience,
        )
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            epochs=max(check_points),
            monitor=monitor,
            patience=patience,
            checkpoint_epochs=checkpoint_epochs,
            on_checkpoint=on_checkpoint,
        )

        y_true, y_prob, _ = trainer.inference(test_dl)
        results[model_name]["test"] = trainer.evaluate(test_dl)
        raw_preds[model_name] = (y_true, y_prob)
        logger.info("Test results: %s", results[model_name])

    return results, raw_preds


def compute_label_positive_rates(train_dl, label_key, logger):
    """Per-label positive rates from the TRAINING dataloader only.

    Returns a 1D tensor of length L (label vocabulary size) where entry l is
    the fraction of train samples in which label l is positive. Used to derive
    a per-label alpha for MultilabelFocalLoss. Iterating only train_dl (never
    val/test) is what keeps focal-loss class balancing free of label leakage
    from the held-out splits. Output is clamped to [eps, 1 - eps] so derived
    alpha values stay strictly inside (0, 1).
    """
    eps = 1e-6
    pos = None
    total = 0
    for batch in train_dl:
        labels = batch[label_key]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.float()
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        batch_pos = labels.sum(dim=0)  # (L,)
        pos = batch_pos if pos is None else pos + batch_pos
        total += int(labels.shape[0])

    if pos is None or total == 0:
        raise ValueError("compute_label_positive_rates: train dataloader was empty")

    rates = (pos / total).clamp(min=eps, max=1.0 - eps)
    logger.info(
        "Train per-label positive rates over %d samples, %d labels: "
        "mean=%.4f, min=%.4f, max=%.4f",
        total, rates.numel(),
        float(rates.mean()), float(rates.min()), float(rates.max()),
    )
    return rates


def run_multilabel_focal_training_loop(models, check_points, sample_dataset,
                                        train_dl, val_dl, test_dl,
                                        metrics, monitor, label_key, logger,
                                        gamma=2.0, alpha=None, patience=None):
    """Same as run_training_loop, but trains with MultilabelFocalLoss instead of
    the model's default per-label binary_cross_entropy_with_logits.

    The loss swap is performed by overriding model.get_loss_function on the
    instance after construction, so the model's existing forward path
    (`self.get_loss_function()(logits, y_true)`) picks up focal loss without
    any further changes.

    Args:
        label_key: Output schema key (e.g. "drugs"). Used to look up labels in
                   batches when computing per-label positive rates.
        gamma:     Focusing parameter for focal loss. Default 2.0.
        alpha:     Either a float, a 1D torch.Tensor of length L, or None.
                   If None, alpha is set to (1 - train_positive_rate_per_label),
                   so rare drugs are up-weighted. The per-label rate is computed
                   only from train_dl to avoid leaking val/test labels into the
                   loss.
        patience:  Early-stopping patience in epochs. None disables it.
    """
    if alpha is None:
        rates = compute_label_positive_rates(train_dl, label_key, logger)
        alpha = 1.0 - rates  # (L,) per-label positive weight
        logger.info(
            "Multilabel focal alpha auto-set per label: mean=%.4f, min=%.4f, max=%.4f",
            float(alpha.mean()), float(alpha.min()), float(alpha.max()),
        )
    elif isinstance(alpha, torch.Tensor):
        logger.info(
            "Multilabel focal alpha (provided tensor): mean=%.4f, min=%.4f, max=%.4f",
            float(alpha.mean()), float(alpha.min()), float(alpha.max()),
        )
    else:
        logger.info("Multilabel focal alpha (scalar): %.4f", float(alpha))
    logger.info("Multilabel focal gamma=%.2f", gamma)

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

        focal = MultilabelFocalLoss(alpha=alpha, gamma=gamma)
        # Override the loss selector on this model instance only.
        model.get_loss_function = lambda focal=focal: focal

        def on_checkpoint(epoch, scores, model_name=model_name):
            results[model_name]["val_by_epoch"][epoch] = scores
            logger.info("Validation results at epoch %d: %s", epoch, scores)

        trainer = CheckpointingTrainer(model=model, metrics=metrics)
        logger.info(
            "Training %s for %d epochs (patience=%s)",
            model_name, max(check_points), patience,
        )
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            epochs=max(check_points),
            monitor=monitor,
            patience=patience,
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
