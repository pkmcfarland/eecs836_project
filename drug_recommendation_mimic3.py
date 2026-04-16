import json
import logging
from pathlib import Path
import numpy as np
import datetime

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN, RETAIN, Transformer
from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

### CONSTANTS ###
MODEL_DICT = {
        "RNN": RNN,
        "RETAIN": RETAIN,
        "Transformer": Transformer
    }

### FUNCTIONS FOR EVALUATION ###
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

if __name__ == "__main__":

    # INITIALIZE: read in configs, setup logging, set random seed, 
    #             and set up paths
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open("config.json") as f:
        config = json.load(f)
    try:
        logging.basicConfig(
        filename= Path(config["LOG_DIR"]) / f"drug_recommendation_{now}.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        )
        logger = logging.getLogger(__name__)
    except KeyError as e:
        print(f"Key not found in config: {e}")
        raise
    try:
        data_path = config["DATA_DIR"]
        tables = config["DRUG_RECOMMENDATION"]["TABLES"]
        cache_dir = config["CACHE_DIR"]
        check_points = config["DRUG_RECOMMENDATION"]["CHECK_POINTS"]
        dev_mode = config["DRUG_RECOMMENDATION"]["DEV_MODE"]
        splits = [config["DRUG_RECOMMENDATION"]["SPLITS"]["TRAIN"],
                  config["DRUG_RECOMMENDATION"]["SPLITS"]["VAL"],
                    config["DRUG_RECOMMENDATION"]["SPLITS"]["TEST"]]
        models = config["DRUG_RECOMMENDATION"]["MODELS"]
    except KeyError as e:
        logger.error("Key not found in config: %s", e)
        raise
    set_seed(42)

    # LOAD DATA (do once for all three models)
    logger.info("Loading MIMIC3 dataset from %s", data_path)
    base_dataset = MIMIC3Dataset(
        root=data_path,
        tables=tables,
        cache_dir=cache_dir,
        dev=dev_mode,
    )
    base_dataset.stats()

    # SET TASK and SPLIT DATA (do once for all three models)
    logger.info("Setting DrugRecommendationMIMIC3 task")
    task = DrugRecommendationMIMIC3()
    sample_dataset = base_dataset.set_task(task)
    drug_vocab_size = len(sample_dataset[0]['drugs'])
    logger.info("Drug vocabulary size (N): %d", drug_vocab_size)

    logger.info("Splitting dataset by patient %s", splits)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, splits
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)
  
    results = {}
    for model_name in models:
        results[model_name] = {"val_by_epoch": {}, "test": {}}
        logger.info("Running model: %s", model_name)        

        for i, epoch in enumerate(check_points):
            logger.info("Training %s for epoch %d", model_name, epoch)

            try:
                model = MODEL_DICT[model_name](dataset=sample_dataset)
            except KeyError:
                logger.error("Model %s not found in MODEL_DICT", model_name)
                raise

            trainer = Trainer(
                model=model,
                metrics=["pr_auc_samples"]
            )
            trainer.train(
                train_dataloader = train_dataloader,
                val_dataloader = val_dataloader,
                epochs = epoch, 
                monitor = "pr_auc_samples",
            )
            results[model_name]["val_by_epoch"][epoch] = trainer.evaluate(
                val_dataloader)
            logger.info("Validation results at epoch %d: %s", 
                        epoch, results[model_name]["val_by_epoch"][epoch])
            
            # keep the last model/trainer available for test evaluation
            # but delete previous ones to save memory
            if i < len(check_points) - 1:
                del model
                del trainer

        # after training, get raw predictions
        y_true, y_prob, loss = trainer.inference(test_dataloader)     #type: ignore 
        logger.info("Evaluating on test set")
        results[model_name]["test"] = trainer.evaluate(test_dataloader)  #type: ignore

        # compute for k=10 and k=20
        results[model_name]["test"]["precision@10"] = precision_at_k(y_true, y_prob, k=10)
        results[model_name]["test"]["precision@20"] = precision_at_k(y_true, y_prob, k=20)
        results[model_name]["test"]["recall@10"] = recall_at_k(y_true, y_prob, k=10)
        results[model_name]["test"]["recall@20"] = recall_at_k(y_true, y_prob, k=20)

        logger.info("Test results: %s", results[model_name])