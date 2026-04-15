import json
import logging
from pathlib import Path
import numpy as np

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN, RETAIN, Transformer
from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

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

    with open("config.json") as f:
        config = json.load(f)

    set_seed(42)

    logging.basicConfig(
        filename= Path(config["LOG_DIR"]) / "drug_recommendation_mimic3.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # STEP 1: load data (do once for all three models)
    logger.info("Loading MIMIC3 dataset from %s", config["MIMIC3"]["data_path"])
    base_dataset = MIMIC3Dataset(
        root=config["MIMIC3"]["data_path"],
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=config["CACHE_DIR"],
        dev=True, #TODO: set to False for full dataset
    )
    base_dataset.stats()

    # STEP 2: set task (do once for all three models)
    logger.info("Setting DrugRecommendationMIMIC3 task")
    task = DrugRecommendationMIMIC3()
    sample_dataset = base_dataset.set_task(task)
    drug_vocab_size = len(sample_dataset[0]['drugs'])
    logger.info("Drug vocabulary size (N): %d", drug_vocab_size)

    logger.info("Splitting dataset by patient [0.8, 0.1, 0.1]")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 3: define models to run
    model_dict = {
        "RNN": {"model": RNN(dataset=sample_dataset),
                "epochs": 1 #TODO: #2 increase number (20-30?) for full training
        },
        "RETAIN": {"model": RETAIN(dataset=sample_dataset),
                   "epochs": 1
        },
        "Transformer": {"model": Transformer(dataset=sample_dataset),
                        "epochs": 1
        }
    }
    results = {}
    for model_name, model_config in model_dict.items():
        logger.info("Initializing %s model", model_name)
        model = model_config["model"]
        epochs = model_config["epochs"]

        # STEP 4: define trainer
        trainer = Trainer(
            model=model,
            metrics=["pr_auc_samples"]
        )

        logger.info("Starting training (epochs=%d, monitor=pr_auc_samples)", 
                    epochs)
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs, 
            monitor="pr_auc_samples",
        )

        # after training, get raw predictions
        y_true, y_prob, loss = trainer.inference(test_dataloader)

        

        # STEP 5: evaluate
        logger.info("Evaluating on test set")
        results[model_name] = trainer.evaluate(test_dataloader)

        # compute for k=10 and k=20
        results[model_name]["precision@10"] = precision_at_k(y_true, y_prob, k=10)
        results[model_name]["precision@20"] = precision_at_k(y_true, y_prob, k=20)
        results[model_name]["recall@10"] = recall_at_k(y_true, y_prob, k=10)
        results[model_name]["recall@20"] = recall_at_k(y_true, y_prob, k=20)

        logger.info("Test results: %s", results[model_name])

        # free memory before next model
        del model
        del trainer