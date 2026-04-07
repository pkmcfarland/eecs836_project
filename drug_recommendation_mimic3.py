import json
import logging

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.trainer import Trainer

logging.basicConfig(
    filename="drug_recommendation_mimic3.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    with open("config.json") as f:
        config = json.load(f)

    # STEP 1: load data
    logger.info("Loading MIMIC3 dataset from %s", config["MIMIC3"]["data_path"])
    base_dataset = MIMIC3Dataset(
        root=config["MIMIC3"]["data_path"],
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=config["CACHE_DIR"],
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: set task
    logger.info("Setting DrugRecommendationMIMIC3 task")
    task = DrugRecommendationMIMIC3()
    sample_dataset = base_dataset.set_task(task)

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

    # STEP 3: define model
    logger.info("Initializing Transformer model")
    model = Transformer(
        dataset=sample_dataset,
    )

    # STEP 4: define trainer
    trainer = Trainer(
        model=model,
        metrics=["jaccard_samples", "f1_samples", "pr_auc_samples"],
    )

    logger.info("Starting training (epochs=1, monitor=pr_auc_samples)")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="pr_auc_samples",
    )

    # STEP 5: evaluate
    logger.info("Evaluating on test set")
    results = trainer.evaluate(test_dataloader)
    logger.info("Test results: %s", results)