import json

from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.utils import set_seed

from eecs836_project_helpers import (
    setup_logging, load_task_config, load_and_split_dataset, run_training_loop,
)

if __name__ == "__main__":

    with open("config.json") as f:
        config = json.load(f)

    logger = setup_logging("mortality_prediction", config)
    cfg = load_task_config(config, "MORTALITY_PREDICTION", logger)
    set_seed(cfg["random_seed"])

    task = MortalityPredictionMIMIC3()
    sample_dataset, train_dl, val_dl, test_dl = load_and_split_dataset(
        data_path=cfg["data_path"],
        tables=cfg["tables"],
        cache_dir=cfg["cache_dir"],
        dev_mode=cfg["dev_mode"],
        task=task,
        splits=cfg["splits"],
        batch_size=cfg["batch_size"],
        logger=logger,
    )

    results, _ = run_training_loop(
        models=cfg["models"],
        check_points=cfg["check_points"],
        sample_dataset=sample_dataset,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        metrics=["pr_auc", "roc_auc"],
        monitor="pr_auc",
        logger=logger,
    )
