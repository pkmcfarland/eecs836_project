import json
from datetime import timedelta

from pyhealth.tasks import ReadmissionPredictionMIMIC3
from pyhealth.utils import set_seed

from eecs836_project_helpers import (
    setup_logging, load_task_config, load_and_split_dataset,
    run_focal_training_loop, save_results,
)

if __name__ == "__main__":

    with open("config.json") as f:
        config = json.load(f)

    logger = setup_logging("readmission_prediction_focal", config)
    cfg = load_task_config(config, "READMISSION_PREDICTION", logger)
    set_seed(cfg["random_seed"])

    task = ReadmissionPredictionMIMIC3(window=timedelta(days=30))
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

    results, _ = run_focal_training_loop(
        models=cfg["models"],
        check_points=cfg["check_points"],
        sample_dataset=sample_dataset,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        metrics=["pr_auc", "roc_auc"],
        monitor="pr_auc",
        label_key="readmission",
        logger=logger,
        alpha=0.25,
        gamma=2.0,
        patience=cfg["patience"],
    )

    save_results(results, "readmission_prediction_focal", config["LOG_DIR"], logger)
