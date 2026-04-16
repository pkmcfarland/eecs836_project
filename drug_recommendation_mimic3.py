import json

from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.utils import set_seed

from eecs836_project_helpers import (
    setup_logging, load_task_config, load_and_split_dataset,
    run_training_loop, precision_at_k, recall_at_k,
)

if __name__ == "__main__":

    with open("config.json") as f:
        config = json.load(f)

    logger = setup_logging("drug_recommendation", config)
    cfg = load_task_config(config, "DRUG_RECOMMENDATION", logger)
    set_seed(cfg["random_seed"])

    task = DrugRecommendationMIMIC3()
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

    drug_vocab_size = len(sample_dataset[0]["drugs"])
    logger.info("Drug vocabulary size (N): %d", drug_vocab_size)

    results, raw_preds = run_training_loop(
        models=cfg["models"],
        check_points=cfg["check_points"],
        sample_dataset=sample_dataset,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        metrics=["pr_auc_samples"],
        monitor="pr_auc_samples",
        logger=logger,
    )

    # add Precision@k and Recall@k to test results (required by assignment)
    for model_name, (y_true, y_prob) in raw_preds.items():
        results[model_name]["test"]["precision@10"] = precision_at_k(y_true, y_prob, k=10)
        results[model_name]["test"]["precision@20"] = precision_at_k(y_true, y_prob, k=20)
        results[model_name]["test"]["recall@10"]    = recall_at_k(y_true, y_prob, k=10)
        results[model_name]["test"]["recall@20"]    = recall_at_k(y_true, y_prob, k=20)
        logger.info("Final test results for %s: %s", model_name, results[model_name]["test"])
