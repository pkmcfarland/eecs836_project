import json

from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.utils import set_seed
import torch

from eecs836_project_helpers_customCost import (
    setup_logging, load_task_config, load_and_split_dataset, run_training_loop, save_results,
)

from pyhealth.tasks import BaseTask
import pandas as pd
from datetime import timedelta

class InHospitalMortalityTask(BaseTask):
    task_name = "in_hospital_mortality"
    input_schema = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs":      "sequence",
    }
    # output_schema = {"mortality": "label"}
    output_schema = {"mortality": "binary"}

    def __call__(self, patient):
        samples = []

        visits = patient.get_events(event_type="admissions")
        if not visits:
            return []

        for visit in visits:
            # --- Label ---
            label = visit.hospital_expire_flag
            try:
                label = int(label)
            except (TypeError, ValueError):
                continue
            if label not in [0, 1]:
                continue

            hadm_id = visit.hadm_id

            # --- Diagnoses (discharge-level, no time filter available) ---
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", hadm_id)]
            )
            conditions = [e.icd9_code for e in diagnoses if e.icd9_code]

            # --- Procedures: filter to those charted within 48h of admission ---
            # admit_time = visit.admittime  # datetime object
            admit_time = visit.timestamp  # PyHealth maps admittime -> timestamp
            cutoff = admit_time + timedelta(hours=48)

            all_procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", hadm_id)]
            )
            procedures_list = [
                e.icd9_code for e in all_procedures
                if e.icd9_code
                and e.timestamp is not None
                and e.timestamp <= cutoff
            ]

            # --- Drugs: filter to those started within 48h of admission ---
            all_prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", hadm_id)]
            )
            drugs = [
                e.drug for e in all_prescriptions
                if e.drug
                and e.timestamp is not None
                and e.timestamp <= cutoff
            ]

            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            samples.append({
                "patient_id": patient.patient_id,
                "visit_id":   str(hadm_id),
                "conditions": conditions,
                "procedures": procedures_list,
                "drugs":      drugs,
                "mortality":  label,
            })

        return samples

if __name__ == "__main__":

    with open("config4.json") as f:
        config = json.load(f)

    logger = setup_logging("mortality_prediction", config)
    cfg = load_task_config(config, "MORTALITY_PREDICTION", logger)
    set_seed(cfg["random_seed"])

    # Use the custom function instead of the class
    task = InHospitalMortalityTask()

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

    print("Sample size:", len(test_dl.dataset))

    print(f"TOTAL SAMPLES GENERATED: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        # print(f"SAMPLE 0 DATA: {sample_dataset.samples[0]}")
        print(f"SAMPLE 0 DATA: {sample_dataset[0]}")

    # Calculate weights: Mortality is rare, so we weight it higher.
    # [Weight for class 0 (lives), Weight for class 1 (dies)]
    weights = torch.tensor([1.0, 5.0])

    results, _ = run_training_loop(
        models=cfg["models"],
        check_points=cfg["check_points"],
        sample_dataset=sample_dataset,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        metrics=["pr_auc", "roc_auc"],
        monitor="pr_auc",
        loss_type="focal", # "weighted_ce" or "focal"
        pos_weight=weights,
        logger=logger,
    )

    save_results(results, "mortality_prediction", config["LOG_DIR"], logger)
