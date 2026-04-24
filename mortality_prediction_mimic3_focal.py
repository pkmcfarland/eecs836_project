import json

from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.utils import set_seed
import torch

from eecs836_project_helpers_customCost import (
    setup_logging, load_task_config, load_and_split_dataset, run_training_loop, save_results,
)

from pyhealth.tasks import BaseTask

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
        partitions = patient.event_type_partitions

        def get_rows(name):
            # PyHealth stores keys as single-element tuples: ('admissions',)
            data = partitions.get((name,), None)
            if data is None or (hasattr(data, "is_empty") and data.is_empty()):
                return []
            return data.to_dicts()

        admissions = get_rows("admissions")
        all_diagnoses  = get_rows("diagnoses_icd")
        all_procedures = get_rows("procedures_icd")
        all_drugs      = get_rows("prescriptions")

        # if admissions:
        #     print(f"[DEBUG] Admission row keys: {list(admissions[0].keys())}")
        # if all_diagnoses:
        #     print(f"[DEBUG] Diagnosis row keys: {list(all_diagnoses[0].keys())}")
        # if admissions:
        #     print(f"[DEBUG] First row keys: {list(admissions[0].keys())}")
        #     print(f"[DEBUG] Sample v_id: {admissions[0].get('admissions/hadm_id')}")
        #     print(f"[DEBUG] Sample label: {admissions[0].get('admissions/hospital_expire_flag')}")
        #     print(f"[DEBUG] Sample diag: {admissions[0].get('diagnoses_icd/icd9_code')}")

        if not admissions:
            return []
        
         # Group rows by visit so we can collect all codes per admission
        from collections import defaultdict
        diag_by_visit  = defaultdict(list)
        proc_by_visit  = defaultdict(list)
        drug_by_visit  = defaultdict(list)

        for row in all_diagnoses:
            hid  = row.get("diagnoses_icd/hadm_id") or row.get("hadm_id")
            code = row.get("diagnoses_icd/icd9_code") or row.get("icd9_code")
            if hid and code:
                diag_by_visit[str(hid)].append(str(code))

        for row in all_procedures:
            hid  = row.get("procedures_icd/hadm_id") or row.get("hadm_id")
            code = row.get("procedures_icd/icd9_code") or row.get("icd9_code")
            if hid and code:
                proc_by_visit[str(hid)].append(str(code))

        for row in all_drugs:
            hid  = row.get("prescriptions/hadm_id") or row.get("hadm_id")
            drug = row.get("prescriptions/drug") or row.get("drug")
            if hid and drug:
                drug_by_visit[str(hid)].append(str(drug))

        for adm in admissions:
            v_id  = adm.get("admissions/hadm_id")
            label = adm.get("admissions/hospital_expire_flag")

            try:
                label = int(label)
            except (TypeError, ValueError):
                continue
            if label not in [0, 1]:
                continue

            v_id_str   = str(v_id)
            conditions = diag_by_visit[v_id_str]
            procedures = proc_by_visit[v_id_str]
            drugs      = drug_by_visit[v_id_str]

            if len(conditions) == 0 or len(procedures) == 0 or len(drugs) == 0:
                continue

            # # If we want more samples by included patients that had a missing field, we could do this, but it will introduce noise.
            # conditions = diag_by_visit[v_id_str] or ["<PAD>"]
            # procedures = proc_by_visit[v_id_str] or ["<PAD>"]
            # drugs      = drug_by_visit[v_id_str] or ["<PAD>"]

            # if len(conditions) + len(procedures) + len(drugs) == 0:
            #     continue

            samples.append({
                "patient_id": patient.patient_id,
                "visit_id":   v_id_str,
                "conditions": conditions,
                "procedures": procedures,
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
