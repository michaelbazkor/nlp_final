"""Run the official MulTaBench evaluator on BIN_TEXT_SEC_MDA_DISTRESS.

Five learners, four conditions, five folds. Fold 0 uses random_state 42.
This file is the training entry point. It is not launched by the dataset build.
Joint TAR uses E5 LoRA at batch size 8.
"""

from __future__ import annotations

import json
import os
import sys
from statistics import mean, pstdev

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WORKSPACE, "MulTaBench"))

from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset
from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.datasets.all_datasets import LocalDatasetID
from multabench.datasets.multimodal import MultimodalState
import torch

from multabench.e5.constants import E5_SMALL_V2

DATASET = LocalDatasetID.BIN_TEXT_SEC_MDA_DISTRESS
MODELS = [LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5]
CONDITIONS = [
    ("structured", MultimodalState.NO_TEXT, False),
    ("text_only", MultimodalState.TEXT_ONLY, False),
    ("joint_frozen", MultimodalState.ALL, False),
    ("joint_tar", MultimodalState.ALL, True),
]
FOLDS = [0, 1, 2, 3, 4]
E5_BATCH_SIZE = 8
RESULT_DIR = os.path.join(WORKSPACE, "results", DATASET.name)


def e5_kwargs() -> dict:
    return {
        "lora_rank": 16,
        "text_layers": 3,
        "learning_rate": 1e-4,
        "epochs": 50,
        "patience": 3,
        "weight_decay": 0.001,
        "batch_size": E5_BATCH_SIZE,
    }


def run_path(model_name: str, condition: str, fold: int) -> str:
    return os.path.join(RESULT_DIR, "json", f"{model_name}_{condition}_fold{fold}.json")


def main() -> None:
    os.makedirs(os.path.join(RESULT_DIR, "json"), exist_ok=True)
    # tabstar's auto picker shells out to grep, which Windows does not provide.
    # E5 LoRA refuses to start unless this variable is set.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    jobs = [
        (model_cls, condition, state, tune_e5, fold)
        for model_cls in MODELS
        for condition, state, tune_e5 in CONDITIONS
        for fold in FOLDS
    ]
    total = len(jobs)
    next_mark = 10
    print(f"progress 0% (0/{total})", flush=True)
    for index, (model_cls, condition, state, tune_e5, fold) in enumerate(jobs, start=1):
        dest = run_path(model_cls.SHORT_NAME, condition, fold)
        if os.path.exists(dest):
            print(f"skip {dest}")
        else:
            summary = evaluate_on_dataset(
                model_cls=model_cls,
                dataset_id=DATASET,
                fold=fold,
                device=device,
                multimodal_state=state,
                tune_e5=tune_e5,
                e5_train_kwargs=e5_kwargs() if tune_e5 else None,
                e5_model_name=E5_SMALL_V2,
            )
            summary["condition"] = condition
            summary["short_name"] = model_cls.SHORT_NAME
            with open(dest, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, default=str, indent=2)
                handle.write("\n")
        percent = 100 * index // total
        if percent >= next_mark:
            print(f"progress {next_mark}% ({index}/{total}) {model_cls.SHORT_NAME} {condition}", flush=True)
            next_mark += 10
    write_report()


def write_report() -> None:
    rows = []
    for model_cls in MODELS:
        for condition, _, _ in CONDITIONS:
            by_fold = {}
            for fold in FOLDS:
                dest = run_path(model_cls.SHORT_NAME, condition, fold)
                if not os.path.exists(dest):
                    continue
                with open(dest, encoding="utf-8") as handle:
                    by_fold[fold] = float(json.load(handle)["test_score"])
            scores = list(by_fold.values())
            row = {
                "model": model_cls.SHORT_NAME,
                "condition": condition,
                "mean_auroc": mean(scores) if scores else None,
                "std_auroc": pstdev(scores) if len(scores) > 1 else 0.0,
                "n_folds": len(scores),
            }
            for fold in FOLDS:
                row[f"fold_{fold}"] = by_fold.get(fold)
            rows.append(row)
    import pandas as pd

    report = os.path.join(RESULT_DIR, "evaluation_report.csv")
    long_frame = pd.DataFrame(rows)
    long_frame.to_csv(report, index=False)
    summary = auroc_summary(long_frame)
    summary_path = os.path.join(RESULT_DIR, "auroc_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {report}")
    print(f"wrote {summary_path}")


def auroc_summary(long_frame):
    """One row per model, plus the average of each column.

    Condition means are rounded to 3 decimals before the deltas, matching MulTaBench.
    delta_joint = joint_frozen - max(structured, text_only)
    delta_awareness = joint_tar - joint_frozen
    """
    import pandas as pd

    conditions = [name for name, _, _ in CONDITIONS]
    wide = (
        long_frame.pivot(index="model", columns="condition", values="mean_auroc")
        .reindex(index=[model.SHORT_NAME for model in MODELS], columns=conditions)
        .round(3)
    )
    wide["delta_joint"] = wide["joint_frozen"] - wide[["structured", "text_only"]].max(axis=1)
    wide["delta_awareness"] = wide["joint_tar"] - wide["joint_frozen"]
    wide[["delta_joint", "delta_awareness"]] = wide[["delta_joint", "delta_awareness"]].round(6)
    average = wide.mean(axis=0)
    average["structured"] = round(float(average["structured"]), 3)
    average["text_only"] = round(float(average["text_only"]), 3)
    average["joint_frozen"] = round(float(average["joint_frozen"]), 3)
    average["joint_tar"] = round(float(average["joint_tar"]), 3)
    average["delta_joint"] = round(float(average["delta_joint"]), 6)
    average["delta_awareness"] = round(float(average["delta_awareness"]), 6)
    summary = pd.concat([wide, average.to_frame("average").T])
    return summary.reset_index(names="model")


if __name__ == "__main__":
    main()
