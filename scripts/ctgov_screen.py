"""Fold 0 screen for the ClinicalTrials.gov pilot.

Waits until the strategy-3 fold-4 process exits, then scores five learners
under the four standard conditions. One fold, one split seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
OUT_DIR = WORKSPACE / "results" / "BIN_TEXT_CTGOV_TRIAL_OUTCOME"
FOLD = 0
POLISHED_JOINT = 0.013
POLISHED_AWARENESS = 0.003
CONDITIONS = [
    ("structured", "no_text", False),
    ("text_only", "text_only", False),
    ("joint_frozen", "all", False),
    ("joint_tar", "all", True),
]


def log(message: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(message, flush=True)
    with (OUT_DIR / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def fold4_running() -> bool:
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" "
        "| Select-Object -ExpandProperty CommandLine"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "could not list python processes")
    return "fold4_both.py" in completed.stdout


def wait_for_fold4() -> None:
    while fold4_running():
        log("waiting for fold4_both.py to exit")
        time.sleep(30)
    log("fold4_both.py is not running")
    time.sleep(15)


def e5_kwargs() -> dict:
    return {
        "lora_rank": 16,
        "text_layers": 3,
        "learning_rate": 1e-4,
        "epochs": 50,
        "patience": 3,
        "weight_decay": 0.001,
        "batch_size": 8,
    }


def gaps_from_scores(scores: dict[str, dict[str, float]], model_names: list[str]) -> list[dict]:
    rows = []
    for model in model_names:
        rounded = {name: round(scores[model][name], 3) for name, _, _ in CONDITIONS}
        row = {"model": model, **rounded}
        row["delta_joint"] = round(
            rounded["joint_frozen"] - max(rounded["structured"], rounded["text_only"]),
            6,
        )
        row["delta_awareness"] = round(rounded["joint_tar"] - rounded["joint_frozen"], 6)
        rows.append(row)
    average = {"model": "average"}
    for column in ["structured", "text_only", "joint_frozen", "joint_tar"]:
        average[column] = round(sum(row[column] for row in rows) / len(rows), 3)
    average["delta_joint"] = round(sum(row["delta_joint"] for row in rows) / len(rows), 6)
    average["delta_awareness"] = round(
        sum(row["delta_awareness"] for row in rows) / len(rows),
        6,
    )
    rows.append(average)
    return rows


def run() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sys.path.insert(0, str(WORKSPACE / "MulTaBench"))
    from dotenv import load_dotenv

    load_dotenv(WORKSPACE / "MulTaBench" / ".env")

    import torch

    from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset
    from multabench.baselines.catboost import CatBoost
    from multabench.baselines.lgbm import LightGBM
    from multabench.baselines.tabm import TabM
    from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
    from multabench.datasets.all_datasets import LocalDatasetID
    from multabench.datasets.multimodal import MultimodalState
    from multabench.e5.constants import E5_SMALL_V2

    states = {
        "no_text": MultimodalState.NO_TEXT,
        "text_only": MultimodalState.TEXT_ONLY,
        "all": MultimodalState.ALL,
    }
    models = [LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5]
    dataset_id = LocalDatasetID.BIN_TEXT_CTGOV_TRIAL_OUTCOME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_dir = OUT_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (model_cls, condition, states[state_name], tune)
        for model_cls in models
        for condition, state_name, tune in CONDITIONS
    ]
    log(f"device {device} fold {FOLD} jobs {len(jobs)}")
    for index, (model_cls, condition, state, tune) in enumerate(jobs, start=1):
        dest = json_dir / f"{model_cls.SHORT_NAME}_{condition}_fold{FOLD}.json"
        if not dest.exists():
            summary = evaluate_on_dataset(
                model_cls=model_cls,
                dataset_id=dataset_id,
                fold=FOLD,
                device=device,
                multimodal_state=state,
                tune_e5=tune,
                e5_train_kwargs=e5_kwargs() if tune else None,
                e5_model_name=E5_SMALL_V2,
            )
            summary["condition"] = condition
            summary["short_name"] = model_cls.SHORT_NAME
            with dest.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, default=str, indent=2)
                handle.write("\n")
        log(f"progress {index}/{len(jobs)} {model_cls.SHORT_NAME} {condition}")
    scores: dict[str, dict[str, float]] = {}
    for model in models:
        scores[model.SHORT_NAME] = {}
        for condition, _, _ in CONDITIONS:
            path = json_dir / f"{model.SHORT_NAME}_{condition}_fold{FOLD}.json"
            with path.open(encoding="utf-8") as handle:
                scores[model.SHORT_NAME][condition] = float(json.load(handle)["test_score"])
    rows = gaps_from_scores(scores, [model.SHORT_NAME for model in models])
    import pandas as pd

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "gap_summary.csv", index=False)
    average = summary.loc[summary["model"] == "average"].iloc[0]
    log(summary.to_string(index=False))
    log(
        f"raw delta_joint {average['delta_joint']:.4f} polished {POLISHED_JOINT:.3f}; "
        f"raw delta_awareness {average['delta_awareness']:.4f} polished {POLISHED_AWARENESS:.3f}"
    )


def main() -> None:
    wait_for_fold4()
    run()


if __name__ == "__main__":
    main()
