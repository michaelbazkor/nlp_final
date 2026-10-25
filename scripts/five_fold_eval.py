"""Official five-fold evaluation of one prepared distress table.

The table replaces data.csv only while this process runs. Completed jobs are
skipped, so a frozen run can be followed by the TAR jobs. Set SKIP_TAR=1 to
score structured, text-only, and joint frozen only.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from statistics import mean

import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "MulTaBench"))

from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset  # noqa: E402
from multabench.baselines.catboost import CatBoost  # noqa: E402
from multabench.baselines.lgbm import LightGBM  # noqa: E402
from multabench.baselines.tabm import TabM  # noqa: E402
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5  # noqa: E402
from multabench.datasets.all_datasets import LocalDatasetID  # noqa: E402
from multabench.datasets.multimodal import MultimodalState  # noqa: E402
from multabench.e5.constants import E5_SMALL_V2  # noqa: E402

DATA_PATH = WORKSPACE / "data" / "BIN_TEXT_SEC_MDA_DISTRESS" / "data.csv"
DATASET = LocalDatasetID.BIN_TEXT_SEC_MDA_DISTRESS
MODELS = [LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5]
CONDITIONS = [
    ("structured", MultimodalState.NO_TEXT, False),
    ("text_only", MultimodalState.TEXT_ONLY, False),
    ("joint_frozen", MultimodalState.ALL, False),
    ("joint_tar", MultimodalState.ALL, True),
]
FOLDS = [0, 1, 2, 3, 4]


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


def log(out_dir: Path, message: str) -> None:
    print(message, flush=True)
    with (out_dir / "eval_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def gaps_from_means(
    scores: dict[str, dict[str, float]],
    conditions: list[str],
    models: list,
) -> pd.DataFrame:
    rows = []
    for model in [item.SHORT_NAME for item in models]:
        rounded = {name: round(scores[model][name], 3) for name in conditions}
        row = {"model": model, **rounded}
        row["delta_joint"] = round(
            rounded["joint_frozen"] - max(rounded["structured"], rounded["text_only"]),
            6,
        )
        if "joint_tar" in rounded:
            row["delta_awareness"] = round(rounded["joint_tar"] - rounded["joint_frozen"], 6)
        rows.append(row)
    frame = pd.DataFrame(rows)
    average = {"model": "average"}
    for column in conditions:
        average[column] = round(float(frame[column].mean()), 3)
    average["delta_joint"] = round(float(frame["delta_joint"].mean()), 6)
    if "delta_awareness" in frame.columns:
        average["delta_awareness"] = round(float(frame["delta_awareness"].mean()), 6)
    return pd.concat([frame, pd.DataFrame([average])], ignore_index=True)


def main() -> None:
    source = Path(os.environ["EVAL_TABLE"]).resolve()
    out_dir = Path(os.environ["EVAL_OUT"]).resolve()
    skip_tar = os.environ.get("SKIP_TAR", "0") == "1"
    selected = {name.strip() for name in os.environ.get("EVAL_MODELS", "").split(",") if name.strip()}
    models = [model for model in MODELS if not selected or model.SHORT_NAME in selected]
    if selected and {model.SHORT_NAME for model in models} != selected:
        known = ", ".join(model.SHORT_NAME for model in MODELS)
        raise SystemExit(f"unknown EVAL_MODELS {sorted(selected)}; known: {known}")
    conditions = [item for item in CONDITIONS if not (skip_tar and item[0] == "joint_tar")]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(exist_ok=True)
    backup = out_dir / "data_csv_backup.csv"
    if not backup.exists():
        shutil.copy2(DATA_PATH, backup)
    table = pd.read_csv(source)
    table.to_csv(DATA_PATH, index=False)
    jobs = [
        (model_cls, condition, state, tune, fold)
        for model_cls in models
        for condition, state, tune in conditions
        for fold in FOLDS
    ]
    log(out_dir, f"device {device} table {source.name} jobs {len(jobs)} skip_tar {skip_tar}")
    try:
        for index, (model_cls, condition, state, tune, fold) in enumerate(jobs, start=1):
            dest = out_dir / "json" / f"{model_cls.SHORT_NAME}_{condition}_fold{fold}.json"
            if not dest.exists():
                summary = evaluate_on_dataset(
                    model_cls=model_cls,
                    dataset_id=DATASET,
                    fold=fold,
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
            if index % 5 == 0 or index == len(jobs):
                log(out_dir, f"progress {index}/{len(jobs)} {model_cls.SHORT_NAME} {condition} fold {fold}")
        present = [name for name, _, _ in conditions]
        scores: dict[str, dict[str, float]] = {}
        for model in models:
            scores[model.SHORT_NAME] = {}
            for condition in present:
                fold_scores = []
                for fold in FOLDS:
                    path = out_dir / "json" / f"{model.SHORT_NAME}_{condition}_fold{fold}.json"
                    with path.open(encoding="utf-8") as handle:
                        fold_scores.append(float(json.load(handle)["test_score"]))
                scores[model.SHORT_NAME][condition] = mean(fold_scores)
        summary = gaps_from_means(scores, present, models)
        name = "frozen_summary.csv" if skip_tar else "five_fold_summary.csv"
        summary.to_csv(out_dir / name, index=False)
        average = summary.loc[summary["model"] == "average"].iloc[0]
        message = f"delta_joint {average['delta_joint']:.4f}"
        if "delta_awareness" in average:
            message += (
                f" delta_awareness {average['delta_awareness']:.4f} "
                f"sum {average['delta_joint'] + average['delta_awareness']:.4f}"
            )
        log(out_dir, summary.to_string(index=False))
        log(out_dir, message)
    finally:
        shutil.copy2(backup, DATA_PATH)
        log(out_dir, "restored data.csv")


if __name__ == "__main__":
    main()
