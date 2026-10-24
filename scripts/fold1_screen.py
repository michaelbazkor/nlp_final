"""Score the closest fold-0 tables on fold 1.

Fold 0 of the capex-dropped table was an easy text split. These variants keep
the same 1,000 rows. The comparison is the mean of fold 0 and fold 1, rounded
to 3 decimals before the gaps.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean

import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "MulTaBench"))

from gap_search import (  # noqa: E402
    clip_debt,
    drop_fiscal_year,
    signed_log_skewed,
    winsor_5_95,
)
from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset  # noqa: E402
from multabench.baselines.catboost import CatBoost  # noqa: E402
from multabench.baselines.lgbm import LightGBM  # noqa: E402
from multabench.baselines.tabm import TabM  # noqa: E402
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5  # noqa: E402
from multabench.datasets.all_datasets import LocalDatasetID  # noqa: E402
from multabench.datasets.multimodal import MultimodalState  # noqa: E402
from multabench.e5.constants import E5_SMALL_V2  # noqa: E402

DATA_PATH = WORKSPACE / "data" / "BIN_TEXT_SEC_MDA_DISTRESS" / "data.csv"
BASELINE_PATH = WORKSPACE / "results" / "gap_search" / "best_data.csv"
SEARCH_DIR = WORKSPACE / "results" / "gap_search"
OUT_DIR = SEARCH_DIR / "fold1_screen"
DATASET = LocalDatasetID.BIN_TEXT_SEC_MDA_DISTRESS
MODELS = [LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5]
CONDITIONS = [
    ("structured", MultimodalState.NO_TEXT, False),
    ("text_only", MultimodalState.TEXT_ONLY, False),
    ("joint_frozen", MultimodalState.ALL, False),
    ("joint_tar", MultimodalState.ALL, True),
]
FOLD = 1


def floor_assets(frame: pd.DataFrame) -> pd.DataFrame:
    updated = frame.copy()
    cut = float(updated["log_total_assets"].quantile(0.20))
    updated["log_total_assets"] = updated["log_total_assets"].clip(lower=cut)
    return updated


VARIANTS = [
    ("winsor_5_95", winsor_5_95),
    ("drop_fiscal_year", drop_fiscal_year),
    ("signed_log_skewed", signed_log_skewed),
    ("floor_assets", floor_assets),
    ("clip_debt_0.80", clip_debt),
]


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


def log(message: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(message, flush=True)
    with (OUT_DIR / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def score_file(directory: Path, model: str, condition: str, fold: int) -> float:
    path = directory / "json" / f"{model}_{condition}_fold{fold}.json"
    with path.open(encoding="utf-8") as handle:
        return float(json.load(handle)["test_score"])


def gaps_from_scores(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model in [item.SHORT_NAME for item in MODELS]:
        rounded = {name: round(scores[model][name], 3) for name, _, _ in CONDITIONS}
        row = {"model": model, **rounded}
        row["delta_joint"] = round(rounded["joint_frozen"] - max(rounded["structured"], rounded["text_only"]), 6)
        row["delta_awareness"] = round(rounded["joint_tar"] - rounded["joint_frozen"], 6)
        rows.append(row)
    frame = pd.DataFrame(rows)
    average = {"model": "average"}
    for column in ["structured", "text_only", "joint_frozen", "joint_tar"]:
        average[column] = round(float(frame[column].mean()), 3)
    average["delta_joint"] = round(float(frame["delta_joint"].mean()), 6)
    average["delta_awareness"] = round(float(frame["delta_awareness"].mean()), 6)
    return pd.concat([frame, pd.DataFrame([average])], ignore_index=True)


def load_fold_scores(directory: Path, fold: int) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for model in MODELS:
        scores[model.SHORT_NAME] = {}
        for condition, _, _ in CONDITIONS:
            scores[model.SHORT_NAME][condition] = score_file(directory, model.SHORT_NAME, condition, fold)
    return scores


def two_fold_mean(fold0: dict, fold1: dict) -> pd.DataFrame:
    merged = {}
    for model, conditions in fold0.items():
        merged[model] = {
            condition: mean([conditions[condition], fold1[model][condition]])
            for condition, _, _ in CONDITIONS
        }
    return gaps_from_scores(merged)


def run_variant(name: str, frame: pd.DataFrame, device) -> None:
    dest_dir = OUT_DIR / name
    json_dir = dest_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    jobs = [
        (model_cls, condition, state, tune)
        for model_cls in MODELS
        for condition, state, tune in CONDITIONS
    ]
    for index, (model_cls, condition, state, tune) in enumerate(jobs, start=1):
        dest = json_dir / f"{model_cls.SHORT_NAME}_{condition}_fold{FOLD}.json"
        if not dest.exists():
            summary = evaluate_on_dataset(
                model_cls=model_cls,
                dataset_id=DATASET,
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
        log(f"{name} {index}/{len(jobs)} {model_cls.SHORT_NAME} {condition}")


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = pd.read_csv(BASELINE_PATH)
    log(f"device {device} variants {len(VARIANTS)} fold {FOLD}")
    try:
        for name, transform in VARIANTS:
            log(f"start {name}")
            run_variant(name, transform(baseline), device)
            fold0 = load_fold_scores(SEARCH_DIR / name, 0)
            fold1 = load_fold_scores(OUT_DIR / name, 1)
            summary = two_fold_mean(fold0, fold1)
            summary.to_csv(OUT_DIR / name / "two_fold_summary.csv", index=False)
            average = summary.loc[summary["model"] == "average"].iloc[0]
            log(
                f"{name} two-fold delta_joint {average['delta_joint']:.4f} "
                f"delta_awareness {average['delta_awareness']:.4f} "
                f"sum {average['delta_joint'] + average['delta_awareness']:.4f}"
            )
    finally:
        baseline.to_csv(DATA_PATH, index=False)
        log("restored capex-dropped table")


if __name__ == "__main__":
    main()
