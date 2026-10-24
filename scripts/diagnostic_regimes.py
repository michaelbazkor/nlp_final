"""Row-level diagnostic for the recreated best-gap split. Fold 0, including LoRA."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WORKSPACE, "MulTaBench"))

from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.datasets.objects import SupervisedTask
from multabench.e5.constants import E5_SMALL_V2
from tabstar.preprocessing.splits import split_to_test

DATA_PATH = os.path.join(WORKSPACE, "data", "BIN_TEXT_SEC_MDA_DISTRESS", "data.csv")
OUT_DIR = os.path.join(WORKSPACE, "results", "gap_search")
OUT_PATH = os.path.join(OUT_DIR, "nocapex_diagnostic_predictions.csv")
TEXT_COLUMN = "risk_excerpt"
TARGET = "distress_next_year"
MODELS = [LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5]
CONDITIONS = ("structured", "text_only", "joint_frozen", "joint_tar")


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


def fit_predict(model_cls, x_train, y_train, x_test, device, tune_e5: bool) -> np.ndarray:
    model = model_cls(
        problem_type=SupervisedTask.BINARY,
        device=device,
        tune_e5=tune_e5,
        e5_train_kwargs=e5_kwargs() if tune_e5 else None,
        e5_model_name=E5_SMALL_V2,
        verbose=False,
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def columns_for(condition: str, features: list[str]) -> list[str]:
    if condition == "structured":
        return features
    if condition == "text_only":
        return [TEXT_COLUMN]
    return features + [TEXT_COLUMN]


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    frame = pd.read_csv(DATA_PATH)
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(object)
    features = [column for column in frame.columns if column not in {TEXT_COLUMN, TARGET}]
    y = frame[TARGET].astype(int)
    x = frame.drop(columns=[TARGET])
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=True, fold=0)
    print(f"device {device} train {len(y_train)} test {len(y_test)} features {features}", flush=True)

    jobs = [(model, condition) for model in MODELS for condition in CONDITIONS]
    total = len(jobs)
    next_mark = 10
    predictions = x_test[features].copy()
    predictions[TARGET] = y_test.to_numpy()
    predictions["word_count"] = x_test[TEXT_COLUMN].fillna("").map(lambda text: len(str(text).split()))
    predictions["sentence_count"] = x_test[TEXT_COLUMN].fillna("").map(
        lambda text: len([part for part in str(text).split(".") if part.strip()])
    )
    print(f"progress 0% (0/{total})", flush=True)
    for index, (model_cls, condition) in enumerate(jobs, start=1):
        cols = columns_for(condition, features)
        probs = fit_predict(
            model_cls,
            x_train[cols],
            y_train,
            x_test[cols],
            device,
            tune_e5=condition == "joint_tar",
        )
        predictions[f"{model_cls.SHORT_NAME}_{condition}"] = probs
        predictions.to_csv(OUT_PATH, index=False)
        percent = 100 * index // total
        if percent >= next_mark:
            print(f"progress {next_mark}% ({index}/{total}) {model_cls.SHORT_NAME} {condition}", flush=True)
            next_mark += 10
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
