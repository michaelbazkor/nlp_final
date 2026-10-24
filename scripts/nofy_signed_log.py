"""Fold 0 and fold 1 for the two changes that beat the baseline average.

Drop fiscal year, then signed-log the skewed ratios. Same 1,000 capex-dropped rows.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "MulTaBench"))

from fold1_screen import (  # noqa: E402
    CONDITIONS,
    DATASET,
    DATA_PATH,
    MODELS,
    e5_kwargs,
    gaps_from_scores,
    two_fold_mean,
)
from gap_search import drop_fiscal_year, signed_log_skewed  # noqa: E402
from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset  # noqa: E402
from multabench.e5.constants import E5_SMALL_V2  # noqa: E402

BASELINE_PATH = WORKSPACE / "results" / "gap_search" / "best_data.csv"
OUT_DIR = WORKSPACE / "results" / "gap_search" / "fold1_screen" / "nofy_signed_log"
FOLDS = [0, 1]


def log(message: str) -> None:
    print(message, flush=True)
    with (OUT_DIR / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "json").mkdir(exist_ok=True)
    baseline = pd.read_csv(BASELINE_PATH)
    table = signed_log_skewed(drop_fiscal_year(baseline))
    table.to_csv(DATA_PATH, index=False)
    jobs = [
        (fold, model_cls, condition, state, tune)
        for fold in FOLDS
        for model_cls in MODELS
        for condition, state, tune in CONDITIONS
    ]
    log(f"device {device} jobs {len(jobs)}")
    try:
        for index, (fold, model_cls, condition, state, tune) in enumerate(jobs, start=1):
            dest = OUT_DIR / "json" / f"{model_cls.SHORT_NAME}_{condition}_fold{fold}.json"
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
            percent = 100 * index // len(jobs)
            if percent % 10 == 0:
                log(f"progress {percent}% ({index}/{len(jobs)}) fold {fold} {model_cls.SHORT_NAME} {condition}")
        scores = {}
        for fold in FOLDS:
            scores[fold] = {}
            for model in MODELS:
                scores[fold][model.SHORT_NAME] = {}
                for condition, _, _ in CONDITIONS:
                    path = OUT_DIR / "json" / f"{model.SHORT_NAME}_{condition}_fold{fold}.json"
                    with path.open(encoding="utf-8") as handle:
                        scores[fold][model.SHORT_NAME][condition] = float(json.load(handle)["test_score"])
        summary = two_fold_mean(scores[0], scores[1])
        summary.to_csv(OUT_DIR / "two_fold_summary.csv", index=False)
        average = summary.loc[summary["model"] == "average"].iloc[0]
        log(summary.to_string(index=False))
        log(
            f"two-fold delta_joint {average['delta_joint']:.4f} "
            f"delta_awareness {average['delta_awareness']:.4f} "
            f"sum {average['delta_joint'] + average['delta_awareness']:.4f}"
        )
    finally:
        baseline.to_csv(DATA_PATH, index=False)
        log("restored capex-dropped table")


if __name__ == "__main__":
    main()
