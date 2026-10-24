"""Fold 3 for the fiscal-year drop and for that drop plus the signed-log.

The mean of folds 0 through 3 is rounded to 3 decimals before the gaps.
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

from fold1_screen import (  # noqa: E402
    CONDITIONS,
    DATASET,
    DATA_PATH,
    MODELS,
    SEARCH_DIR,
    e5_kwargs,
    gaps_from_scores,
)
from gap_search import drop_fiscal_year, signed_log_skewed  # noqa: E402
from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset  # noqa: E402
from multabench.e5.constants import E5_SMALL_V2  # noqa: E402

BASELINE_PATH = WORKSPACE / "results" / "gap_search" / "best_data.csv"
SCREEN = SEARCH_DIR / "fold1_screen"
FOLD = 3


def both_changes(frame: pd.DataFrame) -> pd.DataFrame:
    return signed_log_skewed(drop_fiscal_year(frame))


VARIANTS = [
    ("drop_fiscal_year", drop_fiscal_year, SEARCH_DIR / "drop_fiscal_year"),
    ("nofy_signed_log", both_changes, SCREEN / "nofy_signed_log"),
]


def log(message: str) -> None:
    print(message, flush=True)
    with (SCREEN / "fold3_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def read_score(directory: Path, model: str, condition: str, fold: int) -> float:
    path = directory / "json" / f"{model}_{condition}_fold{fold}.json"
    with path.open(encoding="utf-8") as handle:
        return float(json.load(handle)["test_score"])


def four_fold_summary(fold0_dir: Path, screen_dir: Path) -> pd.DataFrame:
    merged = {}
    for model in MODELS:
        name = model.SHORT_NAME
        merged[name] = {}
        for condition, _, _ in CONDITIONS:
            values = [read_score(fold0_dir, name, condition, 0)]
            values.extend(read_score(screen_dir, name, condition, fold) for fold in (1, 2, 3))
            merged[name][condition] = mean(values)
    return gaps_from_scores(merged)


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = pd.read_csv(BASELINE_PATH)
    total = len(VARIANTS) * len(MODELS) * len(CONDITIONS)
    done = 0
    log(f"device {device} fold {FOLD} jobs {total}")
    try:
        for name, transform, fold0_dir in VARIANTS:
            screen_dir = SCREEN / name
            (screen_dir / "json").mkdir(parents=True, exist_ok=True)
            transform(baseline).to_csv(DATA_PATH, index=False)
            log(f"start {name}")
            for model_cls in MODELS:
                for condition, state, tune in CONDITIONS:
                    done += 1
                    dest = screen_dir / "json" / f"{model_cls.SHORT_NAME}_{condition}_fold{FOLD}.json"
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
                    percent = 100 * done // total
                    if percent % 10 == 0:
                        log(f"progress {percent}% ({done}/{total}) {name} {model_cls.SHORT_NAME} {condition}")
            summary = four_fold_summary(fold0_dir, screen_dir)
            summary.to_csv(screen_dir / "four_fold_summary.csv", index=False)
            average = summary.loc[summary["model"] == "average"].iloc[0]
            log(
                f"{name} four-fold delta_joint {average['delta_joint']:.4f} "
                f"delta_awareness {average['delta_awareness']:.4f} "
                f"sum {average['delta_joint'] + average['delta_awareness']:.4f}"
            )
    finally:
        baseline.to_csv(DATA_PATH, index=False)
        log("restored capex-dropped table")


if __name__ == "__main__":
    main()
