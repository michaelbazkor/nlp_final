"""Column-subset search using frozen E5 embeddings and the official learners.

Earlier LoRA adapters were deleted with their temporary training folders, so this
does not train or reload a LoRA. Frozen embeddings are computed once and reused.
Each learner still applies its own validation split and PCA, matching the benchmark.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WORKSPACE, "MulTaBench"))

import multabench.baselines.preprocessing.text_embeddings as text_embeddings
import multabench.e5.e5_finetune as e5_finetune
from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.datasets.objects import SupervisedTask
from multabench.e5.constants import E5_SMALL_V2
from tabstar.preprocessing.splits import split_to_test

DATA_PATH = os.path.join(WORKSPACE, "data", "BIN_TEXT_SEC_MDA_DISTRESS", "data.csv")
OUT_PATH = os.path.join(WORKSPACE, "results", "BIN_TEXT_SEC_MDA_DISTRESS", "column_search.csv")
TEXT_COLUMN = "risk_excerpt"
TARGET = "distress_next_year"
FOLD = 0
MODELS = [LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5]
EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def cached_encode(texts, col_name, model, tokenizer, device, batch_size=32, max_length=512):
    missing = [text for text in texts if text not in EMBEDDING_CACHE]
    if missing:
        vectors = e5_finetune.encode_texts_with_e5(
            texts=missing,
            col_name=col_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        for text, vector in zip(missing, vectors):
            EMBEDDING_CACHE[text] = vector
    return np.vstack([EMBEDDING_CACHE[text] for text in texts])


def auc(model_cls, x_train, y_train, x_test, y_test, device) -> float:
    model = model_cls(
        problem_type=SupervisedTask.BINARY,
        device=device,
        tune_e5=False,
        e5_model_name=E5_SMALL_V2,
        verbose=False,
    )
    model.fit(x_train, y_train)
    return round(float(model.score_all_metrics(X=x_test, y=y_test).score), 3)


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    text_embeddings.encode_texts_with_e5 = cached_encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(DATA_PATH)
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(object)
    y = frame[TARGET].astype(int)
    x = frame.drop(columns=[TARGET])
    structured = [column for column in x.columns if column != TEXT_COLUMN]
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=True, fold=FOLD)
    print(f"device {device} train {len(y_train)} test {len(y_test)}", flush=True)

    rows = []
    text_only_scores = {}
    for model_cls in MODELS:
        score = auc(model_cls, x_train[[TEXT_COLUMN]], y_train, x_test[[TEXT_COLUMN]], y_test, device)
        text_only_scores[model_cls.SHORT_NAME] = score
        print(f"text_only {model_cls.SHORT_NAME} {score}", flush=True)

    sizes = {int(arg) for arg in sys.argv[1:]}
    subsets = []
    for size in range(1, len(structured) + 1):
        if sizes and size not in sizes:
            continue
        subsets.extend(itertools.combinations(structured, size))
    if not sizes:
        subsets = [tuple(structured)] + [subset for subset in subsets if len(subset) != len(structured)]
    out_path = OUT_PATH
    if sizes:
        suffix = "_".join(str(size) for size in sorted(sizes))
        out_path = OUT_PATH.replace(".csv", f"_{suffix}.csv")
    total = len(subsets)
    next_mark = 10
    print(f"progress 0% (0/{total})", flush=True)
    for index, subset in enumerate(subsets, start=1):
        columns = list(subset)
        joint_columns = columns + [TEXT_COLUMN]
        deltas = []
        for model_cls in MODELS:
            name = model_cls.SHORT_NAME
            structured_score = auc(model_cls, x_train[columns], y_train, x_test[columns], y_test, device)
            joint_score = auc(
                model_cls,
                x_train[joint_columns],
                y_train,
                x_test[joint_columns],
                y_test,
                device,
            )
            text_score = text_only_scores[name]
            delta = round(joint_score - max(structured_score, text_score), 6)
            deltas.append(delta)
            rows.append(
                {
                    "subset": "|".join(columns),
                    "n_cols": len(columns),
                    "model": name,
                    "structured": structured_score,
                    "text_only": text_score,
                    "joint_frozen": joint_score,
                    "delta_joint": delta,
                }
            )
        percent = 100 * index // total
        if percent >= next_mark or index == total:
            print(
                f"progress {min(percent, 100)}% ({index}/{total}) n={len(columns)} mean_delta {float(np.mean(deltas)):.3f}",
                flush=True,
            )
            while next_mark <= percent:
                next_mark += 10
        pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
