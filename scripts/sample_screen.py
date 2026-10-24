"""Screen several 1000-row samples with frozen models only. No LoRA."""

from __future__ import annotations

import importlib.util
import os
import sys

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

OUT_PATH = os.path.join(
    WORKSPACE,
    "results",
    f"sample_screen_{os.environ.get('SCREEN_POS', '25')}_{os.environ.get('SCREEN_NEG', '55')}.csv",
)
TEXT_COLUMN = "risk_excerpt"
TARGET = "distress_next_year"
SEEDS = [0, 1, 2, 7, 11, 21, 42, 99]
MODELS = [LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5]
EMBEDDING_CACHE: dict[str, object] = {}


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_sec_mda",
        os.path.join(WORKSPACE, "curation", "build_sec_mda.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    import numpy as np

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


def prepare_pool(builder):
    builder.POSITIVE_TAIL = float(os.environ.get("SCREEN_POS", builder.POSITIVE_TAIL))
    builder.NEGATIVE_TAIL = float(os.environ.get("SCREEN_NEG", builder.NEGATIVE_TAIL))
    builder.EXTREME_TRIM = float(os.environ.get("SCREEN_TRIM", builder.EXTREME_TRIM))
    filings = builder.add_ratios(builder.load_financials())
    text = pd.concat([builder.load_edgar_year(year) for year in builder.TEXT_YEARS], ignore_index=True)
    text, _info = builder.apply_sentence_idf(text)
    shift, _overlaps = builder.choose_year_shift(filings, text)
    text["fy"] = text["edgar_year"] + shift
    merged = filings.merge(
        text[["cik", "fy", "risk_excerpt", "risk_words", "section_words"]],
        on=["cik", "fy"],
        how="inner",
    )
    eligible = builder.complete_cases(merged, list(builder.NUMERIC_COLUMNS))
    curated, _boundary = builder.purify_boundary(eligible)
    curated = curated.sort_values(["cik", "fy"]).reset_index(drop=True)
    curated["fiscal_year"] = curated["fy"].astype(int)
    print(f"pool {len(curated)}", flush=True)
    return builder, curated


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    text_embeddings.encode_texts_with_e5 = cached_encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder, pool = prepare_pool(load_builder())
    numeric = list(builder.NUMERIC_COLUMNS)
    columns = [
        "log_total_assets",
        "debt_to_assets",
        "current_ratio",
        "cash_to_assets",
        "asset_turnover",
        "capex_to_assets",
        "fiscal_year",
        TEXT_COLUMN,
        TARGET,
    ]
    rows = []
    total = len(SEEDS)
    next_mark = 10
    print(f"progress 0% (0/{total})", flush=True)
    for index, seed in enumerate(SEEDS, start=1):
        sample = pool.sample(n=1000, random_state=seed).reset_index(drop=True)
        builder.winsorize(sample, numeric)
        sample[TEXT_COLUMN] = sample[TEXT_COLUMN].astype(object)
        y = sample[TARGET].astype(int)
        x = sample[columns].drop(columns=[TARGET])
        x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=True, fold=0)
        scores = {}
        for model_cls in MODELS:
            name = model_cls.SHORT_NAME
            structured = auc(
                model_cls,
                x_train.drop(columns=[TEXT_COLUMN]),
                y_train,
                x_test.drop(columns=[TEXT_COLUMN]),
                y_test,
                device,
            )
            text_only = auc(
                model_cls,
                x_train[[TEXT_COLUMN]],
                y_train,
                x_test[[TEXT_COLUMN]],
                y_test,
                device,
            )
            joint = auc(model_cls, x_train, y_train, x_test, y_test, device)
            delta = round(joint - max(structured, text_only), 3)
            scores[name] = delta
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "structured": structured,
                    "text_only": text_only,
                    "joint_frozen": joint,
                    "delta_joint": delta,
                }
            )
        print(
            f"seed {seed} mean_delta {sum(scores.values()) / len(scores):.3f} {scores}",
            flush=True,
        )
        percent = 100 * index // total
        if percent >= next_mark:
            print(f"progress {next_mark}% ({index}/{total})", flush=True)
            next_mark += 10
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
