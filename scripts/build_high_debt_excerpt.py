"""Rebuild the seed-42 sample and rewrite excerpts only where debt-to-assets > 0.8.

The 1,000 rows, the 25/55 label, and the other columns stay aligned with the
archived v3 table. High-debt rows keep up to 10 leverage sentences that already
passed the frequency cut. Rows with none of those sentences keep the original excerpt.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE = os.path.join(
    WORKSPACE,
    "results",
    "BIN_TEXT_SEC_MDA_DISTRESS_v3_recreate",
    "source_table",
    "data.csv",
)
OUT_PATH = os.path.join(WORKSPACE, "data", "BIN_TEXT_SEC_MDA_DISTRESS", "pending_high_debt_excerpt.csv")
COLUMNS = [
    "log_total_assets",
    "debt_to_assets",
    "current_ratio",
    "cash_to_assets",
    "asset_turnover",
    "capex_to_assets",
    "fiscal_year",
    "risk_excerpt",
    "distress_next_year",
]


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_sec_mda",
        os.path.join(WORKSPACE, "curation", "build_sec_mda.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    builder = load_builder()
    builder.MAX_DEBT_TO_ASSETS = None
    print("progress 0% rebuild", flush=True)
    filings = builder.add_ratios(builder.load_financials())
    years = list(builder.TEXT_YEARS)
    frames = []
    for index, year in enumerate(years, start=1):
        frames.append(builder.load_edgar_year(year))
        print(f"progress {100 * index // len(years)}% edgar {year}", flush=True)
    text = pd.concat(frames, ignore_index=True)
    text, _info = builder.apply_sentence_idf(text)
    shift, _overlaps = builder.choose_year_shift(filings, text)
    text["fy"] = text["edgar_year"] + shift
    merged = filings.merge(
        text[["cik", "fy", "risk_excerpt", "risk_words", "section_words", "lead_blob"]],
        on=["cik", "fy"],
        how="inner",
    )
    required = list(builder.NUMERIC_COLUMNS)
    eligible = builder.complete_cases(merged, required)
    curated, _boundary = builder.purify_boundary(eligible)
    if len(curated) > builder.MAX_ROWS:
        curated = builder.random_sample(curated, builder.MAX_ROWS)
    builder.winsorize(curated, required)
    curated["fiscal_year"] = curated["fy"].astype(int)
    curated["distress_next_year"] = curated["distress_next_year"].astype(int)

    archive = pd.read_csv(ARCHIVE)
    if list(archive.columns) != COLUMNS or len(archive) != len(curated):
        raise SystemExit(f"archive shape {archive.shape} vs rebuilt {len(curated)}")
    rebuilt = curated[COLUMNS].reset_index(drop=True)
    excerpt_mismatch = int((rebuilt["risk_excerpt"].fillna("") != archive["risk_excerpt"].fillna("")).sum())
    label_mismatch = int((rebuilt["distress_next_year"].to_numpy() != archive["distress_next_year"].to_numpy()).sum())
    ratio_mismatch = int(
        (
            ~np.isclose(
                rebuilt[required + ["fiscal_year"]].to_numpy(dtype=float),
                archive[required + ["fiscal_year"]].to_numpy(dtype=float),
                rtol=1e-5,
                atol=1e-5,
            )
        ).any(axis=1).sum()
    )
    print(
        f"alignment excerpts {excerpt_mismatch} labels {label_mismatch} ratios {ratio_mismatch}",
        flush=True,
    )
    if excerpt_mismatch or label_mismatch or ratio_mismatch:
        raise SystemExit("rebuilt sample does not match the archived v3 table")

    high = archive["debt_to_assets"] > 0.8
    rewritten = []
    changed = 0
    empty_fallback = 0
    leverage_counts = []
    for blob, original, flag in zip(curated["lead_blob"], archive["risk_excerpt"], high):
        if not flag:
            rewritten.append(original)
            continue
        leverage_counts.append(len(builder.LEVERAGE_TERM_RE.findall(str(blob or ""))))
        updated = builder.leverage_excerpt(blob, original)
        if updated == original:
            empty_fallback += 1
        else:
            changed += 1
        rewritten.append(updated)
    output = archive.copy()
    output["risk_excerpt"] = rewritten
    output.to_csv(OUT_PATH, index=False)
    old_words = archive.loc[high, "risk_excerpt"].fillna("").map(lambda text: len(str(text).split()))
    new_words = output.loc[high, "risk_excerpt"].fillna("").map(lambda text: len(str(text).split()))
    stats = {
        "rows": int(len(output)),
        "high_debt_rows": int(high.sum()),
        "excerpts_rewritten": changed,
        "high_debt_kept_original": empty_fallback,
        "high_debt_rows_with_a_leverage_term": int(sum(count > 0 for count in leverage_counts)),
        "median_leverage_term_hits": float(pd.Series(leverage_counts).median()) if leverage_counts else 0.0,
        "high_debt_median_words_before": float(old_words.median()),
        "high_debt_median_words_after": float(new_words.median()),
        "positive_rate": float(output["distress_next_year"].mean()),
    }
    stats_path = OUT_PATH.replace(".csv", ".json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
