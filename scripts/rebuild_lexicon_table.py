"""Rebuild the seed-42 distress table and attach longer finance excerpts.

The local sentence file is not the current 1,000 rows. This script downloads
the SEC facts and EDGAR Item 1A text, repeats the original sample, and replaces
the ten short leads with up to six longer lexicon sentences. A one-firm table
is written as well. Both are scored with a five-fold text-only screen.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))
OUT_DIR = WORKSPACE / "results" / "gap_search" / "five_fold_loop"
LEXICON = re.compile(
    r"\b(?:debts?|indebtedness|covenants?|liquidity|defaults?|cash flows?|"
    r"margins?|customers?|suppliers?|demand|going concern|credit|"
    r"borrowings?|leverage)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MAX_SENTENCES = 6
MAX_SENTENCE_WORDS = 120
MAX_EXCERPT_WORDS = 380


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_sec_mda",
        WORKSPACE / "curation" / "build_sec_mda.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(message, flush=True)


def keep_long(builder, sentence: str) -> bool:
    words = sentence.split()
    if len(words) < builder.LEAD_MIN_WORDS or len(words) > MAX_SENTENCE_WORDS:
        return False
    lowered = sentence.lower()
    if any(phrase in lowered for phrase in builder.BOILERPLATE_PHRASES):
        return False
    return builder.numeric_fraction(sentence) < builder.LEAD_MAX_NUMERIC_FRACTION


def lexicon_excerpt(builder, section: object) -> str:
    if not isinstance(section, str) or not section.strip():
        return ""
    chosen = []
    seen: set[str] = set()
    words = 0
    for paragraph in section.split("\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in SENTENCE_SPLIT.split(paragraph):
            sentence = builder.DATE_PREFIX_RE.sub("", sentence).strip()
            if not keep_long(builder, sentence) or LEXICON.search(sentence) is None:
                continue
            key = builder.sentence_key(sentence)
            if not key or key in seen:
                continue
            seen.add(key)
            chosen.append(sentence)
            words += len(sentence.split())
            if len(chosen) >= MAX_SENTENCES or words >= MAX_EXCERPT_WORDS:
                return " ".join(chosen)
    if len(chosen) < 2:
        return ""
    return " ".join(chosen)


def load_edgar_year(builder, year: int) -> pd.DataFrame:
    cached = os.path.join(builder.EDGAR_DIR, f"{year}-section1a-lex.parquet")
    if os.path.exists(cached):
        log(f"edgar {year} cached")
        return pd.read_parquet(cached)
    rows = []
    for split in builder.EDGAR_SPLITS:
        dest = os.path.join(builder.EDGAR_DIR, str(year), f"{split}.jsonl")
        builder.download(builder.EDGAR_URL.format(year=year, split=split), dest)
        log(f"parse edgar {year} {split}")
        with open(dest, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                section = item.get("section_1A")
                rows.append(
                    {
                        "cik": builder._digits(item.get("cik")),
                        "edgar_year": int(item["year"]) if str(item.get("year", "")).isdigit() else year,
                        "lead_blob": builder.candidate_leads(section),
                        "section_words": builder.word_count(section),
                        "lexicon_excerpt": lexicon_excerpt(builder, section),
                    }
                )
    frame = pd.DataFrame(rows).dropna(subset=["cik"])
    frame = frame.sort_values("section_words", ascending=False).drop_duplicates(
        ["cik", "edgar_year"], keep="first"
    )
    os.makedirs(builder.EDGAR_DIR, exist_ok=True)
    frame.to_parquet(cached, index=False)
    log(f"edgar {year}: {len(frame)} filings")
    return frame


def champion_columns(builder, frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    from gap_search import drop_fiscal_year, signed_log_skewed

    table = frame.copy()
    table["risk_excerpt"] = table[text_column].where(table[text_column].astype(str).str.len() > 0, table["risk_excerpt"])
    table["distress_next_year"] = table["distress_next_year"].astype(int)
    keep = [column for column in builder.NUMERIC_COLUMNS if column not in builder.DROPPED_FEATURES]
    keep += ["fiscal_year", "risk_excerpt", "distress_next_year"]
    table = table[keep]
    return signed_log_skewed(drop_fiscal_year(table))


def screen(frame: pd.DataFrame, label: str) -> dict:
    from lexicon_screen import firm_holdout_auc, text_auc

    scored = frame.copy()
    if "cik" not in scored.columns:
        scored["cik"] = np.arange(len(scored)).astype(str)
    row_scores = text_auc(scored, "risk_excerpt")
    firm_scores = firm_holdout_auc(scored, "risk_excerpt") if scored["cik"].nunique() < len(scored) else row_scores
    return {
        "table": label,
        "row_mean": round(sum(row_scores) / len(row_scores), 4),
        "row_min": round(min(row_scores), 4),
        "row_folds": [round(score, 4) for score in row_scores],
        "firm_mean": round(float(pd.Series(firm_scores).mean()), 4),
        "firm_min": round(float(pd.Series(firm_scores).min()), 4),
        "firm_folds": [None if score != score else round(float(score), 4) for score in firm_scores],
    }


def sample_one_firm(builder, pool: pd.DataFrame) -> pd.DataFrame:
    capped, info = builder.cap_at_most(pool, 1)
    log(f"one firm per year cap {info}")
    if len(capped) > builder.MAX_ROWS:
        capped = builder.random_sample(capped, builder.MAX_ROWS)
    return capped.reset_index(drop=True)


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builder = load_builder()
    log("progress 0% financials")
    filings = builder.add_ratios(builder.load_financials())
    years = list(builder.TEXT_YEARS)
    frames = []
    for index, year in enumerate(years, start=1):
        frames.append(load_edgar_year(builder, year))
        log(f"progress {20 + 10 * index}% edgar {year}")
    text = pd.concat(frames, ignore_index=True)
    log("progress 70% idf")
    text, idf_info = builder.apply_sentence_idf(text)
    shift, overlaps = builder.choose_year_shift(filings, text)
    text["fy"] = text["edgar_year"] + shift
    merged = filings.merge(
        text[["cik", "fy", "risk_excerpt", "risk_words", "section_words", "lexicon_excerpt"]],
        on=["cik", "fy"],
        how="inner",
    )
    eligible = builder.complete_cases(merged, list(builder.NUMERIC_COLUMNS))
    curated, boundary_info = builder.purify_boundary(eligible)
    log(f"boundary {boundary_info} shift {shift} overlaps {overlaps} idf {idf_info['max_lead_sentences']}")
    if len(curated) > builder.MAX_ROWS:
        curated = builder.random_sample(curated, builder.MAX_ROWS)
    builder.winsorize(curated, list(builder.NUMERIC_COLUMNS))
    curated["fiscal_year"] = curated["fy"].astype(int)
    curated["distress_next_year"] = curated["distress_next_year"].astype(int)
    curated["lexicon_excerpt"] = curated["lexicon_excerpt"].fillna("")
    curated["use_lexicon"] = curated["lexicon_excerpt"].str.len() > 0

    same = champion_columns(builder, curated, "lexicon_excerpt")
    original = champion_columns(builder, curated, "risk_excerpt")
    same_keys = curated[["cik", "fiscal_year", "distress_next_year", "use_lexicon"]].copy()
    same.to_csv(OUT_DIR / "same_rows_lexicon.csv", index=False)
    original.to_csv(OUT_DIR / "same_rows_original.csv", index=False)
    same_keys.to_csv(OUT_DIR / "same_rows_keys.csv", index=False)

    pool, _boundary = builder.purify_boundary(eligible)
    one = sample_one_firm(builder, pool)
    builder.winsorize(one, list(builder.NUMERIC_COLUMNS))
    one["fiscal_year"] = one["fy"].astype(int)
    one["distress_next_year"] = one["distress_next_year"].astype(int)
    one["lexicon_excerpt"] = one["lexicon_excerpt"].fillna("")
    one_table = champion_columns(builder, one, "lexicon_excerpt")
    one_table.to_csv(OUT_DIR / "one_firm_lexicon.csv", index=False)
    one[["cik", "fiscal_year", "distress_next_year"]].to_csv(OUT_DIR / "one_firm_keys.csv", index=False)

    log("progress 90% text screen")
    screens = [
        screen(original.assign(cik=same_keys["cik"].values), "same_rows_original"),
        screen(same.assign(cik=same_keys["cik"].values), "same_rows_lexicon"),
        screen(one_table.assign(cik=one["cik"].values), "one_firm_lexicon"),
    ]
    shock = builder.shock_label(eligible)
    shock_frame = eligible.loc[shock.notna()].copy()
    shock_frame["distress_next_year"] = shock.loc[shock_frame.index].astype(int)
    if shock_frame["distress_next_year"].nunique() == 2 and len(shock_frame) > 200:
        shock_sample = shock_frame
        if len(shock_sample) > builder.MAX_ROWS:
            shock_sample = builder.random_sample(shock_sample, builder.MAX_ROWS)
        shock_sample = shock_sample.copy()
        shock_sample["lexicon_excerpt"] = shock_sample["lexicon_excerpt"].fillna("")
        shock_sample["fiscal_year"] = shock_sample["fy"].astype(int)
        shock_table = champion_columns(builder, shock_sample, "lexicon_excerpt")
        shock_table.to_csv(OUT_DIR / "shock_lexicon.csv", index=False)
        screens.append(screen(shock_table.assign(cik=shock_sample["cik"].values), "shock_lexicon"))
        log(f"shock positive rate {float(shock_sample['distress_next_year'].mean()):.3f} rows {len(shock_sample)}")
    summary = pd.DataFrame(screens)
    summary.to_csv(OUT_DIR / "rebuilt_text_screen.csv", index=False)
    log(summary.to_string(index=False))
    log(
        f"same-row lexicon coverage {int(curated['use_lexicon'].sum())}/{len(curated)} "
        f"positive {float(curated['distress_next_year'].mean()):.3f}"
    )
    log("progress 100% rebuild")


if __name__ == "__main__":
    main()
