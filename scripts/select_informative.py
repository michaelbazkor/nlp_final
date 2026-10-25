"""Pick Item 1A sentences using filings outside the evaluation sample.

A TF-IDF model is fit on the 3,803 firm-years that were not drawn into the
seed-42 table. Each evaluation filing then keeps the six lead sentences whose
wording that model finds most informative. The evaluation labels are not used
to choose the words or the sentences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))

from lexicon_screen import text_auc  # noqa: E402
from rebuild_lexicon_table import load_builder, load_edgar_year, log  # noqa: E402

OUT_DIR = WORKSPACE / "results" / "gap_search" / "five_fold_loop"
KEYS = OUT_DIR / "same_rows_keys.csv"
MAX_SENTENCES = 6


def build_pool(builder) -> pd.DataFrame:
    cached = OUT_DIR / "pool_pre_sample.parquet"
    if cached.exists():
        log("pool cached")
        return pd.read_parquet(cached)
    filings = builder.add_ratios(builder.load_financials())
    text = pd.concat([load_edgar_year(builder, year) for year in builder.TEXT_YEARS], ignore_index=True)
    text, _info = builder.apply_sentence_idf(text)
    shift, _overlaps = builder.choose_year_shift(filings, text)
    text["fy"] = text["edgar_year"] + shift
    merged = filings.merge(
        text[["cik", "fy", "risk_excerpt", "risk_words", "section_words", "survivor_blob"]],
        on=["cik", "fy"],
        how="inner",
    )
    eligible = builder.complete_cases(merged, list(builder.NUMERIC_COLUMNS))
    pool, _boundary = builder.purify_boundary(eligible)
    pool = pool.copy()
    pool["distress_next_year"] = pool["distress_next_year"].astype(int)
    pool.to_parquet(cached, index=False)
    log(f"pool rows {len(pool)}")
    return pool


def informative_excerpt(sentences: list[str], probabilities: np.ndarray) -> str:
    if not sentences:
        return ""
    order = np.argsort(-np.abs(probabilities - 0.5))
    picked = sorted(int(index) for index in order[:MAX_SENTENCES])
    chosen = [sentences[index] for index in picked if abs(float(probabilities[index]) - 0.5) >= 0.02]
    if len(chosen) < 2:
        return ""
    return " ".join(chosen)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builder = load_builder()
    pool = build_pool(builder)
    keys = pd.read_csv(KEYS, dtype={"cik": str})
    keys["cik"] = keys["cik"].str.zfill(10)
    pool["cik"] = pool["cik"].astype(str).str.zfill(10)
    pool["fiscal_year"] = pool["fy"].astype(int)
    marked = pool.merge(keys[["cik", "fiscal_year"]].assign(in_eval=1), on=["cik", "fiscal_year"], how="left")
    eval_rows = marked[marked["in_eval"].eq(1)].copy()
    complement = marked[marked["in_eval"].isna()].copy()
    log(f"eval {len(eval_rows)} complement {len(complement)}")
    if len(eval_rows) != 1000 or complement["distress_next_year"].nunique() < 2:
        raise SystemExit(f"unexpected split eval {len(eval_rows)} complement {len(complement)}")

    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=30000)
    matrix = vectorizer.fit_transform(complement["risk_excerpt"].fillna("").astype(str))
    model = LogisticRegression(max_iter=400, solver="liblinear")
    model.fit(matrix, complement["distress_next_year"].astype(int))

    excerpts = []
    for blob in eval_rows["survivor_blob"].fillna(""):
        sentences = [part.strip() for part in str(blob).split("\n") if part.strip()]
        if not sentences:
            excerpts.append("")
            continue
        probabilities = model.predict_proba(vectorizer.transform(sentences))[:, 1]
        excerpts.append(informative_excerpt(sentences, probabilities))
    eval_rows = eval_rows.copy()
    eval_rows["informative_excerpt"] = excerpts
    eval_rows["fiscal_year"] = eval_rows["fy"].astype(int)
    replaced = eval_rows["informative_excerpt"].str.len() > 0
    log(f"informative excerpts {int(replaced.sum())}/{len(eval_rows)}")

    # Preserve the official row order.
    ordered = keys.merge(
        eval_rows[["cik", "fiscal_year", "informative_excerpt", "risk_excerpt", "distress_next_year"] + [
            column for column in builder.NUMERIC_COLUMNS if column in eval_rows.columns
        ]],
        on=["cik", "fiscal_year"],
        how="left",
        suffixes=("", "_pool"),
    )
    if len(ordered) != 1000 or ordered["distress_next_year"].isna().any():
        raise SystemExit("could not realign the evaluation rows")
    # Winsorized ratios live on the already screened table. Reuse that file's numbers
    # and swap only the text, so the structured columns stay identical.
    base = pd.read_csv(OUT_DIR / "same_rows_original.csv")
    table = base.copy()
    table["risk_excerpt"] = np.where(replaced.to_numpy(), eval_rows["informative_excerpt"].to_numpy(), base["risk_excerpt"])
    # eval_rows order is not the official order. Use ordered instead.
    table["risk_excerpt"] = np.where(
        ordered["informative_excerpt"].fillna("").str.len().to_numpy() > 0,
        ordered["informative_excerpt"].fillna("").to_numpy(),
        base["risk_excerpt"].to_numpy(),
    )
    table.to_csv(OUT_DIR / "informative_lexicon.csv", index=False)
    scores = text_auc(table, "risk_excerpt")
    summary = {
        "table": "informative",
        "row_mean": round(sum(scores) / len(scores), 4),
        "row_min": round(min(scores), 4),
        "row_folds": [round(score, 4) for score in scores],
        "replaced": int((ordered["informative_excerpt"].fillna("").str.len() > 0).sum()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "informative_screen.csv", index=False)
    log(str(summary))


if __name__ == "__main__":
    main()
